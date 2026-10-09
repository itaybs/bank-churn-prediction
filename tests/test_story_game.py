"""The interactive story game: real-data payload, rendered page, and the JavaScript engine.

The JavaScript engine (churn/game_assets/engine.js) is executed with Node and compared with the
Python reference engine (churn/game.py), so the browser game cannot silently drift from the
tested money rules. Node-based tests are skipped if Node is not installed.
"""

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from churn.analysis import run_analysis
from churn.game import PERSONAS, Economics, badge, best_plan, outcomes, success_score, summarise
from churn.story_game import ASSETS, N_ROUNDS, THRESHOLD_GRID, build_payload, render_html, scale_grid

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
COLORS = {"CatBoost": "#e87ba4"}


@pytest.fixture(scope="module")
def analysis():
    return run_analysis(permutation_repeats=None)


@pytest.fixture(scope="module")
def payload(analysis):
    return build_payload(analysis, best="CatBoost", colors=COLORS)


def test_payload_customers_are_real_test_rows(payload, analysis):
    assert len(payload["rounds"]) == N_ROUNDS
    for rnd in payload["rounds"]:
        assert [c["key"] for c in rnd["customers"]] == [p.key for p in PERSONAS]
        for c in rnd["customers"]:
            assert set(c["p"]) == set(analysis.models)
            assert all(0 <= v <= 1 for v in c["p"].values())
            for key in ("idle", "click", "call", "waiver", "vip", "stay_targeted", "stay", "left_targeted", "left"):
                assert c["lines"][key]
    # A customer's probabilities and outcome are exactly the analysis' test-set values.
    first = payload["rounds"][0]["customers"][0]
    match = analysis.X_test[(analysis.X_test["Age"] == first["age"]) & (analysis.X_test["Balance"].round(2) == round(first["balance"], 2))]
    pos = analysis.X_test.index.get_loc(match.index[0])
    assert first["p"]["CatBoost"] == pytest.approx(analysis.test_scores["CatBoost"][pos])
    assert first["churned"] == bool(analysis.y_test.iloc[pos])


def test_payload_models_and_scale_grid_are_consistent(payload, analysis):
    assert payload["best"] == "CatBoost" and len(payload["models"]) == len(analysis.models)
    for m in payload["models"]:
        assert m["threshold"] == pytest.approx(analysis.tuned_thresholds[m["name"]])
    n, churners = payload["scale"]["n"], payload["scale"]["churners"]
    assert n == len(analysis.y_test) and churners == int(analysis.y_test.sum())
    for name, grid in payload["scale"]["models"].items():
        assert set(THRESHOLD_GRID.tolist()) <= set(grid["thresholds"])
        assert analysis.tuned_thresholds[name] in grid["thresholds"]  # the exact optimum, not a rounded step
        assert grid["thresholds"] == sorted(grid["thresholds"])
        tp, fp, fn, tn = (np.array(grid[k]) for k in ("tp", "fp", "fn", "tn"))
        assert (tp + fn == churners).all() and (tp + fp + fn + tn == n).all()
        assert (np.diff(tp + fp) <= 0).all()  # a higher threshold never flags more customers
    assert {i["id"] for i in payload["econ"]["incentives"]} == {"call", "waiver", "vip"}


def test_scale_grid_counts_match_a_direct_computation():
    scores, churned, values = np.array([0.9, 0.6, 0.3, 0.1]), np.array([1, 0, 1, 0]), np.array([10.0, 20, 30, 40])
    g = scale_grid(scores, churned, values)
    i = g["thresholds"].index(0.5)
    assert (g["tp"][i], g["fp"][i], g["fn"][i], g["tn"][i]) == (1, 1, 1, 1)
    assert g["v_tp"][i] == 10 and g["v_fn"][i] == 30


def test_rendered_page_is_self_contained(payload):
    html = render_html(payload)
    assert "/*__ENGINE__*/" not in html and "/*__PAYLOAD__*/" not in html
    assert "The Chief Retention Officer" in html and "window.Engine" not in html.split("<script>")[0]
    data = re.search(r"const DATA = (\{.*?\});\nconst E = ", html, re.S).group(1)
    assert json.loads(data.replace("<\\/", "</"))["best"] == "CatBoost"
    assert not re.search(r"""<(script|link)[^>]+(src|href)=["']https?:""", html)  # no external requests


@needs_node
def test_rendered_scripts_parse(payload, tmp_path):
    html = render_html(payload)
    for k, script in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
        path = tmp_path / f"s{k}.js"
        path.write_text(script, encoding="utf-8")
        run = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr


def _node(code: str) -> dict:
    engine = (ASSETS / "engine.js").read_text(encoding="utf-8")
    # Under Node the engine exports itself through module.exports (in the browser it sets window.Engine).
    script = engine + "\nconst Engine = module.exports;\n" + code
    run = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@needs_node
def test_js_engine_matches_python_engine():
    py_econ = Economics()
    js_econ = {"incentives": [{"id": k, "cost": i.cost, "success": i.success}
                              for k, i in zip(["call", "waiver", "vip"], py_econ.incentives.values())]}
    ids = dict(zip(["none", "call", "waiver", "vip"], py_econ.options()))
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(25):
        churned = rng.integers(0, 2, 5).astype(bool).tolist()
        values = rng.uniform(3_000, 12_000, 5).round(2).tolist()
        weights = rng.uniform(0, 1, 5).round(4).tolist()
        decisions = rng.choice(list(ids), 5).tolist()
        cases.append(dict(churned=churned, values=values, weights=weights, decisions=decisions))
    js = _node(f"""
        const econ = {json.dumps(js_econ)}; const cases = {json.dumps(cases)};
        console.log(JSON.stringify(cases.map((c) => ({{
          summary: Engine.summarise(c.decisions, c.churned, c.values, econ),
          plan: Engine.bestPlan(c.weights, c.values, econ, 10000),
          perfect: Engine.bestPlan(c.churned.map(Number), c.values, econ, 10000),
        }}))));""")
    for case, out in zip(cases, js):
        py = summarise(outcomes([ids[d] for d in case["decisions"]], case["churned"], case["values"], py_econ))
        for key in ("spend", "true_savings", "wasted", "lost_revenue", "net", "roi"):
            assert out["summary"][key] == pytest.approx(py[key], abs=1e-6)
        assert [ids[d] for d in out["plan"]] == best_plan(case["weights"], case["values"], py_econ, 10_000)
        assert [ids[d] for d in out["perfect"]] == best_plan(case["churned"], case["values"], py_econ, 10_000)

    scores = _node("""console.log(JSON.stringify([[50,100],[150,100],[-5,100],[3,0],[-3,0],[85,100],[10,500]].map(
        ([n, b]) => [Engine.successScore(n, b), Engine.badge(Engine.successScore(n, b), n)[1]])));""")
    for (net, best), (js_score, js_title) in zip([(50, 100), (150, 100), (-5, 100), (3, 0), (-3, 0), (85, 100), (10, 500)], scores):
        assert js_score == success_score(net, best) and js_title == badge(success_score(net, best), net)[1]


@needs_node
def test_js_scale_stats_match_real_test_counts(payload, analysis):
    grid = payload["scale"]["models"]["CatBoost"]
    waiver = next(i for i in payload["econ"]["incentives"] if i["id"] == "waiver")
    t = payload["models"][[m["name"] for m in payload["models"]].index("CatBoost")]["threshold"]
    s = _node(f"console.log(JSON.stringify(Engine.scaleStats({json.dumps(grid)}, {t}, {json.dumps(waiver)})))")
    scores, y = analysis.test_scores["CatBoost"], analysis.y_test.to_numpy().astype(bool)
    flagged = scores >= s["threshold"]
    assert s["tp"] == int((flagged & y).sum()) and s["fp"] == int((flagged & ~y).sum())
    assert s["recall"] == pytest.approx((flagged & y).sum() / y.sum())
    assert s["precision"] == pytest.approx((flagged & y).sum() / flagged.sum())
    values = Economics().customer_value(analysis.X_test).to_numpy()
    assert s["net"] == pytest.approx(waiver["success"] * values[flagged & y].sum() - waiver["cost"] * flagged.sum(), abs=0.05)

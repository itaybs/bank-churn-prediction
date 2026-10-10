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


def test_game_page_is_responsive(payload):
    """Mobile-friendly by construction: viewport meta, phone/tablet breakpoints, no fixed overlay."""
    html = render_html(payload)
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    for breakpoint in ("max-width: 1180px", "max-width: 860px", "max-width: 560px", "max-width: 480px"):
        assert f"@media ({breakpoint})" in html
    assert "position: fixed" not in html  # the debrief opens inline, so it works with an auto-height iframe
    assert "max-width: 100%" in html and "minmax(0, 1fr)" in html


def test_assets_fingerprint_tracks_the_game_files():
    from churn.story_game import assets_fingerprint

    fp = assets_fingerprint()
    assert len(fp) == 16 and fp == assets_fingerprint()
    assert all(c in "0123456789abcdef" for c in fp)


@pytest.fixture(scope="module")
def payload_he(analysis):
    return build_payload(analysis, best="CatBoost", colors=COLORS, lang="he")


def test_hebrew_payload_changes_text_not_numbers(payload, payload_he):
    assert payload["lang"] == "en" and payload_he["lang"] == "he"
    for rnd_en, rnd_he in zip(payload["rounds"], payload_he["rounds"]):
        for c_en, c_he in zip(rnd_en["customers"], rnd_he["customers"]):
            assert c_en["p"] == c_he["p"] and c_en["churned"] == c_he["churned"] and c_en["value"] == c_he["value"]
            assert c_en["name"] != c_he["name"] and c_he["lines"]["idle"] != c_en["lines"]["idle"]
    assert payload["scale"] == payload_he["scale"]
    assert [i["cost"] for i in payload["econ"]["incentives"]] == [i["cost"] for i in payload_he["econ"]["incentives"]]
    assert payload_he["econ"]["incentives"][0]["name"] == "שיחה אישית"


@needs_node
def test_rendered_hebrew_scripts_parse(payload_he, tmp_path):
    html = render_html(payload_he)
    assert "/*__I18N__*/" not in html and '"lang": "he"' in html
    for k, script in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
        path = tmp_path / f"he{k}.js"
        path.write_text(script, encoding="utf-8")
        run = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr


@needs_node
def test_game_text_is_complete_in_both_languages():
    """Every interface string exists in English and Hebrew, and every text function renders in both."""
    i18n = (ASSETS / "i18n.js").read_text(encoding="utf-8")
    script = i18n + r"""
const I = module.exports, out = { missing: [], extra: [], errors: [] };
const en = Object.keys(I.en).sort(), he = Object.keys(I.he).sort();
out.missing = en.filter((k) => !(k in I.he)); out.extra = he.filter((k) => !(k in I.en));
const ctx = { base: "20.4%", budget: "$10,000", weeks: 5, nModels: 5, modelList: "A <b>0.8</b>", best: "CatBoost",
  bestAuc: "0.866", lrAuc: "0.763", opt: "28%", n: "2,000", r50: "50%", p50: "82%", rOpt: "70%", pOpt: "58%",
  thresholdBar: "", incentives: [{ icon: "📞", name: "x", cost: "$500", success: "25%", blurb: "b" }] };
const c = { age: 40, products: 2, active: true, tenure: 3 };
for (const lang of ["en", "he"]) {
  for (const [k, v] of Object.entries(I[lang])) {
    try {
      let r = typeof v === "function" ? (k === "chapters" ? v(ctx) : k === "facts" ? v(c, "France", "$1") :
              k === "list" ? v(["a", "b"]) : v("A", "B", "C", "D", "E", "F")) : v;
      const flat = JSON.stringify(r);
      if (!flat || flat.includes("undefined")) out.errors.push(lang + ":" + k);
    } catch (e) { out.errors.push(lang + ":" + k + ":" + e.message); }
  }
  if (I[lang].chapters(ctx).length !== 4 || I[lang].takeaways.length !== 5) out.errors.push(lang + ":counts");
}
out.badges = Object.keys(I.he.badges).sort();
console.log(JSON.stringify(out));"""
    run = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["missing"] == [] and out["extra"] == [], out
    assert out["errors"] == [], out["errors"]
    titles = {b[2] for b in [*__import__("churn.game", fromlist=["BADGES"]).BADGES]} | {"Budget Burner"}
    assert set(out["badges"]) == titles  # every engine badge has a Hebrew title and message

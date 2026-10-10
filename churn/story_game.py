"""The Chief Retention Officer: data payload and HTML for the dashboard's interactive story game.

The game itself runs in the browser (churn/game_assets/game.html + engine.js), so every click
reacts instantly. Python supplies the *real* data it plays with:

- personas: real held-out test customers (one seeded random draw per archetype and round, never
  chosen by score or outcome), with every model's churn probability and their real outcome;
- each model's cross-validated ROC-AUC and its CV-optimised (max-F1) threshold;
- at-scale statistics on all test customers for every model and threshold on a 0.01 grid
  (true/false positives and negatives, and the customer value behind them).

Money values (customer value, incentive costs and success rates) are the game assumptions in
churn.game.Economics; the game shows them as such.
"""

import json
from pathlib import Path

import numpy as np

from churn import config
from churn.game import PERSONAS, Economics, select_personas

ASSETS = Path(__file__).resolve().parent / "game_assets"
N_ROUNDS = 5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)

INCENTIVE_META = {  # Economics key -> (id, icon, short name, blurb)
    "📞 Personal call": ("call", "📞", "Personal call", "A relationship manager calls to listen and fix issues."),
    "💳 Fee waiver": ("waiver", "💳", "Fee waiver", "Twelve months of account fees waived."),
    "👑 VIP package": ("vip", "👑", "VIP package", "Premium rates, a dedicated advisor and lounge access."),
}

PERSONA_LINES = {
    "david": {
        "idle": "I run a company. Make it quick.",
        "click": "My business accounts are here. For now.",
        "call": "A call from the bank? Fine, five minutes.",
        "waiver": "Lower fees on business accounts... that is a start.",
        "vip": "Now that's how you treat a business client.",
        "stay_targeted": "Thanks for the perks, but I wasn't going anywhere.",
        "stay": "Business as usual. I'm staying.",
        "left_targeted": "I was about to move my accounts. Your offer made me think twice.",
        "left": "I've moved my business to a competitor. Nobody called.",
    },
    "sarah": {
        "idle": "Do I even still use this account?",
        "click": "I opened it at university. Haven't logged in for ages.",
        "call": "Oh, someone actually called me? That's... nice.",
        "waiver": "No fees? OK, maybe I'll keep it.",
        "vip": "VIP? For me? I'm a student...",
        "stay_targeted": "Free stuff is cool, but I was never going to bother switching.",
        "stay": "I'll probably just leave it open. Too much hassle to close.",
        "left_targeted": "I was about to switch to a neobank. You got my attention.",
        "left": "Switched to an app-only bank. Didn't even notice you were gone.",
    },
    "michael": {
        "idle": "At my age, I expect good service.",
        "click": "I value stability. And a human being on the phone.",
        "call": "Finally, a real person. Thank you for calling.",
        "waiver": "Saving on fees is always welcome at my age.",
        "vip": "A dedicated advisor? Now we can talk about my portfolio.",
        "stay_targeted": "Kind of you, but I was perfectly happy already.",
        "stay": "I'm staying put. I don't like change.",
        "left_targeted": "My nephew wanted me to move my savings. You changed my mind.",
        "left": "I moved my savings elsewhere. Nobody seemed to care.",
    },
    "emma": {
        "idle": "Two kids, two accounts, zero free time.",
        "click": "We've been with you for years. Everything is fine.",
        "call": "Is something wrong with my account?",
        "waiver": "Fewer fees means more for the kids. Thanks!",
        "vip": "That's generous. I didn't expect anything.",
        "stay_targeted": "Lovely gesture, but we were staying anyway.",
        "stay": "We're happy here. No reason to move.",
        "left_targeted": "We were looking at a mortgage elsewhere. Let's talk.",
        "left": "We took our mortgage and our savings to another bank.",
    },
    "lukas": {
        "idle": "Why am I paying for three products?",
        "click": "Your adviser sold me products I never use.",
        "call": "Good, I have a list of complaints ready.",
        "waiver": "Waiving the fees is the least you can do.",
        "vip": "VIP treatment? You're finally taking me seriously.",
        "stay_targeted": "Fine, I'll stay. I was only grumbling.",
        "stay": "Still annoyed, but I'm not leaving.",
        "left_targeted": "I had the closure forms filled in. Let's see if you deliver.",
        "left": "Closed all three products. Should have listened earlier.",
    },
}


def _round_customers(analysis, econ: Economics, round_no: int) -> list[dict]:
    P = select_personas(analysis.X_test, round_no)
    positions = [analysis.X_test.index.get_loc(i) for i in P.index]
    values = econ.customer_value(P).to_numpy()
    churned = analysis.y_test.loc[P.index].to_numpy().astype(bool)
    customers = []
    for k, (persona, (_, row)) in enumerate(zip(PERSONAS, P.iterrows())):
        customers.append({
            "key": persona.key, "name": persona.name, "avatar": persona.avatar, "title": persona.title,
            "story": persona.story, "lines": PERSONA_LINES[persona.key],
            "age": int(row["Age"]), "geo": str(row["Geography"]), "balance": float(row["Balance"]),
            "products": int(row["NumOfProducts"]), "active": bool(row["IsActiveMember"]), "tenure": int(row["Tenure"]),
            "value": float(values[k]), "churned": bool(churned[k]),
            "p": {name: float(analysis.test_scores[name][positions[k]]) for name in analysis.models},
        })
    return customers


def scale_grid(scores, churned, values, extra: tuple = ()) -> dict:
    """Confusion counts and the customer value behind them for every threshold on THRESHOLD_GRID.

    extra adds exact thresholds (e.g. the model's CV optimum, 0.2776) so the game reports the same
    Precision/Recall at the optimum as the rest of the dashboard, instead of the nearest 0.01 step.
    """
    scores, churned, values = np.asarray(scores), np.asarray(churned, bool), np.asarray(values, float)
    grid = np.unique(np.concatenate([THRESHOLD_GRID, np.asarray(extra, float)]))
    flagged = scores[None, :] >= grid[:, None]
    tp, fp = flagged & churned, flagged & ~churned
    fn, tn = ~flagged & churned, ~flagged & ~churned
    return {
        "thresholds": grid.tolist(),
        "tp": tp.sum(1).tolist(), "fp": fp.sum(1).tolist(), "fn": fn.sum(1).tolist(), "tn": tn.sum(1).tolist(),
        "v_tp": (tp * values).sum(1).round(2).tolist(), "v_fn": (fn * values).sum(1).round(2).tolist(),
    }


def build_payload(analysis, best: str, colors: dict, econ: Economics | None = None, budget: float = 10_000,
                  n_rounds: int = N_ROUNDS, weighting: str = "None") -> dict:
    econ = econ or Economics()
    cv = analysis.cv_summary()
    values = econ.customer_value(analysis.X_test).to_numpy()
    churned = analysis.y_test.to_numpy().astype(bool)
    return {
        "best": best,
        "weighting": weighting,
        "budget": budget,
        "base_rate": float(analysis.base_rate),
        "default_threshold": 0.5,
        "models": [
            {"name": name, "color": colors.get(name, "#8a8984"), "cv_auc": float(cv.loc[name, "ROC_AUC mean"]),
             "cv_auc_std": float(cv.loc[name, "ROC_AUC std"]), "threshold": float(analysis.tuned_thresholds[name])}
            for name in analysis.models
        ],
        "econ": {
            "base": econ.base_value, "per_product": econ.per_product, "share": econ.balance_share,
            "incentives": [
                {"id": INCENTIVE_META[k][0], "icon": INCENTIVE_META[k][1], "name": INCENTIVE_META[k][2],
                 "blurb": INCENTIVE_META[k][3], "cost": inc.cost, "success": inc.success}
                for k, inc in econ.incentives.items()
            ],
        },
        "rounds": [{"no": r, "customers": _round_customers(analysis, econ, r)} for r in range(1, n_rounds + 1)],
        "scale": {
            "n": int(len(churned)), "churners": int(churned.sum()),
            "models": {name: scale_grid(analysis.test_scores[name], churned, values, (analysis.tuned_thresholds[name],))
                       for name in analysis.models},
        },
        "seed": config.RANDOM_STATE,
    }


def assets_fingerprint() -> str:
    """Changes whenever game.html or engine.js changes: part of the dashboard's cache key, so edits show up."""
    import hashlib

    digest = hashlib.sha256()
    for name in ("game.html", "engine.js"):
        digest.update((ASSETS / name).read_bytes())
    return digest.hexdigest()[:16]


def render_html(payload: dict) -> str:
    """The self-contained game page: template + engine + JSON payload (no external requests)."""
    template = (ASSETS / "game.html").read_text(encoding="utf-8")
    engine = (ASSETS / "engine.js").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")  # safe inside <script>
    return template.replace("/*__ENGINE__*/", engine).replace("/*__PAYLOAD__*/null", data)

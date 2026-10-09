"""The Bank Manager Retention Challenge: the engine behind the dashboard's mini-game.

The player is Head of Retention with a fixed budget and spends it on retention
incentives, guided by the model's churn risk scores.

What is real and what is assumed:

- **Real.** Every persona is a real customer from the held-out *test set*
  (never used for training or selection), scored live by the selected model.
  Whether the customer actually left (``Exited``) is their real outcome, so
  true savings, wasted budget and lost revenue are grounded in actual data.
  Persona names, avatars and stories are fictional labels on those rows; each
  round draws one customer per archetype at random (seeded by the round
  number), never by score or outcome.
- **Assumed.** Customer value, incentive costs and incentive success rates
  are not in the dataset. They are game assumptions (``Economics``), shown in
  the dashboard and adjustable by the player. Results are *expected* values:
  an incentive with a 60% success rate saves 60% of a churner's value.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from churn import config

NO_INCENTIVE = "No action"


@dataclass(frozen=True)
class Incentive:
    cost: float  # game dollars per customer
    success: float  # probability that a customer who would churn stays instead


@dataclass
class Economics:
    """Game assumptions (not data). Customer value = base + per_product × products + balance_share × balance."""

    base_value: float = 2_000.0
    per_product: float = 1_000.0
    balance_share: float = 0.04
    incentives: dict = field(default_factory=lambda: {
        "📞 Personal call": Incentive(cost=500.0, success=0.25),
        "💳 Fee waiver": Incentive(cost=1_500.0, success=0.40),
        "👑 VIP package": Incentive(cost=3_500.0, success=0.60),
    })

    def options(self) -> list[str]:
        return [NO_INCENTIVE, *self.incentives]

    def cost(self, option: str) -> float:
        return 0.0 if option == NO_INCENTIVE else self.incentives[option].cost

    def success(self, option: str) -> float:
        return 0.0 if option == NO_INCENTIVE else self.incentives[option].success

    def customer_value(self, X: pd.DataFrame) -> pd.Series:
        """Value lost if the customer leaves (game dollars)."""
        return self.base_value + self.per_product * X["NumOfProducts"] + self.balance_share * X["Balance"]


# --- Personas ------------------------------------------------------------------------
@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    avatar: str
    title: str
    story: str
    rule: str  # pandas query selecting the archetype in the test set


PERSONAS = [
    Persona("david", "David", "👨‍💼", "The High-Value Business Owner",
            "Runs a growing company and keeps a large balance with the bank. Busy, rarely engages with the app.",
            "Gender == 'Male' and 40 <= Age <= 55 and Balance > 120000 and EstimatedSalary > 120000"),
    Persona("sarah", "Sarah", "👩‍🎓", "The Disengaged Student",
            "Opened an account at university. Has not logged in for months.",
            "Gender == 'Female' and Age <= 25 and IsActiveMember == 0"),
    Persona("michael", "Michael", "👴", "The Senior Investor",
            "Retired, with substantial savings. Values personal service and stability.",
            "Gender == 'Male' and Age >= 60 and Balance > 80000"),
    Persona("emma", "Emma", "👩‍👧", "The Established Family Customer",
            "Long-standing, active customer with two products: a current account and a savings plan.",
            "Gender == 'Female' and 30 <= Age <= 45 and NumOfProducts == 2 and IsActiveMember == 1 and Tenure >= 5"),
    Persona("lukas", "Lukas", "🧔", "The Over-Banked Customer",
            "Was sold three or more products in Germany. Starting to question the fees.",
            "Gender == 'Male' and Geography == 'Germany' and NumOfProducts >= 3"),
]


def select_personas(X_test: pd.DataFrame, round_no: int = 1, personas: list[Persona] = PERSONAS) -> pd.DataFrame:
    """One real test customer per persona for a game round: a seeded random draw from each archetype.

    The draw uses only the archetype rule and the round number (seed = RANDOM_STATE + round_no), never
    model scores or outcomes, so personas are not cherry-picked and every round is reproducible.
    Returns the selected rows of X_test (same index), one per persona, in persona order.
    """
    rng = np.random.default_rng(config.RANDOM_STATE + round_no)
    chosen = []
    for persona in personas:
        group = X_test.query(persona.rule)
        if group.empty:
            raise ValueError(f"No test customer matches persona {persona.key!r}")
        chosen.append(group.index[rng.integers(len(group))])
    return X_test.loc[chosen]


def mood(probability: float, threshold: float, base_rate: float) -> tuple[str, str]:
    """Emotional status badge from the model's risk score: (label, colour)."""
    if probability >= threshold:
        return "😠 High risk of churn", "red"
    if probability >= base_rate:
        return "😐 Wavering", "orange"
    return "😊 Loyal", "green"


# --- Outcomes ----------------------------------------------------------------------
OUTCOME_COLUMNS = ["spend", "true_savings", "wasted", "lost_revenue"]


def outcomes(decisions, churned, value, econ: Economics) -> pd.DataFrame:
    """Expected money outcome per customer for a list of decisions (incentive names).

    - targeted churner:  true_savings = success × value, lost_revenue = (1 − success) × value
    - targeted loyal:    wasted = cost (a false positive)
    - untargeted churner: lost_revenue = value (a false negative)
    - untargeted loyal:  nothing happens
    """
    decisions = list(decisions)
    churned = np.asarray(churned, dtype=bool)
    value = np.asarray(value, dtype=float)
    cost = np.array([econ.cost(d) for d in decisions])
    success = np.array([econ.success(d) for d in decisions])
    targeted = cost > 0
    return pd.DataFrame({
        "spend": cost,
        "true_savings": np.where(targeted & churned, success * value, 0.0),
        "wasted": np.where(targeted & ~churned, cost, 0.0),
        "lost_revenue": np.where(churned, (1 - success) * value, 0.0),
    })


def summarise(table: pd.DataFrame) -> dict:
    """Totals plus net value (savings − spend) and ROI ((savings − spend) / spend)."""
    totals = {c: float(table[c].sum()) for c in OUTCOME_COLUMNS}
    totals["net"] = totals["true_savings"] - totals["spend"]
    totals["roi"] = totals["net"] / totals["spend"] if totals["spend"] else 0.0
    return totals


def _all_plans(n: int, econ: Economics, budget: float):
    for plan in product(econ.options(), repeat=n):
        if sum(econ.cost(d) for d in plan) <= budget:
            yield plan


def best_plan(weights, value, econ: Economics, budget: float) -> list[str]:
    """Plan maximising the sum of weight × success × value − cost within budget (exhaustive; for a few personas).

    weights = true outcomes (0/1) gives the perfect-information plan;
    weights = model probabilities gives the model's expected-value plan.
    """
    weights, value = np.asarray(weights, dtype=float), np.asarray(value, dtype=float)

    def gain(plan):
        return sum(w * econ.success(d) * v - econ.cost(d) for d, w, v in zip(plan, weights, value))

    return list(max(_all_plans(len(value), econ, budget), key=lambda plan: (round(gain(plan), 6), -sum(map(econ.cost, plan)))))


def success_score(net: float, best_net: float) -> int:
    """0–100: the share of the best achievable net value the player captured (negative net scores 0)."""
    if best_net <= 0:
        return 100 if net >= 0 else 0
    return int(round(100 * min(max(net / best_net, 0.0), 1.0)))


BADGES = [
    (80, "🏆", "Retention Champion", "You captured most of the value a perfectly informed manager could have saved."),
    (60, "🥈", "Savvy Strategist", "Solid, data-driven decisions with a healthy return."),
    (30, "📈", "Promising Manager", "Some good calls, but money was left on the table or spent on loyal customers."),
    (0, "🪙", "Cautious Starter", "No money lost, but most of the value you could have saved slipped away."),
]
BUDGET_BURNER = ("💸", "Budget Burner", "The budget did more harm than good. Check the risk scores and try again.")


def badge(score: int, net: float) -> tuple[str, str, str]:
    """(icon, title, message); only a negative net value earns the Budget Burner badge."""
    if net < 0:
        return BUDGET_BURNER
    return next(b[1:] for b in BADGES if score >= b[0])


# --- Campaign at scale -----------------------------------------------------------------
def campaign(scores, churned, value, econ: Economics, option: str, budget: float, min_risk: float = 0.0) -> dict:
    """Target the highest-risk customers (score ≥ min_risk) with one incentive until the budget runs out.

    Also returns two references for the same number of targeted customers:
    random targeting (expected value) and perfect information (highest-value true churners).
    """
    scores, churned, value = np.asarray(scores, float), np.asarray(churned, bool), np.asarray(value, float)
    cost, success = econ.cost(option), econ.success(option)
    eligible = np.flatnonzero(scores >= min_risk)
    n = int(min(len(eligible), budget // cost)) if cost > 0 else 0
    target = eligible[np.argsort(-scores[eligible], kind="stable")[:n]]
    decisions = np.full(len(scores), NO_INCENTIVE, dtype=object)
    decisions[target] = option
    model = summarise(outcomes(decisions, churned, value, econ))
    model.update(n_targeted=n, churners_targeted=int(churned[target].sum()), loyal_targeted=int((~churned[target]).sum()),
                 churners_missed=int(churned.sum() - churned[target].sum()))

    spend = n * cost
    random_savings = n / len(scores) * success * value[churned].sum()  # expected under random targeting
    oracle_idx = np.flatnonzero(churned)[np.argsort(-value[churned], kind="stable")[:n]]
    oracle_savings = success * value[oracle_idx].sum()
    return {
        "model": model,
        "random": {"net": random_savings - spend, "true_savings": random_savings, "spend": spend},
        "perfect": {"net": oracle_savings - spend, "true_savings": oracle_savings, "spend": spend},
    }


def campaign_curve(scores, churned, value, econ: Economics, option: str, max_n: int, points: int = 60) -> pd.DataFrame:
    """Net value as the number of targeted customers grows: model ranking vs random targeting."""
    scores, churned, value = np.asarray(scores, float), np.asarray(churned, bool), np.asarray(value, float)
    order = np.argsort(-scores, kind="stable")
    cost, success = econ.cost(option), econ.success(option)
    saved_cum = np.concatenate([[0.0], np.cumsum(np.where(churned[order], success * value[order], 0.0))])
    ns = np.unique(np.linspace(0, min(max_n, len(scores)), points).astype(int))
    random_per = success * value[churned].sum() / len(scores)
    return pd.DataFrame({"n": ns, "model": saved_cum[ns] - ns * cost, "random": ns * (random_per - cost)})

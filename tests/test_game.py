import numpy as np
import pandas as pd
import pytest

from churn.data import make_train_test_split, split_features_target
from churn.game import (
    NO_INCENTIVE,
    PERSONAS,
    Economics,
    Incentive,
    badge,
    best_plan,
    outcomes,
    select_personas,
    success_score,
    summarise,
)

ECON = Economics(base_value=1000, per_product=0, balance_share=0,
                 incentives={"call": Incentive(100, 0.5), "vip": Incentive(400, 0.9)})


def test_outcomes_follow_the_confusion_matrix():
    # targeted churner (TP), targeted loyal (FP), missed churner (FN), untouched loyal (TN)
    t = outcomes(["vip", "call", NO_INCENTIVE, NO_INCENTIVE], [1, 0, 1, 0], [1000] * 4, ECON)
    assert t["true_savings"].tolist() == pytest.approx([900, 0, 0, 0])
    assert t["wasted"].tolist() == pytest.approx([0, 100, 0, 0])
    assert t["lost_revenue"].tolist() == pytest.approx([100, 0, 1000, 0])
    s = summarise(t)
    assert s["spend"] == 500 and s["net"] == pytest.approx(400) and s["roi"] == pytest.approx(0.8)


def test_doing_nothing_has_zero_roi_and_full_losses():
    s = summarise(outcomes([NO_INCENTIVE] * 3, [1, 1, 0], [1000] * 3, ECON))
    assert s["spend"] == 0 and s["roi"] == 0 and s["lost_revenue"] == 2000


def test_perfect_plan_respects_budget_and_beats_every_other_plan():
    churned, value = [1, 0, 1], [1000, 1000, 3000]
    plan = best_plan(churned, value, ECON, budget=500)
    assert sum(ECON.cost(d) for d in plan) <= 500
    assert plan[1] == NO_INCENTIVE  # never pay a loyal customer with perfect information
    best = summarise(outcomes(plan, churned, value, ECON))["net"]
    for other in (["call", NO_INCENTIVE, "call"], ["vip", NO_INCENTIVE, NO_INCENTIVE], [NO_INCENTIVE] * 3):
        assert summarise(outcomes(other, churned, value, ECON))["net"] <= best


def test_model_plan_skips_customers_whose_expected_saving_is_below_cost():
    plan = best_plan([0.05, 0.95], [1000, 1000], ECON, budget=10_000)
    assert plan[0] == NO_INCENTIVE and plan[1] != NO_INCENTIVE


def test_score_and_badges():
    assert success_score(50, 100) == 50 and success_score(150, 100) == 100 and success_score(-10, 100) == 0
    assert badge(90, 10)[1] == "Retention Champion"
    assert badge(90, -1)[1] == "Budget Burner"  # losing money is always a defeat
    assert badge(10, 500)[1] == "Cautious Starter"  # a low score with positive net value is not "budget burning"


def test_personas_are_real_test_customers_matching_their_archetype(small_df):
    X, y = split_features_target(small_df)
    _, X_test, _, _ = make_train_test_split(X, y)
    # widen the synthetic data so every archetype exists
    extra = pd.DataFrame([
        dict(CreditScore=600, Geography="France", Gender="Male", Age=45, Tenure=3, Balance=150000, NumOfProducts=1,
             HasCrCard=1, IsActiveMember=0, EstimatedSalary=150000),
        dict(CreditScore=600, Geography="Spain", Gender="Female", Age=22, Tenure=1, Balance=0, NumOfProducts=1,
             HasCrCard=0, IsActiveMember=0, EstimatedSalary=20000),
        dict(CreditScore=600, Geography="France", Gender="Male", Age=67, Tenure=8, Balance=120000, NumOfProducts=1,
             HasCrCard=1, IsActiveMember=1, EstimatedSalary=50000),
        dict(CreditScore=600, Geography="France", Gender="Female", Age=38, Tenure=7, Balance=50000, NumOfProducts=2,
             HasCrCard=1, IsActiveMember=1, EstimatedSalary=60000),
        dict(CreditScore=600, Geography="Germany", Gender="Male", Age=50, Tenure=2, Balance=90000, NumOfProducts=3,
             HasCrCard=1, IsActiveMember=0, EstimatedSalary=70000),
    ], index=[9001, 9002, 9003, 9004, 9005])[X_test.columns]
    X_test = pd.concat([X_test, extra])
    first, again, other = select_personas(X_test, 1), select_personas(X_test, 1), select_personas(X_test, 2)
    assert list(first.index) == list(again.index)  # reproducible per round
    assert len(first) == len(PERSONAS) and set(first.index) <= set(X_test.index)
    for persona, (_, row) in zip(PERSONAS, first.iterrows()):
        assert len(pd.DataFrame([row]).query(persona.rule)) == 1
    assert isinstance(other, pd.DataFrame)

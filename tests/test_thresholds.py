import numpy as np
import pytest

from churn.analysis import STRATEGY_CUSTOM, STRATEGY_DEFAULT, STRATEGY_OPTIMIZED, run_analysis
from churn.thresholds import apply_threshold, select_cost_threshold, select_f1_threshold, threshold_curve

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1, 0.1, 0.05])


def test_apply_threshold_is_inclusive():
    np.testing.assert_array_equal(apply_threshold([0.2, 0.5, 0.7], 0.5), [0, 1, 1])


def test_f1_threshold_is_the_f1_maximiser():
    curve = threshold_curve(Y, S)
    best = select_f1_threshold(Y, S)
    f1_at = lambda t: curve.loc[curve["threshold"] == t, "f1"].iloc[0]
    assert f1_at(best) == pytest.approx(curve["f1"].max())
    # Brute force over every candidate threshold gives the same best F1.
    from sklearn.metrics import f1_score

    brute = max(f1_score(Y, apply_threshold(S, t)) for t in np.unique(S))
    assert f1_at(best) == pytest.approx(brute)


def test_cost_threshold_follows_cost_ratio():
    # Missing churners very expensive -> flag (almost) everyone; false alarms expensive -> flag few.
    assert select_cost_threshold(Y, S, cost_fn=100, cost_fp=1) <= 0.3
    assert select_cost_threshold(Y, S, cost_fn=1, cost_fp=100) >= 0.7


def test_analysis_thresholds_come_from_training_data_only(small_df, tmp_path):
    path = tmp_path / "data.csv"
    small_df.to_csv(path, index=False)
    a = run_analysis(path, permutation_repeats=2)

    default = a.evaluate_strategy(STRATEGY_DEFAULT)
    tuned = a.evaluate_strategy(STRATEGY_OPTIMIZED)
    custom = a.evaluate_strategy(STRATEGY_CUSTOM, 0.3)
    for name in a.models:
        assert default[name].threshold == 0.5
        # Default strategy reproduces the model's own predict() output (identical to main.py).
        np.testing.assert_array_equal(default[name].y_pred, a.models[name].predict(a.X_test))
        # The tuned threshold is a function of training labels and out-of-fold training scores only.
        assert len(a.oof_scores[name]) == len(a.y_train)
        assert tuned[name].threshold == pytest.approx(select_f1_threshold(a.y_train, a.oof_scores[name]))
        np.testing.assert_array_equal(tuned[name].y_pred, apply_threshold(a.test_scores[name], tuned[name].threshold))
        np.testing.assert_array_equal(custom[name].y_pred, apply_threshold(a.test_scores[name], 0.3))
        # ROC-AUC does not depend on the threshold.
        assert default[name].metrics["ROC_AUC"] == tuned[name].metrics["ROC_AUC"] == custom[name].metrics["ROC_AUC"]
        assert tuned[name].confusion.sum() == len(a.y_test)


def test_custom_strategy_requires_a_value(small_df, tmp_path):
    path = tmp_path / "data.csv"
    small_df.to_csv(path, index=False)
    a = run_analysis(path, permutation_repeats=None)
    assert a.permutation_importance == {}
    with pytest.raises(ValueError):
        a.thresholds_for(STRATEGY_CUSTOM)

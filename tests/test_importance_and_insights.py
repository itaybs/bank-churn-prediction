import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.data import make_train_test_split, split_features_target
from churn.importance import builtin_importance, original_feature, permutation_importance_table
from churn.insights import (
    GAP_STD_RULE,
    analytical_summary,
    best_model,
    churn_rate_by,
    format_segments,
    generalisation_notes,
    generalisation_table,
    is_monotonic,
    paired_fold_comparison,
    segment_stats,
    small_segment_notes,
)
from churn.models import build_models, train_models


@pytest.fixture
def fitted(small_df):
    X, y = split_features_target(small_df)
    X_tr, X_te, y_tr, y_te = make_train_test_split(X, y)
    return train_models(build_models(), X_tr, y_tr), X_tr, X_te, y_tr, y_te


def test_one_hot_columns_map_back_to_original_features():
    assert original_feature("num__Age") == "Age"
    assert original_feature("cat__Geography_Spain") == "Geography"
    assert original_feature("cat__Gender_Female") == "Gender"
    with pytest.raises(ValueError):
        original_feature("cat__Unknown_X")


def test_builtin_importance_is_normalised_and_covers_all_features(fitted):
    models = fitted[0]
    for name in ("Random Forest", "XGBoost", "LightGBM", "CatBoost"):
        imp = builtin_importance(models[name])
        assert set(imp.index) == set(config.FEATURES)
        assert imp.sum() == pytest.approx(1.0)
        onehot = builtin_importance(models[name], aggregate=False)
        assert len(onehot) == 13 and onehot.sum() == pytest.approx(1.0)
        # Aggregation sums the one-hot columns of each categorical feature.
        assert imp["Geography"] == pytest.approx(onehot.filter(like="Geography_").sum())


def test_builtin_importance_rejects_linear_models(fitted):
    with pytest.raises(ValueError):
        builtin_importance(fitted[0]["Logistic Regression"])


def test_permutation_importance_has_one_row_per_raw_feature(fitted):
    models, _, X_te, _, y_te = fitted
    table = permutation_importance_table(models["Logistic Regression"], X_te, y_te, n_repeats=2)
    assert set(table.index) == set(config.FEATURES)
    assert list(table.columns) == ["importance_mean", "importance_std"]
    assert table["importance_mean"].is_monotonic_decreasing


def test_paired_fold_comparison_counts_wins():
    folds = {
        "A": pd.DataFrame({"ROC_AUC": [0.80, 0.82, 0.79]}),
        "B": pd.DataFrame({"ROC_AUC": [0.78, 0.83, 0.75]}),
    }
    p = paired_fold_comparison(folds, "A", "B")
    assert p["wins"] == 2 and p["n"] == 3
    assert p["mean"] == pytest.approx(np.mean([0.02, -0.01, 0.04]))


def test_monotonic_detection_and_churn_rates():
    assert is_monotonic(pd.Series([0.1, 0.2, 0.3]))
    assert not is_monotonic(pd.Series([0.28, 0.07, 0.82, 1.0]))
    X = pd.DataFrame({"NumOfProducts": [1, 1, 2, 2, 3], "Age": [20, 30, 45, 55, 65]})
    y = pd.Series([1, 0, 0, 0, 1])
    rates = churn_rate_by(X, y, "NumOfProducts")
    assert rates.to_dict() == {1: 0.5, 2: 0.0, 3: 1.0}
    assert len(churn_rate_by(X, y, "Age")) == 5  # grouped into bands


def test_summary_names_best_model_from_numbers(fitted):
    _, X_tr, _, y_tr, _ = fitted
    table = pd.DataFrame(
        {"Precision": [0.5, 0.7, 0.75], "Recall": [0.2, 0.4, 0.45], "F1": [0.3, 0.5, 0.55],
         "Accuracy": [0.8, 0.85, 0.86], "ROC_AUC": [0.77, 0.85, 0.87]},
        index=["Logistic Regression", "Random Forest", "XGBoost"],
    )
    cv = pd.DataFrame({"ROC_AUC mean": [0.76, 0.85, 0.86], "ROC_AUC std": [0.02, 0.01, 0.01]}, index=table.index)
    folds = {m: pd.DataFrame({"ROC_AUC": np.full(5, cv.loc[m, "ROC_AUC mean"])}) for m in table.index}
    summary = analytical_summary(table, cv, folds, X_tr, y_tr)
    assert best_model(cv) == "XGBoost"
    assert "XGBoost is selected" in summary["verdict"][0]
    assert "confirms the CV-based choice" in summary["verdict"][1]
    assert set(summary) == {"verdict", "evidence", "why", "caveats"}


def test_selection_uses_cross_validation_not_the_test_set(fitted):
    """A model that wins only on the test set must NOT be selected; the summary reports the disagreement."""
    _, X_tr, _, y_tr, _ = fitted
    index = ["Logistic Regression", "Random Forest", "XGBoost"]
    test = pd.DataFrame(
        {"Precision": [0.5, 0.7, 0.75], "Recall": [0.2, 0.4, 0.45], "F1": [0.3, 0.5, 0.55],
         "Accuracy": [0.8, 0.85, 0.86], "ROC_AUC": [0.90, 0.85, 0.87]},  # LR best on test only
        index=index,
    )
    cv = pd.DataFrame({"ROC_AUC mean": [0.76, 0.85, 0.86], "ROC_AUC std": [0.02, 0.01, 0.01]}, index=index)
    folds = {m: pd.DataFrame({"ROC_AUC": np.full(5, cv.loc[m, "ROC_AUC mean"])}) for m in index}
    summary = analytical_summary(test, cv, folds, X_tr, y_tr)
    assert best_model(cv) == "XGBoost"
    assert "XGBoost is selected" in summary["verdict"][0]
    assert "not confirmed" in summary["verdict"][1] and "Logistic Regression" in summary["verdict"][1]


def test_segment_stats_report_sample_sizes_and_flag_small_segments():
    X = pd.DataFrame({"NumOfProducts": [1] * 150 + [2] * 120 + [4] * 5})
    y = pd.Series([1] * 30 + [0] * 120 + [0] * 120 + [1] * 5)
    stats = segment_stats(X, y, "NumOfProducts")
    assert stats["n"].to_dict() == {1: 150, 2: 120, 4: 5}
    assert stats["churned"].to_dict() == {1: 30, 2: 0, 4: 5}
    assert stats["small"].to_dict() == {1: False, 2: False, 4: True}
    text = format_segments(stats)
    assert "n=150" in text and "5/5, n=5 ⚠ small sample" in text
    notes = small_segment_notes(X, y, ["NumOfProducts"])
    assert len(notes) == 1 and "5 of 5 churned" in notes[0]


def test_generalisation_notes_never_claim_no_overfitting():
    index = ["Logistic Regression", "Random Forest"]
    cv = pd.DataFrame({"ROC_AUC mean": [0.763, 0.849], "ROC_AUC std": [0.022, 0.015]}, index=index)
    test = pd.DataFrame({"ROC_AUC": [0.775, 0.849]}, index=index)
    train = pd.DataFrame({"ROC_AUC": [0.765, 1.000]}, index=index)
    table = generalisation_table(cv, test, train)
    assert table.loc["Random Forest", "Train − CV"] == pytest.approx(0.151)
    assert table[f"Test within {GAP_STD_RULE} CV std"].all()
    text = " ".join(generalisation_notes(table))
    assert "reliable" in text
    assert "Random Forest scores 1.000 on its own training data" in text
    assert "not overfitting" not in text and "does **not** show that a model is free of overfitting" in text

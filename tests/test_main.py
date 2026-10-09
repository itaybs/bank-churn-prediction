import pytest

import main
from churn.data import DataValidationError

MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
EXPECTED = {
    "model_comparison.csv", "cv_results.csv", "cv_folds.csv", "feature_importance.csv",
    "roc_curves.png", "confusion_matrices.png", "feature_importance.png", "report.md",
    "tuning_comparison.csv", "tuning_best_params.csv", "tuning_cv_folds.csv", "tuning_grid_results.csv",
    "tuning_before_after.png", "random_forest_tuned.joblib", "xgboost_tuned.joblib",
}
# Tiny grids keep the test fast; each includes the baseline value and one alternative.
TINY_GRIDS = {
    "Random Forest": {"model__min_samples_leaf": [1, 4]},
    "XGBoost": {"model__max_depth": [3, 4]},
}


def test_full_workflow_creates_all_outputs(small_df, tmp_path):
    data_path = tmp_path / "data.csv"
    small_df.to_csv(data_path, index=False)
    out_dir = tmp_path / "new_outputs_dir"  # does not exist yet

    outputs = main.run(data_path, out_dir, param_grids=TINY_GRIDS)

    assert {p.name for p in outputs} == EXPECTED
    assert all(p.is_file() and p.stat().st_size > 0 for p in outputs)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    for name in MODELS:
        assert name in report
    assert "Cross-validation on the training set" in report and "Feature importance" in report
    # Fix 1: selection on CV, test only as confirmation.
    assert "is selected" in report and "highest mean ROC-AUC in cross-validation" in report
    # Fix 2: the generalisation check shows training scores and never claims "not overfitting".
    assert "| Model | Train | CV mean | CV std | Test | Train − CV | Test − CV |" in report
    assert "not overfitting" not in report
    # Fix 4: segment tables with sample sizes.
    assert "Customers (n)" in report and "Churn rate by segment, with sample sizes" in report

    import pandas as pd

    cv = pd.read_csv(out_dir / "cv_results.csv", index_col="Model")
    assert list(cv.index) == MODELS
    assert not cv.isna().any().any()
    folds = pd.read_csv(out_dir / "cv_folds.csv")
    assert len(folds) == len(MODELS) * 5

    # Tuning: report section, before-vs-after table, and reloadable tuned pipelines.
    assert "Hyperparameter tuning (before vs after)" in report and "Best model overall" in report
    comparison = pd.read_csv(out_dir / "tuning_comparison.csv")
    assert set(comparison["Model"]) == set(TINY_GRIDS) and set(comparison["Stage"]) == {"CV mean", "Test"}
    assert (comparison["Tuned"] - comparison["Baseline"] - comparison["Change"]).abs().max() < 1e-3
    from churn.tuning import load_tuned_models

    tuned = load_tuned_models(out_dir / "models")
    assert set(tuned) == set(TINY_GRIDS)
    X = small_df.drop(columns=["CustomerId", "Surname", "Exited"])
    assert tuned["XGBoost"].predict_proba(X).shape == (len(X), 2)


def test_missing_dataset_stops_with_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        main.run(tmp_path / "missing.csv", tmp_path / "out")
    assert not (tmp_path / "out").exists()  # nothing misleading is written


def test_invalid_target_stops_before_writing_outputs(small_df, tmp_path):
    data_path = tmp_path / "bad.csv"
    small_df.assign(Exited=small_df["Exited"] * 2).to_csv(data_path, index=False)
    with pytest.raises(DataValidationError):
        main.run(data_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()

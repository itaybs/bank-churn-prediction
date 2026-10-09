import numpy as np
import pandas as pd
import pytest

from churn.data import make_train_test_split, split_features_target
from churn.evaluation import build_comparison_table
from churn.models import build_models, predict_models, train_models
from churn.tuning import (
    PARAM_GRIDS,
    TUNED_MODELS,
    best_params,
    candidates_table,
    comparison_table,
    grid_results,
    load_tuned_models,
    params_table,
    save_tuned_models,
    tune_model,
    tuned_cv_folds,
    tuning_insights,
)
from churn.validation import cross_validate_model, summarize_cv

GRID = {"model__max_depth": [3, 4]}  # baseline 4 plus one alternative


@pytest.fixture
def data(small_df):
    X, y = split_features_target(small_df)
    return make_train_test_split(X, y)


@pytest.fixture
def search(data):
    X_tr, _, y_tr, _ = data
    return tune_model("XGBoost", X_tr, y_tr, GRID)


def test_default_grids_contain_the_baseline_settings():
    """The search can then only keep or improve on the baseline's CV score."""
    for name, grid in PARAM_GRIDS.items():
        baseline = build_models()[name].get_params()
        for key, values in grid.items():
            assert baseline[key] in values, f"{name}: baseline {key}={baseline[key]} missing from grid"
    assert TUNED_MODELS == ["Random Forest", "XGBoost"]


def test_search_uses_training_folds_and_refits_best(search, data):
    X_tr, _, _, _ = data
    assert search.n_splits_ == 5 and search.refit == "ROC_AUC"
    assert set(best_params(search)) == {"max_depth"}
    assert hasattr(search.best_estimator_, "predict_proba")
    table = grid_results(search)
    assert len(table) == 2 and table.loc[0, "Rank"] == 1
    assert table.loc[0, "CV ROC-AUC mean"] == pytest.approx(search.best_score_)


def test_tuned_folds_match_baseline_cv_on_the_same_folds(search, data):
    """Baseline settings inside the grid reproduce the baseline CV fold scores exactly (same folds)."""
    X_tr, _, y_tr, _ = data
    baseline_folds, _ = cross_validate_model(build_models()["XGBoost"], X_tr, y_tr)
    r = search.cv_results_
    i = [p["model__max_depth"] for p in r["params"]].index(4)
    grid_baseline = [r[f"split{f}_test_ROC_AUC"][i] for f in range(5)]
    np.testing.assert_allclose(grid_baseline, baseline_folds["ROC_AUC"].values)
    folds = tuned_cv_folds(search)
    assert list(folds.index) == list(baseline_folds.index)
    assert folds["ROC_AUC"].mean() == pytest.approx(search.best_score_)


def test_comparison_table_and_insights(search, data):
    X_tr, X_te, y_tr, y_te = data
    base_models = {n: m for n, m in build_models().items() if n in ("Random Forest", "XGBoost")}
    base_folds = {n: cross_validate_model(m, X_tr, y_tr)[0] for n, m in base_models.items()}
    base_cv = summarize_cv(base_folds)
    base_test = build_comparison_table(y_te, predict_models(train_models(base_models, X_tr, y_tr), X_te))
    tuned_folds = {"XGBoost": tuned_cv_folds(search)}
    tuned_cv = summarize_cv(tuned_folds)
    tuned_test = build_comparison_table(y_te, predict_models({"XGBoost": search.best_estimator_}, X_te))

    comp = comparison_table(base_cv, tuned_cv, base_test, tuned_test)
    assert len(comp) == 1 * 2 * 5  # models x stages x metrics
    assert (comp["Tuned"] - comp["Baseline"]).values == pytest.approx(comp["Change"].values)

    cv_c = candidates_table(base_cv, tuned_cv)
    assert list(cv_c.index) == ["Random Forest", "XGBoost", "XGBoost (tuned)"]
    params = params_table({"XGBoost": search})
    assert list(params["Hyperparameter"]) == ["max_depth"] and params.loc[0, "Baseline"] == "4"
    text = tuning_insights(comp, base_folds, tuned_folds, params, cv_c, candidates_table(base_test, tuned_test))
    assert set(text) == {"per_model", "best", "tradeoffs", "caveats"}
    assert text["per_model"][0].startswith("**XGBoost:**")
    assert "highest mean CV ROC-AUC" in text["best"][0]
    assert any("optimistic" in c for c in text["caveats"])


def test_tuned_models_round_trip(search, data, tmp_path):
    _, X_te, _, _ = data
    paths = save_tuned_models({"XGBoost": search}, tmp_path / "models")
    assert [p.name for p in paths] == ["xgboost_tuned.joblib"]
    loaded = load_tuned_models(tmp_path / "models")
    assert list(loaded) == ["XGBoost"]  # missing Random Forest file is skipped
    np.testing.assert_allclose(loaded["XGBoost"].predict_proba(X_te), search.best_estimator_.predict_proba(X_te))


def test_load_from_missing_directory_returns_empty(tmp_path):
    assert load_tuned_models(tmp_path / "nothing") == {}


def test_grid_results_keep_none_values(data):
    X_tr, _, y_tr, _ = data
    s = tune_model("Random Forest", X_tr, y_tr, {"model__max_depth": [None, 3], "model__n_estimators": [20]})
    assert None in list(grid_results(s)["max_depth"])
    assert isinstance(grid_results(s), pd.DataFrame)

import numpy as np
import pytest

from churn.data import make_train_test_split, split_features_target
from churn.models import build_models
from churn.validation import CV_METRICS, cross_validate_model, format_mean_std, make_cv, summarize_cv


@pytest.fixture
def train(small_df):
    X, y = split_features_target(small_df)
    X_tr, _, y_tr, _ = make_train_test_split(X, y)
    return X_tr, y_tr


def test_cv_returns_one_score_per_fold_and_metric(train):
    X_tr, y_tr = train
    folds, oof = cross_validate_model(build_models()["Logistic Regression"], X_tr, y_tr)
    assert folds.shape == (5, len(CV_METRICS))
    assert list(folds.columns) == CV_METRICS
    assert ((folds >= 0) & (folds <= 1)).all().all()
    # Every training row gets exactly one out-of-fold probability.
    assert oof.shape == (len(X_tr),) and not np.isnan(oof).any()
    assert ((oof >= 0) & (oof <= 1)).all()


def test_folds_are_stratified_disjoint_and_identical_across_calls(train):
    X_tr, y_tr = train
    splits_a = list(make_cv().split(X_tr, y_tr))
    splits_b = list(make_cv().split(X_tr, y_tr))
    seen = np.concatenate([test for _, test in splits_a])
    assert sorted(seen) == list(range(len(X_tr)))  # each row validated exactly once
    for (tr_a, te_a), (tr_b, te_b) in zip(splits_a, splits_b):
        np.testing.assert_array_equal(te_a, te_b)
        assert set(tr_a).isdisjoint(te_a)
        assert y_tr.iloc[te_a].mean() == pytest.approx(y_tr.mean(), abs=0.05)


def test_cv_does_not_fit_the_pipeline_passed_in(train):
    X_tr, y_tr = train
    pipeline = build_models()["Random Forest"]
    cross_validate_model(pipeline, X_tr, y_tr)
    assert not hasattr(pipeline.named_steps["model"], "estimators_")  # a clone was fitted, not the original


def test_summary_and_formatting(train):
    X_tr, y_tr = train
    folds = {name: cross_validate_model(p, X_tr, y_tr)[0] for name, p in build_models().items()}
    summary = summarize_cv(folds)
    assert list(summary.index) == ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    for m in CV_METRICS:
        assert summary.loc["XGBoost", f"{m} mean"] == pytest.approx(folds["XGBoost"][m].mean())
        assert summary.loc["XGBoost", f"{m} std"] == pytest.approx(folds["XGBoost"][m].std(ddof=1))
    text = format_mean_std(summary, sep=" +/- ").loc["Random Forest", "F1"]
    assert " +/- " in text and len(text.split(" +/- ")) == 2

import numpy as np
import pytest

from churn import config
from churn.data import make_train_test_split, split_features_target
from churn.models import build_models, positive_class_proba, predict_models, train_models
from churn.preprocessing import build_preprocessor


@pytest.fixture
def split(small_df):
    X, y = split_features_target(small_df)
    return make_train_test_split(X, y)


def test_scaler_is_fitted_on_training_rows_only(split):
    X_tr, X_te, _, _ = split
    pre = build_preprocessor(scale_numeric=True).fit(X_tr)
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_tr[config.NUMERIC_FEATURES].mean().values)
    assert not np.allclose(scaler.mean_, X_te[config.NUMERIC_FEATURES].mean().values)


def test_encoder_learns_only_training_categories_and_ignores_unknown(split):
    X_tr, X_te, _, _ = split
    pre = build_preprocessor(scale_numeric=False).fit(X_tr)
    unseen = X_te.head(1).assign(Geography="Italy")
    row = pre.transform(unseen)
    geo_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if "Geography" in n]
    assert np.asarray(row)[0, geo_cols].sum() == 0  # unknown -> all zeros, no crash


def test_output_has_8_numeric_plus_5_onehot_columns(split):
    X_tr, _, _, _ = split
    for scale in (True, False):
        assert build_preprocessor(scale).fit_transform(X_tr).shape[1] == 13


def test_random_forest_numeric_columns_are_not_scaled(split):
    X_tr, _, _, _ = split
    out = build_preprocessor(scale_numeric=False).fit_transform(X_tr)
    np.testing.assert_allclose(out[:, :8], X_tr[config.NUMERIC_FEATURES].values)


def test_models_have_required_settings():
    models = build_models()
    lr = models["Logistic Regression"].named_steps["model"]
    rf = models["Random Forest"].named_steps["model"]
    xgb = models["XGBoost"].named_steps["model"]
    assert list(models) == ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    assert lr.max_iter == 1000 and lr.random_state == 42
    assert rf.n_estimators == 200 and rf.random_state == 42
    assert xgb.get_params()["random_state"] == 42 and xgb.get_params()["n_estimators"] == 300
    lgbm = models["LightGBM"].named_steps["model"].get_params()
    cat = models["CatBoost"].named_steps["model"].get_params()
    assert lgbm["random_state"] == 42 and lgbm["n_estimators"] == 300
    assert cat["random_seed"] == 42 and cat["iterations"] == 300
    # The tree models get unscaled inputs; only Logistic Regression is scaled.
    for name in ("Random Forest", "XGBoost", "LightGBM", "CatBoost"):
        assert models[name].named_steps["preprocess"].transformers[0][1] == "passthrough"


def test_preprocessor_output_is_dense(split):
    """XGBoost would treat zeros missing from a sparse matrix as missing values."""
    import scipy.sparse as sp

    X_tr, _, _, _ = split
    assert not sp.issparse(build_preprocessor(scale_numeric=False).fit_transform(X_tr))


def test_class_weighting_reaches_every_model(split):
    _, _, y_tr, _ = split
    balanced = build_models("balanced", y_tr)
    assert balanced["Logistic Regression"].named_steps["model"].class_weight == "balanced"
    assert balanced["Random Forest"].named_steps["model"].class_weight == "balanced"
    expected = (y_tr == 0).sum() / (y_tr == 1).sum()
    assert balanced["XGBoost"].named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(expected)
    assert balanced["LightGBM"].named_steps["model"].class_weight == "balanced"
    assert balanced["CatBoost"].named_steps["model"].get_params()["auto_class_weights"] == "Balanced"
    default = build_models()
    assert default["Random Forest"].named_steps["model"].class_weight is None
    assert default["XGBoost"].named_steps["model"].get_params()["scale_pos_weight"] is None
    assert default["LightGBM"].named_steps["model"].class_weight is None
    assert "auto_class_weights" not in default["CatBoost"].named_steps["model"].get_params()


def test_class_weighting_validates_its_inputs():
    with pytest.raises(ValueError):
        build_models("balanced")  # y_train is required for XGBoost's weight
    with pytest.raises(ValueError):
        build_models("heavy")


def test_predictions_cover_every_test_row_and_are_probabilities(split):
    X_tr, X_te, y_tr, _ = split
    preds = predict_models(train_models(build_models(), X_tr, y_tr), X_te)
    for p in preds.values():
        assert p["y_pred"].shape == (len(X_te),)
        assert p["y_score"].shape == (len(X_te),)
        assert ((p["y_score"] >= 0) & (p["y_score"] <= 1)).all()


def test_positive_class_proba_uses_classes_order_not_column_position(split):
    """If classes_ were [1, 0], the Exited=1 column would be column 0."""
    X_tr, X_te, y_tr, _ = split
    pipe = train_models({"lr": build_models()["Logistic Regression"]}, X_tr, y_tr)["lr"]

    class Reordered:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            return pipe.predict_proba(X)[:, ::-1]

    np.testing.assert_allclose(
        positive_class_proba(Reordered(), X_te), pipe.predict_proba(X_te)[:, 1]
    )


def test_training_is_reproducible(split):
    X_tr, X_te, y_tr, _ = split
    first = predict_models(train_models(build_models(), X_tr, y_tr), X_te)
    second = predict_models(train_models(build_models(), X_tr, y_tr), X_te)
    for name in first:
        np.testing.assert_array_equal(first[name]["y_score"], second[name]["y_score"])

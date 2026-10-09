from churn import config
from churn.data import make_train_test_split, split_features_target
from churn.explain import explain_prediction, typical_customer
from churn.models import build_models, train_models

import pandas as pd


def _fitted_forest(small_df):
    X, y = split_features_target(small_df)
    X_tr, _, y_tr, _ = make_train_test_split(X, y)
    model = train_models({"rf": build_models()["Random Forest"]}, X_tr, y_tr)["rf"]
    return model, typical_customer(X_tr)


def test_typical_customer_uses_training_values(small_df):
    X, y = split_features_target(small_df)
    X_tr, _, _, _ = make_train_test_split(X, y)
    typical = typical_customer(X_tr)
    assert set(typical) == set(config.FEATURES)
    assert typical["Age"] == X_tr["Age"].median()
    assert typical["Geography"] == X_tr["Geography"].mode().iloc[0]


def test_typical_customer_has_nothing_to_explain(small_df):
    model, typical = _fitted_forest(small_df)
    customer = pd.DataFrame([typical])[config.FEATURES]
    assert explain_prediction(model, customer, typical).empty


def test_only_changed_features_are_explained(small_df):
    model, typical = _fitted_forest(small_df)
    customer = pd.DataFrame([typical])[config.FEATURES].assign(Age=typical["Age"] + 20)
    explanation = explain_prediction(model, customer, typical)
    assert list(explanation["feature"]) == ["Age"]


def test_rounded_display_values_count_as_unchanged(small_df):
    """The UI shows the typical Balance with 2 decimals; that rounding must not appear as a driver."""
    model, typical = _fitted_forest(small_df)
    customer = pd.DataFrame([typical])[config.FEATURES].assign(Balance=round(typical["Balance"], 2) + 0.004)
    assert explain_prediction(model, customer, typical).empty

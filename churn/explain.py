"""Simple per-customer explanation of a fitted model's churn probability."""

import numpy as np
import pandas as pd

from churn import config
from churn.models import positive_class_proba

# Numeric columns with few distinct values: "typical" means most common, not median.
DISCRETE_NUMERIC = ["NumOfProducts", "HasCrCard", "IsActiveMember"]


def typical_customer(X_train: pd.DataFrame) -> dict:
    """Median for continuous features, most common value for the rest (training data only)."""
    typical = {}
    for col in config.FEATURES:
        if col in config.CATEGORICAL_FEATURES or col in DISCRETE_NUMERIC:
            typical[col] = X_train[col].mode().iloc[0]
        else:
            typical[col] = X_train[col].median()
    return typical


def _same_value(a, b) -> bool:
    """Equal, or numerically equal up to rounding (e.g. a median shown with 2 decimals)."""
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.isclose(a, b, rtol=1e-6, atol=0.01))


def explain_prediction(model, customer: pd.DataFrame, typical: dict) -> pd.DataFrame:
    """Local "what-if" explanation for one customer.

    For each feature where the customer differs from the typical customer,
    replace that one value with the typical value and measure how the model's
    churn probability changes. A positive effect means the customer's actual
    value raises the predicted risk. This describes the model, not causes of churn.
    """
    base = positive_class_proba(model, customer)[0]
    rows = []
    for col in config.FEATURES:
        if _same_value(customer[col].iloc[0], typical[col]):
            continue
        what_if = customer.copy()
        what_if[col] = typical[col]
        rows.append({"feature": col, "effect": base - positive_class_proba(model, what_if)[0]})
    return pd.DataFrame(rows, columns=["feature", "effect"])

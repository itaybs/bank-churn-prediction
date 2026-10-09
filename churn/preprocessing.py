"""Preprocessing: one-hot encoding for categories, optional scaling for numbers.

The ColumnTransformer returned here is *unfitted*. It is placed inside each
model's Pipeline, so it only ever learns from the training rows passed to
Pipeline.fit(). That is what prevents data leakage from the test set.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the column transformer.

    scale_numeric=True  -> StandardScaler on numeric columns (Logistic Regression
                           is sensitive to feature scale).
    scale_numeric=False -> numeric columns passed through unchanged (tree models
                           such as Random Forest do not need scaling).

    Each column group is its own Pipeline, so extra steps (e.g. a SimpleImputer
    if missing values ever appear) can be added in one place.
    """
    numeric_steps = [("scaler", StandardScaler())] if scale_numeric else []
    numeric_transformer = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # anything not listed (e.g. identifiers) is dropped
        # Always return a dense array: XGBoost treats values missing from a sparse
        # matrix as *missing*, which would corrupt one-hot zeros and Balance = 0.
        sparse_threshold=0,
    )

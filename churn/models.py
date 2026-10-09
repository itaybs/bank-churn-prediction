"""Model pipelines, training, and prediction.

Each model is a Pipeline(preprocessor -> classifier). Calling fit() on the
pipeline fits the preprocessor *and* the classifier on the training rows only;
predict()/predict_proba() then reuse what was learned, without refitting.
"""

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config
from churn.preprocessing import build_preprocessor

POSITIVE_CLASS = 1  # Exited=1 means the customer left

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
TREE_MODELS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
BOOSTING_MODELS = ["XGBoost", "LightGBM", "CatBoost"]

# Baseline settings, documented here in one place. None of them were tuned on
# the test set; they are common, conservative starting values.
LOGISTIC_PARAMS = {
    "max_iter": 1000,  # enough iterations for the lbfgs solver to converge
    "random_state": config.RANDOM_STATE,
}
FOREST_PARAMS = {
    "n_estimators": 200,  # number of trees averaged
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
}
XGBOOST_PARAMS = {
    "n_estimators": 300,  # boosting rounds (trees added one after another)
    "learning_rate": 0.05,  # shrinks each tree's contribution; smaller = more conservative
    "max_depth": 4,  # shallow trees limit how complex each interaction can be
    "subsample": 0.8,  # each tree sees a random 80% of the training rows
    "colsample_bytree": 0.8,  # ...and a random 80% of the features
    "reg_lambda": 1.0,  # L2 penalty on leaf weights (XGBoost default)
    "tree_method": "hist",  # fast histogram-based split finding
    "eval_metric": "logloss",
    "importance_type": "gain",  # feature_importances_ = average loss reduction per split
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,  # boosting rounds
    "learning_rate": 0.05,
    "num_leaves": 15,  # LightGBM grows trees leaf-wise; this caps the leaves per tree
    "subsample": 0.8,  # row subsampling ("bagging" in LightGBM terms)...
    "subsample_freq": 1,  # ...applied at every boosting round
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,  # L2 penalty on leaf weights
    "importance_type": "gain",  # feature_importances_ = total loss reduction per feature
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,  # silence LightGBM's training log
}
CATBOOST_PARAMS = {
    "iterations": 300,  # boosting rounds (CatBoost's default is 1000; 300 keeps training fast)
    "learning_rate": 0.05,
    "depth": 6,  # CatBoost default: symmetric (oblivious) trees of depth 6
    "l2_leaf_reg": 3.0,  # L2 penalty on leaf values (CatBoost default)
    "random_seed": config.RANDOM_STATE,
    "verbose": 0,  # silence the training log
    "allow_writing_files": False,  # do not create a catboost_info/ folder
    "thread_count": -1,
}

CLASS_WEIGHT_OPTIONS = [None, "balanced"]


def balanced_pos_weight(y_train) -> float:
    """Weight for Exited=1 that balances the classes: (#stayed / #churned) in the training labels."""
    y = np.asarray(y_train)
    return float((y == 0).sum() / (y == 1).sum())


def build_models(class_weight: str | None = None, y_train=None) -> dict[str, Pipeline]:
    """Return the unfitted model pipelines (MODEL_NAMES), keyed by display name.

    class_weight=None gives every training row equal weight (the project default).
    class_weight="balanced" up-weights the minority class (churners) so both classes
    contribute equally to the training loss:
      - Logistic Regression / Random Forest: scikit-learn's class_weight="balanced".
      - LightGBM: class_weight="balanced" (same formula as scikit-learn).
      - CatBoost: auto_class_weights="Balanced".
      - XGBoost: scale_pos_weight = #stayed / #churned, computed from y_train
        (the training labels only, so no test information is used).
    To add a new model, add another named Pipeline to the returned dict.
    """
    if class_weight not in CLASS_WEIGHT_OPTIONS:
        raise ValueError(f"class_weight must be one of {CLASS_WEIGHT_OPTIONS}, got {class_weight!r}")
    xgb_params = dict(XGBOOST_PARAMS)
    if class_weight == "balanced":
        if y_train is None:
            raise ValueError("y_train is required to compute XGBoost's balanced scale_pos_weight")
        xgb_params["scale_pos_weight"] = balanced_pos_weight(y_train)
    cat_params = dict(CATBOOST_PARAMS)
    if class_weight == "balanced":
        cat_params["auto_class_weights"] = "Balanced"

    return {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(**LOGISTIC_PARAMS, class_weight=class_weight)),
        ]),
        # Trees split on thresholds, so scaling would change nothing.
        "Random Forest": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(**FOREST_PARAMS, class_weight=class_weight)),
        ]),
        "XGBoost": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", XGBClassifier(**xgb_params)),
        ]),
        "LightGBM": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", LGBMClassifier(**LIGHTGBM_PARAMS, class_weight=class_weight)),
        ]),
        # The categorical features are already one-hot encoded by the shared preprocessor, so CatBoost
        # sees the same input as the other models (its native categorical handling is not used).
        "CatBoost": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", CatBoostClassifier(**cat_params)),
        ]),
    }


def train_models(models: dict[str, Pipeline], X_train, y_train) -> dict[str, Pipeline]:
    """Fit every pipeline on the same training data."""
    for pipeline in models.values():
        pipeline.fit(X_train, y_train)
    return models


def positive_class_proba(pipeline: Pipeline, X) -> np.ndarray:
    """Probability of Exited=1.

    The column is located through classes_ rather than assuming it is column 1.
    """
    classes = list(pipeline.classes_)
    if POSITIVE_CLASS not in classes:
        raise ValueError(f"Positive class {POSITIVE_CLASS} not in model classes {classes}")
    return pipeline.predict_proba(X)[:, classes.index(POSITIVE_CLASS)]


def predict_models(models: dict[str, Pipeline], X_test) -> dict[str, dict[str, np.ndarray]]:
    """Class labels (default 0.5 threshold) and Exited=1 probabilities per model."""
    predictions = {}
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_score = positive_class_proba(pipeline, X_test)
        if len(y_pred) != len(X_test) or len(y_score) != len(X_test):
            raise ValueError(f"{name}: prediction length does not match test rows")
        predictions[name] = {"y_pred": y_pred, "y_score": y_score}
    return predictions

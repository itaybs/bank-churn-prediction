"""Stratified k-fold cross-validation on the training set.

The training set is split into k folds that keep the churn rate of the whole
training set. Each fold is held out once while a fresh copy of the full
pipeline (preprocessing + model) is fitted on the other k-1 folds. This gives:

- k scores per metric, whose mean estimates performance on unseen data and
  whose standard deviation shows how stable that estimate is;
- an out-of-fold probability for every training row, from a model that did
  not see that row, which is what the threshold selection uses.

The test set is never touched here.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn import config
from churn.formatting import fmt_mean_std
from churn.models import POSITIVE_CLASS, positive_class_proba

CV_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
CV_SCORING = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "Recall": make_scorer(recall_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "F1": make_scorer(f1_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "ROC_AUC": "roc_auc",
}


def make_cv(n_splits: int = config.CV_FOLDS) -> StratifiedKFold:
    """The same folds for every model (fixed seed), so fold scores can be compared pairwise."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def cross_validate_model(pipeline, X_train, y_train, n_splits: int = config.CV_FOLDS):
    """Return (fold_scores, out_of_fold_scores) for one unfitted pipeline.

    fold_scores: DataFrame, one row per fold, one column per metric
                 (labels from predict(), i.e. the default 0.5 threshold).
    out_of_fold_scores: P(Exited=1) for every training row, in X_train's order.
    """
    result = cross_validate(
        clone(pipeline), X_train, y_train, cv=make_cv(n_splits), scoring=CV_SCORING,
        return_estimator=True, return_indices=True,
    )
    folds = pd.DataFrame({metric: result[f"test_{metric}"] for metric in CV_METRICS})
    folds.index = pd.Index([f"Fold {i + 1}" for i in range(len(folds))], name="Fold")

    oof = np.full(len(X_train), np.nan)
    for estimator, test_idx in zip(result["estimator"], result["indices"]["test"]):
        oof[test_idx] = positive_class_proba(estimator, X_train.iloc[test_idx])
    if np.isnan(oof).any():
        raise RuntimeError("Some training rows received no out-of-fold prediction")
    return folds, oof


def summarize_cv(folds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean and standard deviation per metric: one row per model, columns like 'F1 mean', 'F1 std'."""
    rows = {}
    for name, folds in folds_by_model.items():
        row = {}
        for metric in CV_METRICS:
            row[f"{metric} mean"] = folds[metric].mean()
            row[f"{metric} std"] = folds[metric].std(ddof=1)
        rows[name] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Model"
    return table


def format_mean_std(summary: pd.DataFrame, sep: str = " ± ") -> pd.DataFrame:
    """'0.812 ± 0.011' strings, for display (pass sep=" +/- " for plain-ASCII consoles)."""
    return pd.DataFrame(
        {m: [fmt_mean_std(r[f"{m} mean"], r[f"{m} std"], sep) for _, r in summary.iterrows()] for m in CV_METRICS},
        index=summary.index,
    )

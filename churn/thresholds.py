"""Decision-threshold selection using cross-validation on the training data only.

A classifier outputs a probability; a threshold turns it into a 0/1 decision.
The default threshold is 0.5. With imbalanced classes (about 20% churners) a
lower threshold often finds more churners at the cost of more false alarms.

The threshold must never be chosen on the test set, or the test score would no
longer be an honest estimate. Instead we use out-of-fold predictions: each
training row is scored by a model that did not see that row during fitting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from churn import config
from churn.models import POSITIVE_CLASS

DEFAULT_THRESHOLD = 0.5
CV_FOLDS = config.CV_FOLDS
# Out-of-fold scores come from churn.validation.cross_validate_model.


def threshold_curve(y_true, y_score) -> pd.DataFrame:
    """Precision, Recall and F1 of Exited=1 at every candidate threshold (score >= threshold)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=POSITIVE_CLASS)
    precision, recall = precision[:-1], recall[:-1]  # the last point has no threshold
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall, "f1": f1})


def select_f1_threshold(y_true, y_score) -> float:
    """The threshold that maximises F1 for Exited=1."""
    curve = threshold_curve(y_true, y_score)
    return float(curve.loc[curve["f1"].idxmax(), "threshold"])


def select_cost_threshold(y_true, y_score, cost_fn: float, cost_fp: float) -> float:
    """The threshold that minimises total cost = cost_fn * missed churners + cost_fp * false alarms."""
    y_true = np.asarray(y_true)
    candidates = np.unique(y_score)
    costs = [
        cost_fn * np.sum((y_score < t) & (y_true == 1)) + cost_fp * np.sum((y_score >= t) & (y_true == 0))
        for t in candidates
    ]
    return float(candidates[int(np.argmin(costs))])


def apply_threshold(y_score, threshold: float) -> np.ndarray:
    """Predict Exited=1 when the score reaches the threshold."""
    return (np.asarray(y_score) >= threshold).astype(int)

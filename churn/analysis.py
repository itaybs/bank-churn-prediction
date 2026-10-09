"""Everything the dashboard shows, computed from the project's own components.

run_analysis() does all expensive work once per class-weighting setting:
training-set cross-validation (fold scores + out-of-fold probabilities),
F1-optimal thresholds chosen from those out-of-fold probabilities, the final
fit on the full training set, test-set probabilities, and feature importances.

Turning probabilities into decisions at a given threshold is cheap, so it is
done on demand with Analysis.evaluate().
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from churn import config
from churn.data import load_dataset, make_train_test_split, split_features_target, validate_dataset
from churn.evaluation import build_comparison_table, compute_metrics
from churn.explain import typical_customer
from churn.importance import builtin_importance, permutation_importance_table
from churn.models import TREE_MODELS, build_models, predict_models, train_models
from churn.thresholds import DEFAULT_THRESHOLD, apply_threshold, select_f1_threshold
from churn.validation import cross_validate_model, summarize_cv

STRATEGY_DEFAULT = "Default (0.5 cutoff)"
STRATEGY_OPTIMIZED = "CV-optimized (max F1)"
STRATEGY_CUSTOM = "Custom (slider)"
STRATEGIES = [STRATEGY_DEFAULT, STRATEGY_OPTIMIZED, STRATEGY_CUSTOM]


@dataclass
class Evaluation:
    threshold: float
    y_pred: np.ndarray
    metrics: dict[str, float]
    confusion: np.ndarray  # [[TN, FP], [FN, TP]]


@dataclass
class Analysis:
    df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    class_weight: str | None
    models: dict
    test_scores: dict[str, np.ndarray]
    default_predictions: dict[str, np.ndarray]
    cv_folds: dict[str, pd.DataFrame]
    oof_scores: dict[str, np.ndarray]
    tuned_thresholds: dict[str, float]
    builtin_importance: pd.DataFrame  # original features x tree models
    builtin_importance_onehot: pd.DataFrame  # one-hot columns x tree models
    permutation_importance: dict[str, pd.DataFrame]  # empty if skipped
    train_table: pd.DataFrame = None  # final models scored on their own training data (threshold 0.5)
    typical: dict = field(default_factory=dict)

    @property
    def base_rate(self) -> float:
        """Churn rate in the training data."""
        return float(self.y_train.mean())

    @property
    def model_names(self) -> list[str]:
        return list(self.models)

    def cv_summary(self) -> pd.DataFrame:
        return summarize_cv(self.cv_folds)

    def thresholds_for(self, strategy: str, custom: float | None = None) -> dict[str, float]:
        if strategy == STRATEGY_DEFAULT:
            return {name: DEFAULT_THRESHOLD for name in self.models}
        if strategy == STRATEGY_OPTIMIZED:
            return dict(self.tuned_thresholds)
        if strategy == STRATEGY_CUSTOM:
            if custom is None:
                raise ValueError("A custom threshold value is required")
            return {name: float(custom) for name in self.models}
        raise ValueError(f"Unknown strategy {strategy!r}")

    def evaluate(self, name: str, threshold: float, use_predict: bool = False) -> Evaluation:
        """Test-set metrics for one model at one threshold.

        use_predict=True uses the model's own predict() labels (the default 0.5 rule,
        identical to main.py) instead of applying the threshold to the probabilities.
        """
        scores = self.test_scores[name]
        y_pred = self.default_predictions[name] if use_predict else apply_threshold(scores, threshold)
        return Evaluation(
            threshold=threshold,
            y_pred=y_pred,
            metrics=compute_metrics(self.y_test, y_pred, scores),
            confusion=confusion_matrix(self.y_test, y_pred, labels=[0, 1]),
        )

    def evaluate_strategy(self, strategy: str, custom: float | None = None) -> dict[str, Evaluation]:
        thresholds = self.thresholds_for(strategy, custom)
        return {
            name: self.evaluate(name, t, use_predict=(strategy == STRATEGY_DEFAULT))
            for name, t in thresholds.items()
        }

    @staticmethod
    def metrics_table(evaluations: dict[str, Evaluation]) -> pd.DataFrame:
        table = pd.DataFrame.from_dict({n: e.metrics for n, e in evaluations.items()}, orient="index")
        table.index.name = "Model"
        return table


def compute_permutation_importance(models: dict, X_test, y_test, n_repeats: int = 10) -> dict[str, pd.DataFrame]:
    """Permutation importance on the test set, for reporting only (nothing is selected with it)."""
    return {name: permutation_importance_table(m, X_test, y_test, n_repeats=n_repeats) for name, m in models.items()}


def run_analysis(
    data_path: Path = config.DATA_PATH, class_weight: str | None = None, permutation_repeats: int | None = 10
) -> Analysis:
    """permutation_repeats=None skips permutation importance (it is the slowest step)."""
    df = load_dataset(data_path)
    validate_dataset(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    # 1. Cross-validation on the training set only (fresh, unfitted pipelines per fold).
    cv_folds, oof_scores, tuned = {}, {}, {}
    for name, pipeline in build_models(class_weight, y_train).items():
        cv_folds[name], oof_scores[name] = cross_validate_model(pipeline, X_train, y_train)
        tuned[name] = select_f1_threshold(y_train, oof_scores[name])

    # 2. Final fit on the full training set, then score the untouched test set.
    models = train_models(build_models(class_weight, y_train), X_train, y_train)
    predictions = predict_models(models, X_test)
    # Training-set scores, only to show how closely each model fits its training data (overfitting gap).
    train_table = build_comparison_table(y_train, predict_models(models, X_train))

    # 3. Feature importance of the final models.
    builtin = pd.DataFrame({n: builtin_importance(models[n]) for n in TREE_MODELS})
    builtin_onehot = pd.DataFrame({n: builtin_importance(models[n], aggregate=False) for n in TREE_MODELS})
    permutation = (
        {} if permutation_repeats is None
        else compute_permutation_importance(models, X_test, y_test, permutation_repeats)
    )

    return Analysis(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        class_weight=class_weight,
        models=models,
        test_scores={name: p["y_score"] for name, p in predictions.items()},
        default_predictions={name: p["y_pred"] for name, p in predictions.items()},
        cv_folds=cv_folds,
        oof_scores=oof_scores,
        tuned_thresholds=tuned,
        builtin_importance=builtin.sort_values(TREE_MODELS[-1], ascending=False),
        builtin_importance_onehot=builtin_onehot.sort_values(TREE_MODELS[-1], ascending=False),
        permutation_importance=permutation,
        train_table=train_table,
        typical=typical_customer(X_train),
    )

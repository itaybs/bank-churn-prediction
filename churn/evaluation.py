"""Metrics, comparison table, confusion matrices, and the ROC plot.

All metrics treat Exited=1 (customer left) as the positive class.
Predictions come from the default 0.5 threshold; the threshold is never tuned
on the test set.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.formatting import fmt3
from churn.models import MODEL_NAMES, POSITIVE_CLASS

CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Stayed (0)", "Exited (1)"]

# Reference categorical palette slots 1-5 in fixed order (colorblind-validated as a set), gray baseline.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BASELINE_COLOR = "#8a8984"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Precision/Recall/F1 for Exited=1, Accuracy, and ROC-AUC from probabilities."""
    return {
        "Precision": precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_score),
    }


def build_comparison_table(y_true, predictions: dict) -> pd.DataFrame:
    """One row per model, one column per metric."""
    rows = {
        name: compute_metrics(y_true, p["y_pred"], p["y_score"])
        for name, p in predictions.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Model"
    if table.isna().any().any():
        raise ValueError("Comparison table contains missing metric values")
    return table


def confusion_matrices(y_true, predictions: dict) -> dict[str, np.ndarray]:
    """Rows = actual class, columns = predicted class, in order [0, 1]."""
    return {
        name: confusion_matrix(y_true, p["y_pred"], labels=CLASS_LABELS)
        for name, p in predictions.items()
    }


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR)


def plot_roc_curves(y_true, predictions: dict, path: Path) -> Path:
    """One ROC plot with every model plus the random-classifier diagonal."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for (name, p), color in zip(predictions.items(), MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y_true, p["y_score"], pos_label=POSITIVE_CLASS)
        auc = roc_auc_score(y_true, p["y_score"])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=BASELINE_COLOR,
            label="Random classifier (AUC = 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (stayed customers flagged as churn)", color=TEXT_COLOR)
    ax.set_ylabel("True positive rate / Recall (churners identified)", color=TEXT_COLOR)
    ax.set_title("ROC curves on the test set (positive class: Exited = 1)", color=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    _style_axes(ax)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrices(matrices: dict[str, np.ndarray], path: Path) -> Path:
    """Side-by-side labeled confusion matrices (counts and row percentages)."""
    ncols = min(len(matrices), 3)
    nrows = int(np.ceil(len(matrices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), dpi=150, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(matrices):]:
        ax.set_visible(False)
    vmax = max(m.max() for m in matrices.values())
    for ax, (name, cm) in zip(axes, matrices.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        row_totals = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / row_totals[i, 0] if row_totals[i, 0] else 0
                dark_cell = cm[i, j] > vmax * 0.55
                ax.text(j, i, f"{cm[i, j]}\n({share:.1%} of actual)", ha="center", va="center",
                        fontsize=11, color="white" if dark_cell else TEXT_COLOR)
        ax.set_xticks([0, 1], labels=CLASS_NAMES)
        ax.set_yticks([0, 1], labels=CLASS_NAMES)
        ax.set_xlabel("Predicted class", color=TEXT_COLOR)
        ax.set_ylabel("Actual class", color=TEXT_COLOR)
        ax.set_title(name, color=TEXT_COLOR)
        ax.tick_params(colors=MUTED_COLOR, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Confusion matrices on the test set (threshold 0.5)", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_comparison_table(table: pd.DataFrame, path: Path) -> Path:
    table.round(3).to_csv(path, float_format="%.3f")
    return path


def plot_feature_importance(builtin: pd.DataFrame, permutation: dict[str, pd.DataFrame], path: Path) -> Path:
    """Left: built-in importance of the tree models. Right: permutation importance of every model."""
    order = builtin.mean(axis=1).sort_values().index
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    y = np.arange(len(order))

    width = 0.8 / len(builtin.columns)
    for i, model in enumerate(builtin.columns):
        color = MODEL_COLORS[list(permutation).index(model)]
        ax1.barh(y + i * width - 0.4 + width / 2, builtin.loc[order, model], height=width, color=color, label=model)
    ax1.set_yticks(y, labels=order)
    ax1.set_xlabel("Share of total importance (RF: impurity decrease; boosting models: gain / prediction change)",
                   color=TEXT_COLOR)
    ax1.set_title("Built-in importance (tree models, training data)", color=TEXT_COLOR)

    order2 = pd.DataFrame({m: t["importance_mean"] for m, t in permutation.items()}).mean(axis=1).sort_values().index
    width = 0.8 / len(permutation)
    for i, (model, table) in enumerate(permutation.items()):
        ax2.barh(y + i * width - 0.4 + width / 2, table.loc[order2, "importance_mean"], height=width,
                 xerr=table.loc[order2, "importance_std"], color=MODEL_COLORS[i], label=model,
                 error_kw=dict(ecolor=MUTED_COLOR, lw=0.8))
    ax2.set_yticks(y, labels=order2)
    ax2.axvline(0, color=GRID_COLOR, linewidth=1)
    ax2.set_xlabel("Drop in test ROC-AUC when the feature is shuffled", color=TEXT_COLOR)
    ax2.set_title("Permutation importance (test set)", color=TEXT_COLOR)

    for ax in (ax1, ax2):
        _style_axes(ax)
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right", labelcolor=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_tuning_before_after(comparison: pd.DataFrame, path: Path) -> Path:
    """Baseline vs tuned scores per tuned model: CV mean (left) and test set (right).

    comparison: churn.tuning.comparison_table() (long format).
    """
    metrics = ["Precision", "Recall", "F1", "ROC_AUC"]
    models = list(dict.fromkeys(comparison["Model"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
    x = np.arange(len(metrics))
    width = 0.8 / (2 * len(models))
    for ax, stage in zip(axes, ["CV mean", "Test"]):
        data = comparison[comparison["Stage"] == stage].set_index(["Model", "Metric"])
        for i, model in enumerate(models):
            color = MODEL_COLORS[MODEL_NAMES.index(model)]
            for j, (version, alpha, hatch) in enumerate([("Baseline", 0.45, "//"), ("Tuned", 1.0, "")]):
                offset = (2 * i + j) * width - 0.4 + width / 2
                values = [data.loc[(model, m), version] for m in metrics]
                bars = ax.bar(x + offset, values, width=width * 0.92, color=color, alpha=alpha, hatch=hatch,
                              edgecolor="white", linewidth=0.5, label=f"{model} · {version}")
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, fmt3(v), ha="center", va="bottom",
                            fontsize=6.5, color=MUTED_COLOR, rotation=90)
        ax.set_xticks(x, labels=[m.replace("_", "-") for m in metrics])
        ax.set_ylim(0, 1.08)
        ax.set_title("Cross-validation on the training set (mean of 5 folds)" if stage == "CV mean"
                     else "Untouched test set (threshold 0.5)", color=TEXT_COLOR)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        _style_axes(ax)
    axes[0].set_ylabel("Score", color=TEXT_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels), fontsize=9, labelcolor=TEXT_COLOR)
    fig.suptitle("Hyperparameter tuning: before (hatched) vs after (solid)", color=TEXT_COLOR)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path

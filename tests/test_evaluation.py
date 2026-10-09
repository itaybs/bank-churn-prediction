import numpy as np
import pandas as pd
import pytest

from churn.evaluation import (
    build_comparison_table,
    compute_metrics,
    confusion_matrices,
    plot_confusion_matrices,
    plot_roc_curves,
    save_comparison_table,
)

# Hand-checkable example: 4 actual churners, 6 stayers.
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])  # TP=3, FN=1, FP=1, TN=5
Y_SCORE = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1, 0.1, 0.05])


def test_metrics_use_exited_1_as_positive_class():
    m = compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)
    assert m["Precision"] == pytest.approx(3 / 4)
    assert m["Recall"] == pytest.approx(3 / 4)
    assert m["F1"] == pytest.approx(0.75)
    assert m["Accuracy"] == pytest.approx(8 / 10)
    # 24 (churner, stayer) pairs; only (0.3 vs 0.6) and (0.3 vs 0.4) are mis-ranked.
    assert m["ROC_AUC"] == pytest.approx(22 / 24)


def test_auc_from_wrong_class_column_would_be_flipped():
    right = compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)["ROC_AUC"]
    wrong = compute_metrics(Y_TRUE, Y_PRED, 1 - Y_SCORE)["ROC_AUC"]
    assert wrong == pytest.approx(1 - right)


def test_no_predicted_churners_gives_zero_precision_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_metrics(Y_TRUE, np.zeros_like(Y_PRED), Y_SCORE)
    assert m["Precision"] == 0 and m["Recall"] == 0 and m["F1"] == 0


def test_confusion_matrix_orientation_rows_actual_columns_predicted():
    cm = confusion_matrices(Y_TRUE, {"m": {"y_pred": Y_PRED, "y_score": Y_SCORE}})["m"]
    tn, fp, fn, tp = cm.ravel()
    assert (tn, fp, fn, tp) == (5, 1, 1, 3)


def test_outputs_are_written(tmp_path):
    preds = {
        "A": {"y_pred": Y_PRED, "y_score": Y_SCORE},
        "B": {"y_pred": np.zeros_like(Y_PRED), "y_score": Y_SCORE[::-1]},
    }
    table = build_comparison_table(Y_TRUE, preds)
    assert list(table.index) == ["A", "B"]
    assert not table.isna().any().any()

    csv = save_comparison_table(table, tmp_path / "t.csv")
    roc = plot_roc_curves(Y_TRUE, preds, tmp_path / "roc.png")
    cmp = plot_confusion_matrices(confusion_matrices(Y_TRUE, preds), tmp_path / "cm.png")
    for path in (csv, roc, cmp):
        assert path.exists() and path.stat().st_size > 0
    saved = pd.read_csv(csv, index_col="Model")
    assert list(saved.columns) == ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC"]

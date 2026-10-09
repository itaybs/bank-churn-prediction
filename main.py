"""Run the full churn-prediction workflow with one command:

    python main.py

Steps: load + validate data -> define X and y -> train/test split ->
build pipelines -> 5-fold cross-validation on the training set ->
train all models -> predict and evaluate on the test set ->
GridSearchCV tuning (Random Forest, XGBoost) and before-vs-after comparison ->
feature importance -> save tables, plots and tuned models -> write the summary report.
"""

import sys
from pathlib import Path

import pandas as pd

from churn import config
from churn.data import (
    DataValidationError,
    class_distribution,
    load_dataset,
    make_train_test_split,
    split_features_target,
    validate_dataset,
)
from churn.evaluation import (
    build_comparison_table,
    confusion_matrices,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_roc_curves,
    plot_tuning_before_after,
    save_comparison_table,
)
from churn.importance import builtin_importance, permutation_importance_table
from churn.insights import best_model
from churn.models import TREE_MODELS, build_models, predict_models, train_models
from churn.report import write_report
from churn.tuning import (
    PARAM_GRIDS,
    candidates_table,
    comparison_table,
    grid_results,
    n_candidates,
    params_table,
    save_tuned_models,
    tune_models,
    tuned_cv_folds,
)
from churn.validation import cross_validate_model, format_mean_std, summarize_cv

STEPS = 11


def step(n: int, text: str) -> None:
    print(f"[{n}/{STEPS}] {text}")


def save_cv_results(summary: pd.DataFrame, folds: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    summary_path = output_dir / "cv_results.csv"
    summary.round(4).to_csv(summary_path, float_format="%.4f")
    folds_path = output_dir / "cv_folds.csv"
    long = pd.concat({name: f for name, f in folds.items()}, names=["Model", "Fold"])
    long.round(4).to_csv(folds_path, float_format="%.4f")
    return [summary_path, folds_path]


def save_importance(builtin: pd.DataFrame, permutation: dict[str, pd.DataFrame], path: Path) -> Path:
    table = builtin.add_prefix("builtin_")
    for name, perm in permutation.items():
        table[f"permutation_{name}"] = perm["importance_mean"]
    table.index.name = "Feature"
    table.sort_values(f"permutation_{TREE_MODELS[-1]}", ascending=False).round(4).to_csv(path, float_format="%.4f")
    return path


def save_tuning_results(comparison: pd.DataFrame, params: pd.DataFrame, searches: dict,
                        tuned_folds: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    comparison_path = output_dir / "tuning_comparison.csv"
    comparison.round(4).to_csv(comparison_path, index=False, float_format="%.4f")
    params_path = output_dir / "tuning_best_params.csv"
    params.to_csv(params_path, index=False)
    folds_path = output_dir / "tuning_cv_folds.csv"
    # Full precision, so the dashboard reproduces the report's numbers exactly (no double rounding).
    pd.concat({name: f for name, f in tuned_folds.items()}, names=["Model", "Fold"]).to_csv(folds_path)
    grid_path = output_dir / "tuning_grid_results.csv"
    pd.concat({name: grid_results(s) for name, s in searches.items()}, names=["Model", "Candidate"]).to_csv(grid_path)
    return [comparison_path, params_path, folds_path, grid_path]


def run(data_path: Path = config.DATA_PATH, output_dir: Path = config.OUTPUT_DIR,
        param_grids: dict | None = None) -> list[Path]:
    """Execute every step and return the list of files written.

    param_grids overrides churn.tuning.PARAM_GRIDS (the tests pass tiny grids to stay fast).
    """
    param_grids = PARAM_GRIDS if param_grids is None else param_grids
    step(1, f"Loading and validating {data_path}")
    df = load_dataset(data_path)
    validate_dataset(df)

    step(2, "Defining features X and target y")
    X, y = split_features_target(df)

    step(3, "Splitting into train/test sets (80/20, stratified)")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    step(4, "Building preprocessing + model pipelines")
    models = build_models()

    step(5, f"{config.CV_FOLDS}-fold stratified cross-validation on the training set")
    cv_folds = {name: cross_validate_model(pipeline, X_train, y_train)[0] for name, pipeline in models.items()}
    cv_summary = summarize_cv(cv_folds)
    print("\n  Cross-validation (mean +/- std over folds, threshold 0.5):")
    print("  " + format_mean_std(cv_summary, sep=" +/- ").to_string().replace("\n", "\n  ") + "\n")

    step(6, "Training every model on the full training set")
    train_models(models, X_train, y_train)

    step(7, "Predicting on the test set and evaluating")
    predictions = predict_models(models, X_test)
    table = build_comparison_table(y_test, predictions)
    matrices = confusion_matrices(y_test, predictions)
    print("\n  Test set (threshold 0.5):")
    print("  " + table.round(3).to_string().replace("\n", "\n  ") + "\n")
    # Selection uses cross-validation on the training set; the test set only confirms it.
    selected = best_model(cv_summary)
    confirmed = table["ROC_AUC"].idxmax() == selected
    print(f"  Selected model (highest CV ROC-AUC): {selected} - test set "
          f"{'confirms' if confirmed else 'does NOT confirm'} the choice.\n")
    # Training-set scores, only to show how closely each model fits its training data.
    train_table = build_comparison_table(y_train, predict_models(models, X_train))

    candidates = sum(n_candidates(g) for g in param_grids.values())
    step(8, f"GridSearchCV tuning of {', '.join(param_grids)} ({candidates} candidates x "
            f"{config.CV_FOLDS} folds, training set only; this is the slowest step)")
    searches = tune_models(X_train, y_train, param_grids)
    tuned_folds = {name: tuned_cv_folds(s) for name, s in searches.items()}
    tuned_cv = summarize_cv(tuned_folds)
    tuned_predictions = predict_models({n: s.best_estimator_ for n, s in searches.items()}, X_test)
    tuned_table = build_comparison_table(y_test, tuned_predictions)
    comparison = comparison_table(cv_summary, tuned_cv, table, tuned_table)
    params = params_table(searches)
    print("\n  Before vs after tuning (ROC-AUC):")
    roc = comparison[comparison["Metric"] == "ROC_AUC"].set_index(["Model", "Stage"])[["Baseline", "Tuned", "Change"]]
    print("  " + roc.round(3).to_string().replace("\n", "\n  ") + "\n")

    step(9, "Computing feature importance (built-in for tree models, permutation for all)")
    builtin = pd.DataFrame({name: builtin_importance(models[name]) for name in TREE_MODELS})
    permutation = {name: permutation_importance_table(m, X_test, y_test) for name, m in models.items()}

    step(10, f"Saving tables, plots and tuned models to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    outputs = [
        save_comparison_table(table, output_dir / "model_comparison.csv"),
        *save_cv_results(cv_summary, cv_folds, output_dir),
        save_importance(builtin, permutation, output_dir / "feature_importance.csv"),
        plot_roc_curves(y_test, predictions, output_dir / "roc_curves.png"),
        plot_confusion_matrices(matrices, output_dir / "confusion_matrices.png"),
        plot_feature_importance(builtin, permutation, output_dir / "feature_importance.png"),
        *save_tuning_results(comparison, params, searches, tuned_folds, output_dir),
        plot_tuning_before_after(comparison, output_dir / "tuning_before_after.png"),
        *save_tuned_models(searches, output_dir / "models"),
        report_path,
    ]

    step(11, "Writing the summary report")
    write_report(
        report_path,
        dataset_name=Path(data_path).name,
        n_rows=len(df),
        n_columns=df.shape[1],
        distribution=class_distribution(y),
        train_distribution=class_distribution(y_train),
        test_distribution=class_distribution(y_test),
        table=table,
        train_table=train_table,
        matrices=matrices,
        cv_summary=cv_summary,
        cv_folds=cv_folds,
        builtin=builtin,
        permutation=permutation,
        X_train=X_train,
        y_train=y_train,
        tuning=dict(comparison=comparison, params=params, tuned_folds=tuned_folds,
                    cv_candidates=candidates_table(cv_summary, tuned_cv),
                    test_candidates=candidates_table(table, tuned_table)),
        output_files=outputs,
    )

    missing = [p for p in outputs if not p.is_file()]
    if missing:
        raise RuntimeError(f"Expected outputs were not created: {missing}")
    return outputs


def main() -> int:
    try:
        outputs = run()
    except (FileNotFoundError, DataValidationError) as error:
        print(f"ERROR: {error}\nStopping: results would be invalid.", file=sys.stderr)
        return 1
    print("\nDone. Files written:")
    for path in outputs:
        print(f"  {path.relative_to(config.PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

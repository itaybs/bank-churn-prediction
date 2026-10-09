"""Hyperparameter tuning with GridSearchCV, and the before-vs-after comparison.

Random Forest and XGBoost are tuned with an exhaustive grid search scored by
ROC-AUC in 5-fold stratified cross-validation on the training set only. The
folds are the same ones used for the baseline cross-validation
(churn.validation.make_cv), so baseline and tuned fold scores are paired.

Each grid contains the baseline settings, so the search can only keep or
improve on the baseline's cross-validated ROC-AUC. Because the best candidate
is chosen on those same folds, its CV score is slightly optimistic (selection
bias). The untouched test set gives the honest before-vs-after comparison;
nested cross-validation would remove the bias (see the roadmap).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from churn import config
from churn.formatting import fmt3, fmt3_signed
from churn.models import build_models
from churn.validation import CV_METRICS, CV_SCORING, make_cv

TUNING_METRIC = "ROC_AUC"  # GridSearchCV refits the candidate with the best mean CV ROC-AUC

# "model__" addresses the classifier step inside each Pipeline. Baseline values are included.
PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators": [200, 400],  # baseline 200
        "model__max_depth": [None, 8, 12],  # baseline None (grow until leaves are pure)
        "model__min_samples_leaf": [1, 4],  # baseline 1
    },
    "XGBoost": {
        "model__n_estimators": [300, 600],  # baseline 300
        "model__learning_rate": [0.03, 0.05, 0.1],  # baseline 0.05
        "model__max_depth": [3, 4, 5],  # baseline 4
    },
}
TUNED_MODELS = list(PARAM_GRIDS)
TUNED_SUFFIX = " (tuned)"
VERSIONS = ["Baseline", "Tuned"]
COMPARISON_STAGES = ["CV mean", "Test"]


def tuned_name(name: str) -> str:
    return name + TUNED_SUFFIX


def model_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def n_candidates(grid: dict) -> int:
    return int(np.prod([len(v) for v in grid.values()]))


def tune_model(name: str, X_train, y_train, grid: dict | None = None) -> GridSearchCV:
    """Grid search for one model on the training data; refits the best candidate on all training rows.

    n_jobs=1 for the search itself: the forest and XGBoost already use every core,
    and parallel search workers would each copy the data (memory).
    """
    grid = PARAM_GRIDS[name] if grid is None else grid
    search = GridSearchCV(
        build_models()[name], grid, scoring=CV_SCORING, refit=TUNING_METRIC, cv=make_cv(), n_jobs=1,
        error_score="raise",
    )
    return search.fit(X_train, y_train)


def tune_models(X_train, y_train, grids: dict | None = None) -> dict[str, GridSearchCV]:
    grids = PARAM_GRIDS if grids is None else grids
    return {name: tune_model(name, X_train, y_train, grid) for name, grid in grids.items()}


def best_params(search: GridSearchCV) -> dict:
    """Best parameters without the 'model__' prefix."""
    return {k.split("__", 1)[-1]: v for k, v in search.best_params_.items()}


def tuned_cv_folds(search: GridSearchCV) -> pd.DataFrame:
    """Fold scores of the selected candidate, in the same format as churn.validation.cross_validate_model."""
    i, k = search.best_index_, search.n_splits_
    folds = pd.DataFrame({m: [search.cv_results_[f"split{f}_test_{m}"][i] for f in range(k)] for m in CV_METRICS})
    folds.index = pd.Index([f"Fold {f + 1}" for f in range(k)], name="Fold")
    return folds


def grid_results(search: GridSearchCV) -> pd.DataFrame:
    """Every candidate the search tried: parameters, mean and std CV ROC-AUC, rank (1 = selected)."""
    r = search.cv_results_
    # dtype=object keeps None (e.g. max_depth=None) instead of turning it into NaN.
    table = pd.DataFrame([{k.split("__", 1)[-1]: v for k, v in p.items()} for p in r["params"]], dtype=object)
    folds = np.column_stack([r[f"split{f}_test_{TUNING_METRIC}"] for f in range(search.n_splits_)])
    table["CV ROC-AUC mean"] = folds.mean(axis=1)
    table["CV ROC-AUC std"] = folds.std(axis=1, ddof=1)
    table["Rank"] = r[f"rank_test_{TUNING_METRIC}"]
    return table.sort_values("Rank").reset_index(drop=True)


def params_table(searches: dict[str, GridSearchCV]) -> pd.DataFrame:
    """Searched values, baseline value and selected value of every tuned hyperparameter."""
    rows = []
    for name, search in searches.items():
        baseline = build_models()[name].get_params()
        for key, values in search.param_grid.items():
            rows.append({
                "Model": name, "Hyperparameter": key.split("__", 1)[-1],
                "Values searched": ", ".join(str(v) for v in values),
                "Baseline": str(baseline[key]), "Tuned": str(search.best_params_[key]),
            })
    table = pd.DataFrame(rows)
    table["Changed"] = table["Baseline"] != table["Tuned"]
    return table


def comparison_table(baseline_cv: pd.DataFrame, tuned_cv: pd.DataFrame,
                     baseline_test: pd.DataFrame, tuned_test: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (model, stage, metric) with the Baseline value, Tuned value and change.

    baseline_cv / tuned_cv: churn.validation.summarize_cv format ('ROC_AUC mean', ...).
    baseline_test / tuned_test: churn.evaluation.build_comparison_table format.
    The tuned frames are indexed by the base model name (without the ' (tuned)' suffix).
    """
    rows = []
    for name in tuned_cv.index:
        for metric in CV_METRICS:
            for stage, before, after in [
                ("CV mean", baseline_cv.loc[name, f"{metric} mean"], tuned_cv.loc[name, f"{metric} mean"]),
                ("Test", baseline_test.loc[name, metric], tuned_test.loc[name, metric]),
            ]:
                rows.append({"Model": name, "Stage": stage, "Metric": metric,
                             "Baseline": float(before), "Tuned": float(after), "Change": float(after - before)})
    return pd.DataFrame(rows)


def candidates_table(baseline: pd.DataFrame, tuned: pd.DataFrame) -> pd.DataFrame:
    """Baseline models plus the tuned versions (renamed 'X (tuned)'), in one table with the same columns."""
    return pd.concat([baseline, tuned.rename(index=tuned_name)])


def save_tuned_models(searches: dict[str, GridSearchCV], directory: Path) -> list[Path]:
    """Persist each refitted best pipeline (preprocessing + model) with joblib."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, search in searches.items():
        path = directory / f"{model_slug(name)}_tuned.joblib"
        joblib.dump(search.best_estimator_, path)
        paths.append(path)
    return paths


def load_tuned_models(directory: Path) -> dict:
    """Load saved tuned pipelines; models whose file is missing are skipped.

    Only load files this project wrote: joblib files can execute code when loaded.
    """
    return {
        name: joblib.load(directory / f"{model_slug(name)}_tuned.joblib")
        for name in TUNED_MODELS if (directory / f"{model_slug(name)}_tuned.joblib").is_file()
    }


def tuning_insights(comparison: pd.DataFrame, baseline_folds: dict[str, pd.DataFrame],
                    tuned_folds: dict[str, pd.DataFrame], params: pd.DataFrame,
                    cv_candidates: pd.DataFrame, test_candidates: pd.DataFrame) -> dict[str, list[str]]:
    """Measured, generated sentences: what tuning changed, whether it helped, and which model is best overall.

    cv_candidates / test_candidates: candidates_table() of the CV summaries and of the test tables.
    """
    out = {"per_model": [], "best": [], "tradeoffs": [], "caveats": []}

    def value(model, stage, metric, col):
        row = comparison[(comparison["Model"] == model) & (comparison["Stage"] == stage) & (comparison["Metric"] == metric)]
        return float(row[col].iloc[0])

    for name in tuned_folds:
        changed = params[(params["Model"] == name) & params["Changed"]]
        change_txt = (", ".join(f"`{r['Hyperparameter']}` {r['Baseline']} → {r['Tuned']}" for _, r in changed.iterrows())
                      if not changed.empty else "the baseline settings were already the best in the grid")
        diff = tuned_folds[name][TUNING_METRIC] - baseline_folds[name][TUNING_METRIC]
        cv_d, test_d = value(name, "CV mean", TUNING_METRIC, "Change"), value(name, "Test", TUNING_METRIC, "Change")
        std = float(baseline_folds[name][TUNING_METRIC].std(ddof=1))
        size = ("larger than" if abs(test_d) > std else "smaller than")
        out["per_model"].append(
            f"**{name}:** {change_txt}. CV ROC-AUC {fmt3(value(name, 'CV mean', TUNING_METRIC, 'Baseline'))} → "
            f"{fmt3(value(name, 'CV mean', TUNING_METRIC, 'Tuned'))} ({fmt3_signed(cv_d)}; better in "
            f"{int((diff > 0).sum())} of {len(diff)} folds). Test ROC-AUC "
            f"{fmt3(value(name, 'Test', TUNING_METRIC, 'Baseline'))} → {fmt3(value(name, 'Test', TUNING_METRIC, 'Tuned'))} "
            f"({fmt3_signed(test_d)}), {size} the baseline's CV standard deviation ({fmt3(std)})."
        )
        moves = [f"{m} {fmt3_signed(value(name, 'Test', m, 'Change'))}" for m in ["Precision", "Recall", "F1"]]
        out["tradeoffs"].append(f"*{name}, test set at threshold 0.5:* " + ", ".join(moves) + ".")

    best = cv_candidates["ROC_AUC mean"].idxmax()
    test_best = test_candidates["ROC_AUC"].idxmax()
    out["best"].append(
        f"**{best}** has the highest mean CV ROC-AUC of all {len(cv_candidates)} candidates "
        f"({fmt3(cv_candidates.loc[best, 'ROC_AUC mean'])} ± {fmt3(cv_candidates.loc[best, 'ROC_AUC std'])})."
    )
    runner_up = cv_candidates["ROC_AUC mean"].drop(best).idxmax()
    gap = cv_candidates.loc[best, "ROC_AUC mean"] - cv_candidates.loc[runner_up, "ROC_AUC mean"]
    gap_txt = "less than 0.001" if gap < 0.0005 else fmt3(gap)
    out["best"].append(
        f"The runner-up, {runner_up}, is {gap_txt} behind: "
        + ("well within one CV standard deviation, so the top models are practically tied on ranking quality."
           if gap < cv_candidates.loc[best, "ROC_AUC std"] else "more than one CV standard deviation.")
    )
    if best.endswith(TUNED_SUFFIX):
        out["best"].append(
            f"{best}'s CV score is the optimistic one (its settings were chosen on these folds), so its lead "
            "should not be read as a real advantage."
        )
    baselines = cv_candidates.loc[[n for n in cv_candidates.index if not n.endswith(TUNED_SUFFIX)], "ROC_AUC mean"]
    if baselines.idxmax() != best:
        out["best"].append(
            f"Among the untuned models compared on equal terms, {baselines.idxmax()} is selected (main comparison). "
            "Both choices are defensible; the measured differences are too small to separate them."
        )
    out["best"].append(
        f"*Final confirmation on the test set:* {test_best} has the highest test ROC-AUC "
        f"({fmt3(test_candidates.loc[test_best, 'ROC_AUC'])})"
        + (", which confirms the CV choice." if test_best == best else
           f"; {best} scores {fmt3(test_candidates.loc[best, 'ROC_AUC'])}. The CV choice is not changed after "
           "looking at the test set, but the disagreement shows the leading models are close.")
    )
    for metric, meaning in [("Recall", "catches the most churners"), ("Precision", "raises the fewest false alarms"),
                            ("F1", "has the best Precision/Recall balance")]:
        leader = test_candidates[metric].idxmax()
        out["tradeoffs"].append(
            f"At threshold 0.5, **{leader}** {meaning} (test {metric} {fmt3(test_candidates.loc[leader, metric])})."
        )
    out["tradeoffs"].append(
        "Precision and Recall depend on the decision threshold, while ROC-AUC does not. A model with higher ROC-AUC "
        "ranks customers better, so with a suitable threshold it can reach better Precision/Recall combinations. "
        "That is why selection uses ROC-AUC and the threshold is chosen separately."
    )
    out["caveats"].append(
        "Tuned CV scores are slightly optimistic: the best grid candidate was chosen on the same folds it is "
        "scored on. The test set is the unbiased comparison; nested cross-validation would remove the bias."
    )
    out["caveats"].append(
        f"Only Random Forest and XGBoost were tuned, over small grids ({', '.join(f'{n}: {n_candidates_from(params, n)} candidates' for n in tuned_folds)}). "
        "LightGBM, CatBoost and Logistic Regression use documented baselines."
    )
    out["caveats"].append(
        "Tuning was done with equal class weights and scored at the default threshold for Precision/Recall/F1; "
        "the ROC-AUC used for selection does not depend on the threshold."
    )
    return out


def n_candidates_from(params: pd.DataFrame, name: str) -> int:
    rows = params[params["Model"] == name]
    return int(np.prod([len(v.split(", ")) for v in rows["Values searched"]]))


def load_tuning_results(output_dir: Path = config.OUTPUT_DIR) -> dict | None:
    """What main.py saved about tuning, or None if any file is missing (main.py has not been run yet).

    Returns models (refitted tuned pipelines), tuned_folds (fold scores at full precision),
    params (searched / baseline / tuned values) and grid (every candidate's CV ROC-AUC).
    """
    files = {k: output_dir / f for k, f in [("folds", "tuning_cv_folds.csv"), ("params", "tuning_best_params.csv"),
                                             ("grid", "tuning_grid_results.csv")]}
    models = load_tuned_models(output_dir / "models")
    if set(models) != set(TUNED_MODELS) or not all(f.is_file() for f in files.values()):
        return None
    folds = pd.read_csv(files["folds"], index_col=["Model", "Fold"])
    params = pd.read_csv(files["params"], dtype=str, keep_default_na=False)
    params["Changed"] = params["Changed"] == "True"
    return {
        "models": models,
        "tuned_folds": {name: folds.loc[name] for name in TUNED_MODELS},
        "params": params,
        "grid": pd.read_csv(files["grid"], index_col=["Model", "Candidate"], keep_default_na=False),
    }

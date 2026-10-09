"""Automated analytical summary: which model is selected, and the evidence for why.

Model selection uses only the training data: the selected model is the one with
the highest mean ROC-AUC in stratified k-fold cross-validation. The test set is
used strictly as a final confirmation of that choice.

Every sentence is generated from measured numbers. Statements are labelled as
either *measured* (computed in this run) or *explanation* (how the algorithms
work in general; not tested by this project).
"""

import pandas as pd

from churn import config
from churn.formatting import fmt3, fmt3_signed, fmt_mean_std
from churn.models import BOOSTING_MODELS, TREE_MODELS

LINEAR_MODEL = "Logistic Regression"
AGE_BINS = [17, 29, 39, 49, 59, 100]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
MIN_SEGMENT_N = 100  # segments smaller than this get a small-sample warning


def paired_fold_comparison(cv_folds: dict[str, pd.DataFrame], a: str, b: str, metric: str = "ROC_AUC") -> dict:
    """Compare two models fold by fold. All models use identical folds, so the differences are paired."""
    diff = cv_folds[a][metric] - cv_folds[b][metric]
    return {"mean": float(diff.mean()), "std": float(diff.std(ddof=1)), "wins": int((diff > 0).sum()), "n": len(diff)}


def _segment_key(X: pd.DataFrame, feature: str) -> pd.Series:
    return pd.cut(X["Age"], AGE_BINS, labels=AGE_LABELS) if feature == "Age" else X[feature]


def segment_stats(X: pd.DataFrame, y: pd.Series, feature: str) -> pd.DataFrame:
    """Per level of a feature: churn rate, number of customers (n), number who churned, small-sample flag."""
    stats = y.groupby(_segment_key(X, feature), observed=True).agg(rate="mean", n="size", churned="sum")
    stats["small"] = stats["n"] < MIN_SEGMENT_N
    return stats


def churn_rate_by(X: pd.DataFrame, y: pd.Series, feature: str) -> pd.Series:
    """Observed churn rate per level of a feature (Age is grouped into bands)."""
    return segment_stats(X, y, feature)["rate"]


def is_monotonic(rates: pd.Series) -> bool:
    return bool(rates.is_monotonic_increasing or rates.is_monotonic_decreasing)


def format_segments(stats: pd.DataFrame) -> str:
    """'1: 28% (n=4,071) → … → 4: 100% (50/50, n=50 ⚠ small sample)'."""
    parts = []
    for level, r in stats.iterrows():
        if r["small"]:
            parts.append(f"{level}: {r['rate']:.0%} ({int(r['churned'])}/{int(r['n'])}, n={int(r['n'])} ⚠ small sample)")
        else:
            parts.append(f"{level}: {r['rate']:.0%} (n={int(r['n']):,})")
    return " → ".join(parts)


def small_segment_notes(X: pd.DataFrame, y: pd.Series, features: list[str]) -> list[str]:
    """One caveat line per feature that has segments with fewer than MIN_SEGMENT_N customers."""
    lines = []
    for feature in features:
        stats = segment_stats(X, y, feature)
        small = stats[stats["small"]]
        if small.empty:
            continue
        detail = "; ".join(f"`{feature} = {lvl}`: {int(r['churned'])} of {int(r['n'])} churned" for lvl, r in small.iterrows())
        lines.append(
            f"Small-sample segments (n < {MIN_SEGMENT_N}, training set): {detail}. Rates based on so few customers are "
            "unstable; a 0% or 100% rate in particular should not be read as a certainty, and may reflect a quirk "
            "of this dataset."
        )
    return lines


GAP_STD_RULE = 2  # a test score within this many CV standard deviations of the CV mean is "consistent"


def generalisation_table(cv_summary: pd.DataFrame, test_table: pd.DataFrame, train_table: pd.DataFrame,
                         metric: str = "ROC_AUC") -> pd.DataFrame:
    """Train vs CV vs test for one metric (all at the default 0.5 threshold for threshold-dependent metrics)."""
    table = pd.DataFrame({
        "Train": train_table[metric],
        "CV mean": cv_summary[f"{metric} mean"],
        "CV std": cv_summary[f"{metric} std"],
        "Test": test_table[metric],
    })
    table["Train − CV"] = table["Train"] - table["CV mean"]
    table["Test − CV"] = table["Test"] - table["CV mean"]
    table[f"Test within {GAP_STD_RULE} CV std"] = (table["Test − CV"].abs() <= GAP_STD_RULE * table["CV std"])
    table.index.name = "Model"
    return table


def generalisation_notes(table: pd.DataFrame, metric: str = "ROC_AUC") -> list[str]:
    """How to read the generalisation table, generated from its numbers. Never claims 'no overfitting'."""
    within = table[f"Test within {GAP_STD_RULE} CV std"]
    metric = metric.replace("_", "-")  # display name, e.g. ROC-AUC
    lines = []
    if within.all():
        lines.append(
            f"*Measured:* every model's test {metric} lies within {GAP_STD_RULE} CV standard deviations of its CV "
            "mean. The cross-validation estimates are therefore **reliable**: they predicted the performance on "
            "the untouched test set well, and the test split is representative."
        )
    else:
        lines.append(
            f"*Measured:* the test {metric} of {', '.join(within.index[~within])} falls outside {GAP_STD_RULE} CV "
            "standard deviations of its CV mean, so for those models the CV estimate did not carry over well "
            "(possible causes: an unrepresentative split or a distribution shift)."
        )
    big = table[table["Train − CV"] > 0.05]
    if not big.empty:
        lines.append(
            "*Measured:* "
            + ", ".join(f"{m} scores {fmt3(r['Train'])} on its own training data vs {fmt3(r['CV mean'])} in CV" for m, r in big.iterrows())
            + ". A training score far above the CV score means the model **fits (overfits) its training data** much "
            "more closely than it generalises. That does not make its CV/test estimates wrong, but it does mean "
            "the training score must never be used as a performance estimate."
        )
    lines.append(
        "Agreement between CV and test shows the estimates are trustworthy; it does **not** show that a model is "
        "free of overfitting. Overfitting is visible in the Train − CV gap."
    )
    return lines


def best_model(cv_summary: pd.DataFrame) -> str:
    """Model selection: highest mean ROC-AUC in cross-validation on the training set (never the test set)."""
    return str(cv_summary["ROC_AUC mean"].idxmax())


def analytical_summary(
    test_table: pd.DataFrame,
    cv_summary: pd.DataFrame,
    cv_folds: dict[str, pd.DataFrame],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    permutation: dict[str, pd.DataFrame] | None = None,
) -> dict[str, list[str]]:
    """Markdown bullet lists under the keys 'verdict', 'evidence', 'why', 'caveats'."""
    best = best_model(cv_summary)
    test_best = str(test_table["ROC_AUC"].idxmax())
    others = [m for m in cv_summary.index if m != best]
    out = {"verdict": [], "evidence": [], "why": [], "caveats": []}

    # --- Verdict: selection on CV, confirmation on test ---------------------------
    out["verdict"].append(
        f"**{best} is selected as the best model.** It has the highest mean ROC-AUC in "
        f"{config.CV_FOLDS}-fold cross-validation on the training set "
        f"({fmt_mean_std(cv_summary.loc[best, 'ROC_AUC mean'], cv_summary.loc[best, 'ROC_AUC std'])})."
    )
    if test_best == best:
        out["verdict"].append(
            f"*Final confirmation on the untouched test set:* it also has the highest test ROC-AUC "
            f"({fmt3(test_table.loc[best, 'ROC_AUC'])}), so the test set confirms the CV-based choice."
        )
    else:
        out["verdict"].append(
            f"*Final confirmation on the untouched test set:* **not confirmed**: {test_best} has the higher test "
            f"ROC-AUC ({fmt3(test_table.loc[test_best, 'ROC_AUC'])} vs {fmt3(test_table.loc[best, 'ROC_AUC'])}). "
            "The selection is not changed after looking at the test set; the disagreement means the evidence "
            "for a single winner is weak."
        )
    leads = [m for m in ["Precision", "Recall", "F1", "Accuracy"] if test_table[m].idxmax() == best]
    if leads:
        out["verdict"].append(f"On the test set, at the selected thresholds, it also leads on {', '.join(leads)}.")
    for m in ["Precision", "Recall", "F1"]:
        if test_table[m].idxmax() != best:
            out["verdict"].append(
                f"{test_table[m].idxmax()} has the higher test {m} at the selected thresholds "
                f"({fmt3(test_table[m].max())} vs {fmt3(test_table.loc[best, m])}); threshold choice shifts this trade-off."
            )

    # --- Robustness evidence -----------------------------------------------------
    for other in others:
        p = paired_fold_comparison(cv_folds, best, other)
        out["evidence"].append(
            f"**{best} vs {other}:** higher CV ROC-AUC in **{p['wins']} of {p['n']} folds** "
            f"(mean difference {fmt3_signed(p['mean'])} ± {fmt3(p['std'])}); test-set difference "
            f"{fmt3_signed(test_table.loc[best, 'ROC_AUC'] - test_table.loc[other, 'ROC_AUC'])}."
        )

    # --- Why: measured patterns + algorithm explanation -------------------------
    trees_beat_linear = LINEAR_MODEL in cv_summary.index and all(
        cv_summary.loc[m, "ROC_AUC mean"] > cv_summary.loc[LINEAR_MODEL, "ROC_AUC mean"]
        for m in TREE_MODELS if m in cv_summary.index
    )
    nonmono = []
    for feature in ["NumOfProducts", "Age"]:
        stats = segment_stats(X_train, y_train, feature)
        if not is_monotonic(stats["rate"]):
            nonmono.append(feature)
            out["why"].append(
                f"*Measured:* churn rate by `{feature}` (training set) is **non-monotonic**: {format_segments(stats)}."
            )
    if permutation and LINEAR_MODEL in permutation:
        for feature in nonmono:
            lin = permutation[LINEAR_MODEL].loc[feature, "importance_mean"]
            trees = {m: permutation[m].loc[feature, "importance_mean"] for m in TREE_MODELS if m in permutation}
            if trees and min(trees.values()) > lin + 0.01:
                tree_txt = ", ".join(f"{m} {fmt3(v)}" for m, v in trees.items())
                out["why"].append(
                    f"*Measured:* shuffling `{feature}` lowers test ROC-AUC by {tree_txt}, but only {fmt3(lin)} for "
                    f"{LINEAR_MODEL}: the linear model barely uses this signal."
                )
    zero_share = float((X_train["Balance"] == 0).mean())
    zero_rate = float(y_train[X_train["Balance"] == 0].mean())
    pos_rate = float(y_train[X_train["Balance"] > 0].mean())
    out["why"].append(
        f"*Measured:* {zero_share:.0%} of customers have `Balance = 0` (churn {zero_rate:.0%}) versus "
        f"{pos_rate:.0%} churn for positive balances: a spike in the distribution rather than a smooth trend."
    )
    if trees_beat_linear:
        out["why"].append(
            "*Explanation (how the algorithms work; not tested here):* Logistic Regression fits **one slope per "
            "feature** on the log-odds scale, so it can only say “more products → more (or less) churn”. A "
            "U-shaped or threshold pattern such as the ones above is lost unless features are engineered by hand. "
            "Tree ensembles split each feature into ranges (e.g. `NumOfProducts ≤ 2` vs `≥ 3`) and combine splits "
            "along a path, so they capture **non-linear effects and interactions** (e.g. age × activity) "
            "automatically, and are unaffected by skewed scales or spikes like zero balances."
        )
    boosters = [m for m in BOOSTING_MODELS if m in cv_summary.index]
    if boosters and "Random Forest" in cv_summary.index:
        rf = cv_summary.loc["Random Forest", "ROC_AUC mean"]
        better = [m for m in boosters if cv_summary.loc[m, "ROC_AUC mean"] > rf]
        if better:
            out["why"].append(
                f"*Measured:* {len(better)} of {len(boosters)} gradient-boosting models ({', '.join(better)}) have a "
                f"higher mean CV ROC-AUC than Random Forest ({fmt3(rf)})."
            )
            out["why"].append(
                "*Explanation (not tested here):* Random Forest **averages** many deep, independent trees "
                "(bagging), which mainly reduces variance. Gradient boosting (XGBoost, LightGBM, CatBoost) **adds** "
                "shallow trees one at a time, each fitted to the errors of the ensemble so far, with shrinkage "
                "(learning rate 0.05), subsampling and L2 penalties as regularisation. On structured tabular data "
                "this often produces a better ranking of risk, which is what ROC-AUC measures."
            )
        else:
            out["why"].append(
                "*Measured:* boosting did not outperform bagging in cross-validation; Random Forest's averaged deep "
                "trees ranked customers at least as well as the boosted models."
            )
        if len(boosters) > 1:
            spread = cv_summary.loc[boosters, "ROC_AUC mean"]
            out["why"].append(
                f"*Measured:* the {len(boosters)} boosting implementations are within "
                f"{fmt3(spread.max() - spread.min())} CV ROC-AUC of each other. They grow trees differently "
                "(depth-wise, leaf-wise, symmetric), but on this data they reach practically the same ranking quality."
            )

    # --- Caveats --------------------------------------------------------------
    out["caveats"].append(
        "This comparison uses documented baseline hyperparameters. Random Forest and XGBoost are also tuned with "
        "GridSearchCV (reported separately); the other models are not tuned, which could favour the tuned ones."
    )
    out["caveats"].append(
        f"With {config.CV_FOLDS} folds, winning every fold is consistent but modest evidence (a one-sided sign test "
        f"gives p = {0.5 ** config.CV_FOLDS:.3f}), and folds share training data, so they are not independent."
    )
    out["caveats"].append(
        "The test set is a single split and was viewed during development, so its scores may be slightly "
        "optimistic; the cross-validation results are the primary evidence."
    )
    out["caveats"].extend(small_segment_notes(X_train, y_train, ["NumOfProducts", "Age"]))
    out["caveats"].append("Importance and churn-rate patterns are associations in this dataset, not causes.")
    return out


def importance_insights(builtin: pd.DataFrame, permutation: dict[str, pd.DataFrame], top_n: int = 5) -> list[str]:
    """Bullet points about which features the models rely on, and where the two methods disagree."""
    lines = []
    tree_perm = pd.DataFrame({m: permutation[m]["importance_mean"] for m in TREE_MODELS if m in permutation})
    if not tree_perm.empty:
        consensus = tree_perm.mean(axis=1).sort_values(ascending=False)
        top = [f for f in consensus.index[:4] if consensus[f] > 0.005]
        lines.append(
            "*Measured:* by permutation importance, the tree models rely most on "
            + ", ".join(f"`{f}` ({fmt3(consensus[f])})" for f in top)
            + " (mean drop in test ROC-AUC when the feature is shuffled)."
        )
        negligible = [f for f in consensus.index if consensus[f] <= 0.002]
        if negligible:
            lines.append(
                "*Measured:* shuffling " + ", ".join(f"`{f}`" for f in negligible)
                + " changes ROC-AUC by ≤ 0.002; the models gain almost nothing from these features."
            )
    for model in builtin.columns:
        top_builtin = builtin[model].sort_values(ascending=False).index[:top_n]
        perm = permutation.get(model)
        if perm is None:
            continue
        inflated = [f for f in top_builtin if perm.loc[f, "importance_mean"] <= max(2 * perm.loc[f, "importance_std"], 0.005)]
        if inflated:
            lines.append(
                f"*Measured:* {model}'s built-in importance ranks "
                + ", ".join(f"`{f}` ({builtin.loc[f, model]:.1%})" for f in inflated)
                + f" in its top {top_n}, yet their permutation importance is indistinguishable from zero. "
                "Impurity/gain importance is computed on training data and favours continuous features with many "
                "split points, so it can overstate such features."
            )
    return lines

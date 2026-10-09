"""Feature importance: built-in tree importances and permutation importance.

Two complementary views:

- Built-in (tree models only). Random Forest: mean decrease in impurity across
  all splits. XGBoost: average gain (loss reduction) of the splits that use the
  feature. Both are computed on the training data during fitting and are known
  to favour continuous features with many possible split points.
- Permutation importance (any model). On held-out data, shuffle one feature at
  a time and measure how much ROC-AUC drops. A feature the model truly relies on
  causes a large drop; an irrelevant one causes about zero.

Neither shows that a feature *causes* churn; they describe what the fitted
model uses. One-hot columns are summed back to their original feature
(e.g. Geography_France + Geography_Germany + Geography_Spain -> Geography).
"""

import pandas as pd
from sklearn.inspection import permutation_importance

from churn import config


def original_feature(output_name: str) -> str:
    """Map a ColumnTransformer output name ('num__Age', 'cat__Geography_Spain') to its source column."""
    name = output_name.split("__", 1)[-1]
    if name in config.FEATURES:
        return name
    for col in config.CATEGORICAL_FEATURES:
        if name.startswith(col + "_"):
            return col
    raise ValueError(f"Cannot map output column {output_name!r} to an original feature")


def builtin_importance(pipeline, aggregate: bool = True) -> pd.Series:
    """Normalised built-in importances (sum = 1) of a fitted tree pipeline."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__} has no built-in feature importances")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    imp = pd.Series(model.feature_importances_, index=names, dtype=float)
    if aggregate:
        imp = imp.groupby(original_feature).sum()
    else:
        imp.index = [n.split("__", 1)[-1] for n in imp.index]
    return (imp / imp.sum()).sort_values(ascending=False)


def permutation_importance_table(pipeline, X, y, n_repeats: int = 10) -> pd.DataFrame:
    """Drop in ROC-AUC when each original feature is shuffled (mean and std over repeats)."""
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=config.RANDOM_STATE,
    )
    table = pd.DataFrame(
        {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
        index=pd.Index(X.columns, name="Feature"),
    )
    return table.sort_values("importance_mean", ascending=False)

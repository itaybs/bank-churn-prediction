"""Loading, validating, and splitting the churn dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


class DataValidationError(ValueError):
    """Raised when the dataset cannot safely be used for modelling."""


def load_dataset(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV file, failing with a clear message if it does not exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> None:
    """Stop early if the data would produce misleading results.

    Checks: required columns exist, the target is strictly binary (0/1) with
    both classes present, and the features contain no missing values.
    """
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    target_values = set(df[config.TARGET].dropna().unique())
    if df[config.TARGET].isna().any() or not target_values <= {0, 1}:
        raise DataValidationError(
            f"Target '{config.TARGET}' must contain only 0 and 1 with no missing "
            f"values; found {sorted(target_values)}"
        )
    if target_values != {0, 1}:
        raise DataValidationError(
            f"Target '{config.TARGET}' must contain both classes; found {sorted(target_values)}"
        )

    # The current data has no missing values, so no imputation is used.
    # If this check ever fails, add a SimpleImputer step in preprocessing.py.
    na_counts = df[config.FEATURES].isna().sum()
    if na_counts.any():
        raise DataValidationError(
            f"Missing values found in features: {na_counts[na_counts > 0].to_dict()}"
        )


def class_distribution(y: pd.Series) -> pd.DataFrame:
    """Counts and percentages of each target class."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({"count": counts, "percent": counts / counts.sum() * 100})


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select X by explicit column names (never "all but the last column")."""
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()

    # Explicit checks (not `assert`, which Python skips when run with -O).
    if config.TARGET in X.columns:
        raise DataValidationError("Target leaked into features")
    if set(config.ID_COLUMNS) & set(X.columns):
        raise DataValidationError("Identifier column used as a feature")
    if len(X) != len(y):
        raise DataValidationError("X and y have different numbers of rows")
    return X, y


def make_train_test_split(X: pd.DataFrame, y: pd.Series):
    """One shared, stratified 80/20 split used by every model.

    This happens before any preprocessing is fitted, so the test rows never
    influence encoders, scalers, or models.
    """
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

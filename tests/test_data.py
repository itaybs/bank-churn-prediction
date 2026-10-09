import numpy as np
import pytest

from churn import config
from churn.data import (
    DataValidationError,
    load_dataset,
    make_train_test_split,
    split_features_target,
    validate_dataset,
)


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_valid_data_passes(small_df):
    validate_dataset(small_df)


def test_missing_column_is_rejected(small_df):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(small_df.drop(columns=["Age"]))


@pytest.mark.parametrize("bad_value", [2, -1, np.nan])
def test_invalid_target_values_are_rejected(small_df, bad_value):
    df = small_df.astype({"Exited": float})
    df.loc[0, "Exited"] = bad_value
    with pytest.raises(DataValidationError, match="only 0 and 1"):
        validate_dataset(df)


def test_single_class_target_is_rejected(small_df):
    df = small_df.assign(Exited=0)
    with pytest.raises(DataValidationError, match="both classes"):
        validate_dataset(df)


def test_missing_feature_values_are_rejected(small_df):
    df = small_df.astype({"Balance": float})
    df.loc[3, "Balance"] = np.nan
    with pytest.raises(DataValidationError, match="Missing values"):
        validate_dataset(df)


def test_features_exclude_target_and_identifiers(small_df):
    X, y = split_features_target(small_df)
    assert list(X.columns) == config.FEATURES
    assert "Exited" not in X.columns
    assert "CustomerId" not in X.columns and "Surname" not in X.columns
    assert len(X) == len(y)
    # The source DataFrame keeps its identifier columns.
    assert {"CustomerId", "Surname"} <= set(small_df.columns)


def test_split_is_80_20_stratified_disjoint_and_reproducible(small_df):
    X, y = split_features_target(small_df)
    X_tr, X_te, y_tr, y_te = make_train_test_split(X, y)
    assert len(X_te) == 40 and len(X_tr) == 160
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert y_tr.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_te.mean() == pytest.approx(y.mean(), abs=0.01)
    _, X_te_again, _, _ = make_train_test_split(X, y)
    assert list(X_te.index) == list(X_te_again.index)

import pandas as pd
import pytest

from churn.formatting import fmt3, fmt3_signed, fmt_mean_std, round3

# Exact ties seen in this project's results (e.g. XGBoost test accuracy 1733/2000 = 0.8665).
TIES = [0.8665, 0.7655, 0.8265]


@pytest.mark.parametrize("value", TIES)
def test_fmt3_matches_the_csv_rounding(value):
    """The CSV files are written with pandas round(3); every display must show the same digits."""
    csv_text = f"{pd.Series([value]).round(3).iloc[0]:.3f}"
    assert fmt3(value) == csv_text


def test_known_tie_values():
    assert fmt3(0.8665) == "0.866"  # f"{0.8665:.3f}" would give 0.867
    assert round3(0.8665) == 0.866


def test_no_negative_zero():
    assert fmt3(-0.0004) == "0.000"
    assert fmt3_signed(-0.0004) == "+0.000"
    assert fmt3_signed(-0.002) == "-0.002"
    assert fmt3_signed(0.012) == "+0.012"


def test_mean_std_formatting():
    assert fmt_mean_std(0.8660, 0.0121) == "0.866 ± 0.012"
    assert fmt_mean_std(0.8660, 0.0121, sep=" +/- ") == "0.866 +/- 0.012"

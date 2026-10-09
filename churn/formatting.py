"""One rounding rule for every metric the project displays.

Metrics are shown with three decimals. Python's f"{x:.3f}" and numpy's
np.round(x, 3) disagree on exact ties such as 0.8665 (f-format gives 0.867,
numpy gives 0.866). The CSV files are written with pandas' round(), which uses
numpy's rule, so every displayed value goes through numpy's rule as well:
the report, CLI, CSVs and dashboard then always show the same digits.
"""

import numpy as np

DECIMALS = 3


def round3(x: float) -> float:
    """Round to three decimals with numpy's rule; -0.0 becomes 0.0."""
    return float(np.round(x, DECIMALS)) + 0.0


def fmt3(x: float) -> str:
    """'0.866' for 0.8665 (consistent with the CSV files)."""
    return f"{round3(x):.3f}"


def fmt3_signed(x: float) -> str:
    """'+0.012' / '-0.002' / '+0.000'."""
    return f"{round3(x):+.3f}"


def fmt_mean_std(mean: float, std: float, sep: str = " ± ") -> str:
    return f"{fmt3(mean)}{sep}{fmt3(std)}"

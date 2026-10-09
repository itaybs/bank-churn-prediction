import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_df() -> pd.DataFrame:
    """A small synthetic dataset with the same columns as churn_modelling.csv."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "CustomerId": np.arange(n),
            "Surname": [f"Name{i}" for i in range(n)],
            "CreditScore": rng.integers(350, 850, n),
            "Geography": rng.choice(["France", "Germany", "Spain"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 90, n),
            "Tenure": rng.integers(0, 11, n),
            "Balance": rng.uniform(0, 250_000, n),
            "NumOfProducts": rng.integers(1, 5, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(10, 200_000, n),
            "Exited": np.r_[np.ones(40, dtype=int), np.zeros(n - 40, dtype=int)],
        }
    )

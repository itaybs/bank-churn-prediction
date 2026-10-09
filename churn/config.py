"""Central configuration: paths, column names, and fixed random settings.

Paths are built relative to the project folder, so the project runs on any
machine without editing hard-coded paths.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "churn_modelling.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- Columns -----------------------------------------------------------------
TARGET = "Exited"

# Identifiers carry no general information about churn and could let a model
# memorise individual customers, so they are never used as features.
# (They are only dropped from X; the CSV file itself is never modified.)
ID_COLUMNS = ["CustomerId", "Surname"]

# HasCrCard and IsActiveMember are binary 0/1 flags; they can stay numeric.
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = ID_COLUMNS + FEATURES + [TARGET]

# --- Split and reproducibility ------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Stratified k-fold cross-validation on the training set (model assessment and threshold selection).
CV_FOLDS = 5

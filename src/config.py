from pathlib import Path

# ==============================
# Project Paths
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RANKING_DIR = PROJECT_ROOT / "feature_rankings"

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

RANKING_METHODS = [
    "ANOVA_Only",
    "KW_Only",
    "Min",
    "Max",
    "GeoMean",
    "ANOVA_First",
    "KW_First",
]
TARGETS = [
    "Subject_ID",
    "Task_ID",
]

FEATURE_SIZES = [50, 75, 100, 150, 200, 250, 500]
RANDOM_STATE = 42

CV_FOLDS = 10

SCORING = "accuracy"

DT_PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
}
N_JOBS = -1
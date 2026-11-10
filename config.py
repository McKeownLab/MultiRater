"""Project paths and rater / task constants for the public release."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = Path(os.environ.get("MULTI_RATER_DATA_RAW", PROJECT_ROOT / "data" / "raw"))
DATA_PROCESSED = Path(os.environ.get("MULTI_RATER_DATA_PROCESSED", PROJECT_ROOT / "data" / "processed"))
DATA_EXTERNAL = Path(os.environ.get("MULTI_RATER_DATA_EXTERNAL", PROJECT_ROOT / "data" / "external"))
CLEAN_DATA_DEFAULT = DATA_PROCESSED / "camera_booth_clean_confident.csv"
CLEAN_DATA_FULL = DATA_PROCESSED / "camera_booth_clean.csv"
PHASE2_FEATURES_DIR = DATA_PROCESSED / "phase2_features"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"

# Rating-sheet column order (fixed): rater 1, 2, 3, excluded, 4, 5.
RATERS = ["R1", "R2", "R3", "EXCLUDED", "R4", "R5"]
ACTIVE_RATERS = ["R1", "R2", "R3", "R4", "R5"]
PAPER_RATERS = list(ACTIVE_RATERS)
RATER_ROLE = {
    "R1": "movement disorder neurologist",
    "R2": "trained research rater (non-clinician)",
    "R3": "movement disorder neurologist",
    "R4": "movement disorder neurologist",
    "R5": "movement disorder neurologist",
}


def paper_rater(name: str) -> str:
    """Identity for paper labels (already R1..R5 in this release)."""
    return name


TASK_FAMILY = {
    "FT_R": "FT", "FT_L": "FT", "OC_R": "OC", "OC_L": "OC",
    "PS_R": "PS", "PS_L": "PS", "PT_R": "PT", "PT_L": "PT",
}

# Override with MULTI_RATER_XLSX or place ratings.xlsx under data/raw/.
XLSX_FILENAME = os.environ.get("MULTI_RATER_XLSX", "ratings.xlsx")
RATINGS_SHEET = "All Ratings_Adjusted"
DEMOGRAPHICS_SHEET = "Demographics"

TASKS = [
    ("Finger Tapping", "R"), ("Finger Tapping", "L"),
    ("Open - Close", "R"), ("Open - Close", "L"),
    ("Pronation - Supination", "R"), ("Pronation - Supination", "L"),
    ("Postural Tremor", "R"), ("Postural Tremor", "L"),
]
TASK_SHORT = {
    ("Finger Tapping", "R"): "FT_R", ("Finger Tapping", "L"): "FT_L",
    ("Open - Close", "R"): "OC_R", ("Open - Close", "L"): "OC_L",
    ("Pronation - Supination", "R"): "PS_R", ("Pronation - Supination", "L"): "PS_L",
    ("Postural Tremor", "R"): "PT_R", ("Postural Tremor", "L"): "PT_L",
}

HIGH_DISAGREEMENT_RANGE = 2
WILLIAMS_ICC_BENCHMARK = 0.53

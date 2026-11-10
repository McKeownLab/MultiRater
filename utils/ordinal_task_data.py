"""Per session×task ordinal rows for ratings-only ordinal models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from utils.phase2_data import load_feature_tables

ACTIVE_RATERS = config.ACTIVE_RATERS
TASK_ORDER = list(config.TASK_SHORT.values())

LOTO_COLS = [f"x_{i}" for i in range(len(TASK_ORDER) - 1)]
FEATURE_MODES = ("ratings",)


def build_task_level_table(
    min_raters: int = 3,
    features_dir: Path | None = None,
    attach_video: bool = False,
    video_root: Path | None = None,
) -> pd.DataFrame:
    """
    One row per session×task.

    Model inputs (no label leakage):
      - x_0..x_6: leave-one-task-out consensus from *other* tasks

    Metadata (not model inputs):
      - score_range, score_std: disagreement on the target task
    """
    if attach_video:
        raise ValueError("Video features are not part of this release; use ratings-only features.")

    tables = load_feature_tables(features_dir)
    consensus = tables["consensus"]
    full_grid = tables["full_grid"]
    # One row per session in each table.
    merged = consensus.merge(full_grid, on="session_id", suffixes=("_meta", ""), validate="one_to_one")

    rows: list[dict] = []

    for _, row in merged.iterrows():
        split_val = row["split_meta"] if "split_meta" in row.index else row.get("split", "train")
        booth_id = int(row["booth_id"])
        visit_date = str(row["visit_date"])

        for task in TASK_ORDER:
            score_cols = [f"{task}_{r}" for r in ACTIVE_RATERS]
            scores = row[score_cols]
            n_valid = int(scores.notna().sum())
            if n_valid < min_raters:
                continue

            valid_scores = scores.dropna().astype(float)
            rec: dict = {
                "session_id": row["session_id"],
                "subject_id": int(row["subject_id"]),
                "booth_id": booth_id,
                "visit_date": visit_date,
                "split": split_val,
                "task": task,
                "y_consensus": float(row[f"consensus_{task}"]),
                "n_raters": n_valid,
                "score_range": float(valid_scores.max() - valid_scores.min()),
                "score_std": float(valid_scores.std(ddof=0)) if n_valid > 1 else 0.0,
            }
            for rater in ACTIVE_RATERS:
                val = row[f"{task}_{rater}"]
                rec[f"y_{rater}"] = float(val) if pd.notna(val) else np.nan
                rec[f"m_{rater}"] = float(pd.notna(val))
            for i, other_task in enumerate(t for t in TASK_ORDER if t != task):
                rec[f"x_{i}"] = float(row[f"consensus_{other_task}"])

            rows.append(rec)

    return pd.DataFrame(rows)


def model_feature_cols(df: pd.DataFrame, mode: str = "ratings") -> list[str]:
    if mode != "ratings":
        raise ValueError("This release supports feature_mode='ratings' only")
    loto = [c for c in df.columns if c.startswith("x_") and c.split("_")[1].isdigit()]
    return sorted(loto, key=lambda c: int(c.split("_")[1]))


def row_weights_from_disagreement(train_df: pd.DataFrame, scale: float = 1.0) -> np.ndarray:
    if "score_range" not in train_df.columns:
        return np.ones(len(train_df), dtype=float)
    return (1.0 + scale * train_df["score_range"].values.astype(float)).astype(float)


def split_arrays(
    df: pd.DataFrame,
    train_sessions: set[str],
    test_sessions: set[str],
    feature_mode: str = "ratings",
    disagreement_weight_scale: float = 1.0,
    use_video_imputation: bool = True,
) -> dict:
    del use_video_imputation  # ratings-only release; kept for call-site compatibility
    train_df = df[df["session_id"].isin(train_sessions)].copy()
    test_df = df[df["session_id"].isin(test_sessions)].copy()
    x_cols = model_feature_cols(df, mode=feature_mode)

    x_train = train_df[x_cols].values.astype(float)
    x_test = test_df[x_cols].values.astype(float)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_consensus_train = train_df["y_consensus"].values.astype(float)
    y_consensus_test = test_df["y_consensus"].values.astype(float)

    y_train_by_rater: dict[str, np.ndarray] = {}
    y_test_by_rater: dict[str, np.ndarray] = {}
    mask_train: dict[str, np.ndarray] = {}
    mask_test: dict[str, np.ndarray] = {}

    for rater in ACTIVE_RATERS:
        y_train_by_rater[rater] = train_df[f"y_{rater}"].values.astype(float)
        mask_train[rater] = train_df[f"m_{rater}"].values.astype(bool)
        y_test_by_rater[rater] = test_df[f"y_{rater}"].values.astype(float)
        mask_test[rater] = test_df[f"m_{rater}"].values.astype(bool)

    y_train_by_rater["consensus"] = y_consensus_train

    score_range_test = test_df["score_range"].values.astype(float)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "feature_mode": feature_mode,
        "y_consensus_train": y_consensus_train,
        "y_consensus_test": y_consensus_test,
        "y_train_by_rater": y_train_by_rater,
        "y_test_by_rater": y_test_by_rater,
        "mask_train": mask_train,
        "mask_test": mask_test,
        "row_weights": row_weights_from_disagreement(train_df, scale=disagreement_weight_scale),
        "score_range_test": score_range_test,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

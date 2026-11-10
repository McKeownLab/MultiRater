"""Load Phase 2 feature artifacts produced by phase2_build_features.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

FEATURE_COLS_CONSENSUS = [f"consensus_{t}" for t in config.TASK_SHORT.values()]
FEATURE_COLS_LORO = [f"loro_{t}" for t in config.TASK_SHORT.values()]
ACTIVE_RATERS = config.ACTIVE_RATERS


def load_feature_tables(features_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    features_dir = features_dir or config.PHASE2_FEATURES_DIR
    return {
        "consensus": pd.read_csv(features_dir / "consensus_features.csv"),
        "full_grid": pd.read_csv(features_dir / "full_grid_features.csv"),
        "loro": pd.read_csv(features_dir / "loro_features.csv"),
        "rater_targets": pd.read_csv(features_dir / "rater_targets.csv"),
    }


def scale_consensus_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[object, object, StandardScaler]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLS_CONSENSUS].values.astype(float))
    x_test = scaler.transform(test_df[FEATURE_COLS_CONSENSUS].values.astype(float))
    return x_train, x_test, scaler


def scale_loro_features(loro_df: pd.DataFrame, scaler: StandardScaler):
    return scaler.transform(loro_df[FEATURE_COLS_LORO].values.astype(float))

"""Phase 2 Step 1: Build ratings-only feature matrices (no video)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from utils.ratings_features import (
    build_consensus_features,
    build_full_grid_features,
    build_loro_features,
    build_rater_targets,
    completeness_report,
    filter_complete_sessions,
    session_task_wide,
)

FEATURE_COLS_CONSENSUS = [f"consensus_{t}" for t in config.TASK_SHORT.values()]
FEATURE_COLS_LORO = [f"loro_{t}" for t in config.TASK_SHORT.values()]


def assign_train_test_split(sessions: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Subject-level stratified split so repeat visits stay in the same fold."""
    subject_dx = sessions.groupby("subject_id")["dx_label"].first()
    train_subj, test_subj = train_test_split(
        subject_dx.index,
        test_size=test_size,
        random_state=seed,
        stratify=subject_dx.values,
    )
    train_set, test_set = set(train_subj), set(test_subj)
    out = sessions.copy()
    out["split"] = out["subject_id"].apply(lambda s: "test" if s in test_set else "train")
    return out


def run(
    clean_path: Path | None = None,
    output_dir: Path | None = None,
    min_raters_per_task: int = 3,
) -> dict[str, Path]:
    clean_path = clean_path or config.CLEAN_DATA_DEFAULT
    output_dir = output_dir or (config.DATA_PROCESSED / "phase2_features")
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(clean_path, parse_dates=["visit_date"])
    wide = session_task_wide(long_df)

    consensus = build_consensus_features(wide)
    full_grid = build_full_grid_features(wide)
    loro = build_loro_features(wide)
    rater_targets = build_rater_targets(long_df)
    completeness = completeness_report(wide)

    consensus = filter_complete_sessions(consensus, min_raters_per_task=min_raters_per_task)
    full_grid = full_grid[full_grid["session_id"].isin(consensus["session_id"])]
    loro = filter_complete_sessions(loro, min_raters_per_task=min_raters_per_task)
    rater_targets = rater_targets[rater_targets["session_id"].isin(consensus["session_id"])]

    consensus = assign_train_test_split(consensus)
    split_map = consensus.drop_duplicates("session_id").set_index("session_id")["split"].to_dict()
    full_grid["split"] = full_grid["session_id"].map(split_map)
    loro["split"] = loro["session_id"].map(split_map)
    rater_targets["split"] = rater_targets["session_id"].map(split_map)

    paths = {
        "consensus": output_dir / "consensus_features.csv",
        "full_grid": output_dir / "full_grid_features.csv",
        "loro": output_dir / "loro_features.csv",
        "rater_targets": output_dir / "rater_targets.csv",
        "completeness": output_dir / "session_completeness.csv",
        "manifest": output_dir / "feature_manifest.json",
    }

    consensus.to_csv(paths["consensus"], index=False)
    full_grid.to_csv(paths["full_grid"], index=False)
    loro.to_csv(paths["loro"], index=False)
    rater_targets.to_csv(paths["rater_targets"], index=False)
    completeness.to_csv(paths["completeness"], index=False)

    manifest = {
        "data_source": str(clean_path),
        "n_sessions_raw": int(len(wide)),
        "n_sessions_after_filter": int(len(consensus)),
        "n_loro_rows": int(len(loro)),
        "n_subjects": int(consensus["subject_id"].nunique()),
        "train_sessions": int((consensus["split"] == "train").sum()),
        "test_sessions": int((consensus["split"] == "test").sum()),
        "feature_sets": {
            "consensus": {
                "description": "8 per-task means across all raters",
                "columns": FEATURE_COLS_CONSENSUS,
            },
            "full_grid": {
                "description": "40 raw task×rater scores",
                "n_columns": 40,
            },
            "loro": {
                "description": "8 per-task means excluding held-out rater; one row per session×rater",
                "columns": FEATURE_COLS_LORO,
            },
        },
        "target_column": "dx_label",
        "target_meaning": "PD=1, HC=0 from demographics",
        "min_raters_per_task": min_raters_per_task,
        "split": "subject-level stratified 80/20",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Phase 2 Step 1 — ratings-only features")
    print(f"Source: {clean_path}")
    print(f"Sessions: {manifest['n_sessions_after_filter']} / {manifest['n_sessions_raw']} after completeness filter")
    print(f"Subjects: {manifest['n_subjects']} | Train: {manifest['train_sessions']} | Test: {manifest['test_sessions']}")
    print(f"LORO rows: {manifest['n_loro_rows']} (= sessions × {len(config.ACTIVE_RATERS)} raters)")
    print(f"\nWrote to {output_dir}:")
    for name, path in paths.items():
        print(f"  {name}: {path.name}")

    return paths


if __name__ == "__main__":
    run()

"""Export the 1,559-row modeling table to data/processed/ordinal_task_rows.csv.

One row per session x task item with at least three raters: per-rater labels (y_*), masks (m_*),
consensus, leave-one-task-out features (x_0..x_6) and inter-rater range.

Run after phase0_preprocessing.py and phase2_build_features.py. Downstream scripts such as
scripts/rater_axis_weighting_insight_pd.py and the multi-seed modelling runs read this table.

Usage: python scripts/export_ordinal_task_rows.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from utils.ordinal_task_data import build_task_level_table  # noqa: E402

EXPECTED = {"rows": 1559, "sessions": 195, "subjects": 191}


def main() -> None:
    df = build_task_level_table(min_raters=3)
    got = {"rows": len(df), "sessions": df["session_id"].nunique(), "subjects": df["subject_id"].nunique()}
    assert not df.duplicated(["session_id", "task"]).any(), "duplicate session x task rows"
    if got != EXPECTED:
        raise SystemExit(f"modeling table is {got}, expected {EXPECTED}; check the demographics merge fix "
                         "in phase0_preprocessing.merge_demographics")
    path = config.DATA_PROCESSED / "ordinal_task_rows.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} ({got['rows']} rows, {got['sessions']} sessions, {got['subjects']} subjects)")


if __name__ == "__main__":
    main()

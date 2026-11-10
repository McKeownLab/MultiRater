"""Phase 0: Parse CAMERA Booth xlsx into clean long-form data."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import config
from utils.id_mapping import build_id_mapping
from utils.parsing import parse_score

RATERS = config.RATERS
ACTIVE_RATERS = config.ACTIVE_RATERS
TASKS = config.TASKS
TASK_SHORT = config.TASK_SHORT


def load_ratings_wide(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    wide = raw.iloc[2:].copy()
    wide = wide[wide.iloc[:, 0].notna()].copy()
    wide.iloc[:, 0] = wide.iloc[:, 0].astype(int)
    return wide


def ratings_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    col_idx = 2
    for task_name, side in TASKS:
        task_key = TASK_SHORT[(task_name, side)]
        for rater in RATERS:
            values = wide_df.iloc[:, col_idx]
            for row_i, value in enumerate(values):
                score, dual_attempt, raw_value = parse_score(value)
                records.append(
                    {
                        "booth_id": int(wide_df.iloc[row_i, 0]),
                        "date": wide_df.iloc[row_i, 1],
                        "task": task_key,
                        "task_name": task_name,
                        "side": side,
                        "rater": rater,
                        "score": score,
                        "dual_attempt": dual_attempt,
                        "raw_value": raw_value,
                    }
                )
            col_idx += 1

    long_df = pd.DataFrame(records)
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["visit_date"] = long_df["date"].dt.date
    return long_df


def merge_demographics(
    long_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    demo = demographics_df.rename(
        columns={
            "CAMERA ID": "camera_id",
            "Date of Visit ": "demo_date",
            "Dx": "dx",
            "Sex": "sex",
            "Age": "age",
        }
    )
    mapping_cols = [
        "booth_id",
        "visit_date",
        "camera_id",
        "mapping_method",
        "confidence",
        "candidate_camera_ids",
        "booth_id_reused",
    ]
    merged = long_df.merge(mapping_df[mapping_cols], on=["booth_id", "visit_date"], how="left")
    merged["_rating_row"] = np.arange(len(merged))
    demo = demo[["camera_id", "demo_date", "dx", "sex", "age"]].copy()
    demo["demo_date"] = pd.to_datetime(demo["demo_date"], errors="coerce")
    merged = merged.merge(demo, on="camera_id", how="left")
    # The Demographics sheet has one row per visit. Keep the row dated closest to the rating visit.
    gap = (merged["demo_date"] - pd.to_datetime(merged["visit_date"], errors="coerce")).abs()
    merged = (
        merged.assign(_gap=gap)
        .sort_values(["_rating_row", "_gap"], na_position="last", kind="stable")
        .drop_duplicates("_rating_row", keep="first")
        .sort_values("_rating_row", kind="stable")
        .drop(columns=["_rating_row", "_gap", "demo_date"])
        .reset_index(drop=True)
    )
    assert len(merged) == len(long_df), "demographics merge changed the number of rating rows"
    merged["subject_id"] = merged["camera_id"]
    merged = merged.rename(columns={"date": "visit_date"})
    return merged


def quadratic_weighted_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    mask = ~(np.isnan(y1) | np.isnan(y2))
    if mask.sum() < 2:
        return np.nan
    y1 = np.round(y1[mask]).astype(int)
    y2 = np.round(y2[mask]).astype(int)
    if len(np.unique(y1)) < 2 and len(np.unique(y2)) < 2:
        return 1.0 if np.array_equal(y1, y2) else 0.0
    return float(cohen_kappa_score(y1, y2, weights="quadratic"))


def compute_disagreement_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    active = long_df[long_df["rater"].isin(ACTIVE_RATERS)].copy()
    summary_rows = []
    kappa_rows = []

    grouped = active.groupby(["subject_id", "task"], dropna=False)
    for (subject_id, task), group in grouped:
        scores = group["score"].dropna()
        summary_rows.append(
            {
                "subject_id": subject_id,
                "task": task,
                "n_raters": int(scores.count()),
                "mean_score": float(scores.mean()) if len(scores) else np.nan,
                "std_score": float(scores.std(ddof=0)) if len(scores) > 1 else 0.0,
                "min_score": float(scores.min()) if len(scores) else np.nan,
                "max_score": float(scores.max()) if len(scores) else np.nan,
                "score_range": float(scores.max() - scores.min()) if len(scores) else np.nan,
                "high_disagreement": bool(
                    len(scores) >= 2 and (scores.max() - scores.min()) >= config.HIGH_DISAGREEMENT_RANGE
                ),
            }
        )

    for task, task_df in active.groupby("task"):
        for r1, r2 in itertools.combinations(ACTIVE_RATERS, 2):
            left = task_df[task_df["rater"] == r1][["subject_id", "score"]].rename(
                columns={"score": "s1"}
            )
            right = task_df[task_df["rater"] == r2][["subject_id", "score"]].rename(
                columns={"score": "s2"}
            )
            pair = left.merge(right, on="subject_id", how="inner")
            kappa = quadratic_weighted_kappa(pair["s1"].values, pair["s2"].values)
            kappa_rows.append(
                {
                    "task": task,
                    "rater_1": r1,
                    "rater_2": r2,
                    "quadratic_weighted_kappa": kappa,
                    "n_pairs": len(pair),
                }
            )

    summary = pd.DataFrame(summary_rows)
    kappa_df = pd.DataFrame(kappa_rows)
    summary = summary.merge(
        kappa_df.groupby("task")["quadratic_weighted_kappa"].mean().rename("mean_pairwise_kappa"),
        on="task",
        how="left",
    )
    return summary, kappa_df


def run(
    xlsx_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xlsx_path = xlsx_path or (config.DATA_RAW / config.XLSX_FILENAME)
    output_dir = output_dir or config.DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    wide = load_ratings_wide(xlsx_path, config.RATINGS_SHEET)
    long_df = ratings_to_long(wide)

    demographics = pd.read_excel(xlsx_path, sheet_name=config.DEMOGRAPHICS_SHEET)
    ratings_for_map = wide.rename(columns={0: "booth_id", 1: "date"})
    ratings_for_map["sheet_row"] = ratings_for_map.index
    demographics["sheet_row"] = demographics.index
    mapping = build_id_mapping(ratings_for_map, demographics, method="sheet_row_order")
    mapping.to_csv(output_dir / "booth_camera_id_mapping.csv", index=False)

    clean = merge_demographics(long_df, demographics, mapping)
    clean = clean[clean["rater"].isin(ACTIVE_RATERS)].copy()
    confident = clean[clean["confidence"].isin(["high", "medium"])].copy()

    disagreement_summary, pairwise_kappa = compute_disagreement_summary(clean)

    clean_path = output_dir / "camera_booth_clean.csv"
    confident_path = output_dir / "camera_booth_clean_confident.csv"
    summary_path = output_dir / "disagreement_summary.csv"
    kappa_path = output_dir / "pairwise_kappa_summary.csv"

    clean.to_csv(clean_path, index=False)
    confident.to_csv(confident_path, index=False)
    disagreement_summary.to_csv(summary_path, index=False)
    pairwise_kappa.to_csv(kappa_path, index=False)

    print(f"Wrote {clean_path} ({len(clean)} rows)")
    print(f"Wrote {confident_path} ({len(confident)} rows, high+medium confidence only)")
    print(f"Wrote {summary_path} ({len(disagreement_summary)} rows)")
    n_subjects = clean.loc[clean["subject_id"].notna(), "subject_id"].nunique()
    n_confident = confident.loc[confident["subject_id"].notna(), "subject_id"].nunique()
    print(f"Subjects mapped: {n_subjects} (confident subset: {n_confident})")
    print(f"Needs review: {(mapping['confidence'] == 'needs_review').sum()} sessions")
    print(f"High-disagreement cells: {disagreement_summary['high_disagreement'].sum()}")

    return clean, disagreement_summary, pairwise_kappa


if __name__ == "__main__":
    run()

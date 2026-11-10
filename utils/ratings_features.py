"""Ratings-only feature builders for Phase 2 (no video)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

ACTIVE_RATERS = config.ACTIVE_RATERS
TASK_ORDER = list(config.TASK_SHORT.values())


@dataclass(frozen=True)
class SessionKey:
    subject_id: int
    visit_date: str
    booth_id: int

    @property
    def session_id(self) -> str:
        return f"{int(self.subject_id)}_{self.visit_date}"


def prepare_sessions(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per session × task × rater with normalized types."""
    df = long_df.copy()
    df = df[df["rater"].isin(ACTIVE_RATERS)].dropna(subset=["subject_id", "visit_date"])
    df["subject_id"] = df["subject_id"].astype(int)
    df["booth_id"] = df["booth_id"].astype(int)
    df["visit_date"] = pd.to_datetime(df["visit_date"]).dt.strftime("%Y-%m-%d")
    return df


def session_task_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows = sessions, columns = MultiIndex (task, rater) scores."""
    df = prepare_sessions(long_df)
    wide = df.pivot_table(
        index=["subject_id", "visit_date", "booth_id", "dx", "sex", "age"],
        columns=["task", "rater"],
        values="score",
        aggfunc="mean",
    )
    wide = wide.reindex(columns=pd.MultiIndex.from_product([TASK_ORDER, ACTIVE_RATERS]))
    return wide


def _session_metadata(wide: pd.DataFrame) -> pd.DataFrame:
    meta = wide.index.to_frame(index=False)
    meta["session_id"] = meta["subject_id"].astype(str) + "_" + meta["visit_date"]
    meta["dx_label"] = meta["dx"].map({"PD": 1, "HC": 0})
    return meta


def build_consensus_features(wide: pd.DataFrame) -> pd.DataFrame:
    """8 features: per-task mean UPDRS across all available raters."""
    meta = _session_metadata(wide)
    features = pd.DataFrame(index=wide.index)
    for task in TASK_ORDER:
        task_scores = wide[task]
        features[f"consensus_{task}"] = task_scores.mean(axis=1, skipna=True)
        features[f"consensus_{task}_n_raters"] = task_scores.notna().sum(axis=1)

    out = pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out["feature_set"] = "consensus"
    out["n_features"] = len(TASK_ORDER)
    return out


def build_full_grid_features(wide: pd.DataFrame) -> pd.DataFrame:
    """40 features: every task × rater score (no aggregation)."""
    meta = _session_metadata(wide)
    features = pd.DataFrame(index=wide.index)
    for task in TASK_ORDER:
        for rater in ACTIVE_RATERS:
            features[f"{task}_{rater}"] = wide[(task, rater)].values

    out = pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out["feature_set"] = "full_grid"
    out["n_features"] = len(TASK_ORDER) * len(ACTIVE_RATERS)
    return out


def build_loro_features(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-rater-out features for Shapley attribution.

    One row per (session, held_out_rater): 8 task means computed from the other raters only.
    """
    meta = _session_metadata(wide)
    rows: list[dict] = []

    for row_idx, _ in enumerate(wide.index):
        session_meta = meta.iloc[row_idx].to_dict()
        for held_out in ACTIVE_RATERS:
            row = {**session_meta, "held_out_rater": held_out}
            other_ratings: list[float] = []
            for task in TASK_ORDER:
                task_scores = wide.iloc[row_idx][task]
                other = task_scores.drop(labels=held_out, errors="ignore").dropna()
                row[f"loro_{task}"] = float(other.mean()) if len(other) else np.nan
                row[f"loro_{task}_n_other"] = int(len(other))
                other_ratings.extend(other.tolist())

            row["loro_mean_updrs"] = float(np.mean(other_ratings)) if other_ratings else np.nan
            row["feature_set"] = "loro"
            row["n_features"] = len(TASK_ORDER)
            rows.append(row)

    return pd.DataFrame(rows)


def build_rater_targets(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-rater session targets for Phase 2 losses.

    - rater_mean_updrs: mean score across tasks for that rater
    - rater_total_updrs: sum across tasks
    - rater_pd_proxy: 1 if mean UPDRS > 0 (weak severity signal)
    """
    df = prepare_sessions(long_df)
    rows = []
    for keys, g in df.groupby(["subject_id", "visit_date", "booth_id", "dx", "sex", "age", "rater"]):
        scores = g.groupby("task")["score"].mean()
        mean_updrs = float(scores.mean(skipna=True))
        total_updrs = float(scores.sum(skipna=True))
        rows.append(
            {
                "subject_id": keys[0],
                "visit_date": keys[1],
                "booth_id": keys[2],
                "dx": keys[3],
                "sex": keys[4],
                "age": keys[5],
                "rater": keys[6],
                "session_id": f"{int(keys[0])}_{keys[1]}",
                "dx_label": {"PD": 1, "HC": 0}.get(keys[3], np.nan),
                "rater_mean_updrs": mean_updrs,
                "rater_total_updrs": total_updrs,
                "rater_pd_proxy": int(mean_updrs > 0),
                "rater_n_tasks": int(scores.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def completeness_report(wide: pd.DataFrame) -> pd.DataFrame:
    """How many raters scored each task per session."""
    rows = []
    meta = _session_metadata(wide)
    for i, task in enumerate(TASK_ORDER):
        n_raters = wide[task].notna().sum(axis=1)
        for row_idx, n in enumerate(n_raters):
            rows.append(
                {
                    "session_id": meta.iloc[row_idx]["session_id"],
                    "task": task,
                    "n_raters": int(n),
                }
            )
    summary = pd.DataFrame(rows)
    return (
        summary.groupby(["session_id", "n_raters"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def filter_complete_sessions(
    df: pd.DataFrame,
    min_raters_per_task: int = 3,
    min_tasks: int = 6,
) -> pd.DataFrame:
    """Drop sessions with too few ratings for stable features."""
    if "loro_FT_R_n_other" in df.columns:
        task_cols = [c for c in df.columns if c.startswith("loro_") and c.endswith("_n_other")]
        ok = (df[task_cols] >= min_raters_per_task - 1).sum(axis=1) >= min_tasks
        return df.loc[ok].copy()

    consensus_cols = [c for c in df.columns if c.endswith("_n_raters") and c.startswith("consensus_")]
    if consensus_cols:
        ok = (df[consensus_cols] >= min_raters_per_task).sum(axis=1) >= min_tasks
        return df.loc[ok].copy()

    return df

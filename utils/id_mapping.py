"""Build and validate booth_id + visit_date to CAMERA ID mappings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MappingResult:
    mapping: pd.DataFrame
    flags: pd.DataFrame
    summary: pd.DataFrame


def _prepare_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings_df.copy()
    ratings["visit_date"] = pd.to_datetime(ratings["date"], errors="coerce").dt.date
    if "sheet_row" not in ratings.columns:
        ratings["sheet_row"] = ratings.index
    return ratings


def _prepare_demographics(demographics_df: pd.DataFrame) -> pd.DataFrame:
    demo = demographics_df.copy()
    demo["visit_date"] = pd.to_datetime(demo["Date of Visit "], errors="coerce").dt.date
    demo = demo[demo["visit_date"].notna()].copy()
    if "sheet_row" not in demo.columns:
        demo["sheet_row"] = demo.index
    return demo


def build_id_mapping(
    ratings_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    method: str = "sheet_row_order",
) -> pd.DataFrame:
    """
    Map booth sessions to CAMERA IDs.

    Preferred method is `sheet_row_order`: within each visit date, pair ratings rows
    with demographics rows in the order they appear in the workbook. This matches the
    README "Initial check on order of IDs" and correctly links repeat booth IDs across
    follow-up visits to the same CAMERA ID.
    """
    ratings = _prepare_ratings(ratings_df)
    demo = _prepare_demographics(demographics_df)

    sort_ratings = "sheet_row" if method == "sheet_row_order" else "booth_id"
    sort_demo = "sheet_row" if method == "sheet_row_order" else "CAMERA ID"

    mapping_rows: list[dict] = []
    for visit_date, r_group in ratings.groupby("visit_date"):
        d_group = demo[demo["visit_date"] == visit_date]
        r_sorted = r_group.sort_values(sort_ratings)
        d_sorted = d_group.sort_values(sort_demo)

        if d_group.empty:
            for _, row in r_sorted.iterrows():
                mapping_rows.append(
                    {
                        "booth_id": int(row["booth_id"]),
                        "visit_date": visit_date,
                        "camera_id": np.nan,
                        "mapping_method": "no_demographics_on_date",
                        "confidence": "unmapped",
                        "candidate_camera_ids": "",
                        "notes": "No demographics rows on this visit date.",
                    }
                )
            continue

        if len(r_sorted) != len(d_sorted):
            candidate_ids = ",".join(str(int(x)) for x in d_sorted["CAMERA ID"].tolist())
            for _, row in r_sorted.iterrows():
                mapping_rows.append(
                    {
                        "booth_id": int(row["booth_id"]),
                        "visit_date": visit_date,
                        "camera_id": np.nan,
                        "mapping_method": "date_count_mismatch",
                        "confidence": "needs_review",
                        "candidate_camera_ids": candidate_ids,
                        "notes": (
                            f"Ratings={len(r_sorted)} vs demographics={len(d_sorted)} "
                            "on this date."
                        ),
                    }
                )
            continue

        confidence = "high" if len(r_sorted) == 1 else "medium"
        for (_, r_row), (_, d_row) in zip(r_sorted.iterrows(), d_sorted.iterrows()):
            mapping_rows.append(
                {
                    "booth_id": int(r_row["booth_id"]),
                    "visit_date": visit_date,
                    "camera_id": int(d_row["CAMERA ID"]),
                    "mapping_method": method,
                    "confidence": confidence,
                    "candidate_camera_ids": str(int(d_row["CAMERA ID"])),
                    "notes": "",
                }
            )

    mapping = pd.DataFrame(mapping_rows)
    if mapping.empty:
        return mapping

    mapping = _annotate_repeat_booth_ids(mapping)
    return mapping


def _annotate_repeat_booth_ids(mapping: pd.DataFrame) -> pd.DataFrame:
    """Flag booth IDs reused across dates; downgrade confidence if camera IDs disagree."""
    mapping = mapping.copy()
    mapping["booth_id_reused"] = mapping.groupby("booth_id")["visit_date"].transform("nunique") > 1

    for booth_id, group in mapping.groupby("booth_id"):
        if len(group) <= 1:
            continue
        mapped = group.dropna(subset=["camera_id"])
        if mapped.empty:
            continue
        if mapped["camera_id"].nunique() == 1:
            note = "Repeat booth_id; same CAMERA ID on all mapped visits."
            mapping.loc[group.index, "notes"] = mapping.loc[group.index, "notes"].fillna("").replace("", note)
            if not (mapping.loc[group.index, "notes"] == note).all():
                mapping.loc[group.index, "notes"] = (
                    mapping.loc[group.index, "notes"].fillna("").str.strip() + " " + note
                ).str.strip()
        else:
            mapping.loc[group.index, "confidence"] = "needs_review"
            mapping.loc[group.index, "notes"] = (
                "Repeat booth_id maps to different CAMERA IDs across visits."
            )

    return mapping


def validate_mapping(
    mapping_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
) -> MappingResult:
    """Create review flags and summary statistics for a mapping table."""
    demo = _prepare_demographics(demographics_df)
    ratings = _prepare_ratings(ratings_df)

    flags: list[dict] = []
    for _, row in mapping_df.iterrows():
        if row["confidence"] in {"needs_review", "unmapped"}:
            flags.append(
                {
                    "booth_id": row["booth_id"],
                    "visit_date": row["visit_date"],
                    "issue": row["mapping_method"],
                    "confidence": row["confidence"],
                    "candidate_camera_ids": row.get("candidate_camera_ids", ""),
                    "notes": row.get("notes", ""),
                }
            )

    mapped = mapping_df.dropna(subset=["camera_id"]).copy()
    duplicate_camera = (
        mapped.groupby(["camera_id", "visit_date"])["booth_id"]
        .nunique()
        .reset_index(name="n_booths")
    )
    duplicate_camera = duplicate_camera[duplicate_camera["n_booths"] > 1]
    for _, row in duplicate_camera.iterrows():
        flags.append(
            {
                "booth_id": "",
                "visit_date": row["visit_date"],
                "issue": "duplicate_camera_on_date",
                "confidence": "needs_review",
                "candidate_camera_ids": str(int(row["camera_id"])),
                "notes": f"{int(row['n_booths'])} booth rows map to CAMERA ID {int(row['camera_id'])}.",
            }
        )

    summary = pd.DataFrame(
        [
            {"metric": "rating_sessions", "value": len(ratings)},
            {"metric": "demographics_rows", "value": len(demo)},
            {"metric": "mapped_sessions", "value": int(mapping_df["camera_id"].notna().sum())},
            {"metric": "high_confidence", "value": int((mapping_df["confidence"] == "high").sum())},
            {"metric": "medium_confidence", "value": int((mapping_df["confidence"] == "medium").sum())},
            {
                "metric": "needs_review",
                "value": int((mapping_df["confidence"] == "needs_review").sum()),
            },
            {"metric": "unmapped", "value": int((mapping_df["confidence"] == "unmapped").sum())},
            {
                "metric": "repeat_booth_ids",
                "value": int(mapping_df.loc[mapping_df["booth_id_reused"], "booth_id"].nunique()),
            },
        ]
    )

    return MappingResult(
        mapping=mapping_df,
        flags=pd.DataFrame(flags),
        summary=summary,
    )


def compare_mapping_methods(
    ratings_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare booth_id sort vs sheet row order mappings."""
    booth_map = build_id_mapping(ratings_df, demographics_df, method="booth_id_sort")
    row_map = build_id_mapping(ratings_df, demographics_df, method="sheet_row_order")

    merged = booth_map.merge(
        row_map,
        on=["booth_id", "visit_date"],
        suffixes=("_booth_sort", "_row_order"),
    )
    merged["same_camera_id"] = merged["camera_id_booth_sort"] == merged["camera_id_row_order"]
    return merged

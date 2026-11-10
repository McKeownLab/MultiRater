"""Characterise demographics of high-disagreement (range >= 2) cells.

Questions (collaborator):
  1. Are disputed cells more common in early-stage PD (shorter disease duration)?
  2. Do older patients produce more inter-rater disagreement?

Outputs:
  results/tables/disagreement_by_demographics_summary.csv
  results/tables/disagreement_demographics_cell_level.csv
  results/tables/disagreement_demographics_subject_level.csv
  results/figures/disagreement_vs_age.png
  results/figures/disagreement_vs_disease_duration.png
  results/figures/disputed_cells_dx_breakdown.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _month_to_int(val) -> float:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        return v if 1 <= v <= 12 else np.nan
    s = str(val).strip().lower()
    return float(MONTH_MAP.get(s, np.nan))


def load_demographics() -> pd.DataFrame:
    xlsx = config.DATA_RAW / config.XLSX_FILENAME
    demo = pd.read_excel(xlsx, sheet_name=config.DEMOGRAPHICS_SHEET)
    demo.columns = [c.strip() for c in demo.columns]
    demo = demo.rename(
        columns={
            "CAMERA ID": "subject_id",
            "Date of Visit": "visit_date",
            "Dx": "dx",
            "Sex": "sex",
            "Age": "age",
            "Month of diagnosis": "month_dx",
            "Year of diagnosis": "year_dx",
        }
    )
    demo["subject_id"] = pd.to_numeric(demo["subject_id"], errors="coerce")
    demo["visit_date"] = pd.to_datetime(demo["visit_date"], errors="coerce")
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
    demo["year_dx"] = pd.to_numeric(demo["year_dx"], errors="coerce")
    demo["month_dx_num"] = demo["month_dx"].map(_month_to_int)
    # Mid-year if month missing (July = 7)
    month_fill = demo["month_dx_num"].fillna(7).clip(1, 12)
    diag = pd.to_datetime(
        {
            "year": demo["year_dx"],
            "month": month_fill,
            "day": 1,
        },
        errors="coerce",
    )
    # Only compute duration when year is known
    demo["disease_duration_years"] = np.where(
        demo["year_dx"].notna() & demo["visit_date"].notna(),
        (demo["visit_date"] - diag).dt.days / 365.25,
        np.nan,
    )
    demo["dx"] = demo["dx"].astype(str).str.strip().str.upper()
    demo.loc[~demo["dx"].isin(["PD", "HC"]), "dx"] = np.nan
    # One row per subject (first visit if duplicates)
    demo = demo.sort_values("visit_date").drop_duplicates("subject_id", keep="first")
    return demo[
        ["subject_id", "dx", "sex", "age", "year_dx", "month_dx_num", "disease_duration_years", "visit_date"]
    ]


def load_cells() -> pd.DataFrame:
    path = config.RESULTS_TABLES / "per_subject_task_disagreement.csv"
    cells = pd.read_csv(path)
    cells["subject_id"] = pd.to_numeric(cells["subject_id"], errors="coerce")
    cells["high_disagreement"] = cells["score_range"] >= config.HIGH_DISAGREEMENT_RANGE
    return cells


def _or_ci(table: np.ndarray) -> tuple[float, float, float, float]:
    """Odds ratio + Fisher exact p from 2x2 [[a,b],[c,d]]."""
    a, b = table[0]
    c, d = table[1]
    oddsratio, p = stats.fisher_exact(table)
    # Woolf logit CI
    if min(a, b, c, d) == 0:
        return float(oddsratio), np.nan, np.nan, float(p)
    log_or = np.log(oddsratio)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(oddsratio), float(np.exp(log_or - 1.96 * se)), float(np.exp(log_or + 1.96 * se)), float(p)


def analyse() -> dict:
    demo = load_demographics()
    cells = load_cells()
    merged = cells.merge(demo, on="subject_id", how="left", suffixes=("_cell", ""))
    # Prefer demographics dx when present
    if "dx_cell" in merged.columns:
        merged["dx"] = merged["dx"].fillna(merged["dx_cell"])
        merged = merged.drop(columns=["dx_cell"], errors="ignore")

    disputed = merged[merged["high_disagreement"]].copy()
    n_cells = len(merged)
    n_disp = len(disputed)
    n_subj_disp = disputed["subject_id"].nunique()

    # ---- Subject-level: any disputed cell? ----
    subj = (
        merged.groupby("subject_id", as_index=False)
        .agg(
            n_cells=("task", "count"),
            n_disputed=("high_disagreement", "sum"),
            mean_range=("score_range", "mean"),
            max_range=("score_range", "max"),
            dx=("dx", "first"),
            age=("age", "first"),
            disease_duration_years=("disease_duration_years", "first"),
            sex=("sex", "first"),
        )
    )
    subj["has_disputed"] = subj["n_disputed"] > 0
    subj["pct_disputed"] = 100.0 * subj["n_disputed"] / subj["n_cells"]

    # ---- Coverage ----
    dur_cov_all = int(subj["disease_duration_years"].notna().sum())
    dur_cov_pd = int(subj.loc[subj["dx"] == "PD", "disease_duration_years"].notna().sum())
    n_pd = int((subj["dx"] == "PD").sum())
    age_cov = int(subj["age"].notna().sum())

    # ---- Q1: disease duration (PD only) ----
    pd_subj = subj[subj["dx"] == "PD"].copy()
    pd_with_dur = pd_subj.dropna(subset=["disease_duration_years"]).copy()
    # Early = below median duration among PD with known duration
    med_dur = float(pd_with_dur["disease_duration_years"].median()) if len(pd_with_dur) else np.nan
    pd_with_dur["early_stage"] = pd_with_dur["disease_duration_years"] < med_dur

    # Cell-level among PD with duration
    pd_cells = merged[(merged["dx"] == "PD") & merged["disease_duration_years"].notna()].copy()
    pd_cells["early_stage"] = pd_cells["disease_duration_years"] < med_dur

    # Subject-level: disputed rate early vs late
    early = pd_with_dur[pd_with_dur["early_stage"]]
    late = pd_with_dur[~pd_with_dur["early_stage"]]
    # Mann-Whitney: duration among subjects with vs without disputed cells
    has = pd_with_dur[pd_with_dur["has_disputed"]]["disease_duration_years"]
    no = pd_with_dur[~pd_with_dur["has_disputed"]]["disease_duration_years"]
    mw_dur = stats.mannwhitneyu(has, no, alternative="two-sided") if len(has) and len(no) else None
    # Point-biserial / Spearman: duration vs has_disputed / n_disputed / mean_range
    spear_dur_range = stats.spearmanr(
        pd_with_dur["disease_duration_years"], pd_with_dur["mean_range"]
    )
    spear_dur_ndisp = stats.spearmanr(
        pd_with_dur["disease_duration_years"], pd_with_dur["n_disputed"]
    )
    # Cell-level: disputed fraction early vs late + Fisher
    cell_early_disp = int(pd_cells.loc[pd_cells["early_stage"], "high_disagreement"].sum())
    cell_early_n = int(pd_cells["early_stage"].sum())
    cell_late_disp = int(pd_cells.loc[~pd_cells["early_stage"], "high_disagreement"].sum())
    cell_late_n = int((~pd_cells["early_stage"]).sum())
    fisher_early = _or_ci(
        np.array(
            [
                [cell_early_disp, cell_early_n - cell_early_disp],
                [cell_late_disp, cell_late_n - cell_late_disp],
            ]
        )
    )

    # Continuous: Spearman cell score_range vs duration
    spear_cell_dur = stats.spearmanr(pd_cells["disease_duration_years"], pd_cells["score_range"])

    # ---- Q2: age ----
    age_subj = subj.dropna(subset=["age"])
    older_med = float(age_subj["age"].median())
    age_subj = age_subj.copy()
    age_subj["older"] = age_subj["age"] >= older_med
    has_a = age_subj[age_subj["has_disputed"]]["age"]
    no_a = age_subj[~age_subj["has_disputed"]]["age"]
    mw_age = stats.mannwhitneyu(has_a, no_a, alternative="two-sided") if len(has_a) and len(no_a) else None
    spear_age_range = stats.spearmanr(age_subj["age"], age_subj["mean_range"])
    spear_age_ndisp = stats.spearmanr(age_subj["age"], age_subj["n_disputed"])

    age_cells = merged.dropna(subset=["age"]).copy()
    age_cells["older"] = age_cells["age"] >= older_med
    cell_old_disp = int(age_cells.loc[age_cells["older"], "high_disagreement"].sum())
    cell_old_n = int(age_cells["older"].sum())
    cell_young_disp = int(age_cells.loc[~age_cells["older"], "high_disagreement"].sum())
    cell_young_n = int((~age_cells["older"]).sum())
    fisher_age = _or_ci(
        np.array(
            [
                [cell_old_disp, cell_old_n - cell_old_disp],
                [cell_young_disp, cell_young_n - cell_young_disp],
            ]
        )
    )
    spear_cell_age = stats.spearmanr(age_cells["age"], age_cells["score_range"])

    # Also within PD only for age
    pd_age = age_subj[age_subj["dx"] == "PD"]
    spear_age_pd = stats.spearmanr(pd_age["age"], pd_age["mean_range"]) if len(pd_age) > 5 else None

    # ---- Dx breakdown of disputed cells ----
    dx_counts = disputed["dx"].value_counts(dropna=False)
    all_dx = merged["dx"].value_counts(dropna=False)

    # Severity context: mean score of disputed vs not
    sev = merged.groupby("high_disagreement")["mean_score"].agg(["mean", "median", "count"])

    summary_rows = [
        {"question": "coverage", "metric": "n_cells_total", "value": n_cells},
        {"question": "coverage", "metric": "n_disputed_cells", "value": n_disp},
        {"question": "coverage", "metric": "n_subjects_with_disputed", "value": n_subj_disp},
        {"question": "coverage", "metric": "age_known_subjects", "value": age_cov},
        {"question": "coverage", "metric": "duration_known_subjects_all", "value": dur_cov_all},
        {"question": "coverage", "metric": "duration_known_subjects_PD", "value": dur_cov_pd},
        {"question": "coverage", "metric": "n_PD_subjects", "value": n_pd},
        {"question": "coverage", "metric": "median_duration_years_PD", "value": med_dur},
        {"question": "coverage", "metric": "median_age", "value": older_med},
        {
            "question": "dx",
            "metric": "disputed_cells_PD",
            "value": int(dx_counts.get("PD", 0)),
        },
        {
            "question": "dx",
            "metric": "disputed_cells_HC",
            "value": int(dx_counts.get("HC", 0)),
        },
        {
            "question": "dx",
            "metric": "pct_disputed_among_PD_cells",
            "value": 100.0 * merged.loc[merged["dx"] == "PD", "high_disagreement"].mean(),
        },
        {
            "question": "dx",
            "metric": "pct_disputed_among_HC_cells",
            "value": 100.0 * merged.loc[merged["dx"] == "HC", "high_disagreement"].mean(),
        },
        {
            "question": "duration",
            "metric": "mean_duration_subjects_with_disputed",
            "value": float(has.mean()) if len(has) else np.nan,
        },
        {
            "question": "duration",
            "metric": "mean_duration_subjects_without_disputed",
            "value": float(no.mean()) if len(no) else np.nan,
        },
        {
            "question": "duration",
            "metric": "mannwhitney_U_p",
            "value": float(mw_dur.pvalue) if mw_dur else np.nan,
        },
        {
            "question": "duration",
            "metric": "spearman_duration_vs_mean_range_rho",
            "value": float(spear_dur_range.statistic),
        },
        {
            "question": "duration",
            "metric": "spearman_duration_vs_mean_range_p",
            "value": float(spear_dur_range.pvalue),
        },
        {
            "question": "duration",
            "metric": "spearman_duration_vs_n_disputed_rho",
            "value": float(spear_dur_ndisp.statistic),
        },
        {
            "question": "duration",
            "metric": "spearman_duration_vs_n_disputed_p",
            "value": float(spear_dur_ndisp.pvalue),
        },
        {
            "question": "duration",
            "metric": "cell_pct_disputed_early",
            "value": 100.0 * cell_early_disp / cell_early_n if cell_early_n else np.nan,
        },
        {
            "question": "duration",
            "metric": "cell_pct_disputed_late",
            "value": 100.0 * cell_late_disp / cell_late_n if cell_late_n else np.nan,
        },
        {
            "question": "duration",
            "metric": "fisher_OR_early_vs_late",
            "value": fisher_early[0],
        },
        {
            "question": "duration",
            "metric": "fisher_OR_CI_lo",
            "value": fisher_early[1],
        },
        {
            "question": "duration",
            "metric": "fisher_OR_CI_hi",
            "value": fisher_early[2],
        },
        {
            "question": "duration",
            "metric": "fisher_p",
            "value": fisher_early[3],
        },
        {
            "question": "duration",
            "metric": "spearman_cell_duration_vs_range_rho",
            "value": float(spear_cell_dur.statistic),
        },
        {
            "question": "duration",
            "metric": "spearman_cell_duration_vs_range_p",
            "value": float(spear_cell_dur.pvalue),
        },
        {
            "question": "age",
            "metric": "mean_age_subjects_with_disputed",
            "value": float(has_a.mean()) if len(has_a) else np.nan,
        },
        {
            "question": "age",
            "metric": "mean_age_subjects_without_disputed",
            "value": float(no_a.mean()) if len(no_a) else np.nan,
        },
        {
            "question": "age",
            "metric": "mannwhitney_U_p",
            "value": float(mw_age.pvalue) if mw_age else np.nan,
        },
        {
            "question": "age",
            "metric": "spearman_age_vs_mean_range_rho",
            "value": float(spear_age_range.statistic),
        },
        {
            "question": "age",
            "metric": "spearman_age_vs_mean_range_p",
            "value": float(spear_age_range.pvalue),
        },
        {
            "question": "age",
            "metric": "spearman_age_vs_n_disputed_rho",
            "value": float(spear_age_ndisp.statistic),
        },
        {
            "question": "age",
            "metric": "spearman_age_vs_n_disputed_p",
            "value": float(spear_age_ndisp.pvalue),
        },
        {
            "question": "age",
            "metric": "cell_pct_disputed_older",
            "value": 100.0 * cell_old_disp / cell_old_n if cell_old_n else np.nan,
        },
        {
            "question": "age",
            "metric": "cell_pct_disputed_younger",
            "value": 100.0 * cell_young_disp / cell_young_n if cell_young_n else np.nan,
        },
        {
            "question": "age",
            "metric": "fisher_OR_older_vs_younger",
            "value": fisher_age[0],
        },
        {
            "question": "age",
            "metric": "fisher_OR_CI_lo",
            "value": fisher_age[1],
        },
        {
            "question": "age",
            "metric": "fisher_OR_CI_hi",
            "value": fisher_age[2],
        },
        {
            "question": "age",
            "metric": "fisher_p",
            "value": fisher_age[3],
        },
        {
            "question": "age",
            "metric": "spearman_cell_age_vs_range_rho",
            "value": float(spear_cell_age.statistic),
        },
        {
            "question": "age",
            "metric": "spearman_cell_age_vs_range_p",
            "value": float(spear_cell_age.pvalue),
        },
        {
            "question": "age",
            "metric": "spearman_age_vs_mean_range_PD_only_rho",
            "value": float(spear_age_pd.statistic) if spear_age_pd else np.nan,
        },
        {
            "question": "age",
            "metric": "spearman_age_vs_mean_range_PD_only_p",
            "value": float(spear_age_pd.pvalue) if spear_age_pd else np.nan,
        },
        {
            "question": "severity",
            "metric": "mean_score_disputed_cells",
            "value": float(sev.loc[True, "mean"]) if True in sev.index else np.nan,
        },
        {
            "question": "severity",
            "metric": "mean_score_agreed_cells",
            "value": float(sev.loc[False, "mean"]) if False in sev.index else np.nan,
        },
    ]

    summary = pd.DataFrame(summary_rows)
    return {
        "merged": merged,
        "subj": subj,
        "summary": summary,
        "med_dur": med_dur,
        "older_med": older_med,
        "pd_with_dur": pd_with_dur,
        "pd_cells": pd_cells,
        "age_subj": age_subj,
        "age_cells": age_cells,
        "disputed": disputed,
    }


def plot_age(age_subj: pd.DataFrame, age_cells: pd.DataFrame, older_med: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for flag, label, color in [
        (True, "Has disputed cell", "#c0392b"),
        (False, "No disputed cell", "#16a085"),
    ]:
        sub = age_subj[age_subj["has_disputed"] == flag]["age"].dropna()
        ax.hist(sub, bins=15, alpha=0.55, label=f"{label} (n={len(sub)})", color=color)
    ax.axvline(older_med, color="black", ls="--", lw=1, label=f"median age={older_med:.0f}")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Subjects")
    ax.set_title("Age by whether subject has any\nhigh-disagreement cell (range>=2)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(
        age_subj["age"],
        age_subj["mean_range"],
        c=np.where(age_subj["has_disputed"], "#c0392b", "#16a085"),
        alpha=0.65,
        s=28,
        edgecolors="none",
    )
    # trend line
    ok = age_subj[["age", "mean_range"]].dropna()
    if len(ok) > 2:
        z = np.polyfit(ok["age"], ok["mean_range"], 1)
        xs = np.linspace(ok["age"].min(), ok["age"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="black", lw=1.5)
    rho, p = stats.spearmanr(age_subj["age"], age_subj["mean_range"])
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Mean inter-rater range\n(across tasks)")
    ax.set_title(f"Age vs mean disagreement\nSpearman rho={rho:.2f}, p={p:.3g}")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_duration(pd_with_dur: pd.DataFrame, pd_cells: pd.DataFrame, med_dur: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for flag, label, color in [
        (True, "Has disputed cell", "#c0392b"),
        (False, "No disputed cell", "#16a085"),
    ]:
        sub = pd_with_dur[pd_with_dur["has_disputed"] == flag]["disease_duration_years"]
        ax.hist(sub, bins=12, alpha=0.55, label=f"{label} (n={len(sub)})", color=color)
    ax.axvline(med_dur, color="black", ls="--", lw=1, label=f"median={med_dur:.1f}y")
    ax.set_xlabel("Disease duration (years)")
    ax.set_ylabel("PD subjects (with known duration)")
    ax.set_title("Duration by disputed-cell status\n(PD only; duration known)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(
        pd_with_dur["disease_duration_years"],
        pd_with_dur["mean_range"],
        c=np.where(pd_with_dur["has_disputed"], "#c0392b", "#16a085"),
        alpha=0.7,
        s=36,
        edgecolors="none",
    )
    ok = pd_with_dur[["disease_duration_years", "mean_range"]].dropna()
    if len(ok) > 2:
        z = np.polyfit(ok["disease_duration_years"], ok["mean_range"], 1)
        xs = np.linspace(ok["disease_duration_years"].min(), ok["disease_duration_years"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="black", lw=1.5)
    rho, p = stats.spearmanr(pd_with_dur["disease_duration_years"], pd_with_dur["mean_range"])
    ax.set_xlabel("Disease duration (years)")
    ax.set_ylabel("Mean inter-rater range")
    ax.set_title(f"Duration vs mean disagreement\nSpearman rho={rho:.2f}, p={p:.3g}")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_dx(disputed: pd.DataFrame, merged: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    counts = disputed["dx"].fillna("Unknown").value_counts()
    colors = {"PD": "#c0392b", "HC": "#2980b9", "Unknown": "#95a5a6"}
    ax.bar(counts.index.astype(str), counts.values, color=[colors.get(str(i), "#7f8c8d") for i in counts.index])
    ax.set_ylabel("Disputed cells (range >= 2)")
    ax.set_title(f"Who owns the {len(disputed)} disputed cells?")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 1, str(v), ha="center", fontsize=10)

    ax = axes[1]
    rates = (
        merged.dropna(subset=["dx"])
        .groupby("dx")["high_disagreement"]
        .mean()
        .mul(100)
        .reindex(["PD", "HC"])
    )
    ax.bar(rates.index.astype(str), rates.values, color=["#c0392b", "#2980b9"])
    ax.set_ylabel("% of cells with range >= 2")
    ax.set_title("Disagreement rate by diagnosis")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_report(out: dict) -> None:
    s = out["summary"].set_index(["question", "metric"])["value"]

    def g(q, m, fmt=".3f"):
        v = s.loc[(q, m)]
        if pd.isna(v):
            return "NA"
        if fmt == "d":
            return str(int(v))
        return f"{v:{fmt}}"

    print("=" * 70)
    print("DISPUTED CELLS — DEMOGRAPHIC CHARACTERISATION")
    print("=" * 70)
    print(f"Disputed cells (range>=2): {g('coverage','n_disputed_cells','d')} / {g('coverage','n_cells_total','d')}")
    print(f"Subjects with >=1 disputed cell: {g('coverage','n_subjects_with_disputed','d')}")
    print(f"Age known: {g('coverage','age_known_subjects','d')} subjects")
    print(
        f"Disease duration known: {g('coverage','duration_known_subjects_PD','d')} / "
        f"{g('coverage','n_PD_subjects','d')} PD subjects "
        f"(median duration {g('coverage','median_duration_years_PD','.1f')} y)"
    )
    print()
    print("--- Dx of disputed cells ---")
    print(f"  PD: {g('dx','disputed_cells_PD','d')} | HC: {g('dx','disputed_cells_HC','d')}")
    print(f"  Rate among PD cells: {g('dx','pct_disputed_among_PD_cells','.1f')}%")
    print(f"  Rate among HC cells: {g('dx','pct_disputed_among_HC_cells','.1f')}%")
    print(f"  Mean severity disputed vs agreed: {g('severity','mean_score_disputed_cells','.2f')} vs {g('severity','mean_score_agreed_cells','.2f')}")
    print()
    print("--- Q1: Early-stage PD (shorter duration)? ---")
    print(f"  Mean duration WITH disputed:    {g('duration','mean_duration_subjects_with_disputed','.2f')} y")
    print(f"  Mean duration WITHOUT disputed: {g('duration','mean_duration_subjects_without_disputed','.2f')} y")
    print(f"  Mann-Whitney p: {g('duration','mannwhitney_U_p','.3g')}")
    print(
        f"  Spearman duration vs mean range: rho={g('duration','spearman_duration_vs_mean_range_rho')} "
        f"p={g('duration','spearman_duration_vs_mean_range_p','.3g')}"
    )
    print(
        f"  Cell % disputed early (<median) vs late: "
        f"{g('duration','cell_pct_disputed_early','.1f')}% vs {g('duration','cell_pct_disputed_late','.1f')}%"
    )
    print(
        f"  Fisher OR (early vs late): {g('duration','fisher_OR_early_vs_late')} "
        f"[{g('duration','fisher_OR_CI_lo')}, {g('duration','fisher_OR_CI_hi')}] "
        f"p={g('duration','fisher_p','.3g')}"
    )
    print()
    print("--- Q2: Older patients more disagreement? ---")
    print(f"  Mean age WITH disputed:    {g('age','mean_age_subjects_with_disputed','.1f')}")
    print(f"  Mean age WITHOUT disputed: {g('age','mean_age_subjects_without_disputed','.1f')}")
    print(f"  Mann-Whitney p: {g('age','mannwhitney_U_p','.3g')}")
    print(
        f"  Spearman age vs mean range: rho={g('age','spearman_age_vs_mean_range_rho')} "
        f"p={g('age','spearman_age_vs_mean_range_p','.3g')}"
    )
    print(
        f"  Cell % disputed older (>=median) vs younger: "
        f"{g('age','cell_pct_disputed_older','.1f')}% vs {g('age','cell_pct_disputed_younger','.1f')}%"
    )
    print(
        f"  Fisher OR (older vs younger): {g('age','fisher_OR_older_vs_younger')} "
        f"[{g('age','fisher_OR_CI_lo')}, {g('age','fisher_OR_CI_hi')}] "
        f"p={g('age','fisher_p','.3g')}"
    )
    print(
        f"  PD-only Spearman age vs mean range: rho={g('age','spearman_age_vs_mean_range_PD_only_rho')} "
        f"p={g('age','spearman_age_vs_mean_range_PD_only_p','.3g')}"
    )


def main() -> None:
    config.RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    config.RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)

    out = analyse()
    out["summary"].to_csv(config.RESULTS_TABLES / "disagreement_by_demographics_summary.csv", index=False)
    out["merged"].to_csv(config.RESULTS_TABLES / "disagreement_demographics_cell_level.csv", index=False)
    out["subj"].to_csv(config.RESULTS_TABLES / "disagreement_demographics_subject_level.csv", index=False)

    plot_age(out["age_subj"], out["age_cells"], out["older_med"], config.RESULTS_FIGURES / "disagreement_vs_age.png")
    plot_duration(
        out["pd_with_dur"], out["pd_cells"], out["med_dur"],
        config.RESULTS_FIGURES / "disagreement_vs_disease_duration.png",
    )
    plot_dx(out["disputed"], out["merged"], config.RESULTS_FIGURES / "disputed_cells_dx_breakdown.png")

    print_report(out)
    print()
    print(f"Tables: {config.RESULTS_TABLES}")
    print(f"Figures: {config.RESULTS_FIGURES}")


if __name__ == "__main__":
    main()

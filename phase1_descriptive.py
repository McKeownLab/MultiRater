"""Phase 1: Descriptive inter-rater structure analysis."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from utils.icc import icc_c1_per_task
from phase0_preprocessing import quadratic_weighted_kappa

ACTIVE_RATERS = config.ACTIVE_RATERS
TASK_ORDER = list(config.TASK_SHORT.values())
TASK_FAMILY_ORDER = ["FT", "OC", "PS", "PT"]


def load_clean_data(path: Path | None = None) -> pd.DataFrame:
    path = path or config.CLEAN_DATA_DEFAULT
    df = pd.read_csv(path, parse_dates=["visit_date"])
    return df[df["subject_id"].notna()].copy()


def rater_bias_profiles(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for rater, r_df in long_df.groupby("rater"):
        rows.append(
            {
                "rater": rater,
                "mean_score": r_df["score"].mean(),
                "median_score": r_df["score"].median(),
                "std_score": r_df["score"].std(),
                "n_scores": int(r_df["score"].notna().sum()),
            }
        )

    task_rows = []
    for (rater, task), g in long_df.groupby(["rater", "task"]):
        task_rows.append(
            {
                "rater": rater,
                "task": task,
                "task_family": config.TASK_FAMILY.get(task, task),
                "mean_score": g["score"].mean(),
                "std_score": g["score"].std(),
                "n_scores": int(g["score"].notna().sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(task_rows)


def pairwise_kappa_matrix(long_df: pd.DataFrame, task: str | None = None) -> pd.DataFrame:
    data = long_df if task is None else long_df[long_df["task"] == task]
    matrix = pd.DataFrame(index=ACTIVE_RATERS, columns=ACTIVE_RATERS, dtype=float)
    for r1, r2 in itertools.combinations(ACTIVE_RATERS, 2):
        left = data[data["rater"] == r1][["subject_id", "task", "score"]].rename(columns={"score": "s1"})
        right = data[data["rater"] == r2][["subject_id", "task", "score"]].rename(columns={"score": "s2"})
        merge_on = ["subject_id"] if task else ["subject_id", "task"]
        pair = left.merge(right, on=merge_on, how="inner")
        kappa = quadratic_weighted_kappa(pair["s1"].values, pair["s2"].values)
        matrix.loc[r1, r2] = kappa
        matrix.loc[r2, r1] = kappa
    np.fill_diagonal(matrix.values, 1.0)
    return matrix.astype(float)


def pairwise_kappa_by_task(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        matrix = pairwise_kappa_matrix(long_df, task=task)
        for r1, r2 in itertools.combinations(ACTIVE_RATERS, 2):
            rows.append(
                {
                    "task": task,
                    "task_family": config.TASK_FAMILY[task],
                    "rater_1": r1,
                    "rater_2": r2,
                    "quadratic_weighted_kappa": matrix.loc[r1, r2],
                }
            )
    return pd.DataFrame(rows)


def per_subject_disagreement(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject_id, task), g in long_df.groupby(["subject_id", "task"]):
        scores = g["score"].dropna()
        if scores.empty:
            continue
        rows.append(
            {
                "subject_id": int(subject_id),
                "task": task,
                "task_family": config.TASK_FAMILY.get(task, task),
                "dx": g["dx"].iloc[0],
                "n_raters": int(scores.count()),
                "mean_score": float(scores.mean()),
                "std_score": float(scores.std(ddof=0)) if len(scores) > 1 else 0.0,
                "score_range": float(scores.max() - scores.min()) if len(scores) > 1 else 0.0,
                "high_disagreement": bool(
                    len(scores) >= 2 and (scores.max() - scores.min()) >= config.HIGH_DISAGREEMENT_RANGE
                ),
            }
        )
    return pd.DataFrame(rows)


def disagreement_by_group(long_df: pd.DataFrame) -> pd.DataFrame:
    cell_df = per_subject_disagreement(long_df)
    rows = []
    for (dx, task), g in cell_df.groupby(["dx", "task"]):
        rows.append(
            {
                "dx": dx,
                "task": task,
                "task_family": config.TASK_FAMILY.get(task, task),
                "mean_range": g["score_range"].mean(),
                "std_range": g["score_range"].std(),
                "mean_std": g["std_score"].mean(),
                "n_subjects": len(g),
                "n_high_disagreement": int(g["high_disagreement"].sum()),
            }
        )
    return pd.DataFrame(rows)


def task_level_summary(cell_df: pd.DataFrame) -> pd.DataFrame:
    task_summary = (
        cell_df.groupby("task")
        .agg(
            mean_range=("score_range", "mean"),
            mean_std=("std_score", "mean"),
            n_subjects=("subject_id", "nunique"),
            n_high_disagreement=("high_disagreement", "sum"),
        )
        .reset_index()
    )
    task_summary["task_family"] = task_summary["task"].map(config.TASK_FAMILY)

    family_summary = (
        cell_df.groupby("task_family")
        .agg(
            mean_range=("score_range", "mean"),
            mean_std=("std_score", "mean"),
            n_subjects=("subject_id", "nunique"),
            n_high_disagreement=("high_disagreement", "sum"),
        )
        .reset_index()
    )
    return task_summary, family_summary


def save_heatmap(matrix: pd.DataFrame, path: Path, title: str, vmin: float = 0, vmax: float = 1) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_icc_by_task(icc_df: pd.DataFrame, path: Path) -> None:
    plot_df = icc_df.sort_values("icc", ascending=True)
    plt.figure(figsize=(8, 5))
    colors = ["#2ecc71" if v >= config.WILLIAMS_ICC_BENCHMARK else "#e74c3c" for v in plot_df["icc"]]
    plt.barh(plot_df["task"], plot_df["icc"], color=colors)
    plt.axvline(config.WILLIAMS_ICC_BENCHMARK, color="black", linestyle="--", label="Williams et al. (0.53)")
    plt.xlabel("ICC(2,1)")
    plt.ylabel("Task")
    plt.title("Intraclass Correlation by Task (5 raters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_rater_bias(bias_df: pd.DataFrame, path: Path) -> None:
    plot_df = bias_df.sort_values("mean_score")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=plot_df, x="rater", y="mean_score", order=plot_df["rater"], color="#3498db")
    plt.ylabel("Mean UPDRS score")
    plt.xlabel("Rater")
    plt.title("Rater bias profiles (all tasks)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_rater_task_heatmap(task_bias_df: pd.DataFrame, path: Path) -> None:
    pivot = task_bias_df.pivot(index="rater", columns="task", values="mean_score")
    pivot = pivot.reindex(columns=TASK_ORDER)
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=4)
    plt.title("Mean UPDRS score by rater and task")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_score_distributions(long_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=long_df, x="rater", y="score", order=ACTIVE_RATERS)
    plt.title("UPDRS score distribution by rater")
    plt.ylabel("Score (0–4)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_disagreement_by_task(task_summary: pd.DataFrame, path: Path) -> None:
    plot_df = task_summary.sort_values("mean_range", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="task", y="mean_range", order=plot_df["task"], color="#8e44ad")
    plt.ylabel("Mean inter-rater range")
    plt.xlabel("Task")
    plt.title("Disagreement by task (mean score range across raters)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_disagreement_by_dx(dx_df: pd.DataFrame, path: Path) -> None:
    family_df = (
        dx_df.groupby(["dx", "task_family"])["mean_range"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=family_df, x="task_family", y="mean_range", hue="dx", order=TASK_FAMILY_ORDER)
    plt.ylabel("Mean inter-rater range")
    plt.xlabel("Task family")
    plt.title("Disagreement by diagnosis (PD vs HC)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_task_family_kappa(kappa_by_task: pd.DataFrame, figures_dir: Path) -> None:
    family_kappa = (
        kappa_by_task.groupby(["task_family", "rater_1", "rater_2"])["quadratic_weighted_kappa"]
        .mean()
        .reset_index()
    )
    for family in TASK_FAMILY_ORDER:
        sub = family_kappa[family_kappa["task_family"] == family]
        if sub.empty:
            continue
        mat = pairwise_kappa_matrix_from_long(sub, "task_family", family)
        save_heatmap(
            mat,
            figures_dir / f"pairwise_kappa_family_{family}.png",
            f"Pairwise κ — {family} tasks (avg L/R)",
        )


def pairwise_kappa_matrix_from_long(df: pd.DataFrame, group_col: str, group_val: str) -> pd.DataFrame:
    sub = df[df[group_col] == group_val] if group_col in df.columns else df
    matrix = pd.DataFrame(index=ACTIVE_RATERS, columns=ACTIVE_RATERS, dtype=float)
    for _, row in sub.iterrows():
        matrix.loc[row["rater_1"], row["rater_2"]] = row["quadratic_weighted_kappa"]
        matrix.loc[row["rater_2"], row["rater_1"]] = row["quadratic_weighted_kappa"]
    np.fill_diagonal(matrix.values, 1.0)
    return matrix.astype(float)


def write_phase1_summary(
    path: Path,
    data_source: str,
    long_df: pd.DataFrame,
    icc_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    kappa_matrix: pd.DataFrame,
    cell_df: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 1 Summary — Inter-Rater Structure",
        "",
        f"- Data source: `{data_source}`",
        f"- Subjects with demographics: {long_df['subject_id'].nunique()}",
        f"- Raters: {', '.join(ACTIVE_RATERS)}",
        f"- Rating rows: {len(long_df)}",
        "",
        "## ICC(2,1) per task",
        f"- Mean ICC: **{icc_df['icc'].mean():.3f}**",
        f"- Williams et al. benchmark: **{config.WILLIAMS_ICC_BENCHMARK}**",
        f"- Tasks above benchmark: {(icc_df['icc'] >= config.WILLIAMS_ICC_BENCHMARK).sum()} / {len(icc_df)}",
        "",
        "## Pairwise agreement (quadratic κ, all tasks)",
    ]
    for r1, r2 in itertools.combinations(ACTIVE_RATERS, 2):
        lines.append(f"- {r1}–{r2}: {kappa_matrix.loc[r1, r2]:.3f}")

    strictest = bias_df.sort_values("mean_score").iloc[0]
    most_lenient = bias_df.sort_values("mean_score").iloc[-1]
    lines.extend(
        [
            "",
            "## Rater bias",
            f"- Most strict (lowest mean): **{strictest['rater']}** ({strictest['mean_score']:.2f})",
            f"- Most lenient (highest mean): **{most_lenient['rater']}** ({most_lenient['mean_score']:.2f})",
            "",
            "## Disagreement",
            f"- High-disagreement cells (range ≥ {config.HIGH_DISAGREEMENT_RANGE}): **{int(cell_df['high_disagreement'].sum())}**",
            f"- Task family with highest mean range: **{family_summary.sort_values('mean_range', ascending=False).iloc[0]['task_family']}**",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(clean_path: Path | None = None) -> pd.DataFrame:
    config.RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    config.RESULTS_TABLES.mkdir(parents=True, exist_ok=True)

    clean_path = clean_path or config.CLEAN_DATA_DEFAULT
    long_df = load_clean_data(clean_path)
    long_df = long_df[long_df["rater"].isin(ACTIVE_RATERS)].copy()

    icc_df = icc_c1_per_task(long_df, ACTIVE_RATERS)
    bias_df, task_bias_df = rater_bias_profiles(long_df)
    kappa_matrix = pairwise_kappa_matrix(long_df)
    kappa_by_task = pairwise_kappa_by_task(long_df)
    cell_df = per_subject_disagreement(long_df)
    dx_df = disagreement_by_group(long_df)
    task_summary, family_summary = task_level_summary(cell_df)

    high_disagreement = cell_df[cell_df["high_disagreement"]].sort_values(
        "score_range", ascending=False
    )

    bias_df.to_csv(config.RESULTS_TABLES / "rater_statistics.csv", index=False)
    icc_df.to_csv(config.RESULTS_TABLES / "icc_by_task.csv", index=False)
    task_summary.to_csv(config.RESULTS_TABLES / "task_variance.csv", index=False)
    family_summary.to_csv(config.RESULTS_TABLES / "task_family_variance.csv", index=False)
    dx_df.to_csv(config.RESULTS_TABLES / "disagreement_by_dx.csv", index=False)
    task_bias_df.to_csv(config.RESULTS_TABLES / "rater_task_bias.csv", index=False)
    kappa_matrix.to_csv(config.RESULTS_TABLES / "pairwise_kappa_matrix.csv", index=False)
    kappa_by_task.to_csv(config.RESULTS_TABLES / "pairwise_kappa_by_task.csv", index=False)
    cell_df.to_csv(config.RESULTS_TABLES / "per_subject_task_disagreement.csv", index=False)
    high_disagreement.to_csv(config.RESULTS_TABLES / "high_disagreement_cases.csv", index=False)

    save_heatmap(
        kappa_matrix,
        config.RESULTS_FIGURES / "pairwise_kappa_heatmap.png",
        "Pairwise Quadratic-Weighted κ (All Tasks)",
    )
    for task in TASK_ORDER:
        task_mat = pairwise_kappa_matrix(long_df, task=task)
        save_heatmap(
            task_mat,
            config.RESULTS_FIGURES / f"pairwise_kappa_{task}.png",
            f"Pairwise κ — {task}",
        )

    plot_task_family_kappa(kappa_by_task, config.RESULTS_FIGURES)
    plot_icc_by_task(icc_df, config.RESULTS_FIGURES / "icc_by_task.png")
    plot_rater_bias(bias_df, config.RESULTS_FIGURES / "rater_bias.png")
    plot_rater_task_heatmap(task_bias_df, config.RESULTS_FIGURES / "rater_task_heatmap.png")
    plot_score_distributions(long_df, config.RESULTS_FIGURES / "score_distribution_by_rater.png")
    plot_disagreement_by_task(task_summary, config.RESULTS_FIGURES / "disagreement_by_task.png")
    plot_disagreement_by_dx(dx_df, config.RESULTS_FIGURES / "disagreement_by_dx.png")

    write_phase1_summary(
        config.RESULTS_TABLES / "phase1_summary.md",
        clean_path.name,
        long_df,
        icc_df,
        bias_df,
        kappa_matrix,
        cell_df,
        family_summary,
    )

    print(f"Data: {clean_path}")
    print(f"Subjects: {long_df['subject_id'].nunique()} | Rows: {len(long_df)}")
    print("\nICC by task:")
    print(icc_df.to_string(index=False))
    print(f"\nMean ICC: {icc_df['icc'].mean():.3f} (benchmark {config.WILLIAMS_ICC_BENCHMARK})")
    print("\nRater bias (mean score):")
    print(bias_df.sort_values("mean_score")[["rater", "mean_score"]].to_string(index=False))
    print(f"\nHigh-disagreement cells: {int(cell_df['high_disagreement'].sum())}")
    print(f"\nTables: {config.RESULTS_TABLES}")
    print(f"Figures: {config.RESULTS_FIGURES}")

    return bias_df


if __name__ == "__main__":
    run()

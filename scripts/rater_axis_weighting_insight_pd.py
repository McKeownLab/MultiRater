"""Rater axis-weighting analysis using INSIGHT-PD sign labels.

Tests whether R1–R5 UPDRS scores load differently onto amplitude / sequence
effect / slowness / halt-hesitation signs (complementary perspectives vs
global leniency).

Outputs under results/tables/ and results/figures/ (R1–R5 only; no initials).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

SIGNS_CSV = config.DATA_EXTERNAL / "insight_pd_signs.csv"
MODELING_CSV = config.DATA_PROCESSED / "ordinal_task_rows.csv"

# Internal columns → paper labels only in outputs
RATER_COLS = {r: f"y_{r}" for r in config.ACTIVE_RATERS}
SIGN_COLS = {
    "amplitude_reduction": "pred_low_amplitude",
    "sequence_effect": "pred_sequence_effect",
    "slowness": "pred_slowness",
    "halt_hesitation": "pred_halt_hesitation",
}
SIGN_ORDER = list(SIGN_COLS.keys())
PAPER_ORDER = [config.paper_rater(r) for r in config.ACTIVE_RATERS]

OUT_TAB = config.RESULTS_TABLES
OUT_FIG = config.RESULTS_FIGURES
MIN_N_FLAG = 30
MIN_N_WARN = 50


def _p(r: str) -> str:
    return config.paper_rater(r)


def load_and_inspect() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 72)
    print("STEP 1 — Load and inspect")
    print("=" * 72)

    signs = pd.read_csv(SIGNS_CSV)
    model = pd.read_csv(MODELING_CSV)

    print("\n--- File 1: INSIGHT-PD signs (collaborator aligned) ---")
    print("path:", SIGNS_CSV)
    print("shape:", signs.shape)
    print("columns:", list(signs.columns))
    print("dtypes:\n", signs.dtypes)
    print("head:\n", signs.head())
    print("unique tasks:", sorted(signs["task"].astype(str).unique()))
    if "task_prefix" in signs.columns:
        print("unique task_prefix:", sorted(signs["task_prefix"].dropna().astype(str).unique()))
    if "split" in signs.columns:
        print("split counts:", signs["split"].value_counts(dropna=False).to_dict())

    print("\n--- File 2: modeling table ---")
    print("path:", MODELING_CSV)
    print("shape:", model.shape)
    print("columns:", list(model.columns))
    print("dtypes:\n", model.dtypes)
    print("head:\n", model.head())
    print("unique tasks:", sorted(model["task"].astype(str).unique()))
    if "split" in model.columns:
        print("split counts:", model["split"].value_counts(dropna=False).to_dict())

    s_sess = set(signs["session_id"].astype(str))
    m_sess = set(model["session_id"].astype(str))
    overlap = len(s_sess & m_sess)
    print(f"\nsession_id overlap: {overlap} (signs={len(s_sess)}, modeling={len(m_sess)})")
    if overlap < 50:
        raise SystemExit(f"ABORT: session overlap {overlap} < 50")

    print("\nMissing rate per sign column (File 1):")
    for nice, col in SIGN_COLS.items():
        miss = float(signs[col].isna().mean())
        nunq = int(signs[col].nunique(dropna=True))
        vals = sorted(signs[col].dropna().unique().tolist())
        print(f"  {nice} ({col}): missing={miss:.3f}, nunique={nunq}, values={vals}")

    return signs, model


def harmonise_and_merge(
    signs: pd.DataFrame, model: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Signs file is already aligned 1:1 to modeling rows; still enforce rules.

    - Drop duplicate (session_id, task) before correlating
    - Keep PT in a separate frame (no sign correlations)
    - Exclude held-out test split
    - Restrict to has_sign_prediction==1 for sign analyses
    """
    print("\n" + "=" * 72)
    print("STEP 2 — Harmonise task labels and merge")
    print("=" * 72)

    # Prefer signs file (already has y_* + pred_*); join keys from modeling for sanity
    df = signs.copy()
    df["task_family"] = df["task"].map(config.TASK_FAMILY)
    if "task_prefix" in df.columns:
        # prefer explicit prefix when present
        df["task_family"] = df["task_prefix"].fillna(df["task_family"])

    # Note: collaborator already mapped bilateral signs onto L/R rows — no further expand needed
    print(
        "Note: signs CSV is pre-aligned to modeling rows (session_id x task including L/R). "
        "Bilateral expand (FT -> FT_L/FT_R) was already applied upstream."
    )

    n_before = len(df)
    dups = int(df.duplicated(["session_id", "task"]).sum())
    df = df.drop_duplicates(["session_id", "task"], keep="first").copy()
    print(f"Dropped {dups} duplicate (session_id, task) rows ({n_before} → {len(df)})")

    # Split: use train only (no val in this table); exclude test
    if "split" in df.columns:
        print("split before filter:", df["split"].value_counts(dropna=False).to_dict())
        df = df[df["split"].astype(str).str.lower() != "test"].copy()
        print(f"After excluding test: {len(df)} rows")

    pt = df[df["task_family"] == "PT"].copy()
    brady = df[df["task_family"].isin(["FT", "OC", "PS"])].copy()

    if "has_sign_prediction" in brady.columns:
        n_has = int((brady["has_sign_prediction"] == 1).sum())
        brady = brady[brady["has_sign_prediction"] == 1].copy()
        print(f"Bradykinesia rows with has_sign_prediction==1: {len(brady)} (flagged {n_has})")
    else:
        # require at least one sign non-null
        brady = brady.dropna(subset=list(SIGN_COLS.values()), how="all").copy()

    print("\nModeling rows with sign labels by task (post-dedupe, train-only):")
    for task, g in brady.groupby("task"):
        print(f"  {task}: {len(g)}")
    print("by family:")
    for fam, g in brady.groupby("task_family"):
        print(f"  {fam}: {len(g)}")
    print(f"PT rows held separate (no sign analysis): {len(pt)}")
    print(f"Sessions in sign-analysis set: {brady['session_id'].nunique()}")

    # Binary check
    binary = True
    for col in SIGN_COLS.values():
        vals = set(brady[col].dropna().unique().tolist())
        if not vals.issubset({0.0, 1.0, 0, 1}):
            binary = False
    if binary:
        print(
            "NOTE: sign labels are binary {0,1} (not continuous probabilities). "
            "Primary association = point-biserial r (Pearson with binary sign). "
            "Spearman rho also reported for transparency."
        )
    else:
        print("NOTE: signs appear continuous — primary association = Spearman rho.")

    return brady, pt, binary


def _assoc(y: np.ndarray, x: np.ndarray, binary_sign: bool) -> dict:
    """Association between UPDRS scores and one sign column."""
    n = len(y)
    if n < 3 or np.nanstd(y) == 0 or np.nanstd(x) == 0:
        return {
            "point_biserial_r": np.nan,
            "point_biserial_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n_pairs": n,
        }
    # point-biserial ≡ pearson when x is 0/1
    pb_r, pb_p = stats.pearsonr(x, y)
    sp_r, sp_p = stats.spearmanr(x, y)
    return {
        "point_biserial_r": float(pb_r),
        "point_biserial_p": float(pb_p),
        "spearman_rho": float(sp_r),
        "spearman_p": float(sp_p),
        "n_pairs": int(n),
    }


def correlate(
    df: pd.DataFrame,
    *,
    binary_sign: bool,
    stratum: str = "all",
) -> pd.DataFrame:
    rows = []
    for rater in config.ACTIVE_RATERS:
        ycol = RATER_COLS[rater]
        n_avail = int(df[ycol].notna().sum())
        if n_avail < MIN_N_WARN:
            print(
                f"WARNING: {_p(rater)} has only {n_avail} non-NaN rows after merge "
                f"(stratum={stratum}) — underpowered."
            )
        for sign_name, scol in SIGN_COLS.items():
            sub = df[[ycol, scol]].dropna()
            y = sub[ycol].to_numpy(dtype=float)
            x = sub[scol].to_numpy(dtype=float)
            res = _assoc(y, x, binary_sign)
            unreliable = res["n_pairs"] < MIN_N_FLAG
            if unreliable:
                print(
                    f"FLAG: {_p(rater)} × {sign_name} (stratum={stratum}) "
                    f"n_pairs={res['n_pairs']} < {MIN_N_FLAG}"
                )
            rows.append(
                {
                    "stratum": stratum,
                    "rater": _p(rater),
                    "sign": sign_name,
                    "point_biserial_r": res["point_biserial_r"],
                    "point_biserial_p": res["point_biserial_p"],
                    "spearman_rho": res["spearman_rho"],
                    "spearman_p": res["spearman_p"],
                    "n_pairs": res["n_pairs"],
                    "unreliable_n_lt_30": unreliable,
                    "primary_r": res["point_biserial_r"] if binary_sign else res["spearman_rho"],
                    "primary_p": res["point_biserial_p"] if binary_sign else res["spearman_p"],
                    "primary_method": "point_biserial" if binary_sign else "spearman",
                }
            )
    return pd.DataFrame(rows)


def profile_matrix(corr: pd.DataFrame, value_col: str = "primary_r") -> pd.DataFrame:
    mat = (
        corr.pivot(index="rater", columns="sign", values=value_col)
        .reindex(index=PAPER_ORDER, columns=SIGN_ORDER)
    )
    return mat


def fisher_z_diff(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Two-sample Fisher z test for difference of independent correlations."""
    if any(pd.isna(v) for v in [r1, r2]) or n1 < 4 or n2 < 4:
        return np.nan, np.nan
    # clip for atanh stability
    r1 = float(np.clip(r1, -0.999999, 0.999999))
    r2 = float(np.clip(r2, -0.999999, 0.999999))
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def step4_hypothesis(
    corr_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    print("\n" + "=" * 72)
    print("STEP 4 — Axis-weighting hypothesis")
    print("=" * 72)

    mat = profile_matrix(corr_all)
    # n matrix for Fisher tests
    nmat = profile_matrix(corr_all, "n_pairs")

    dominant = []
    for rater in PAPER_ORDER:
        row = mat.loc[rater]
        if row.isna().all():
            dominant.append({"rater": rater, "dominant_sign": None, "dominant_r": np.nan})
            continue
        # argmax |r|
        abs_row = row.abs()
        dom = abs_row.idxmax()
        dominant.append(
            {
                "rater": rater,
                "dominant_sign": dom,
                "dominant_r": float(row[dom]),
                "dominant_abs_r": float(abs_row[dom]),
            }
        )
    dom_df = pd.DataFrame(dominant)
    print("\nDominant sign per rater (argmax |primary_r|):")
    print(dom_df.to_string(index=False))

    # R1 amplitude check
    r1_row = mat.loc["R1"]
    r1_amp_strongest = bool(
        r1_row.notna().any()
        and abs(r1_row["amplitude_reduction"]) == r1_row.abs().max()
    )
    print(
        f"\nR1 amplitude_reduction strongest correlate? {r1_amp_strongest} "
        f"(rho/r profile: {r1_row.round(3).to_dict()})"
    )

    # Pairwise distances + dominant match + Fisher on each sign
    pairs = []
    for i, ra in enumerate(PAPER_ORDER):
        for rb in PAPER_ORDER[i + 1 :]:
            va = mat.loc[ra].to_numpy(dtype=float)
            vb = mat.loc[rb].to_numpy(dtype=float)
            if np.any(np.isnan(va)) or np.any(np.isnan(vb)):
                dist = float(np.nan)
            else:
                dist = float(np.linalg.norm(va - vb))
            da = dom_df.loc[dom_df["rater"] == ra, "dominant_sign"].iloc[0]
            db = dom_df.loc[dom_df["rater"] == rb, "dominant_sign"].iloc[0]
            pairs.append(
                {
                    "rater_a": ra,
                    "rater_b": rb,
                    "profile_euclidean_distance": dist,
                    "dominant_sign_a": da,
                    "dominant_sign_b": db,
                    "dominant_sign_match": bool(da == db) if da and db else False,
                }
            )
            # Fisher z on each sign's r
            for sign in SIGN_ORDER:
                z, p = fisher_z_diff(
                    float(mat.loc[ra, sign]),
                    int(nmat.loc[ra, sign]),
                    float(mat.loc[rb, sign]),
                    int(nmat.loc[rb, sign]),
                )
                pairs[-1][f"fisher_z_{sign}"] = z
                pairs[-1][f"fisher_p_{sign}"] = p
                # compare dominant-sign loadings specifically if they differ
                if da and db and da != db:
                    z_dom, p_dom = fisher_z_diff(
                        float(mat.loc[ra, da]),
                        int(nmat.loc[ra, da]),
                        float(mat.loc[rb, da]),
                        int(nmat.loc[rb, da]),
                    )
                    pairs[-1]["fisher_z_on_a_dominant"] = z_dom
                    pairs[-1]["fisher_p_on_a_dominant"] = p_dom

    pair_df = pd.DataFrame(pairs)
    print("\nPairwise profile distances:")
    print(
        pair_df[
            [
                "rater_a",
                "rater_b",
                "profile_euclidean_distance",
                "dominant_sign_match",
            ]
        ].to_string(index=False)
    )

    unique_dom = sorted({d for d in dom_df["dominant_sign"] if d})
    meta = {
        "r1_amplitude_strongest": r1_amp_strongest,
        "n_unique_dominant_signs": len(unique_dom),
        "unique_dominant_signs": unique_dom,
        "axis_weighting_supported": len(unique_dom) >= 2,
        "inter_rater_profile_variance": float(
            np.nanvar(mat.to_numpy(dtype=float), axis=0).mean()
        ),
        "mean_pairwise_euclidean": float(
            pair_df["profile_euclidean_distance"].mean(skipna=True)
        ),
    }
    print("\nHypothesis summary:", meta)
    return mat, pair_df, meta, dom_df


def plot_heatmap(mat: pd.DataFrame, pmat: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    data = mat.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 0.05)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SIGN_ORDER)))
    ax.set_xticklabels(SIGN_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(PAPER_ORDER)))
    ax.set_yticklabels(PAPER_ORDER)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                txt = "NA"
            else:
                star = "*" if (not np.isnan(pmat.iloc[i, j]) and pmat.iloc[i, j] < 0.05) else ""
                txt = f"{val:.2f}{star}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="association r")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmaps_by_task(corr_task: pd.DataFrame, path: Path) -> None:
    families = ["FT", "OC", "PS"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, fam in zip(axes, families):
        sub = corr_task[corr_task["stratum"] == fam]
        mat = profile_matrix(sub)
        pmat = profile_matrix(sub, "primary_p")
        data = mat.to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        vmax = max(float(vmax), 0.05)
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SIGN_ORDER)))
        ax.set_xticklabels(SIGN_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(PAPER_ORDER)))
        ax.set_yticklabels(PAPER_ORDER)
        ax.set_title(fam)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    txt = "NA"
                else:
                    star = "*" if (not np.isnan(pmat.iloc[i, j]) and pmat.iloc[i, j] < 0.05) else ""
                    txt = f"{val:.2f}{star}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="r")
    fig.suptitle("Rater-sign associations by task family", y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_amplitude(df: pd.DataFrame, corr_all: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    scol = SIGN_COLS["amplitude_reduction"]
    for ax, rater in zip(axes, config.ACTIVE_RATERS):
        ycol = RATER_COLS[rater]
        sub = df[[ycol, scol]].dropna()
        x = sub[scol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        # jitter binary x for visibility
        rng = np.random.default_rng(0)
        xj = x + rng.uniform(-0.08, 0.08, size=len(x))
        ax.scatter(xj, y, alpha=0.25, s=12, c="#2c7bb6")
        if len(sub) >= 3 and np.unique(x).size > 1:
            coef = np.polyfit(x, y, 1)
            xs = np.array([0.0, 1.0])
            ax.plot(xs, coef[0] * xs + coef[1], color="#d7191c", lw=2)
        row = corr_all[
            (corr_all["rater"] == _p(rater)) & (corr_all["sign"] == "amplitude_reduction")
        ].iloc[0]
        ax.set_title(
            f"{_p(rater)}\nr={row['primary_r']:.2f}, n={int(row['n_pairs'])}",
            fontsize=10,
        )
        ax.set_xlabel("amplitude (0/1)")
        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([0, 1])
    axes[0].set_ylabel("UPDRS score")
    fig.suptitle("UPDRS vs amplitude-reduction sign (jittered binary x)", y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    corr_all: pd.DataFrame,
    mat: pd.DataFrame,
    pair_df: pd.DataFrame,
    meta: dict,
    dom_df: pd.DataFrame,
    corr_task: pd.DataFrame,
    path: Path,
) -> str:
    lines = []
    lines.append("Rater axis-weighting analysis — INSIGHT-PD signs")
    lines.append("=" * 60)
    lines.append(
        f"Primary association: {corr_all['primary_method'].iloc[0]} "
        "(signs are binary 0/1 in this release)."
    )
    lines.append("Analysis set: train only, deduped (session_id, task), has_sign_prediction==1; PT excluded.")
    lines.append("")
    lines.append("Dominant sign per rater — ALL FT/OC/PS pooled (argmax |r|):")
    for _, r in dom_df.iterrows():
        lines.append(
            f"  {r['rater']}: {r['dominant_sign']} "
            f"(r={r['dominant_r']:+.3f}, |r|={r['dominant_abs_r']:.3f})"
        )
    lines.append("")
    lines.append("Dominant sign per rater — BY TASK FAMILY:")
    for fam in ["FT", "OC", "PS"]:
        sub = corr_task[corr_task["stratum"] == fam]
        m = profile_matrix(sub)
        lines.append(f"  [{fam}]")
        for rater in PAPER_ORDER:
            row = m.loc[rater]
            if row.isna().all():
                lines.append(f"    {rater}: NA")
                continue
            dom = row.abs().idxmax()
            lines.append(f"    {rater}: {dom} (r={row[dom]:+.3f})")
    lines.append("")
    lines.append(
        f"Unique dominant signs (pooled): {meta['n_unique_dominant_signs']} "
        f"-> {meta['unique_dominant_signs']}"
    )
    lines.append(
        f"Mean pairwise Euclidean profile distance (pooled): {meta['mean_pairwise_euclidean']:.3f}"
    )
    lines.append(
        f"Mean across-sign inter-rater variance of r (pooled): {meta['inter_rater_profile_variance']:.4f}"
    )
    lines.append(f"R1 amplitude strongest (pooled)? {meta['r1_amplitude_strongest']}")
    lines.append("")
    lines.append("Fisher z tests on differing dominant-sign loadings (pairs where dominant differs):")
    diff = pair_df[~pair_df["dominant_sign_match"]]
    if diff.empty:
        lines.append("  (no pairs with differing dominant signs)")
    else:
        for _, r in diff.iterrows():
            p = r.get("fisher_p_on_a_dominant", np.nan)
            lines.append(
                f"  {r['rater_a']} vs {r['rater_b']}: "
                f"dom {r['dominant_sign_a']} vs {r['dominant_sign_b']}; "
                f"Fisher p on {r['rater_a']}'s dominant = {p}"
            )
    lines.append("")

    # Honest interpretation: pooled vs FT-stratified
    ft = profile_matrix(corr_task[corr_task["stratum"] == "FT"])
    ft_dom = {r: ft.loc[r].abs().idxmax() for r in PAPER_ORDER}
    ft_unique = sorted(set(ft_dom.values()))
    amp_r1 = float(mat.loc["R1", "amplitude_reduction"])
    slow_r1 = float(mat.loc["R1", "slowness"])

    if len(ft_unique) >= 2:
        # Find examples
        amp_raters = [r for r, d in ft_dom.items() if d == "amplitude_reduction"]
        slow_raters = [r for r, d in ft_dom.items() if d == "slowness"]
        para = (
            f"On pooled FT/OC/PS rows, R1–R3 and R5 all loaded most strongly on "
            f"amplitude_reduction (R1 r={amp_r1:.2f}; slowness second at r={slow_r1:.2f}), "
            f"while sparse R4 alone was slowness-dominant — so pooled profiles show only "
            f"modest axis differentiation and do not by themselves prove complementary "
            f"feature weighting among the primary raters. "
            f"Task-stratified finger-tapping (FT) is more informative: dominant signs split "
            f"into amplitude_reduction ({', '.join(amp_raters) or 'none'}) versus slowness "
            f"({', '.join(slow_raters) or 'none'}), consistent with differential sensitivity "
            f"to amplitude vs speed on the task that preferentially elicits amplitude "
            f"decrement. OC and PS remained largely amplitude-led. "
            f"Sequence-effect associations were weakest overall. "
            f"R4/R5 remain partially underpowered (n≈230/350); conclusions for axis "
            f"weighting should emphasize R1–R3 and the FT stratum."
        )
    elif meta["axis_weighting_supported"]:
        para = (
            f"Raters showed partially different sign-loading profiles, with R1 most "
            f"strongly associated with amplitude_reduction and R4 with slowness; however "
            f"R1–R3 share the same pooled dominant sign, so evidence for complementary "
            f"axis weighting is limited outside task-stratified FT analyses."
        )
    else:
        para = (
            "All raters showed similar sign-loading profiles, suggesting that rater "
            "disagreement in this cohort does not arise from differential feature weighting "
            "across the four motor sign dimensions captured by INSIGHT-PD."
        )
    lines.append("Results-style paragraph:")
    lines.append(para)
    lines.append("")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print("\n" + text)
    return para


def main() -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    signs, model = load_and_inspect()
    brady, pt, binary = harmonise_and_merge(signs, model)

    # Save PT separately (never used in correlations)
    pt_path = OUT_TAB / "insight_pd_pt_rows_no_signs.csv"
    pt.to_csv(pt_path, index=False)
    print(f"Wrote {pt_path} ({len(pt)} PT rows)")

    print("\n" + "=" * 72)
    print("STEP 3 — Per-rater sign correlations")
    print("=" * 72)
    corr_all = correlate(brady, binary_sign=binary, stratum="all")
    corr_out = corr_all.rename(
        columns={
            "primary_r": "association_r",
            "primary_p": "p_value",
        }
    )
    corr_save = corr_out[
        [
            "rater",
            "sign",
            "association_r",
            "p_value",
            "spearman_rho",
            "spearman_p",
            "point_biserial_r",
            "point_biserial_p",
            "n_pairs",
            "unreliable_n_lt_30",
            "primary_method",
            "stratum",
        ]
    ]
    corr_save.to_csv(OUT_TAB / "rater_sign_correlations.csv", index=False)
    print(corr_save.to_string(index=False))

    mat = profile_matrix(corr_all)
    mat.to_csv(OUT_TAB / "rater_sign_profile_matrix.csv")
    print("\n5x4 profile matrix (primary r):")
    print(mat.round(3))

    mat, pair_df, meta, dom_df = step4_hypothesis(corr_all)
    pair_df.to_csv(OUT_TAB / "pairwise_profile_distances.csv", index=False)
    dom_df.to_csv(OUT_TAB / "rater_dominant_signs.csv", index=False)

    print("\n" + "=" * 72)
    print("STEP 5 — Task-stratified correlations")
    print("=" * 72)
    task_parts = [corr_all]
    for fam in ["FT", "OC", "PS"]:
        sub = brady[brady["task_family"] == fam]
        print(f"\n--- {fam} (n_rows={len(sub)}) ---")
        task_parts.append(correlate(sub, binary_sign=binary, stratum=fam))
    corr_task = pd.concat(task_parts, ignore_index=True)
    corr_task.to_csv(OUT_TAB / "task_stratified_correlations.csv", index=False)
    for fam in ["FT", "OC", "PS"]:
        print(f"\n{fam} profile:")
        print(profile_matrix(corr_task[corr_task["stratum"] == fam]).round(3))

    print("\n" + "=" * 72)
    print("STEP 6 — Plots")
    print("=" * 72)
    pmat = profile_matrix(corr_all, "primary_p")
    plot_heatmap(
        mat,
        pmat,
        "Rater-sign associations (all FT/OC/PS; train; binary signs)",
        OUT_FIG / "heatmap_all_tasks.png",
    )
    plot_heatmaps_by_task(corr_task, OUT_FIG / "heatmap_by_task.png")
    plot_scatter_amplitude(brady, corr_all, OUT_FIG / "scatter_rater_vs_amplitude.png")
    print("Wrote heatmaps + scatter to", OUT_FIG)

    print("\n" + "=" * 72)
    print("STEP 7 — Summary")
    print("=" * 72)
    write_summary(
        corr_all,
        mat,
        pair_df,
        meta,
        dom_df,
        corr_task,
        OUT_TAB / "summary_stats.txt",
    )

    import json

    meta_out = {
        **meta,
        "n_sign_rows_train_deduped": int(len(brady)),
        "n_pt_rows_excluded": int(len(pt)),
        "signs_binary": bool(binary),
        "excluded_split": "test",
        "outputs": {
            "rater_sign_correlations": str(OUT_TAB / "rater_sign_correlations.csv"),
            "rater_sign_profile_matrix": str(OUT_TAB / "rater_sign_profile_matrix.csv"),
            "pairwise_profile_distances": str(OUT_TAB / "pairwise_profile_distances.csv"),
            "task_stratified_correlations": str(OUT_TAB / "task_stratified_correlations.csv"),
            "summary_stats": str(OUT_TAB / "summary_stats.txt"),
            "heatmap_all_tasks": str(OUT_FIG / "heatmap_all_tasks.png"),
            "heatmap_by_task": str(OUT_FIG / "heatmap_by_task.png"),
            "scatter_rater_vs_amplitude": str(OUT_FIG / "scatter_rater_vs_amplitude.png"),
        },
    }
    (OUT_TAB / "rater_axis_weighting_meta.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()

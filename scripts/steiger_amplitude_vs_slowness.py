"""Steiger (1980) test: within-rater amplitude vs slowness association.

For each rater, r1 = corr(UPDRS, amplitude), r2 = corr(UPDRS, slowness)
share the same UPDRS scores; signs are dependent → Steiger test of H0: r1 = r2.

Uses the same analysis set as the INSIGHT-PD axis-weighting analysis:
train-only, deduped (session_id, task), has_sign_prediction==1, PT excluded.
Signs are binary → point-biserial (= Pearson). R1–R5 only in outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

SIGNS_CSV = config.DATA_EXTERNAL / "insight_pd_signs.csv"
AMP = "pred_low_amplitude"
SLOW = "pred_slowness"
OUT = config.RESULTS_TABLES / "steiger_amplitude_vs_slowness.csv"
OUT_TXT = config.RESULTS_TABLES / "steiger_amplitude_vs_slowness_summary.txt"


def load_analysis_frame() -> pd.DataFrame:
    df = pd.read_csv(SIGNS_CSV)
    df["task_family"] = df["task"].map(config.TASK_FAMILY)
    if "task_prefix" in df.columns:
        df["task_family"] = df["task_prefix"].fillna(df["task_family"])
    df = df.drop_duplicates(["session_id", "task"], keep="first")
    df = df[df["split"].astype(str).str.lower() != "test"]
    df = df[df["task_family"].isin(["FT", "OC", "PS"])]
    if "has_sign_prediction" in df.columns:
        df = df[df["has_sign_prediction"] == 1]
    return df.copy()


def point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r (point-biserial when one variable is 0/1)."""
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    r, _ = stats.pearsonr(x, y)
    return float(r)


def steiger_t(r1: float, r2: float, r3: float, n: int) -> tuple[float, float]:
    """Steiger (1980) t for H0: r1 == r2 with dependent correlations sharing X.

    Formula as specified in the task brief.
    """
    if n < 5 or any(np.isnan(v) for v in [r1, r2, r3]):
        return float("nan"), float("nan")
    # numerical safety for atanh-adjacent edge cases in R
    r1 = float(np.clip(r1, -0.999999, 0.999999))
    r2 = float(np.clip(r2, -0.999999, 0.999999))
    r3 = float(np.clip(r3, -0.999999, 0.999999))

    R = 1.0 - r1**2 - r2**2 - r3**2 + 2.0 * r1 * r2 * r3
    # R can be slightly negative from floating error
    R = max(R, 1e-12)
    numer = (r1 - r2) * np.sqrt((n - 1) * (1.0 + r3))
    denom_inner = 2.0 * ((n - 1) / (n - 3)) * R + ((r1 + r2) / 2.0) ** 2 * (1.0 - r3) ** 3
    if denom_inner <= 0:
        return float("nan"), float("nan")
    t = numer / np.sqrt(denom_inner)
    df = n - 3
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=df))
    return float(t), float(p)


def sentence(rater: str, r1: float, r2: float, n: int, t: float, p: float) -> str:
    df = n - 3
    if np.isnan(t) or np.isnan(p):
        return f"{rater}: Steiger test not computable (n={n})."
    if p < 0.05:
        stronger = "amplitude" if r1 > r2 else "slowness"
        weaker = "slowness" if stronger == "amplitude" else "amplitude"
        rs, rw = (r1, r2) if stronger == "amplitude" else (r2, r1)
        return (
            f"{rater}'s {stronger} correlation (r={rs:.2f}) was significantly stronger "
            f"than its {weaker} correlation (r={rw:.2f}), "
            f"Steiger t({df})={t:.2f}, p={p:.4g}."
        )
    return (
        f"{rater}'s amplitude (r={r1:.2f}) and slowness (r={r2:.2f}) correlations "
        f"did not differ significantly, Steiger t({df})={t:.2f}, p={p:.4g}."
    )


def approx_gap_fisher(r1_a: float, r2_a: float, n_a: int, r1_b: float, r2_b: float, n_b: int):
    """Approximate comparison of (r_amp - r_slow) gaps via Fisher z differences.

    Not a formal Steiger-of-Steigers; gaps are differences of dependent correlations.
    """
    def gap_z(r_amp, r_slow):
        return float(np.arctanh(np.clip(r_amp, -0.999999, 0.999999)) - np.arctanh(
            np.clip(r_slow, -0.999999, 0.999999)
        ))

    z1 = gap_z(r1_a, r2_a)
    z2 = gap_z(r1_b, r2_b)
    # Approximate SE as if comparing two independent Fisher-z quantities with df~n-3
    se = np.sqrt(1.0 / (n_a - 3) + 1.0 / (n_b - 3))
    z = (z1 - z2) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return {
        "gap_a": r1_a - r2_a,
        "gap_b": r1_b - r2_b,
        "z_gap_a": z1,
        "z_gap_b": z2,
        "z_diff": float(z),
        "p_approx": float(p),
    }


def main() -> None:
    df = load_analysis_frame()
    print(f"Analysis rows (train, FT/OC/PS, signs): {len(df)}")

    rows = []
    sentences = []
    by_paper = {}

    for rater in config.ACTIVE_RATERS:
        ycol = f"y_{rater}"
        paper = config.paper_rater(rater)
        sub = df[[ycol, AMP, SLOW]].dropna()
        y = sub[ycol].to_numpy(dtype=float)
        amp = sub[AMP].to_numpy(dtype=float)
        slow = sub[SLOW].to_numpy(dtype=float)
        n = len(sub)

        r1 = point_biserial(amp, y)
        r2 = point_biserial(slow, y)
        r3 = point_biserial(amp, slow)
        t, p = steiger_t(r1, r2, r3, n)

        row = {
            "rater": paper,
            "r1_amplitude": r1,
            "r2_slowness": r2,
            "r3_amp_slow": r3,
            "n": n,
            "steiger_t": t,
            "steiger_p": p,
            "significant_p_lt_05": bool(p < 0.05) if not np.isnan(p) else False,
            "gap_amp_minus_slow": r1 - r2,
        }
        rows.append(row)
        by_paper[paper] = row
        s = sentence(paper, r1, r2, n, t, p)
        sentences.append(s)
        print(s)

    out_df = pd.DataFrame(rows)
    print("\nSteiger table:")
    print(out_df.to_string(index=False))
    out_df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

    # STEP 3 — R1 vs R3 gap (approximate)
    print("\n" + "=" * 72)
    print("STEP 3 — Approximate R1 vs R3 amplitude-slowness gap comparison")
    print("=" * 72)
    r1 = by_paper["R1"]
    r3 = by_paper["R3"]
    gap = approx_gap_fisher(
        r1["r1_amplitude"],
        r1["r2_slowness"],
        int(r1["n"]),
        r3["r1_amplitude"],
        r3["r2_slowness"],
        int(r3["n"]),
    )
    gap_line = (
        f"Approximate Fisher-z comparison of gaps "
        f"(R1 gap={gap['gap_a']:+.3f} vs R3 gap={gap['gap_b']:+.3f}): "
        f"z={gap['z_diff']:.3f}, p≈{gap['p_approx']:.4g}. "
        f"This treats dependent-correlation gaps as if independent; interpret cautiously."
    )
    print(gap_line)

    # Also R1 vs R2 for context (both amplitude-led)
    r2 = by_paper["R2"]
    gap12 = approx_gap_fisher(
        r1["r1_amplitude"],
        r1["r2_slowness"],
        int(r1["n"]),
        r2["r1_amplitude"],
        r2["r2_slowness"],
        int(r2["n"]),
    )
    gap12_line = (
        f"Same approximate test R1 vs R2 gaps "
        f"({gap12['gap_a']:+.3f} vs {gap12['gap_b']:+.3f}): "
        f"z={gap12['z_diff']:.3f}, p≈{gap12['p_approx']:.4g}."
    )
    print(gap12_line)

    # Append gap columns to a companion row file / summary
    summary = [
        "Steiger (1980) within-rater test: amplitude vs slowness association with UPDRS",
        "H0: r(UPDRS, amplitude) == r(UPDRS, slowness); dependent correlations sharing UPDRS.",
        "Analysis set: train, deduped, FT/OC/PS, has_sign_prediction==1; point-biserial.",
        "",
        out_df.to_string(index=False),
        "",
        "Plain-English:",
        *sentences,
        "",
        gap_line,
        gap12_line,
        "",
        f"R1 gap (amp-slow) = {r1['gap_amp_minus_slow']:+.3f}",
        f"R3 gap (amp-slow) = {r3['gap_amp_minus_slow']:+.3f}",
    ]
    OUT_TXT.write_text("\n".join(summary) + "\n", encoding="utf-8")

    # Extend CSV with gap meta as extra rows? Better: write gap JSON-ish columns in a small side CSV
    gap_df = pd.DataFrame(
        [
            {
                "contrast": "R1_vs_R3_gap_approx",
                "gap_R1": gap["gap_a"],
                "gap_R3": gap["gap_b"],
                "z_approx": gap["z_diff"],
                "p_approx": gap["p_approx"],
                "note": "approximate; gaps are differences of dependent correlations",
            },
            {
                "contrast": "R1_vs_R2_gap_approx",
                "gap_R1": gap12["gap_a"],
                "gap_R2": gap12["gap_b"],
                "z_approx": gap12["z_diff"],
                "p_approx": gap12["p_approx"],
                "note": "approximate; gaps are differences of dependent correlations",
            },
        ]
    )
    gap_path = config.RESULTS_TABLES / "steiger_gap_contrasts_approx.csv"
    gap_df.to_csv(gap_path, index=False)
    print(f"Wrote {gap_path}")
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()

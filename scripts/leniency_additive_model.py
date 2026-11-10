"""Definitive leniency estimate via two-way additive model (cell + rater effects).

score_{cell,rater} = mu + cell_difficulty_{cell} + rater_offset_{rater} + eps

The rater_offset is each rater's systematic tendency to score high/low AFTER
controlling for how hard the specific cells they rated were. Estimated jointly by
alternating least squares (handles unbalanced/missing design), so it is NOT
confounded by which colleagues co-rated a cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from phase1_descriptive import load_clean_data

ACTIVE_RATERS = config.ACTIVE_RATERS
RNG = np.random.default_rng(0)


def build_long_cells(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["session_id"] = df["subject_id"].astype(str) + "_" + df["visit_date"].astype(str)
    df["cell"] = df["session_id"] + "|" + df["task"].astype(str)
    df = df.dropna(subset=["score"])
    df = df[df["rater"].isin(ACTIVE_RATERS)]
    return df[["cell", "rater", "score"]]


def fit_additive(df: pd.DataFrame, n_iter: int = 200, tol: float = 1e-9) -> pd.Series:
    """Alternating least squares for additive cell + rater effects. Returns rater offsets (sum-to-zero)."""
    cells = df["cell"].to_numpy()
    raters = df["rater"].to_numpy()
    y = df["score"].to_numpy(dtype=float)

    cell_ids, cell_idx = np.unique(cells, return_inverse=True)
    rater_ids, rater_idx = np.unique(raters, return_inverse=True)

    cell_eff = np.zeros(len(cell_ids))
    rater_eff = np.zeros(len(rater_ids))
    mu = y.mean()

    for _ in range(n_iter):
        prev = rater_eff.copy()
        # cell effects given rater effects
        resid = y - mu - rater_eff[rater_idx]
        cell_eff = np.bincount(cell_idx, weights=resid) / np.bincount(cell_idx)
        # rater effects given cell effects
        resid = y - mu - cell_eff[cell_idx]
        rater_eff = np.bincount(rater_idx, weights=resid) / np.bincount(rater_idx)
        # identifiability: center cell effects, fold into mu
        rater_eff = rater_eff - rater_eff.mean() + 0.0
        if np.max(np.abs(rater_eff - prev)) < tol:
            break

    return pd.Series(rater_eff, index=rater_ids).reindex(ACTIVE_RATERS)


def bootstrap_additive(df: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    cells = df["cell"].unique()
    base = fit_additive(df)
    boots = np.zeros((n_boot, len(ACTIVE_RATERS)))
    for b in range(n_boot):
        samp_cells = RNG.choice(cells, len(cells), replace=True)
        # rebuild by concatenating sampled cells (cluster bootstrap on cells)
        sub = df[df["cell"].isin(samp_cells)]  # note: duplicates collapsed; approx cluster boot
        boots[b] = fit_additive(sub).to_numpy()
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return pd.DataFrame({"rater": ACTIVE_RATERS, "offset": base.to_numpy(), "ci_lo": lo, "ci_hi": hi})


def downsample_stability(df: pd.DataFrame, n_draws: int = 300) -> None:
    print("=" * 70)
    print("DOWNSAMPLING: refit additive model with high-count raters cut to R4/R5 n")
    print("=" * 70)
    counts = df.groupby("rater")["cell"].count().reindex(ACTIVE_RATERS)
    print("Ratings per rater:", counts.to_dict())
    targets = {"R4": int(counts["R4"]), "R5": int(counts["R5"])}
    base = fit_additive(df)
    for label, tn in targets.items():
        print(f"\n-- cap high-count raters at n={tn} ({label}'s count) --")
        for r in ["R2", "R3", "R1"]:
            offs = []
            r_rows = df[df["rater"] == r]
            other = df[df["rater"] != r]
            for _ in range(n_draws):
                keep = r_rows.sample(tn, random_state=None)
                sub = pd.concat([other, keep])
                offs.append(fit_additive(sub)[r])
            offs = np.array(offs)
            print(
                f"   {r}: full offset={base[r]:+.3f} | downsampled={offs.mean():+.3f} "
                f"[{np.percentile(offs,2.5):+.3f}, {np.percentile(offs,97.5):+.3f}]"
            )
        print(f"   (R4 offset={base['R4']:+.3f}, R5 offset={base['R5']:+.3f})")


def pairwise_matrix(long_df: pd.DataFrame) -> None:
    print("=" * 70)
    print("FULL PAIRWISE same-cell mean differences (row - col), positive = row higher")
    print("=" * 70)
    df = long_df.copy()
    df["session_id"] = df["subject_id"].astype(str) + "_" + df["visit_date"].astype(str)
    wide = df.pivot_table(index=["session_id", "task"], columns="rater", values="score", aggfunc="mean")
    mat = pd.DataFrame(index=ACTIVE_RATERS, columns=ACTIVE_RATERS, dtype=float)
    for a in ACTIVE_RATERS:
        for b in ACTIVE_RATERS:
            if a == b:
                mat.loc[a, b] = 0.0
                continue
            valid = wide[a].notna() & wide[b].notna()
            mat.loc[a, b] = (wide.loc[valid, a] - wide.loc[valid, b]).mean()
    print(mat.round(3).to_string())
    print("\nMean of each row (avg leniency vs all others, same-cell):")
    print((mat.mean(axis=1)).round(3).sort_values().to_string())


def main() -> None:
    long_df = load_clean_data()
    df = build_long_cells(long_df)
    print(f"Observations: {len(df)}, cells: {df['cell'].nunique()}, raters: {df['rater'].nunique()}")

    raw = df.groupby("rater")["score"].mean().reindex(ACTIVE_RATERS)
    print("\nRAW mean score per rater (confounded by case-mix):")
    print(raw.round(3).sort_values().to_string())

    print("\n" + "=" * 70)
    print("ADDITIVE MODEL rater offsets (cell-difficulty controlled)")
    print("Positive = lenient (scores high for the difficulty); negative = strict")
    print("=" * 70)
    boot = bootstrap_additive(df)
    boot = boot.sort_values("offset")
    for _, r in boot.iterrows():
        tag = "lenient" if r.offset > 0.03 else ("strict" if r.offset < -0.03 else "neutral")
        print(f"  {r.rater}: {r.offset:+.3f}  95%=[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  ({tag})")

    corr = np.corrcoef(raw.reindex(boot.rater).to_numpy(), boot["offset"].to_numpy())[0, 1]
    print(f"\nCorrelation(raw mean, additive offset) = {corr:.3f}")

    print()
    pairwise_matrix(long_df)
    print()
    downsample_stability(df)


if __name__ == "__main__":
    main()

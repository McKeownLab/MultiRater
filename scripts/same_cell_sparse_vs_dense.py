"""Same-cell paired tests: sparse raters (R4, R5) vs dense raters (R1–R3) on shared observations.

Backs the Methods robustness check (2) in the paper brief. Uses the 1,559-row modeling table
(one row per session×task, ≥3 raters). Paired Wilcoxon signed-rank on per-cell score differences.

Usage: python scripts/same_cell_sparse_vs_dense.py
Writes results/tables/same_cell_sparse_vs_dense.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import config  # noqa: E402
from utils.ordinal_task_data import build_task_level_table  # noqa: E402

PAPER = {r: config.paper_rater(r) for r in config.ACTIVE_RATERS}
BY_PAPER = {v: k for k, v in PAPER.items()}


def main() -> None:
    t = build_task_level_table(min_raters=3).drop_duplicates(["session_id", "task"]).reset_index(drop=True)
    assert len(t) == 1559, len(t)
    rows = []
    for sparse in ("R4", "R5"):
        for dense in ("R1", "R2", "R3"):
            a, b = t[f"y_{BY_PAPER[sparse]}"], t[f"y_{BY_PAPER[dense]}"]
            m = a.notna() & b.notna()
            d = (a[m] - b[m]).astype(float)
            p = wilcoxon(d, zero_method="wilcox").pvalue if (d != 0).any() else 1.0
            rows.append({"sparse": sparse, "dense": dense, "n_cells": int(m.sum()),
                         "mean_diff": float(d.mean()), "wilcoxon_p": float(p)})
    out = pd.DataFrame(rows)
    path = ROOT / "results" / "tables" / "same_cell_sparse_vs_dense.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

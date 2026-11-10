"""Small derived numbers quoted in the paper brief that are not direct model outputs.

  - Bonferroni alpha for the sign-correlation tests (5 raters x 4 INSIGHT-PD signs = 20 tests)
  - rating workload saved by dropping the two sparse raters (R4, R5)
  - mean and range of the Nash - consensus kappa difference across the ten three-rater panels
  - disagreement bands on the modeling rows, including the half-point labels from dual attempts

Usage: python scripts/derived_numbers.py
Needs: results/tables/dedup_multiseed/triples_dedup_multiseed.csv (scripts/dedup_triples_multiseed.py)
Writes: results/tables/derived_numbers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from utils.ordinal_task_data import build_task_level_table  # noqa: E402

N_SIGNS = 4  # amplitude reduction, sequence effect, slowness, halt/hesitation
ALPHA = 0.05


def main() -> None:
    t = build_task_level_table(min_raters=3).drop_duplicates(["session_id", "task"]).reset_index(drop=True)
    paper = {r: config.paper_rater(r) for r in config.ACTIVE_RATERS}

    n_tests = len(config.ACTIVE_RATERS) * N_SIGNS
    bonferroni = {"n_raters": len(config.ACTIVE_RATERS), "n_signs": N_SIGNS, "n_tests": n_tests,
                  "alpha_family": ALPHA, "alpha_per_test": ALPHA / n_tests}

    counts = {paper[r]: int(t[f"m_{r}"].sum()) for r in config.ACTIVE_RATERS}
    total = sum(counts.values())
    saved = (counts["R4"] + counts["R5"]) / total
    workload = {"n_rows": len(t), "ratings_per_rater": counts,
                "coverage": {k: v / len(t) for k, v in counts.items()},
                "total_ratings": total, "fraction_saved_without_R4_R5": saved,
                "fraction_kept": 1 - saved}

    tri = pd.read_csv(ROOT / "results" / "tables" / "dedup_multiseed" / "triples_dedup_multiseed.csv")
    assert tri["raters"].nunique() == 10, "triples run incomplete"
    tp = tri.pivot_table(index="raters", columns="method", values="kappa_fold_mean", aggfunc="mean")
    d = (tp["nash"] - tp["consensus"]).sort_values()
    triples = {"n_panels": int(len(d)), "n_seeds": int(tri["seed"].nunique()),
               "delta_kappa_mean": float(d.mean()), "delta_kappa_min": float(d.min()),
               "delta_kappa_max": float(d.max()), "n_nash_higher": int((d > 0).sum()),
               "per_panel": {k: float(v) for k, v in d.items()}}

    r = t["score_range"].to_numpy()
    labels = t[[f"y_{x}" for x in config.ACTIVE_RATERS]].stack()
    bands = {"unanimous_r0": int((r == 0).sum()), "adjacent_0_lt_r_lt_2": int(((r > 0) & (r < 2)).sum()),
             "disputed_r_ge_2": int((r >= 2).sum()), "extreme_r_ge_3": int((r >= 3).sum()),
             "fraction_r_ge_2": float(np.mean(r >= 2)),
             "half_point_ranges": {str(k): int(v) for k, v in pd.Series(r[r % 1 != 0]).value_counts().items()},
             "half_point_labels": int((labels % 1 != 0).sum()), "n_labels": int(len(labels))}

    out = {"bonferroni": bonferroni, "workload": workload, "three_rater_panels": triples, "bands": bands}
    path = ROOT / "results" / "tables" / "derived_numbers.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Bonferroni alpha {bonferroni['alpha_per_test']:.4f} ({n_tests} tests)")
    print(f"workload saved {saved:.1%} (kept {1 - saved:.1%}); coverage R4 {workload['coverage']['R4']:.0%}, "
          f"R5 {workload['coverage']['R5']:.0%}")
    print(f"three-rater panels: mean {d.mean():+.3f}, range {d.min():+.3f} to {d.max():+.3f}, "
          f"Nash higher in {(d > 0).sum()}/{len(d)}")
    print(f"bands {bands}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

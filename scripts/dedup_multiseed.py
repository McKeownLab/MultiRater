"""Multi-seed run of the comparators and the R1-R3 panel-size test.

Usage: python dedup_multiseed.py <repo_root> <out_dir> <seed> [<seed> ...]
Writes per-seed OOF CSVs to out_dir. Nothing is written inside the repo.
"""
from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
SEEDS = [int(s) for s in sys.argv[3:]]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import utils.nash_training as nt  # noqa: E402
import comparators as p1  # noqa: E402
import panel_size as cr  # noqa: E402
from utils.ordinal_task_data import build_task_level_table  # noqa: E402
from utils.phase2_data import load_feature_tables  # noqa: E402

ORIG = {n: getattr(nt, n) for n in ("train_ordinal_supervised", "train_ordinal_uniform", "train_ordinal_nash")}


def pin_seed(seed: int) -> None:
    for mod in (p1, cr):
        for name, fn in ORIG.items():
            setattr(mod, name, functools.partial(fn, seed=seed))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = build_task_level_table(min_raters=3)
    base = base.drop_duplicates(["session_id", "task"]).reset_index(drop=True)
    print(f"modelling rows {len(base)}", flush=True)

    grid = load_feature_tables()["full_grid"].drop_duplicates("session_id").reset_index(drop=True)
    r123 = ["R1", "R2", "R3"]
    sub = cr.build_subset_table(grid, r123)
    print(f"R1-R3 rows {len(sub)}", flush=True)
    fold_map = cr.fixed_fold_map(base)

    for seed in SEEDS:
        pin_seed(seed)
        t0 = time.time()
        oof, folds, _ = p1.run_oof(base)
        oof["seed"] = seed
        oof.to_csv(OUT / f"p1_dedup_seed{seed}.csv", index=False)
        print(f"seed {seed} P1 done {time.time() - t0:.0f}s "
              + str(folds.groupby('method').mean_rater_kappa.mean().round(3).to_dict()), flush=True)
        parts = []
        for method in ("consensus", "multitask", "nash"):
            f, o, _ = cr.run_cv_condition(sub, fold_map, r123, method, collect_allfold_preds=False)
            parts.append(o)
            print(f"  seed {seed} R1-R3 {method}: {f.mean_rater_kappa.mean():.3f}", flush=True)
        s = pd.concat(parts, ignore_index=True)
        s["seed"] = seed
        s.to_csv(OUT / f"r123_dedup_seed{seed}.csv", index=False)
        print(f"seed {seed} all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

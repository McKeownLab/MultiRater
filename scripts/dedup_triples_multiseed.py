"""All ten three-rater panels, consensus vs Nash, several seeds.

Each subset is the rows scored by all three raters, with leave-one-task-out features
from those raters only. Folds follow the session map in scripts/panel_size.py.

Usage: python scripts/dedup_triples_multiseed.py <out_dir> <seed> [<seed> ...]
"""
from __future__ import annotations

import functools
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import utils.nash_training as nt  # noqa: E402
import panel_size as cr  # noqa: E402
from utils.ordinal_task_data import build_task_level_table  # noqa: E402
from utils.phase2_data import load_feature_tables  # noqa: E402

OUT = Path(sys.argv[1]).resolve()
SEEDS = [int(s) for s in sys.argv[2:]]
ORIG = {n: getattr(nt, n) for n in ("train_ordinal_supervised", "train_ordinal_uniform", "train_ordinal_nash")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = build_task_level_table(min_raters=3).drop_duplicates(["session_id", "task"]).reset_index(drop=True)
    grid = load_feature_tables()["full_grid"].drop_duplicates("session_id").reset_index(drop=True)
    fold_map = cr.fixed_fold_map(base)
    rows = []
    for seed in SEEDS:
        for name, fn in ORIG.items():
            setattr(cr, name, functools.partial(fn, seed=seed))
        for triple in itertools.combinations(cr.ACTIVE_RATERS, 3):
            raters = list(triple)
            sub = cr.build_subset_table(grid, raters)
            t0 = time.time()
            for method in ("consensus", "nash"):
                folds, oof, _ = cr.run_cv_condition(sub, fold_map, raters, method, collect_allfold_preds=False)
                rows.append({
                    "seed": seed, "raters": "+".join(cr.PAPER[r] for r in raters), "method": method,
                    "n_rows": len(sub), "kappa_fold_mean": float(folds["mean_rater_kappa"].mean()),
                })
            print(f"seed {seed} {rows[-1]['raters']} n={len(sub)} "
                  f"cons {rows[-2]['kappa_fold_mean']:.3f} nash {rows[-1]['kappa_fold_mean']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "triples_dedup_multiseed.csv", index=False)
    print("TRIPLES DONE", flush=True)


if __name__ == "__main__":
    main()

"""Where does the Nash gain come from? Disagreement strata and per-rater kappa.

Reads the de-duplicated five-seed out-of-fold predictions (results/tables/dedup_multiseed/p1_dedup_seed*.csv)
and reports, for each method, mean per-rater quadratic kappa within inter-rater-range strata and kappa
against each rater separately. Values are fold means averaged over seeds; CIs are 400 paired subject
resamples (kappa averaged over seeds inside each resample). No models are trained.

Usage: python scripts/dedup_strata_per_rater.py
Writes results/tables/dedup_multiseed/strata_per_rater.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "tables" / "dedup_multiseed"
sys.path.insert(0, str(ROOT / "figures"))
from fig2_results import qwk  # noqa: E402  (validated against sklearn)

RATERS = {"R1": "R1", "R2": "R2", "R3": "R3", "R4": "R4", "R5": "R5"}
METHODS = {"consensus_mean": "Consensus", "reliability_weighted": "Reliability",
           "multitask_uniform": "Multitask", "nash": "Nash"}
STRATA = {"range 0": lambda r: r == 0, "range 1–1.5": lambda r: (r > 0) & (r < 2), "range ≥2": lambda r: r >= 2}
KEY = ["session_id", "task"]
N_BOOT, SEED = 400, 42


def load():
    out = {}
    for f in sorted(glob.glob(str(RUN / "p1_dedup_seed*.csv"))):
        d = pd.read_csv(f)
        for m, g in d.groupby("method"):
            out.setdefault(m, []).append(g.sort_values(KEY).reset_index(drop=True))
    return out


def kappa(frame_pred, y, m, rows, raters):
    vals = []
    for r in raters:
        idx = rows[m[r][rows]]
        if idx.size >= 2:
            vals.append(qwk(y[r][idx], frame_pred[idx]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    data = load()
    base = data["nash"][0]
    y = {r: np.rint(np.nan_to_num(base[f"y_{r}"].to_numpy())).astype(int) for r in RATERS}
    m = {r: base[f"m_{r}"].to_numpy().astype(bool) for r in RATERS}
    fold = base["fold"].to_numpy()
    rng_vals = base["score_range"].to_numpy()
    subj = base["subject_id"].to_numpy()
    preds = {k: [np.rint(f["y_pred"].to_numpy()).astype(int) for f in v] for k, v in data.items()}
    for v in data.values():
        for f in v:
            assert (f[KEY].values == base[KEY].values).all()

    def fold_mean(pred, mask, raters):
        return float(np.nanmean([kappa(pred, y, m, np.flatnonzero((fold == k) & mask), raters)
                                 for k in np.unique(fold)]))

    def seed_mean(meth, mask, raters):
        return float(np.mean([fold_mean(p, mask, raters) for p in preds[meth]]))

    uniq = np.unique(subj)
    rows_of = {s: np.flatnonzero(subj == s) for s in uniq}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([rows_of[s] for s in rng.choice(uniq, uniq.size, replace=True)]) for _ in range(N_BOOT)]

    def boot_delta(meth, mask, raters):
        out = np.empty(N_BOOT)
        for b, rows in enumerate(draws):
            rows = rows[mask[rows]]
            k = lambda mm: np.mean([kappa(p, y, m, rows, raters) for p in preds[mm]])  # noqa: E731
            out[b] = k(meth) - k("consensus_mean")
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

    all_raters = list(RATERS)
    res = {"strata": {}, "per_rater": {}, "n_seeds": len(preds["nash"]), "n_boot": N_BOOT}
    for sname, fn in STRATA.items():
        mask = fn(rng_vals)
        row = {"n_rows": int(mask.sum())}
        for meth, label in METHODS.items():
            row[label] = {"kappa": seed_mean(meth, mask, all_raters)}
            if meth != "consensus_mean":
                row[label]["delta_vs_consensus"] = row[label]["kappa"] - row["Consensus"]["kappa"]
                row[label]["delta_ci95"] = boot_delta(meth, mask, all_raters)
        res["strata"][sname] = row
        print(sname, {k: (round(v["kappa"], 3) if isinstance(v, dict) else v) for k, v in row.items()},
              "Nash Δ CI", [round(x, 3) for x in row["Nash"]["delta_ci95"]])
    everything = np.ones(len(base), bool)
    for code, label in RATERS.items():
        row = {"n_rows": int(m[code].sum())}
        for meth, mlabel in METHODS.items():
            row[mlabel] = {"kappa": seed_mean(meth, everything, [code])}
        row["Nash"]["delta_vs_consensus"] = row["Nash"]["kappa"] - row["Consensus"]["kappa"]
        row["Nash"]["delta_ci95"] = boot_delta("nash", everything, [code])
        res["per_rater"][label] = row
        print(label, row["n_rows"], {k: round(v["kappa"], 3) for k, v in row.items() if isinstance(v, dict)},
              "Nash Δ CI", [round(x, 3) for x in row["Nash"]["delta_ci95"]])
    (RUN / "strata_per_rater.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

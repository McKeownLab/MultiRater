"""Empirical cost-reduction test: fewer raters + multitask/Nash vs 5-rater consensus.

Tests whether kappa(Nash/multitask, k<5) can match kappa(consensus, 5).
Fixed GroupKFold subject splits (same protocol as P1). R1–R5 only in outputs.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from models.ordinal_model import coral_predict
from utils.settings import DISAGREEMENT_WEIGHT_SCALE, LR, N_EPOCHS, NASH_KWARGS
from utils.nash_training import (
    ACTIVE_RATERS,
    train_ordinal_nash,
    train_ordinal_supervised,
    train_ordinal_uniform,
)
from utils.ordinal_metrics import ordinal_metrics, quadratic_kappa
from utils.ordinal_task_data import build_task_level_table
from utils.phase2_data import load_feature_tables

N_FOLDS = 5
N_BOOT = 400
N_RANDOM_DRAWS = 20
RNG = np.random.default_rng(42)
OUT_TAB = config.RESULTS_TABLES
OUT_FIG = config.RESULTS_FIGURES
TASK_ORDER = list(config.TASK_SHORT.values())
P = config.paper_rater
MIN_ROWS = 300

# Paper labels for reporting
PAPER = {r: P(r) for r in ACTIVE_RATERS}


def _predict(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return coral_predict(model(torch.tensor(x, dtype=torch.float32))).numpy().astype(float)


def mean_kappa_for_raters(
    df: pd.DataFrame, pred: np.ndarray, raters: list[str]
) -> float:
    vals = []
    for r in raters:
        m = df[f"m_{r}"].to_numpy(dtype=bool)
        if int(m.sum()) < 2:
            continue
        vals.append(quadratic_kappa(df[f"y_{r}"].to_numpy()[m], pred[m]))
    return float(np.mean(vals)) if vals else float("nan")


def fixed_fold_map(base_df: pd.DataFrame) -> dict[str, int]:
    """Session → fold id from GroupKFold on the full modelling table (P1 protocol)."""
    session_meta = base_df.drop_duplicates("session_id")[["session_id", "subject_id"]]
    session_ids = session_meta["session_id"].to_numpy()
    groups = session_meta["subject_id"].to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    fmap = {}
    for fold, (_, te) in enumerate(gkf.split(session_ids, groups=groups)):
        for s in session_ids[te]:
            fmap[str(s)] = fold
    return fmap


def build_subset_table(full_grid: pd.DataFrame, raters: list[str]) -> pd.DataFrame:
    """Rows where ALL listed raters scored the target task; LOTO from those raters only."""
    rows = []
    for _, row in full_grid.iterrows():
        for task in TASK_ORDER:
            scores = [row.get(f"{task}_{r}", np.nan) for r in raters]
            if any(pd.isna(s) for s in scores):
                continue
            scores_f = [float(s) for s in scores]
            rec = {
                "session_id": row["session_id"],
                "subject_id": int(row["subject_id"]),
                "booth_id": int(row["booth_id"]) if pd.notna(row.get("booth_id")) else -1,
                "visit_date": str(row.get("visit_date", "")),
                "task": task,
                "y_consensus": float(np.mean(scores_f)),
                "n_raters": len(raters),
                "score_range": float(np.max(scores_f) - np.min(scores_f)),
                "score_std": float(np.std(scores_f, ddof=0)),
            }
            for r in ACTIVE_RATERS:
                if r in raters:
                    rec[f"y_{r}"] = float(row[f"{task}_{r}"])
                    rec[f"m_{r}"] = 1.0
                else:
                    rec[f"y_{r}"] = np.nan
                    rec[f"m_{r}"] = 0.0
            ok = True
            for i, other in enumerate(t for t in TASK_ORDER if t != task):
                oth = [row.get(f"{other}_{r}", np.nan) for r in raters]
                oth = [float(x) for x in oth if pd.notna(x)]
                if not oth:
                    ok = False
                    break
                rec[f"x_{i}"] = float(np.mean(oth))
            if not ok:
                continue
            rows.append(rec)
    return pd.DataFrame(rows)


def sanity_subset(name: str, base: pd.DataFrame, sub: pd.DataFrame, raters: list[str]) -> bool:
    print(f"\n--- Sanity: {name} | raters={[PAPER[r] for r in raters]} ---")
    print(f"  n_rows={len(sub)} (base={len(base)})")
    if len(sub) == 0:
        print("  FLAG: empty")
        return False
    dropped = base.merge(
        sub[["session_id", "task"]], on=["session_id", "task"], how="left", indicator=True
    )
    # rough: compare severity on overlapping definition
    if "y_consensus" in base.columns and len(sub):
        print(
            f"  retained y_consensus mean={sub['y_consensus'].mean():.3f} "
            f"range_mean={sub['score_range'].mean():.3f}"
        )
        # rows in base with all raters present vs not — for subset builds, compare to all-base
        print(
            f"  base y_consensus mean={base['y_consensus'].mean():.3f} "
            f"range_mean={base['score_range'].mean():.3f}"
        )
    ok = len(sub) >= MIN_ROWS
    if not ok:
        print(f"  FLAG UNDERPOWERED: n_rows={len(sub)} < {MIN_ROWS}")
    return ok


def run_cv_condition(
    df: pd.DataFrame,
    fold_map: dict[str, int],
    raters: list[str],
    method: str,
    *,
    collect_allfold_preds: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Train/eval with fixed session→fold map. Returns folds, oof, mean_pred_var."""
    df = df.copy()
    df["fold"] = df["session_id"].astype(str).map(fold_map)
    if df["fold"].isna().any():
        # sessions not in base map (shouldn't happen if subset of same sessions)
        df = df.dropna(subset=["fold"])
        df["fold"] = df["fold"].astype(int)

    x_cols = [f"x_{i}" for i in range(len(TASK_ORDER) - 1)]
    fold_rows = []
    oof_parts = []
    # matrix of predictions from each fold-model on all rows
    all_preds = np.full((len(df), N_FOLDS), np.nan) if collect_allfold_preds else None
    df = df.reset_index(drop=True)

    for fold in range(N_FOLDS):
        train_df = df[df["fold"] != fold]
        test_df = df[df["fold"] == fold]
        if len(train_df) < 20 or len(test_df) < 5:
            continue
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(train_df[x_cols].to_numpy(dtype=float))
        x_te = scaler.transform(test_df[x_cols].to_numpy(dtype=float))
        x_all = scaler.transform(df[x_cols].to_numpy(dtype=float))
        y_cons_tr = train_df["y_consensus"].to_numpy(dtype=float)
        y_cons_te = test_df["y_consensus"].to_numpy(dtype=float)
        row_w = 1.0 + DISAGREEMENT_WEIGHT_SCALE * train_df["score_range"].to_numpy(dtype=float)

        y_by = {r: train_df[f"y_{r}"].to_numpy(dtype=float) for r in ACTIVE_RATERS}
        y_by["consensus"] = y_cons_tr
        mask_tr = {r: train_df[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS}
        # zero masks for unused raters
        for r in ACTIVE_RATERS:
            if r not in raters:
                mask_tr[r] = np.zeros(len(train_df), dtype=bool)

        if method == "consensus":
            model, _, _ = train_ordinal_supervised(
                x_tr, y_cons_tr, x_te, y_cons_te, n_epochs=N_EPOCHS, lr=LR, row_weights=row_w
            )
        elif method == "multitask":
            model, _, _ = train_ordinal_uniform(
                x_tr,
                y_by,
                x_te,
                y_cons_te,
                n_epochs=N_EPOCHS,
                lr=LR,
                mask_train=mask_tr,
                row_weights=row_w,
            )
        elif method == "nash":
            model, _, _, _, _ = train_ordinal_nash(
                x_tr,
                y_by,
                x_te,
                y_cons_te,
                n_epochs=N_EPOCHS,
                lr=LR,
                shapley_weights=None,
                mask_train=mask_tr,
                row_weights=row_w,
                **NASH_KWARGS,
            )
        elif method == "single":
            # train on the single rater's labels
            r0 = raters[0]
            y_tr = train_df[f"y_{r0}"].to_numpy(dtype=float)
            model, _, _ = train_ordinal_supervised(
                x_tr, y_tr, x_te, y_cons_te, n_epochs=N_EPOCHS, lr=LR, row_weights=row_w
            )
        else:
            raise ValueError(method)

        pred = _predict(model, x_te)
        kappa = mean_kappa_for_raters(test_df, pred, raters)
        mae = float(np.mean(np.abs(pred - y_cons_te)))
        fold_rows.append(
            {
                "fold": fold,
                "method": method,
                "mean_rater_kappa": kappa,
                "test_mae": mae,
                "n_test": len(test_df),
            }
        )
        part = test_df[
            ["session_id", "subject_id", "task", "score_range", "y_consensus", "fold"]
            + [f"y_{r}" for r in ACTIVE_RATERS]
            + [f"m_{r}" for r in ACTIVE_RATERS]
        ].copy()
        part["y_pred"] = pred
        part["method"] = method
        oof_parts.append(part)

        if collect_allfold_preds:
            all_preds[:, fold] = _predict(model, x_all)

    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    if collect_allfold_preds and all_preds is not None:
        # variance across fold-models per row, then mean
        row_var = np.nanvar(all_preds, axis=1)
        mean_var = float(np.nanmean(row_var))
    else:
        mean_var = float("nan")
    return folds, oof, mean_var


def bootstrap_kappa(oof: pd.DataFrame, raters: list[str]) -> tuple[float, float, float]:
    subjects = oof["subject_id"].drop_duplicates().to_numpy()
    by = {s: g for s, g in oof.groupby("subject_id")}
    stats = []
    for _ in range(N_BOOT):
        samp = RNG.choice(subjects, size=len(subjects), replace=True)
        parts = [by[s] for s in samp if s in by]
        if not parts:
            continue
        sub = pd.concat(parts, ignore_index=True)
        k = mean_kappa_for_raters(sub, sub["y_pred"].to_numpy(), raters)
        if np.isfinite(k):
            stats.append(k)
    arr = np.asarray(stats, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def summarize_condition(
    label: str,
    n_raters: int,
    raters: list[str],
    method: str,
    folds: pd.DataFrame,
    oof: pd.DataFrame,
    mean_var: float,
    powered: bool,
) -> dict:
    mu = float(folds["mean_rater_kappa"].mean()) if len(folds) else float("nan")
    mae = float(folds["test_mae"].mean()) if len(folds) else float("nan")
    if len(oof):
        _, lo, hi = bootstrap_kappa(oof, raters)
    else:
        lo = hi = float("nan")
    return {
        "condition": label,
        "method": method,
        "n_raters": n_raters,
        "raters": "+".join(PAPER[r] for r in raters),
        "n_rows": int(len(oof)) if len(oof) else 0,
        "kappa": mu,
        "kappa_ci_lo": lo,
        "kappa_ci_hi": hi,
        "mae": mae,
        "mean_pred_variance": mean_var,
        "powered": powered,
        "row_universe_note": (
            "Kappa values from different row subsets reflect different "
            "patient-task combinations and should not be directly compared without adjustment"
        ),
    }


def main() -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 0 — Load base data + fixed splits")
    print("=" * 72)
    base = build_task_level_table(min_raters=3)
    print(f"Base modeling rows (min 3 raters): {len(base)}")
    fold_map = fixed_fold_map(base)
    print(f"Fixed folds: {len(fold_map)} sessions → {N_FOLDS} folds")
    # verify fold counts
    fc = pd.Series(fold_map).value_counts().sort_index()
    print("sessions per fold:", fc.to_dict())

    tables = load_feature_tables()
    full_grid = tables["full_grid"]

    comparison_rows = []
    variance_rows = []
    oof_store = {}

    # ----- Condition 1: 5-rater panel (modelling table, all ACTIVE_RATERS) -----
    print("\n" + "=" * 72)
    print("CONDITION 1 — All 5 raters (min≥3 present; P1-style)")
    print("=" * 72)
    # Use existing LOTO from base table (all-rater consensus LOTO) for fidelity to P1
    c1 = base.copy()
    for r in ACTIVE_RATERS:
        c1[f"m_{r}"] = c1[f"m_{r}"].astype(bool)
    sanity_subset("5-rater base", base, c1, ACTIVE_RATERS)
    for method in ["consensus", "multitask", "nash"]:
        print(f"  running {method}...")
        folds, oof, mvar = run_cv_condition(c1, fold_map, ACTIVE_RATERS, method)
        row = summarize_condition(
            "5 raters (min≥3)", 5, ACTIVE_RATERS, method, folds, oof, mvar, True
        )
        comparison_rows.append(row)
        variance_rows.append(
            {
                "condition": row["condition"],
                "method": method,
                "n_raters": 5,
                "mean_pred_variance": mvar,
                "n_rows": row["n_rows"],
            }
        )
        oof_store[("5", method)] = oof
        print(f"    κ={row['kappa']:.3f} [{row['kappa_ci_lo']:.3f},{row['kappa_ci_hi']:.3f}] var={mvar:.4f}")

    # ----- Condition 2: R1+R2+R3 -----
    print("\n" + "=" * 72)
    print("CONDITION 2 — R1+R2+R3 (all three present)")
    print("=" * 72)
    r123 = ACTIVE_RATERS[:3]  # R1, R2, R3 = R1,R2,R3
    c2 = build_subset_table(full_grid, r123)
    powered2 = sanity_subset("R1+R2+R3", base, c2, r123)
    for method in ["consensus", "multitask", "nash"]:
        print(f"  running {method}...")
        folds, oof, mvar = run_cv_condition(c2, fold_map, r123, method)
        row = summarize_condition(
            "3 raters (R1+R2+R3)", 3, r123, method, folds, oof, mvar, powered2
        )
        comparison_rows.append(row)
        variance_rows.append(
            {
                "condition": row["condition"],
                "method": method,
                "n_raters": 3,
                "mean_pred_variance": mvar,
                "n_rows": row["n_rows"],
            }
        )
        oof_store[("3_R123", method)] = oof
        print(f"    κ={row['kappa']:.3f} [{row['kappa_ci_lo']:.3f},{row['kappa_ci_hi']:.3f}]")

    # ----- Condition 4: R1+R3 -----
    print("\n" + "=" * 72)
    print("CONDITION 4 — R1+R3 (divergent profiles)")
    print("=" * 72)
    r13 = [ACTIVE_RATERS[0], ACTIVE_RATERS[2]]
    c4 = build_subset_table(full_grid, r13)
    powered4 = sanity_subset("R1+R3", base, c4, r13)
    for method in ["consensus", "multitask", "nash"]:
        print(f"  running {method}...")
        folds, oof, mvar = run_cv_condition(c4, fold_map, r13, method)
        row = summarize_condition(
            "2 raters (R1+R3)", 2, r13, method, folds, oof, mvar, powered4
        )
        comparison_rows.append(row)
        variance_rows.append(
            {
                "condition": row["condition"],
                "method": method,
                "n_raters": 2,
                "mean_pred_variance": mvar,
                "n_rows": row["n_rows"],
            }
        )
        oof_store[("2_R13", method)] = oof
        print(f"    κ={row['kappa']:.3f}")

    # Q5 proxy: R1+R2
    print("\nCONDITION 4b — R1+R2 (more similar profiles; Q5)")
    r12 = ACTIVE_RATERS[:2]
    c4b = build_subset_table(full_grid, r12)
    powered4b = sanity_subset("R1+R2", base, c4b, r12)
    for method in ["multitask"]:
        folds, oof, mvar = run_cv_condition(c4b, fold_map, r12, method)
        row = summarize_condition(
            "2 raters (R1+R2)", 2, r12, method, folds, oof, mvar, powered4b
        )
        comparison_rows.append(row)
        oof_store[("2_R12", method)] = oof
        print(f"    κ={row['kappa']:.3f}")

    # ----- Condition 5: R1 only -----
    print("\n" + "=" * 72)
    print("CONDITION 5 — R1 single-rater")
    print("=" * 72)
    r1 = [ACTIVE_RATERS[0]]
    c5 = build_subset_table(full_grid, r1)
    powered5 = sanity_subset("R1 only", base, c5, r1)
    folds, oof, mvar = run_cv_condition(c5, fold_map, r1, "single")
    row = summarize_condition("1 rater (R1 only)", 1, r1, "single", folds, oof, mvar, powered5)
    comparison_rows.append(row)
    variance_rows.append(
        {
            "condition": row["condition"],
            "method": "single",
            "n_raters": 1,
            "mean_pred_variance": mvar,
            "n_rows": row["n_rows"],
        }
    )
    print(f"    κ={row['kappa']:.3f}")

    # ----- Condition 3: random 3-rater draws -----
    print("\n" + "=" * 72)
    print(f"CONDITION 3 — {N_RANDOM_DRAWS} random 3-rater subsets")
    print("=" * 72)
    # all combinations of 3 from 5 = 10; with 20 draws sample with replacement of combos
    combos = list(itertools.combinations(ACTIVE_RATERS, 3))
    stability_rows = []
    for draw in range(N_RANDOM_DRAWS):
        raters = list(combos[int(RNG.integers(0, len(combos)))])
        # ensure variety: also shuffle order
        sub = build_subset_table(full_grid, raters)
        if len(sub) < MIN_ROWS:
            print(f"  draw {draw}: {[PAPER[r] for r in raters]} n={len(sub)} SKIP underpowered")
            continue
        for method in ["consensus", "multitask", "nash"]:
            folds, oof, _ = run_cv_condition(
                sub, fold_map, raters, method, collect_allfold_preds=False
            )
            mu = float(folds["mean_rater_kappa"].mean())
            mae = float(folds["test_mae"].mean())
            stability_rows.append(
                {
                    "draw": draw,
                    "raters": "+".join(PAPER[r] for r in raters),
                    "method": method,
                    "n_rows": len(oof),
                    "kappa": mu,
                    "mae": mae,
                }
            )
            print(f"  draw {draw} {[PAPER[r] for r in raters]} {method}: κ={mu:.3f} n={len(oof)}")

    stab = pd.DataFrame(stability_rows)
    stab.to_csv(OUT_TAB / "random_subset_stability.csv", index=False)

    # stability plot
    if len(stab):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        methods = ["consensus", "multitask", "nash"]
        data = [stab.loc[stab["method"] == m, "kappa"].to_numpy() for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True)
        for i, m in enumerate(methods):
            y = stab.loc[stab["method"] == m, "kappa"].to_numpy()
            x = np.random.default_rng(0).normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.55, s=22, zorder=3)
        ax.set_ylabel("Mean per-rater quadratic κ")
        ax.set_title("Random 3-rater subsets (20 draws)")
        k5c = next(
            r["kappa"]
            for r in comparison_rows
            if r["condition"].startswith("5") and r["method"] == "consensus"
        )
        ax.axhline(k5c, color="gray", ls="--", lw=1, label=f"5-rater consensus κ={k5c:.3f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "stability_plot.png", dpi=150)
        plt.close(fig)

    # ----- Step 6 stratified 3-rater -----
    print("\n" + "=" * 72)
    print("STEP 6 — Disagreement-stratified (R1+R2+R3)")
    print("=" * 72)
    strat_rows = []
    for method in ["consensus", "multitask", "nash"]:
        oof = oof_store.get(("3_R123", method))
        if oof is None or not len(oof):
            continue
        for label, mask in [
            ("unanimous_3", oof["score_range"] == 0),
            ("disputed_3", oof["score_range"] >= 1),
            ("high_3", oof["score_range"] >= 2),
            ("all", np.ones(len(oof), dtype=bool)),
        ]:
            sub = oof[mask]
            if len(sub) < 10:
                k = float("nan")
            else:
                k = mean_kappa_for_raters(sub, sub["y_pred"].to_numpy(), r123)
            strat_rows.append(
                {
                    "condition": "3 raters (R1+R2+R3)",
                    "method": method,
                    "stratum": label,
                    "n": int(len(sub)),
                    "kappa": k,
                }
            )
            print(f"  {method} {label}: κ={k:.3f} n={len(sub)}")
    # overlap comparison with 5-rater consensus on same session-task keys
    oof5c = oof_store.get(("5", "consensus"))
    oof3 = oof_store.get(("3_R123", "multitask"))
    if oof5c is not None and oof3 is not None:
        keys = set(zip(oof3["session_id"], oof3["task"]))
        ov = oof5c[oof5c.apply(lambda r: (r["session_id"], r["task"]) in keys, axis=1)]
        if len(ov):
            # recompute range among R1-3 on overlap for strata? use 5-rater score_range for info
            for label, mask in [
                ("all_overlap", np.ones(len(ov), dtype=bool)),
            ]:
                sub = ov[mask]
                k = mean_kappa_for_raters(sub, sub["y_pred"].to_numpy(), ACTIVE_RATERS)
                strat_rows.append(
                    {
                        "condition": "5-rater consensus on R123-row overlap",
                        "method": "consensus",
                        "stratum": label,
                        "n": int(len(sub)),
                        "kappa": k,
                    }
                )

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(OUT_TAB / "stratified_3rater_kappa.csv", index=False)

    # ----- Save main tables -----
    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(OUT_TAB / "cost_reduction_comparison_table.csv", index=False)
    var_df = pd.DataFrame(variance_rows)
    var_df.to_csv(OUT_TAB / "prediction_variance_table.csv", index=False)

    print("\n" + "=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)
    print(
        comp[
            ["condition", "method", "n_raters", "n_rows", "kappa", "kappa_ci_lo", "kappa_ci_hi", "mae"]
        ].to_string(index=False)
    )

    # ----- Q1–Q5 -----
    def get(cond_prefix, method):
        hits = [r for r in comparison_rows if r["condition"].startswith(cond_prefix) and r["method"] == method]
        return hits[0] if hits else None

    k5c = get("5 raters", "consensus")
    k5m = get("5 raters", "multitask")
    k5n = get("5 raters", "nash")
    k3c = get("3 raters", "consensus")
    k3m = get("3 raters", "multitask")
    k3n = get("3 raters", "nash")
    k2_13m = get("2 raters (R1+R3)", "multitask")
    k2_12m = get("2 raters (R1+R2)", "multitask")
    k1 = get("1 rater", "single")

    lines = []
    lines.append("COST REDUCTION SUMMARY (Q1–Q5)")
    lines.append("=" * 60)
    lines.append(
        "ROW UNIVERSE NOTE: Kappa values from different row subsets reflect different "
        "patient-task combinations and should not be directly compared without adjustment."
    )
    lines.append("")

    # Q1
    lines.append("Q1: Does 3-rater Nash beat 5-rater consensus?")
    if k3n and k5c:
        gap = k3n["kappa"] - k5c["kappa"]
        yes = k3n["kappa"] >= k5c["kappa"]
        lines.append(
            f"{'YES' if yes else 'NO'}. κ(Nash, R1+R2+R3)={k3n['kappa']:.3f} "
            f"[{k3n['kappa_ci_lo']:.3f},{k3n['kappa_ci_hi']:.3f}] on n={k3n['n_rows']} vs "
            f"κ(consensus,5)={k5c['kappa']:.3f} [{k5c['kappa_ci_lo']:.3f},{k5c['kappa_ci_hi']:.3f}] "
            f"on n={k5c['n_rows']}; gap={gap:+.3f}. "
            + (
                "Supports the cost-reduction pitch for Nash specifically, with the row-universe caveat."
                if yes
                else "Does not support '3-rater Nash ≥ 5-rater averaging' as a headline claim."
            )
        )
    lines.append("")

    # Q2
    lines.append("Q2: Does 3-rater multitask beat 5-rater consensus?")
    if k3m and k5c:
        gap = k3m["kappa"] - k5c["kappa"]
        yes = k3m["kappa"] >= k5c["kappa"]
        lines.append(
            f"{'YES' if yes else 'NO'}. κ(multitask, R1+R2+R3)={k3m['kappa']:.3f} "
            f"[{k3m['kappa_ci_lo']:.3f},{k3m['kappa_ci_hi']:.3f}] vs "
            f"κ(consensus,5)={k5c['kappa']:.3f}; gap={gap:+.3f}. "
            + (
                "Supports a cost argument for multi-rater training (not Nash-specific)."
                if yes
                else "3-rater multitask does not match 5-rater consensus in this run."
            )
        )
    lines.append("")

    # Q3
    lines.append("Q3: At what rater count does multitask/Nash first match 5-rater consensus?")
    thresh_m = thresh_n = None
    for nlab, km, kn in [
        (1, k1, k1),
        (2, k2_13m, get("2 raters (R1+R3)", "nash")),
        (3, k3m, k3n),
        (5, k5m, k5n),
    ]:
        if km and k5c and km["kappa"] >= k5c["kappa"] and thresh_m is None:
            thresh_m = nlab
        if kn and k5c and kn["kappa"] >= k5c["kappa"] and thresh_n is None:
            thresh_n = nlab
    lines.append(
        f"Using κ(consensus,5)={k5c['kappa']:.3f} as threshold: "
        f"multitask first matches at n≈{thresh_m}; Nash first matches at n≈{thresh_n}. "
        f"Single-rater R1 κ={k1['kappa']:.3f}; 2-rater R1+R3 multitask κ={k2_13m['kappa']:.3f}; "
        f"3-rater multitask κ={k3m['kappa']:.3f}."
        if k5c and k1 and k2_13m and k3m
        else "Insufficient rows to determine threshold."
    )
    lines.append("")

    # Q4
    lines.append("Q4: Kappa loss per rater removed (5→3 multitask)?")
    if k5m and k3m:
        loss_per = (k5m["kappa"] - k3m["kappa"]) / 2.0
        lines.append(
            f"(κ_5_multitask − κ_3_multitask)/2 = ({k5m['kappa']:.3f} − {k3m['kappa']:.3f})/2 "
            f"= {loss_per:+.3f} κ per rater removed. "
            "Interpret cautiously: row universes differ (full modelling table vs R1+R2+R3-complete rows)."
        )
    lines.append("")

    # Q5
    lines.append("Q5: Do divergent R1+R3 outperform more-similar R1+R2 (multitask)?")
    if k2_13m and k2_12m:
        gap = k2_13m["kappa"] - k2_12m["kappa"]
        lines.append(
            f"κ(multitask, R1+R3)={k2_13m['kappa']:.3f} vs κ(multitask, R1+R2)={k2_12m['kappa']:.3f} "
            f"(gap={gap:+.3f}). "
            + (
                "Divergent profiles look better — tentative support for complementary expertise."
                if gap > 0.01
                else (
                    "Similar or worse — no clear gain from pairing the most divergent Steiger profiles "
                    "over R1+R2 in this proxy."
                    if gap < -0.01
                    else "Essentially similar — no strong complementary-pair advantage in this proxy."
                )
            )
        )
    lines.append("")

    # Outcome A/B/C
    lines.append("OVERALL OUTCOME")
    var5c = next((v["mean_pred_variance"] for v in variance_rows if v["condition"].startswith("5") and v["method"] == "consensus"), np.nan)
    var3n = next((v["mean_pred_variance"] for v in variance_rows if "R1+R2+R3" in v["condition"] and v["method"] == "nash"), np.nan)
    var3m = next((v["mean_pred_variance"] for v in variance_rows if "R1+R2+R3" in v["condition"] and v["method"] == "multitask"), np.nan)
    lines.append(f"Prediction variance: consensus_5={var5c:.4f}, Nash_3={var3n:.4f}, multitask_3={var3m:.4f}")

    if k3n and k5c and k3n["kappa"] >= k5c["kappa"] and var3n <= var5c:
        outcome = "A"
        pitch = (
            "OUTCOME A: 3-rater Nash matches/beats 5-rater consensus on κ and has "
            "comparable-or-lower prediction variance. Headline cost pitch is supportable "
            "with row-universe caveats."
        )
    elif k3m and k5c and k3m["kappa"] >= k5c["kappa"]:
        outcome = "B"
        pitch = (
            "OUTCOME B: 3-rater multitask matches/beats 5-rater consensus; Nash does not "
            "uniquely add over multitask. Cost argument should emphasize retaining individual "
            "labels, not Nash-specific gains."
        )
    else:
        outcome = "C"
        pitch = (
            "OUTCOME C: Even 3-rater multitask/Nash do not clearly match 5-rater consensus "
            "under fair caveats. Do not claim 40% cost reduction as a headline; all five "
            "raters appear to contribute measurable value in this cohort."
        )
    # refine if Nash beats consensus but multitask also does
    if outcome == "A" and k3m and k3n and abs(k3n["kappa"] - k3m["kappa"]) < 0.02:
        outcome = "B"
        pitch = (
            "OUTCOME B (refined): 3-rater multi-rater training matches 5-rater consensus, "
            "but Nash ≈ multitask. Prefer the cost pitch without overclaiming Nash."
        )
    lines.append(pitch)
    lines.append(f"(Outcome code: {outcome})")

    # Stability summary
    if len(stab):
        lines.append("")
        lines.append("RANDOM 3-RATER STABILITY")
        for m in ["consensus", "multitask", "nash"]:
            s = stab[stab["method"] == m]["kappa"]
            lines.append(f"  {m}: mean κ={s.mean():.3f} ± {s.std():.3f} (n_draws={len(s)})")
        # pairwise within draw
        piv = stab.pivot_table(index="draw", columns="method", values="kappa")
        if {"nash", "consensus"}.issubset(piv.columns):
            lines.append(
                f"  P(Nash > consensus)={float((piv['nash'] > piv['consensus']).mean()):.2f}"
            )
        if {"multitask", "consensus"}.issubset(piv.columns):
            lines.append(
                f"  P(multitask > consensus)={float((piv['multitask'] > piv['consensus']).mean()):.2f}"
            )

    summary = "\n".join(lines) + "\n"
    (OUT_TAB / "cost_reduction_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    meta = {
        "n_folds": N_FOLDS,
        "n_boot": N_BOOT,
        "n_random_draws": N_RANDOM_DRAWS,
        "outcome": outcome,
        "base_n_rows": int(len(base)),
    }
    (OUT_TAB / "cost_reduction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote cost_reduction_comparison_table.csv and cost_reduction_summary.txt")


if __name__ == "__main__":
    main()

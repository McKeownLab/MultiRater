"""P1: same-split comparators, strata, CIs, referral, ECE.

Four methods, identical GroupKFold (subject) splits and CORAL architecture:

  1. consensus_mean       — train on mean of available raters
  2. reliability_weighted — train on LORO-κ weighted mean (weights from TRAIN only)
  3. multitask_uniform    — individual rater losses, equal gradient mix
  4. nash                 — same individual losses, Nash gradient mix (no Shapley prior)

Reliability recipe (not circular vs the training target):
  For rater r on training rows, quadratic κ vs the leave-one-rater-out mean of
  *other* raters present on that row. Clip at 0.01, normalize. Per-row target
  re-normalizes over raters who scored that row.

Outputs under results/tables and results/figures with prefix comparators_*.
Paper labels R1–R5 only in tables/figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.settings import (
    DISAGREEMENT_WEIGHT_SCALE,
    LR,
    MIN_RATERS,
    N_EPOCHS,
    NASH_KWARGS,
)
from utils.uncertainty import predict_uncertainty
from utils.nash_training import (
    ACTIVE_RATERS,
    train_ordinal_nash,
    train_ordinal_supervised,
    train_ordinal_uniform,
)
from utils.ordinal_metrics import ordinal_metrics, per_rater_kappa_table, quadratic_kappa
from utils.ordinal_task_data import build_task_level_table, split_arrays

N_FOLDS = 5
N_BOOT = 500
RNG = np.random.default_rng(42)

METHODS = (
    "consensus_mean",
    "reliability_weighted",
    "multitask_uniform",
    "nash",
)
METHOD_LABEL = {
    "consensus_mean": "Consensus mean",
    "reliability_weighted": "Reliability-weighted",
    "multitask_uniform": "Multitask (equal-weight)",
    "nash": "Nash",
}

STRATA = {
    "unanimous": lambda r: r == 0,
    "adjacent": lambda r: (r > 0) & (r < 2),  # 0 < range < 2 (one cell has 1.5)
    "range_ge_2": lambda r: r >= 2,
    "all": lambda r: np.ones_like(r, dtype=bool),
}


def _paper(r: str) -> str:
    return config.paper_rater(r)


def loro_reliability_weights(train_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Train-only weights: κ(rater vs LORO mean of other raters)."""
    y = np.column_stack(
        [np.nan_to_num(train_df[f"y_{r}"].to_numpy(dtype=float), nan=0.0) for r in ACTIVE_RATERS]
    )
    m = np.column_stack([train_df[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS])
    kappas: dict[str, float] = {}
    for i, rater in enumerate(ACTIVE_RATERS):
        m_oth = m.copy()
        m_oth[:, i] = False
        n_oth = m_oth.sum(axis=1)
        loro = np.divide(
            np.where(m_oth, y, 0.0).sum(axis=1),
            np.maximum(n_oth, 1),
        )
        use = m[:, i] & (n_oth >= 1)
        if int(use.sum()) < 20:
            kappas[rater] = 0.0
        else:
            kappas[rater] = quadratic_kappa(y[use, i], loro[use])
    raw = np.array([kappas[r] for r in ACTIVE_RATERS], dtype=float)
    w = np.maximum(raw, 0.01)
    w = w / w.sum()
    return w, kappas


def reliability_weighted_target(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    y = np.column_stack(
        [np.nan_to_num(df[f"y_{r}"].to_numpy(dtype=float), nan=0.0) for r in ACTIVE_RATERS]
    )
    m = np.column_stack([df[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS])
    w_row = np.where(m, weights.reshape(1, -1), 0.0)
    denom = np.maximum(w_row.sum(axis=1), 1e-8)
    return (y * w_row).sum(axis=1) / denom


def mean_rater_kappa(
    y_by_rater: dict[str, np.ndarray],
    pred: np.ndarray,
    mask_by_rater: dict[str, np.ndarray],
    row_mask: np.ndarray | None = None,
) -> float:
    vals = []
    for rater in ACTIVE_RATERS:
        m = mask_by_rater[rater].astype(bool)
        if row_mask is not None:
            m = m & row_mask
        if m.sum() < 2:
            continue
        vals.append(quadratic_kappa(y_by_rater[rater][m], pred[m]))
    return float(np.mean(vals)) if vals else float("nan")


def severity_band(y: np.ndarray) -> np.ndarray:
    g = np.clip(np.round(y), 0, 4).astype(int)
    out = np.array(["grade_0"] * len(g), dtype=object)
    out[g == 1] = "grade_1"
    out[g == 2] = "grade_2"
    out[g == 3] = "grade_3"
    out[g == 4] = "grade_4"
    return out


def train_one(method: str, arrays: dict, y_rel_train: np.ndarray):
    train_kw = dict(
        mask_train=arrays.get("mask_train"),
        row_weights=arrays.get("row_weights"),
    )
    if method == "consensus_mean":
        model, _, _ = train_ordinal_supervised(
            arrays["x_train"],
            arrays["y_consensus_train"],
            arrays["x_test"],
            arrays["y_consensus_test"],
            n_epochs=N_EPOCHS,
            lr=LR,
            row_weights=arrays.get("row_weights"),
        )
        return model
    if method == "reliability_weighted":
        model, _, _ = train_ordinal_supervised(
            arrays["x_train"],
            y_rel_train,
            arrays["x_test"],
            arrays["y_consensus_test"],
            n_epochs=N_EPOCHS,
            lr=LR,
            row_weights=arrays.get("row_weights"),
        )
        return model
    if method == "multitask_uniform":
        model, _, _ = train_ordinal_uniform(
            arrays["x_train"],
            arrays["y_train_by_rater"],
            arrays["x_test"],
            arrays["y_consensus_test"],
            n_epochs=N_EPOCHS,
            lr=LR,
            **train_kw,
        )
        return model
    if method == "nash":
        model, _, _, _, _ = train_ordinal_nash(
            arrays["x_train"],
            arrays["y_train_by_rater"],
            arrays["x_test"],
            arrays["y_consensus_test"],
            n_epochs=N_EPOCHS,
            lr=LR,
            shapley_weights=None,
            **NASH_KWARGS,
            **train_kw,
        )
        return model
    raise ValueError(method)


def run_oof(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_meta = df.drop_duplicates("session_id")[["session_id", "subject_id"]]
    groups = session_meta["subject_id"].values
    session_ids = session_meta["session_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    weight_rows: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(session_ids, groups=groups)):
        train_sessions = set(session_ids[tr_idx])
        test_sessions = set(session_ids[te_idx])
        train_df = df[df["session_id"].isin(train_sessions)].copy()
        test_df = df[df["session_id"].isin(test_sessions)].copy().reset_index(drop=True)
        rel_w, rel_k = loro_reliability_weights(train_df)
        y_rel_train = reliability_weighted_target(train_df, rel_w)
        for rater, w, k in zip(ACTIVE_RATERS, rel_w, [rel_k[r] for r in ACTIVE_RATERS]):
            weight_rows.append(
                {
                    "fold": fold,
                    "rater": _paper(rater),
                    "loro_kappa": k,
                    "reliability_weight": float(w),
                }
            )

        arrays = split_arrays(
            df,
            train_sessions,
            test_sessions,
            feature_mode="ratings",
            disagreement_weight_scale=DISAGREEMENT_WEIGHT_SCALE,
        )
        assert len(test_df) == arrays["n_test"]

        for method in METHODS:
            print(f"fold {fold}  {method}  n_test={len(test_df)}", flush=True)
            model = train_one(method, arrays, y_rel_train)
            unc = predict_uncertainty(model, arrays["x_test"])
            pred = unc["pred"]
            kappas = per_rater_kappa_table(
                arrays["y_test_by_rater"], pred, mask_by_rater=arrays.get("mask_test")
            )
            mean_k = float(np.nanmean(list(kappas.values())))
            mae = ordinal_metrics(arrays["y_consensus_test"], pred)["mae"]
            fold_row = {
                "fold": fold,
                "method": method,
                "mean_rater_kappa": mean_k,
                "test_mae": float(mae),
                "n_test": int(len(test_df)),
            }
            for rater in ACTIVE_RATERS:
                fold_row[f"kappa_{_paper(rater)}"] = float(kappas[rater])
            rng = arrays["score_range_test"]
            for sname, fn in STRATA.items():
                mask = fn(rng) if sname != "all" else np.ones(len(rng), dtype=bool)
                fold_row[f"kappa_{sname}"] = mean_rater_kappa(
                    arrays["y_test_by_rater"], pred, arrays["mask_test"], mask
                )
                if mask.sum():
                    fold_row[f"mae_{sname}"] = float(
                        ordinal_metrics(arrays["y_consensus_test"][mask], pred[mask])["mae"]
                    )
                else:
                    fold_row[f"mae_{sname}"] = float("nan")
            fold_rows.append(fold_row)

            out = test_df[
                ["session_id", "subject_id", "task", "score_range", "y_consensus"]
                + [f"y_{r}" for r in ACTIVE_RATERS]
                + [f"m_{r}" for r in ACTIVE_RATERS]
            ].copy()
            out["task_family"] = out["task"].astype(str).str[:2]
            out["severity_band"] = severity_band(out["y_consensus"].to_numpy())
            out["fold"] = fold
            out["method"] = method
            out["y_pred"] = pred
            out["entropy"] = unc["entropy"]
            out["confidence"] = unc["confidence"]
            out["abs_err"] = np.abs(pred - out["y_consensus"].to_numpy())
            oof_parts.append(out)

    oof = pd.concat(oof_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    weights = pd.DataFrame(weight_rows)
    return oof, folds, weights


def _resample_by_subject(g: pd.DataFrame, uniq: np.ndarray, idx_map: dict) -> pd.DataFrame:
    samp = RNG.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([idx_map[s] for s in samp])
    return g.iloc[idx]


def bootstrap_method_kappa(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        g = oof[oof["method"] == method].reset_index(drop=True)
        subj = g["subject_id"].to_numpy()
        uniq = np.unique(subj)
        idx_map = {s: np.flatnonzero(subj == s) for s in uniq}
        stats = []
        for _ in range(N_BOOT):
            sub = _resample_by_subject(g, uniq, idx_map)
            y_by = {r: sub[f"y_{r}"].to_numpy() for r in ACTIVE_RATERS}
            masks = {r: sub[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS}
            stats.append(mean_rater_kappa(y_by, sub["y_pred"].to_numpy(), masks))
        arr = np.array(stats, dtype=float)
        rows.append(
            {
                "method": method,
                "kappa_boot_mean": float(np.nanmean(arr)),
                "kappa_ci95_lo": float(np.nanpercentile(arr, 2.5)),
                "kappa_ci95_hi": float(np.nanpercentile(arr, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_deltas(oof: pd.DataFrame, base: str = "multitask_uniform") -> pd.DataFrame:
    rows = []
    packed = {}
    for method in METHODS:
        g = oof[oof["method"] == method].reset_index(drop=True)
        subj = g["subject_id"].to_numpy()
        uniq = np.unique(subj)
        packed[method] = (g, uniq, {s: np.flatnonzero(subj == s) for s in uniq})
    uniq = packed[base][1]
    for method in METHODS:
        if method == base:
            continue
        stats = []
        g_m, _, map_m = packed[method]
        g_b, _, map_b = packed[base]
        for _ in range(N_BOOT):
            samp = RNG.choice(uniq, size=len(uniq), replace=True)
            sub_m = g_m.iloc[np.concatenate([map_m[s] for s in samp])]
            sub_b = g_b.iloc[np.concatenate([map_b[s] for s in samp])]
            def _k(sub):
                y_by = {r: sub[f"y_{r}"].to_numpy() for r in ACTIVE_RATERS}
                masks = {r: sub[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS}
                return mean_rater_kappa(y_by, sub["y_pred"].to_numpy(), masks)
            stats.append(_k(sub_m) - _k(sub_b))
        arr = np.array(stats, dtype=float)
        rows.append(
            {
                "method": method,
                "minus": base,
                "delta_kappa_boot_mean": float(np.nanmean(arr)),
                "delta_ci95_lo": float(np.nanpercentile(arr, 2.5)),
                "delta_ci95_hi": float(np.nanpercentile(arr, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def paired_fold_deltas(folds: pd.DataFrame) -> pd.DataFrame:
    piv = folds.pivot(index="fold", columns="method", values="mean_rater_kappa")
    rows = []
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1 :]:
            d = piv[a] - piv[b]
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "mean_delta_kappa": float(d.mean()),
                    "std_delta_kappa": float(d.std(ddof=1)),
                    "n_folds_a_gt_b": int((d > 0).sum()),
                    "n_folds": int(len(d)),
                }
            )
    return pd.DataFrame(rows)


def stratified_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in oof.groupby("method"):
        rng = g["score_range"].to_numpy()
        y_by = {r: g[f"y_{r}"].to_numpy() for r in ACTIVE_RATERS}
        masks = {r: g[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS}
        pred = g["y_pred"].to_numpy()
        ycons = g["y_consensus"].to_numpy()
        for sname, fn in STRATA.items():
            mask = fn(rng) if sname != "all" else np.ones(len(g), dtype=bool)
            n = int(mask.sum())
            rows.append(
                {
                    "method": method,
                    "stratum": sname,
                    "n": n,
                    "mean_rater_kappa": mean_rater_kappa(y_by, pred, masks, mask),
                    "mae": float(ordinal_metrics(ycons[mask], pred[mask])["mae"]) if n else np.nan,
                }
            )
        for fam, sub in g.groupby("task_family"):
            yb = {r: sub[f"y_{r}"].to_numpy() for r in ACTIVE_RATERS}
            mb = {r: sub[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS}
            rows.append(
                {
                    "method": method,
                    "stratum": f"task_{fam}",
                    "n": int(len(sub)),
                    "mean_rater_kappa": mean_rater_kappa(yb, sub["y_pred"].to_numpy(), mb),
                    "mae": float(ordinal_metrics(sub["y_consensus"], sub["y_pred"])["mae"]),
                }
            )
        for band, sub in g.groupby("severity_band"):
            yb = {r: sub[f"y_{r}"].to_numpy() for r in ACTIVE_RATERS}
            mb = {r: sub[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS}
            rows.append(
                {
                    "method": method,
                    "stratum": f"sev_{band}",
                    "n": int(len(sub)),
                    "mean_rater_kappa": mean_rater_kappa(yb, sub["y_pred"].to_numpy(), mb),
                    "mae": float(ordinal_metrics(sub["y_consensus"], sub["y_pred"])["mae"]),
                }
            )
    return pd.DataFrame(rows)


def referral_curve(oof: pd.DataFrame) -> pd.DataFrame:
    qs = np.round(np.linspace(0.0, 0.50, 11), 2)
    rows = []
    for method, g in oof.groupby("method"):
        g = g.sort_values("entropy", ascending=False).reset_index(drop=True)
        n = len(g)
        for q in qs:
            n_defer = int(round(q * n))
            kept = g.iloc[n_defer:]
            y_by = {r: kept[f"y_{r}"].to_numpy() for r in ACTIVE_RATERS}
            masks = {r: kept[f"m_{r}"].to_numpy(dtype=bool) for r in ACTIVE_RATERS}
            rows.append(
                {
                    "method": method,
                    "defer_frac": float(q),
                    "n_retained": int(len(kept)),
                    "mean_rater_kappa": mean_rater_kappa(y_by, kept["y_pred"].to_numpy(), masks),
                    "mae": float(ordinal_metrics(kept["y_consensus"], kept["y_pred"])["mae"]),
                }
            )
    return pd.DataFrame(rows)


def ece_table(oof: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for method, g in oof.groupby("method"):
        conf = g["confidence"].to_numpy()
        correct = (np.round(g["y_pred"]) == np.round(g["y_consensus"])).to_numpy().astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_rows = []
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            if b == n_bins - 1:
                sel = (conf >= lo) & (conf <= hi)
            else:
                sel = (conf >= lo) & (conf < hi)
            n = int(sel.sum())
            if n == 0:
                acc = np.nan
                mc = np.nan
            else:
                acc = float(correct[sel].mean())
                mc = float(conf[sel].mean())
                ece += (n / len(g)) * abs(acc - mc)
            bin_rows.append(
                {
                    "method": method,
                    "bin": b,
                    "conf_lo": lo,
                    "conf_hi": hi,
                    "n": n,
                    "mean_confidence": mc,
                    "accuracy": acc,
                }
            )
        for br in bin_rows:
            br["ece"] = ece
        rows.extend(bin_rows)
    return pd.DataFrame(rows)


def plot_methods(folds: pd.DataFrame, boot: pd.DataFrame, path: Path) -> None:
    agg = folds.groupby("method")["mean_rater_kappa"].agg(["mean", "std"]).reindex(METHODS)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(METHODS))
    ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"].fillna(0),
        color=["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad"],
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel("Mean quadratic κ vs each rater")
    ax.set_title("P1 — same-split comparators (5-fold subject CV)")
    for i, m in enumerate(METHODS):
        br = boot[boot["method"] == m].iloc[0]
        ax.text(
            i,
            agg["mean"].iloc[i] + 0.01,
            f"[{br['kappa_ci95_lo']:.3f}, {br['kappa_ci95_hi']:.3f}]",
            ha="center",
            fontsize=8,
        )
    ax.set_ylim(0, max(0.75, float(agg["mean"].max()) + 0.08))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_strata(strata: pd.DataFrame, path: Path) -> None:
    keep = ["unanimous", "adjacent", "range_ge_2"]
    sub = strata[strata["stratum"].isin(keep)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.18
    x = np.arange(len(keep))
    colors = ["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad"]
    for i, method in enumerate(METHODS):
        vals = [
            float(sub[(sub.method == method) & (sub.stratum == s)]["mean_rater_kappa"].iloc[0])
            for s in keep
        ]
        ax.bar(x + i * width, vals, width, label=METHOD_LABEL[method], color=colors[i])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(["Unanimous (range=0)", "Range = 1", "Range ≥ 2"])
    ax.set_ylabel("Mean rater quadratic κ")
    ax.legend(fontsize=8)
    ax.set_title("P1 — performance by disagreement stratum (OOF pooled)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_referral(ref: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    colors = {
        "consensus_mean": "#7f8c8d",
        "reliability_weighted": "#2980b9",
        "multitask_uniform": "#27ae60",
        "nash": "#8e44ad",
    }
    for method in METHODS:
        sub = ref[ref["method"] == method]
        ax.plot(
            sub["defer_frac"],
            sub["mean_rater_kappa"],
            "o-",
            color=colors[method],
            label=METHOD_LABEL[method],
        )
    ax.set_xlabel("Fraction deferred (highest entropy first)")
    ax.set_ylabel("Mean rater quadratic κ on retained rows")
    ax.set_title("P1 — selective prediction / referral (uncertainty signal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ece(ece: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, method in zip(axes, METHODS):
        sub = ece[ece["method"] == method].dropna(subset=["accuracy", "mean_confidence"])
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(sub["mean_confidence"], sub["accuracy"], "o-", color="#8e44ad")
        ece_val = float(sub["ece"].iloc[0]) if len(sub) else np.nan
        ax.set_title(f"{METHOD_LABEL[method]}\nECE={ece_val:.3f}")
        ax.set_xlabel("Mean confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy (pred = round consensus)")
    fig.suptitle("P1 — confidence reliability (not entropy calibration)", y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    folds: pd.DataFrame,
    boot: pd.DataFrame,
    deltas: pd.DataFrame,
    fold_d: pd.DataFrame,
    ece: pd.DataFrame,
    weights: pd.DataFrame,
    path: Path,
) -> None:
    agg = folds.groupby("method")[["mean_rater_kappa", "test_mae"]].mean().reindex(METHODS)
    lines = [
        "# lab P1 comparator results",
        "",
        "Same 5-fold GroupKFold by subject; CORAL; ratings-only LOTO features.",
        "Nash uses individual rater losses with Nash mixing (no Shapley prior).",
        "Multitask uses the same losses with equal-weight gradient mixing.",
        "Reliability weights = train-only quadratic κ vs leave-one-rater-out mean of other raters.",
        "Rater labels in tables are R1–R5.",
        "",
        "## Mean rater quadratic κ (fold mean) and subject-bootstrap 95% CI",
        "",
    ]
    for m in METHODS:
        b = boot[boot["method"] == m].iloc[0]
        lines.append(
            f"- **{METHOD_LABEL[m]}**: {agg.loc[m, 'mean_rater_kappa']:.3f} "
            f"(MAE {agg.loc[m, 'test_mae']:.3f}); "
            f"bootstrap CI [{b['kappa_ci95_lo']:.3f}, {b['kappa_ci95_hi']:.3f}]"
        )
    lines += ["", "## Δκ vs multitask (subject bootstrap)", ""]
    for _, r in deltas.iterrows():
        lines.append(
            f"- {METHOD_LABEL[r['method']]} − multitask: "
            f"{r['delta_kappa_boot_mean']:+.3f} "
            f"[{r['delta_ci95_lo']:+.3f}, {r['delta_ci95_hi']:+.3f}]"
        )
    ece_one = ece.groupby("method")["ece"].first()
    lines += ["", "## ECE (confidence vs round(consensus) accuracy)", ""]
    for m in METHODS:
        lines.append(f"- {METHOD_LABEL[m]}: ECE = {ece_one.get(m, float('nan')):.3f}")
    wmean = weights.groupby("rater")["reliability_weight"].mean()
    lines += ["", "## Mean train-fold reliability weights (R1–R5)", ""]
    for r, v in wmean.items():
        lines.append(f"- {r}: {v:.3f}")
    lines += [
        "",
        "## Interpretation guardrails (lab)",
        "- Do not report relative % κ.",
        "- Entropy referral is an uncertainty signal; ECE is reported separately.",
        "- Nash is not claimed superior in general; compare strata, especially range ≥ 2.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config.RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    config.RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    df = build_task_level_table(min_raters=MIN_RATERS)
    print(f"Rows={len(df)} sessions={df['session_id'].nunique()}")

    oof, folds, weights = run_oof(df)
    oof.to_csv(config.RESULTS_TABLES / "comparators_oof.csv", index=False)
    folds.to_csv(config.RESULTS_TABLES / "comparators_cv_folds.csv", index=False)
    weights.to_csv(config.RESULTS_TABLES / "comparators_reliability_weights.csv", index=False)

    boot = bootstrap_method_kappa(oof)
    deltas = bootstrap_deltas(oof, base="multitask_uniform")
    fold_d = paired_fold_deltas(folds)
    strata = stratified_table(oof)
    ref = referral_curve(oof)
    ece = ece_table(oof)

    boot.to_csv(config.RESULTS_TABLES / "comparators_bootstrap_kappa.csv", index=False)
    deltas.to_csv(config.RESULTS_TABLES / "comparators_delta_vs_multitask.csv", index=False)
    fold_d.to_csv(config.RESULTS_TABLES / "comparators_paired_fold_deltas.csv", index=False)
    strata.to_csv(config.RESULTS_TABLES / "comparators_stratified.csv", index=False)
    ref.to_csv(config.RESULTS_TABLES / "comparators_referral.csv", index=False)
    ece.to_csv(config.RESULTS_TABLES / "comparators_ece.csv", index=False)

    plot_methods(folds, boot, config.RESULTS_FIGURES / "comparators_method_comparison.png")
    plot_strata(strata, config.RESULTS_FIGURES / "comparators_stratified_kappa.png")
    plot_referral(ref, config.RESULTS_FIGURES / "comparators_referral_curve.png")
    plot_ece(ece, config.RESULTS_FIGURES / "comparators_reliability_ece.png")

    write_summary(
        folds,
        boot,
        deltas,
        fold_d,
        ece,
        weights,
        config.RESULTS_TABLES / "comparators_summary.md",
    )
    meta = {
        "n_rows": int(len(df)),
        "n_folds": N_FOLDS,
        "n_epochs": N_EPOCHS,
        "methods": list(METHODS),
        "reliability_recipe": "train-only quadratic kappa vs LORO mean of other raters; floor 0.01",
        "nash_prior": None,
    }
    (config.RESULTS_TABLES / "comparators_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nFold means:")
    print(folds.groupby("method")[["mean_rater_kappa", "test_mae"]].mean().reindex(METHODS))
    print("Wrote results/tables/comparators_summary.md")


if __name__ == "__main__":
    main()

"""Figure 2 for the brief (multi-seed). 180 mm x 118 mm.

Inputs: results/tables/dedup_multiseed/
  p1_dedup_seed*.csv    five-rater training, 4 methods, 1,559 session-by-task rows
  r123_dedup_seed*.csv  R1-R3 panel (consensus, multitask, Nash), 1,547 rows all three scored

Statistics
  kappa     fold mean of mean per-rater quadratic kappa, averaged over seeds
  CIs       400 subject resamples (seed 42); in each resample kappa is computed for every
            seed's out-of-fold predictions and averaged over seeds; methods share resamples
  rho       Spearman(entropy, inter-rater range) per seed, averaged over seeds

Writes fig2_results.svg / .pdf / .png and results/tables/dedup_multiseed/summary.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "results" / "tables" / "dedup_multiseed"

NAVY, TEAL, LIGHT, GREYTEAL = "#1E2761", "#028090", "#CADCFC", "#7EAEB8"
SLATE, GRID, WHITE = "#5D6D7B", "#E5E7EB", "#FFFFFF"
ALL5 = ("R1", "R2", "R3", "R4", "R5")
R123 = ("R1", "R2", "R3")
KEY = ["session_id", "task"]
N_BOOT, BOOT_SEED = 400, 42
FIG_W, FIG_H = 180.0, 118.0

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.unicode_minus": False, "figure.facecolor": WHITE, "savefig.facecolor": WHITE,
    "axes.edgecolor": SLATE, "axes.linewidth": 0.6, "xtick.color": NAVY, "ytick.color": NAVY,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 7, "axes.labelcolor": NAVY,
})


# --------------------------------------------------------------------------- kappa

def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa matching sklearn.metrics.cohen_kappa_score(weights='quadratic')."""
    labels = np.union1d(y_true, y_pred)
    n = labels.size
    if n < 2:
        return 0.0
    ti, pi = np.searchsorted(labels, y_true), np.searchsorted(labels, y_pred)
    conf = np.bincount(ti * n + pi, minlength=n * n).reshape(n, n).astype(float)
    expected = np.outer(conf.sum(1), conf.sum(0)) / conf.sum()
    w = (np.arange(n)[:, None] - np.arange(n)[None, :]) ** 2
    return float(1.0 - (w * conf).sum() / (w * expected).sum())


class Arm:
    """One method's out-of-fold predictions for every seed, row-aligned on KEY."""

    def __init__(self, frames: list[pd.DataFrame], raters):
        self.raters = raters
        self.pred = np.stack([np.rint(f["y_pred"].to_numpy()).astype(int) for f in frames])
        f0 = frames[0]
        self.fold = f0["fold"].to_numpy()
        self.y = {r: np.rint(np.nan_to_num(f0[f"y_{r}"].to_numpy())).astype(int) for r in raters}
        self.m = {r: f0[f"m_{r}"].to_numpy().astype(bool) for r in raters}
        self.entropy = np.stack([f["entropy"].to_numpy() for f in frames]) if "entropy" in f0 else None

    def kappa(self, s: int, rows: np.ndarray) -> float:
        vals = []
        for r in self.raters:
            idx = rows[self.m[r][rows]]
            if idx.size >= 2:
                vals.append(qwk(self.y[r][idx], self.pred[s, idx]))
        return float(np.mean(vals))

    def fold_mean(self, s: int) -> float:
        return float(np.mean([self.kappa(s, np.flatnonzero(self.fold == f)) for f in np.unique(self.fold)]))

    def per_seed(self) -> np.ndarray:
        return np.array([self.fold_mean(s) for s in range(self.pred.shape[0])])

    def boot_stat(self, rows: np.ndarray) -> float:
        return float(np.mean([self.kappa(s, rows) for s in range(self.pred.shape[0])]))


def load(pattern: str) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(glob.glob(str(RUN / pattern))):
        d = pd.read_csv(f)
        for m, g in d.groupby("method"):
            out.setdefault(m, []).append(g.sort_values(KEY).reset_index(drop=True))
    return out


def ci(a: np.ndarray) -> list[float]:
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def evaluate(arms: dict[str, Arm], subjects: np.ndarray, base: str) -> dict:
    uniq = np.unique(subjects)
    rows_of = {s: np.flatnonzero(subjects == s) for s in uniq}
    rng = np.random.default_rng(BOOT_SEED)
    boot = {n: np.empty(N_BOOT) for n in arms}
    for b in range(N_BOOT):
        rows = np.concatenate([rows_of[s] for s in rng.choice(uniq, uniq.size, replace=True)])
        for n, arm in arms.items():
            boot[n][b] = arm.boot_stat(rows)
    out = {"n_rows": int(subjects.size), "n_subjects": int(uniq.size), "baseline": base, "arms": {}}
    seeds = {n: arm.per_seed() for n, arm in arms.items()}
    for n in arms:
        d = seeds[n] - seeds[base]
        out["arms"][n] = {
            "kappa": float(seeds[n].mean()), "kappa_sd_seeds": float(seeds[n].std(ddof=1)),
            "kappa_per_seed": [float(v) for v in seeds[n]], "ci95": ci(boot[n]),
            "delta_vs_base": float(d.mean()), "delta_ci95": ci(boot[n] - boot[base]),
            "seeds_above_base": int((d > 0).sum()), "n_seeds": int(d.size),
        }
    out["_boot"] = boot
    out["_seeds"] = seeds
    return out


def pair(res: dict, a: str, b: str) -> dict:
    d = np.array(res["_seeds"][a]) - np.array(res["_seeds"][b])
    return {"delta": float(d.mean()), "ci95": ci(res["_boot"][a] - res["_boot"][b]),
            "seeds_positive": int((d > 0).sum()), "n_seeds": int(d.size)}


def rho_block(p1: dict[str, list[pd.DataFrame]]) -> dict:
    rng_key = p1["nash"][0]["score_range"].to_numpy()
    subj = p1["nash"][0]["subject_id"].to_numpy()
    ent = {m: np.stack([f["entropy"].to_numpy() for f in p1[m]]) for m in ("nash", "multitask_uniform", "consensus_mean", "reliability_weighted")}
    point = {m: [float(spearmanr(e[s], rng_key).statistic) for s in range(e.shape[0])] for m, e in ent.items()}
    uniq = np.unique(subj)
    rows_of = {s: np.flatnonzero(subj == s) for s in uniq}
    rng = np.random.default_rng(BOOT_SEED)
    boot = {m: np.empty(N_BOOT) for m in ent}
    for b in range(N_BOOT):
        rows = np.concatenate([rows_of[s] for s in rng.choice(uniq, uniq.size, replace=True)])
        for m, e in ent.items():
            boot[m][b] = np.mean([spearmanr(e[s, rows], rng_key[rows]).statistic for s in range(e.shape[0])])
    out = {m: {"rho": float(np.mean(v)), "rho_per_seed": v, "ci95": ci(boot[m])} for m, v in point.items()}
    out["nash_minus_consensus"] = {"delta": out["nash"]["rho"] - out["consensus_mean"]["rho"],
                                   "ci95": ci(boot["nash"] - boot["consensus_mean"])}
    out["nash_minus_multitask"] = {"delta": out["nash"]["rho"] - out["multitask_uniform"]["rho"],
                                   "ci95": ci(boot["nash"] - boot["multitask_uniform"])}
    out["_scatter"] = {"range": rng_key, **{m: e.mean(axis=0) for m, e in ent.items()}}
    return out


# --------------------------------------------------------------------------- drawing

def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, pad=2)


def f3(v):
    return f"{v:.3f}"


def fs(v):
    return f"{'+' if v >= 0 else '−'}{abs(v):.3f}"


def fci(lo, hi):
    def s(v):
        if abs(v) < 0.0005:  # show the sign of values that round to zero
            return f"{'−' if v < 0 else ''}{abs(v):.4f}"
        return f"−{abs(v):.3f}" if v < 0 else f"{v:.3f}"
    return f"{s(lo)} to {s(hi)}" if lo < 0 else f"{s(lo)}–{s(hi)}"


def bars(ax, specs, res, ylim=0.92):
    for x, name, face, edge, tcol in specs:
        a = res["arms"][name]
        ax.bar(x, a["kappa"], width=0.68, color=face, edgecolor=edge, linewidth=0.6, zorder=2)
        lo, hi = a["ci95"]
        ax.errorbar(x, a["kappa"], yerr=[[a["kappa"] - lo], [hi - a["kappa"]]], fmt="none",
                    ecolor="#334155", elinewidth=0.7, capsize=2.2, capthick=0.7, zorder=3)
        jitter = np.linspace(-0.16, 0.16, len(a["kappa_per_seed"]))
        ax.scatter(x + jitter, a["kappa_per_seed"], s=5, color=WHITE, edgecolor="#334155",
                   linewidths=0.5, zorder=4)
        ax.text(x, 0.03, f3(a["kappa"]), ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                color=tcol, zorder=5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, ylim)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.spines["left"].set_bounds(0, 0.8)
    ax.set_ylabel("Quadratic weighted κ")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    style(ax)


def bracket(ax, x0, x1, y, text):
    ax.plot([x0, x0, x1, x1], [y - 0.015, y, y, y - 0.015], color=NAVY, lw=0.6, clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.012, text, ha="center", va="bottom", fontsize=6, color=NAVY)


def main():
    p1 = load("p1_dedup_seed*.csv")
    r3 = load("r123_dedup_seed*.csv")
    n_seeds = len(p1["nash"])

    # A. Five-rater training on all unique rows.
    names_a = {"consensus_mean": "Consensus", "reliability_weighted": "Reliability",
               "multitask_uniform": "Multitask", "nash": "Nash"}
    arms_a = {names_a[m]: Arm(p1[m], ALL5) for m in names_a}
    res_a = evaluate(arms_a, p1["nash"][0]["subject_id"].to_numpy(), "Consensus")

    # B. Panel size on the rows R1-R3 all scored; scored against R1-R3 and against all raters.
    keys = r3["consensus"][0][KEY]
    labels = keys.merge(p1["consensus_mean"][0][KEY + ["subject_id"] + [f"{p}_{r}" for p in "ym" for r in ALL5]],
                        on=KEY, how="left")
    def restrict(frames):
        return [keys.merge(f, on=KEY, how="left") for f in frames]
    def relabel(frames):
        return [labels.merge(f[KEY + ["fold", "y_pred"]], on=KEY, how="left") for f in frames]
    frames_b = {
        "Consensus 5 raters": restrict(p1["consensus_mean"]),
        "Consensus R1–R3": relabel(r3["consensus"]),
        "Multitask R1–R3": relabel(r3["multitask"]),
        "Nash R1–R3": relabel(r3["nash"]),
        "Nash 5 raters": restrict(p1["nash"]),
    }
    for n, fr in frames_b.items():
        assert all(f["y_pred"].notna().all() and len(f) == len(keys) for f in fr), n
    subj_b = labels["subject_id"].to_numpy()
    res_b = evaluate({n: Arm(fr, R123) for n, fr in frames_b.items()}, subj_b, "Consensus 5 raters")
    res_b_all = evaluate({n: Arm(fr, ALL5) for n, fr in frames_b.items()}, subj_b, "Consensus 5 raters")
    rho = rho_block(p1)

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
    ov = fig.add_axes([0, 0, 1, 1], zorder=10)
    ov.set_xlim(0, FIG_W); ov.set_ylim(0, FIG_H); ov.axis("off"); ov.patch.set_alpha(0)

    ax = mm_axes(fig, 12, 16, 78, 90)
    bars(ax, [(0, "Consensus", NAVY, NAVY, WHITE), (1, "Reliability", SLATE, SLATE, WHITE),
              (2, "Multitask", LIGHT, NAVY, NAVY), (3, "Nash", TEAL, TEAL, WHITE)], res_a, ylim=1.10)
    ax.axhline(res_a["arms"]["Consensus"]["kappa"], color=SLATE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(4), ["Consensus", "Reliability", "Multitask", "Nash"])
    ax.set_xlim(-0.55, 3.55)
    for i, n in enumerate(("Nash", "Multitask", "Reliability")):
        a = res_a["arms"][n]
        ax.text(3.5, 1.04 - i * 0.05, f"{n} − consensus  {fs(a['delta_vs_base'])}  ({fci(*a['delta_ci95'])})",
                ha="right", va="center", fontsize=6, color=NAVY)

    bx = mm_axes(fig, 104, 64, 72, 42)
    order = [(0, "Consensus 5 raters", NAVY, NAVY, WHITE), (1, "Consensus R1–R3", LIGHT, NAVY, NAVY),
             (2, "Nash R1–R3", TEAL, TEAL, WHITE), (3, "Nash 5 raters", GREYTEAL, GREYTEAL, WHITE)]
    bars(bx, order, res_b, ylim=1.0)
    bx.set_xticks(range(4), ["Consensus\n5 raters", "Consensus\nR1–R3", "Nash\nR1–R3", "Nash\n5 raters"])
    bx.set_xlim(-0.55, 3.55)
    d1 = pair(res_b, "Nash R1–R3", "Consensus R1–R3")
    d2 = pair(res_b, "Nash R1–R3", "Consensus 5 raters")
    bracket(bx, 1, 2, 0.715, f"{fs(d1['delta'])} ({fci(*d1['ci95'])})")
    bracket(bx, 0, 2, 0.825, f"{fs(d2['delta'])} ({fci(*d2['ci95'])})")

    ins = mm_axes(fig, 106, 11, 70, 38)
    sc = rho["_scatter"]
    jit = np.random.default_rng(0)
    for m, color, off, z in (("consensus_mean", NAVY, -0.18, 2), ("multitask_uniform", GREYTEAL, 0.0, 3), ("nash", TEAL, 0.18, 4)):
        x = sc["range"] + off + jit.uniform(-0.08, 0.08, sc["range"].size)
        ins.scatter(x, sc[m], s=1.5, color=color, alpha=0.25, linewidths=0, zorder=z)
        slope, icpt = np.polyfit(sc["range"], sc[m], 1)
        ins.plot([0, 4], icpt + slope * np.array([0.0, 4.0]), color=color, lw=1.3, zorder=5)
    ins.set_xlim(-0.55, 4.55); ins.set_xticks(range(5)); ins.set_ylim(0, 1.65); ins.set_yticks([0, 0.5, 1.0, 1.5])
    ins.set_xlabel("Inter-rater range (MDS-UPDRS points)", labelpad=1)
    ins.set_ylabel("Predictive entropy (nats)", labelpad=1)
    ins.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ins.set_axisbelow(True); style(ins)
    ins.legend(handles=[
        Line2D([0], [0], color=TEAL, lw=1.3, label=f"Nash ρ = {f3(rho['nash']['rho'])}"),
        Line2D([0], [0], color=GREYTEAL, lw=1.3, label=f"Multitask ρ = {f3(rho['multitask_uniform']['rho'])}"),
        Line2D([0], [0], color=NAVY, lw=1.3, label=f"Consensus ρ = {f3(rho['consensus_mean']['rho'])}"),
    ], frameon=True, fontsize=6, loc="upper left", edgecolor=GRID, framealpha=1, borderpad=0.2,
        handlelength=1.2, labelcolor=NAVY)

    ov.text(2, 112, "A", fontsize=8, fontweight="bold", color=NAVY, va="center")
    ov.text(100, 112, "B", fontsize=8, fontweight="bold", color=NAVY, va="center")
    small = [t.get_text() for t in fig.findobj(plt.Text) if t.get_text().strip() and t.get_fontsize() < 6]
    assert not small, small
    for ext in ("svg", "pdf"):
        fig.savefig(HERE / f"fig2_results.{ext}", format=ext)
    fig.savefig(HERE / "fig2_results.png", dpi=300)
    plt.close(fig)

    def clean(res):
        return {k: v for k, v in res.items() if not k.startswith("_")}
    summary = {
        "n_seeds": n_seeds, "seeds": [int(f["seed"].iloc[0]) for f in p1["nash"]],
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "five_rater_all_rows": clean(res_a),
        "five_rater_pairs": {"nash_minus_multitask": pair(res_a, "Nash", "Multitask"),
                             "nash_minus_reliability": pair(res_a, "Nash", "Reliability")},
        "panel_size_scored_R1R3": clean(res_b),
        "panel_size_scored_all": clean(res_b_all),
        "panel_size_pairs_R1R3": {"nash_r123_minus_consensus_r123": d1, "nash_r123_minus_consensus_5": d2,
                                  "nash_5_minus_nash_r123": pair(res_b, "Nash 5 raters", "Nash R1–R3"),
                                  "consensus_5_minus_consensus_r123": pair(res_b, "Consensus 5 raters", "Consensus R1–R3")},
        "panel_size_pairs_all": {"nash_r123_minus_consensus_5": pair(res_b_all, "Nash R1–R3", "Consensus 5 raters"),
                                 "nash_r123_minus_consensus_r123": pair(res_b_all, "Nash R1–R3", "Consensus R1–R3")},
        "rho": {k: v for k, v in rho.items() if not k.startswith("_")},
    }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("five_rater_pairs", "panel_size_pairs_R1R3", "panel_size_pairs_all")}, indent=1, ensure_ascii=False))
    print({n: (round(a["kappa"], 3), [round(x, 3) for x in a["ci95"]], round(a["delta_vs_base"], 3), [round(x, 3) for x in a["delta_ci95"]]) for n, a in summary["five_rater_all_rows"]["arms"].items()})
    print({n: (round(a["kappa"], 3), [round(x, 3) for x in a["ci95"]]) for n, a in summary["panel_size_scored_R1R3"]["arms"].items()})
    print({m: (round(v["rho"], 3), [round(x, 3) for x in v["ci95"]]) if "rho" in v else (round(v["delta"], 3), [round(x, 3) for x in v["ci95"]]) for m, v in summary["rho"].items()})


if __name__ == "__main__":
    main()

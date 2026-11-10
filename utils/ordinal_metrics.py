"""Metrics for ordinal UPDRS prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


def quadratic_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.round(np.asarray(y_true)).astype(int)
    y_pred = np.round(np.asarray(y_pred)).astype(int)
    if len(np.unique(y_true)) < 2 and len(np.unique(y_pred)) < 2:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_kappa": quadratic_kappa(y_true, y_pred),
    }


def per_rater_kappa_table(
    y_true_by_rater: dict[str, np.ndarray],
    y_pred: np.ndarray,
    mask_by_rater: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for rater, y_true in y_true_by_rater.items():
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        if mask_by_rater is not None and rater in mask_by_rater:
            m = mask_by_rater[rater].astype(bool)
            yt, yp = yt[m], yp[m]
        else:
            valid = ~np.isnan(yt)
            yt, yp = yt[valid], yp[valid]
        if len(yt) < 2:
            out[rater] = float("nan")
            continue
        out[rater] = quadratic_kappa(yt, yp)
    return out

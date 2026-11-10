"""Loss-based rater weighting with a Nash-bargaining-style log-gain objective.

Inspired by Nash-MTL (Navon et al. 2022) but not equivalent to it: weights are computed from
per-rater loss values, not from inner products of rater gradients, so the Nash-MTL
Pareto-stationarity guarantees do not apply.
"""

from __future__ import annotations

import numpy as np


def _project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    return w


def solve_nash_weights(
    utilities: np.ndarray,
    disagreement: np.ndarray | None = None,
    max_iter: int = 250,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Truncated ascent of J(alpha) = sum_r alpha_r * log(u_r - d_r) over the probability simplex.

    Each step is a projected-gradient step with Frank-Wolfe step sizes,
        alpha <- (1 - g_k) * alpha + g_k * Proj_simplex(alpha + grad J),  g_k = 2 / (k + 2),
    starting from uniform weights, for at most `max_iter` steps or until the L1 change is
    below `tol`. J is linear in alpha, so its maximiser puts all weight on the rater with
    the largest u_r. The truncated iterate returned here is a soft weighting that favours
    the largest u_r; it is not that maximiser.

    utilities: per-rater values u_r. Training passes scaled losses (loss / EMA of loss),
        so raters the model currently fits worst receive the most weight. Despite the
        name, a larger value is not "better" here.
    disagreement: per-rater disagreement point d_r; defaults to min(utilities) - 1,
        so every gain u_r - d_r is at least 1.
    Returns weights over raters that sum to one.
    """
    u = np.asarray(utilities, dtype=float)
    if disagreement is None:
        d0 = np.full_like(u, u.min() - 1.0)
    else:
        d0 = np.asarray(disagreement, dtype=float)

    gains = np.maximum(u - d0, 1e-8)
    log_gains = np.log(gains)
    n = len(u)
    alpha = np.ones(n) / n

    for _ in range(max_iter):
        grad = log_gains - np.dot(alpha, log_gains)
        direction = _project_simplex(alpha + grad, z=1.0)
        gamma = 2.0 / (2.0 + _)
        alpha_new = (1.0 - gamma) * alpha + gamma * direction
        if np.linalg.norm(alpha_new - alpha, ord=1) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    alpha = np.maximum(alpha, 0.0)
    total = alpha.sum()
    if total <= 0:
        return np.ones(n) / n
    return alpha / total


def aggregate_gradients(gradients: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted sum of per-rater gradients."""
    stacked = np.stack(gradients, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))

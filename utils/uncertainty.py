"""Predictive uncertainty from the ordinal head (CORAL logits)."""
from __future__ import annotations

import numpy as np
import torch

from models.ordinal_model import CoralOrdinalModel


def coral_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Recover P(Y=k) from cumulative logits P(Y>k)=sigmoid(z_k).

    Negative differences are clamped to zero and the distribution is renormalised,
    as described in the paper.
    """
    p_gt = torch.sigmoid(logits)
    parts = [1.0 - p_gt[:, :1]]
    for k in range(p_gt.shape[1] - 1):
        parts.append(p_gt[:, k : k + 1] - p_gt[:, k + 1 : k + 2])
    parts.append(p_gt[:, -1:])
    probs = torch.cat(parts, dim=1).clamp(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs


def uncertainty_from_logits(logits: torch.Tensor) -> dict[str, np.ndarray]:
    probs = coral_class_probs(logits).cpu().numpy()
    ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    ks = np.arange(probs.shape[1], dtype=float)
    ey = (probs * ks).sum(axis=1)
    ey2 = (probs * (ks**2)).sum(axis=1)
    std = np.sqrt(np.clip(ey2 - ey**2, 0, None))
    conf = probs.max(axis=1)
    p_gt = torch.sigmoid(logits).cpu().numpy()
    bin_ent = -(p_gt * np.log(p_gt + 1e-12) + (1 - p_gt) * np.log(1 - p_gt + 1e-12))
    thr_ent = bin_ent.mean(axis=1)
    preds = (p_gt > 0.5).sum(axis=1).astype(float)
    return {
        "pred": preds,
        "entropy": ent,
        "pred_std": std,
        "confidence": conf,
        "threshold_entropy": thr_ent,
        "expected_y": ey,
    }


@torch.no_grad()
def predict_uncertainty(model: CoralOrdinalModel, x: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    logits = model(torch.tensor(x, dtype=torch.float32))
    return uncertainty_from_logits(logits)

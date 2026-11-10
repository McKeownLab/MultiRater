"""Ordinal regression head for MDS-UPDRS scores 0..K-1, trained with the CORAL loss.

Note: this is NOT rank-consistent CORAL (Cao et al., 2020). CORAL shares one weight vector across
the K-1 binary tasks and learns only K-1 biases; here `self.coral = nn.Linear(hidden, K-1)` gives
each threshold its own weights (ordinal binary decomposition, as in Niu et al., 2016). The CORAL
target encoding and loss are used, and the prediction is the number of outputs with p > 0.5.
All reported results use this head; the paper describes it this way.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoralOrdinalModel(nn.Module):
    """K-1 sigmoid binary threshold outputs with independent weights (CORAL loss, no weight sharing)."""

    def __init__(self, input_dim: int, num_classes: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.coral = nn.Linear(hidden_dim, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coral(self.backbone(x))


def coral_labels(y: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Convert class labels [N] to cumulative binary targets [N, K-1]."""
    thresholds = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()


def coral_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    targets = coral_labels(y, num_classes=num_classes)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def coral_loss_masked(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 5,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CORAL loss on a subset of rows (e.g. where a rater actually scored)."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    logits_m = logits[mask]
    y_m = y[mask]
    targets = coral_labels(y_m, num_classes=num_classes)
    per_row = nn.functional.binary_cross_entropy_with_logits(
        logits_m, targets, reduction="none"
    ).mean(dim=1)
    if row_weights is not None:
        w = row_weights[mask]
        return (per_row * w).sum() / w.sum().clamp(min=1e-8)
    return per_row.mean()


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).long()

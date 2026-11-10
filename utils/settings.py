"""Training settings used for every paper run."""
from __future__ import annotations

N_EPOCHS = 300
LR = 0.01
N_FOLDS = 5
MIN_RATERS = 3
DISAGREEMENT_WEIGHT_SCALE = 2.5
NASH_KWARGS = {"min_weight": 0.1, "weight_ema": 0.8, "warmup_epochs": 40}

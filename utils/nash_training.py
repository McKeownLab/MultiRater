"""Stable Nash-MTL training for CORAL ordinal regression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config
from models.ordinal_model import CoralOrdinalModel, coral_loss, coral_loss_masked, coral_predict
from utils.nash_solver import aggregate_gradients, solve_nash_weights
from utils.ordinal_metrics import ordinal_metrics, per_rater_kappa_table

ACTIVE_RATERS = config.ACTIVE_RATERS
NUM_CLASSES = 5
DEFAULT_SEED = 42


def _seed_all(seed: int = DEFAULT_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _clip_ordinal(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(y, dtype=float)), 0, NUM_CLASSES - 1).astype(int)


def _flat_grad_list(model: nn.Module) -> list[np.ndarray]:
    return [p.grad.detach().cpu().numpy().copy() for p in model.parameters() if p.grad is not None]


def _normalize_grad_list(grads: list[np.ndarray]) -> list[np.ndarray]:
    flat = np.concatenate([g.reshape(-1) for g in grads])
    norm = np.linalg.norm(flat)
    if norm < 1e-8:
        return grads
    scale = 1.0 / norm
    return [g * scale for g in grads]


def _set_grads(model: nn.Module, grads: list[np.ndarray], device: torch.device) -> None:
    for param, grad in zip(model.parameters(), grads):
        param.grad = torch.tensor(grad, dtype=torch.float32, device=device)


def _eval_coral(
    model: CoralOrdinalModel,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        preds = coral_predict(logits).cpu().numpy()
    y_int = _clip_ordinal(y)
    out = ordinal_metrics(y_int, preds)
    out["n_samples"] = int(len(y))
    return out


def train_ordinal_supervised(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.01,
    hidden_dim: int = 32,
    row_weights: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[CoralOrdinalModel, dict[str, float], dict[str, float]]:
    _seed_all(seed)
    device = torch.device("cpu")
    y_tr = torch.tensor(_clip_ordinal(y_train), dtype=torch.long, device=device)
    x_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    w_tr = (
        torch.tensor(row_weights, dtype=torch.float32, device=device)
        if row_weights is not None
        else None
    )
    all_mask = torch.ones(len(y_train), dtype=torch.bool, device=device)

    model = CoralOrdinalModel(x_train.shape[1], num_classes=NUM_CLASSES, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = coral_loss_masked(
            model(x_tr), y_tr, all_mask, num_classes=NUM_CLASSES, row_weights=w_tr
        )
        loss.backward()
        optimizer.step()

    return model, _eval_coral(model, x_train, y_train), _eval_coral(model, x_test, y_test)


def _rater_tensors(
    y_train_by_rater: dict[str, np.ndarray],
    mask_train: dict[str, np.ndarray] | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    y_tensors: dict[str, torch.Tensor] = {}
    mask_tensors: dict[str, torch.Tensor] = {}
    for rater in ACTIVE_RATERS:
        y_clean = np.nan_to_num(y_train_by_rater[rater], nan=0.0)
        y_tensors[rater] = torch.tensor(_clip_ordinal(y_clean), dtype=torch.long, device=device)
        if mask_train is not None:
            mask_tensors[rater] = torch.tensor(mask_train[rater], dtype=torch.bool, device=device)
        else:
            mask_tensors[rater] = torch.ones(len(y_clean), dtype=torch.bool, device=device)
    return y_tensors, mask_tensors


def train_ordinal_uniform(
    x_train: np.ndarray,
    y_train_by_rater: dict[str, np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.01,
    hidden_dim: int = 32,
    mask_train: dict[str, np.ndarray] | None = None,
    row_weights: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[CoralOrdinalModel, dict[str, float], dict[str, float]]:
    """Equal-weight multi-rater gradient mix (ablation vs Nash)."""
    _seed_all(seed)
    device = torch.device("cpu")
    x_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensors, mask_tensors = _rater_tensors(y_train_by_rater, mask_train, device)
    w_tr = (
        torch.tensor(row_weights, dtype=torch.float32, device=device)
        if row_weights is not None
        else None
    )
    model = CoralOrdinalModel(x_train.shape[1], num_classes=NUM_CLASSES, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = np.ones(len(ACTIVE_RATERS)) / len(ACTIVE_RATERS)

    for _ in range(n_epochs):
        per_rater_grads: list[list[np.ndarray] | None] = [None] * len(ACTIVE_RATERS)
        active: list[int] = []
        for i, rater in enumerate(ACTIVE_RATERS):
            if mask_tensors[rater].sum() == 0:
                continue
            model.zero_grad()
            loss = coral_loss_masked(
                model(x_tr),
                y_tensors[rater],
                mask_tensors[rater],
                num_classes=NUM_CLASSES,
                row_weights=w_tr,
            )
            loss.backward()
            per_rater_grads[i] = _normalize_grad_list(_flat_grad_list(model))
            active.append(i)
        if not active:
            continue

        sub_w = np.ones(len(active), dtype=float) / len(active)
        # Aggregate only active raters (weights length must match gradient list)
        active_grads = [per_rater_grads[i] for i in active]
        n_params = len(active_grads[0])
        agg = [
            aggregate_gradients([g[param_idx] for g in active_grads], sub_w)
            for param_idx in range(n_params)
        ]
        model.zero_grad()
        _set_grads(model, agg, device)
        optimizer.step()

    train_metrics = _eval_coral(model, x_train, y_train_by_rater["consensus"])
    test_metrics = _eval_coral(model, x_test, y_test)
    return model, train_metrics, test_metrics


def train_ordinal_nash(
    x_train: np.ndarray,
    y_train_by_rater: dict[str, np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int = 200,
    lr: float = 0.01,
    hidden_dim: int = 32,
    warmup_epochs: int = 20,
    weight_ema: float = 0.85,
    min_weight: float = 0.08,
    shapley_weights: np.ndarray | None = None,
    mask_train: dict[str, np.ndarray] | None = None,
    row_weights: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
    diagnostics: dict | None = None,
) -> tuple[CoralOrdinalModel, dict[str, float], dict[str, float], np.ndarray, list[np.ndarray]]:
    _seed_all(seed)
    device = torch.device("cpu")
    x_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensors, mask_tensors = _rater_tensors(y_train_by_rater, mask_train, device)
    w_tr = (
        torch.tensor(row_weights, dtype=torch.float32, device=device)
        if row_weights is not None
        else None
    )

    model = CoralOrdinalModel(x_train.shape[1], num_classes=NUM_CLASSES, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weight_history: list[np.ndarray] = []
    ema_weights = np.ones(len(ACTIVE_RATERS)) / len(ACTIVE_RATERS)
    loss_ema = np.ones(len(ACTIVE_RATERS), dtype=float)

    for epoch in range(n_epochs):
        per_rater_losses: list[float] = []
        per_rater_grads: list[list[np.ndarray]] = []
        active_idx: list[int] = []
        for i, rater in enumerate(ACTIVE_RATERS):
            if mask_tensors[rater].sum() == 0:
                continue
            model.zero_grad()
            loss = coral_loss_masked(
                model(x_tr),
                y_tensors[rater],
                mask_tensors[rater],
                num_classes=NUM_CLASSES,
                row_weights=w_tr,
            )
            loss.backward()
            per_rater_losses.append(float(loss.item()))
            raw_grads = _flat_grad_list(model)
            if diagnostics is not None:
                flat = np.concatenate([g.reshape(-1) for g in raw_grads])
                diagnostics.setdefault("loss_sum", np.zeros(len(ACTIVE_RATERS)))
                diagnostics.setdefault("grad_norm_sum", np.zeros(len(ACTIVE_RATERS)))
                diagnostics.setdefault("n_active", np.zeros(len(ACTIVE_RATERS)))
                diagnostics["loss_sum"][i] += float(loss.item())
                diagnostics["grad_norm_sum"][i] += float(np.linalg.norm(flat))
                diagnostics["n_active"][i] += 1
            per_rater_grads.append(_normalize_grad_list(raw_grads))
            active_idx.append(i)

        if not active_idx:
            continue

        losses = np.array(per_rater_losses, dtype=float)
        loss_ema_sub = loss_ema[np.array(active_idx)]
        loss_ema_sub = 0.9 * loss_ema_sub + 0.1 * losses
        loss_ema[np.array(active_idx)] = loss_ema_sub
        scaled = losses / np.maximum(loss_ema_sub, 1e-6)
        utilities = scaled.copy()
        if shapley_weights is not None:
            utilities = utilities * (1.0 + shapley_weights[np.array(active_idx)])

        if epoch < warmup_epochs:
            sub_w = np.ones(len(active_idx)) / len(active_idx)
        else:
            solved = solve_nash_weights(utilities)
            if diagnostics is not None:
                diagnostics.setdefault("floor_hits", np.zeros(len(ACTIVE_RATERS)))
                diagnostics.setdefault("n_post_warmup", 0)
                diagnostics["n_post_warmup"] += 1
                for j, i in enumerate(active_idx):
                    if min_weight > 0 and solved[j] < min_weight - 1e-8:
                        diagnostics["floor_hits"][i] += 1
            sub_w = np.maximum(solved, min_weight) if min_weight > 0 else solved
            sub_w = sub_w / sub_w.sum()

        weights = np.zeros(len(ACTIVE_RATERS))
        for j, i in enumerate(active_idx):
            weights[i] = sub_w[j]
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(ACTIVE_RATERS)) / len(ACTIVE_RATERS)

        ema_weights = weight_ema * ema_weights + (1.0 - weight_ema) * weights
        weight_history.append(ema_weights.copy())

        ref = per_rater_grads[0]
        padded = [None] * len(ACTIVE_RATERS)
        for j, i in enumerate(active_idx):
            padded[i] = per_rater_grads[j]
        for i in range(len(ACTIVE_RATERS)):
            if padded[i] is None:
                padded[i] = [np.zeros_like(g) for g in ref]
        w_agg = ema_weights / ema_weights.sum()
        n_params = len(ref)
        agg = [
            aggregate_gradients([g[param_idx] for g in padded], w_agg)
            for param_idx in range(n_params)
        ]
        model.zero_grad()
        _set_grads(model, agg, device)
        optimizer.step()

    mean_weights = np.mean(np.stack(weight_history[-10:]), axis=0)
    train_metrics = _eval_coral(model, x_train, y_train_by_rater["consensus"])
    test_metrics = _eval_coral(model, x_test, y_test)
    return model, train_metrics, test_metrics, mean_weights, weight_history

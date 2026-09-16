"""Metrics of the Cola-GNN / EpiGNN protocol on the count scale."""
from __future__ import annotations

import numpy as np


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def colagnn_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: (n_samples, n_regions) on the raw scale.

    `rmse` / `pcc` are pooled over all samples and regions (the numbers printed in the Cola-GNN and EpiGNN
    papers); `rmse_states` / `pcc_states` average the per-region values (also printed by their code)."""
    err = y_pred - y_true
    per_region_pcc = [pearson(y_true[:, k], y_pred[:, k]) for k in range(y_true.shape[1])]
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pcc": pearson(y_true.reshape(-1), y_pred.reshape(-1)),
        "rmse_states": float(np.mean(np.sqrt(np.mean(err ** 2, axis=0)))),
        "pcc_states": float(np.nanmean(per_region_pcc)),
    }

#!/usr/bin/env python
"""Seasonal references under the Cola-GNN protocol (raw scale, pooled and per-region metrics).

Writes results/<dataset>/<tag>/h{h}_s0.json for
    SeasonalNaive52   x[i - 52]                         (last season)
    Climatology2      mean over the two previous seasons of a +-2-week window around week i - 52 k
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import DATASETS, ColaGNNData  # noqa: E402
from utils import colagnn_metrics  # noqa: E402

PERIOD = 52


def climatology(raw: np.ndarray, i: int, seasons: int = 2, width: int = 2) -> np.ndarray:
    vals = [raw[i - PERIOD * k - width:i - PERIOD * k + width + 1].mean(0)
            for k in range(1, seasons + 1) if i - PERIOD * k - width >= 0]
    return np.mean(vals, 0)


def main() -> None:
    for ds in DATASETS:
        for h in (3, 5, 10, 15):
            data = ColaGNNData(ds, ROOT / "data" / "colagnn", window=20, horizon=h)
            raw, idx = data.raw.astype(float), data.test_idx
            y = raw[idx]
            for tag, pred in {"SeasonalNaive52": np.array([raw[i - PERIOD] for i in idx]),
                              "Climatology2": np.array([climatology(raw, i) for i in idx])}.items():
                metrics = colagnn_metrics(y, pred)
                out = ROOT / "results" / ds / tag / f"h{h}_s0.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"dataset": ds, "model": tag, "tag": tag, "seed": 0, "horizon": h,
                                           "window": -1, "n_params": 0, "metrics": metrics, "reference": True}, indent=1))
                print(f"{ds:10s} h{h:2d} {tag:16s} rmse {metrics['rmse']:7.1f} pcc {metrics['pcc']:.3f}")


if __name__ == "__main__":
    main()

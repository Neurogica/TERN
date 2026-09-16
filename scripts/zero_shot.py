#!/usr/bin/env python
"""Zero-shot foundation-model baselines (Chronos-2, TimesFM) under the Cola-GNN protocol.

For every test target index i and lead time h the context is the raw count history of each region, either the
protocol window (20 weeks ending at i - h) or the full history up to i - h.
    python scripts/zero_shot.py chronos --dataset japan --context full
    python scripts/zero_shot.py timesfm --dataset japan --context full
Requires `uv pip install chronos-forecasting` or `uv pip install timesfm`. Results go to results/<dataset>/<tag>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import ColaGNNData  # noqa: E402
from utils import colagnn_metrics  # noqa: E402


def chronos_forecaster(model_id: str, batch: int):
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16)

    def predict(contexts: list[np.ndarray], h: int) -> np.ndarray:
        preds = []
        for s in range(0, len(contexts), batch):
            ctx = [torch.tensor(c, dtype=torch.float32) for c in contexts[s:s + batch]]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=h, quantile_levels=[0.5])
            q = torch.stack([torch.as_tensor(t) for t in q]) if isinstance(q, (list, tuple)) else torch.as_tensor(q)
            if q.dim() == 4:
                q = q[:, 0]
            preds.append(q[:, h - 1, 0].float().cpu().numpy())
        return np.concatenate(preds)

    return predict


def timesfm_forecaster(model_id: str, batch: int):
    import timesfm

    fc = timesfm.TimesFM3Forecaster.from_pretrained(model_id, device="cuda")

    def point(out, h: int) -> float:
        for name in ("point_forecast", "mean_forecast", "mean", "forecast"):
            if hasattr(out, name):
                arr = np.asarray(getattr(out, name))
                return float(arr.reshape(-1, arr.shape[-1])[0, h - 1])
        return float(np.asarray(out[0]).reshape(-1)[h - 1])

    def predict(contexts: list[np.ndarray], h: int) -> np.ndarray:
        preds = []
        for s in range(0, len(contexts), batch):
            outs = fc.predict_batch(contexts[s:s + batch], horizon=h, return_quantiles=False, make_positive=True)
            preds.extend(point(o, h) for o in outs)
        return np.asarray(preds)

    return predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", choices=["chronos", "timesfm"])
    ap.add_argument("--dataset", default="japan")
    ap.add_argument("--horizons", default="3,5,10,15")
    ap.add_argument("--context", choices=["window", "full"], default="full")
    ap.add_argument("--model", default="", help="model id (default: amazon/chronos-2 or google/timesfm-3.0-pytorch)")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    model_id = args.model or {"chronos": "amazon/chronos-2", "timesfm": "google/timesfm-3.0-pytorch"}[args.family]
    predict = (chronos_forecaster if args.family == "chronos" else timesfm_forecaster)(model_id, args.batch)
    tag = {"chronos": "Chronos2", "timesfm": "TimesFM3"}[args.family] + ("_zs_full" if args.context == "full" else "_zs")
    for h in [int(x) for x in args.horizons.split(",")]:
        data = ColaGNNData(args.dataset, ROOT / "data" / "colagnn", window=20, horizon=h)
        raw, idx = data.raw, data.test_idx
        if args.context == "window":
            ctx_len = 20
        else:  # one fixed context length so that every batch has the same shape
            ctx_len = max(32, (min(i - h + 1 for i in idx) // 32) * 32)
        contexts = [raw[max(0, i - h + 1 - ctx_len):i - h + 1, k].astype(np.float32) for i in idx for k in range(data.m)]
        t0 = time.time()
        y_pred = np.clip(predict(contexts, h).reshape(len(idx), data.m), 0, None)
        metrics = colagnn_metrics(raw[idx].astype(np.float64), y_pred.astype(np.float64))
        out = ROOT / "results" / args.dataset / tag / f"h{h}_s0.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"dataset": args.dataset, "model": tag, "tag": tag, "seed": 0, "horizon": h,
                                   "window": ctx_len, "n_params": 0, "metrics": metrics, "zero_shot": True,
                                   "model_kwargs": {"model": model_id, "context": args.context},
                                   "time_s": time.time() - t0}, indent=1))
        print(json.dumps({"dataset": args.dataset, "tag": tag, "h": h, **{k: round(v, 4) for k, v in metrics.items()}}),
              flush=True)


if __name__ == "__main__":
    main()

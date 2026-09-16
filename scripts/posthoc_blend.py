#!/usr/bin/env python
"""Apply TERN's online blend post hoc to saved baseline forecasts (fairness check).

For every preds/<dataset>/<tag>/h{h}_s{seed}.npz the forecast is combined with a seasonal reference r_i as
alpha * r_i + (1 - alpha) * f_i, where alpha in {0, 0.1, ..., 1} is re-selected at every test origin from the
scale-weighted squared error of both on the last W test targets already observed at that origin (the rule of
train.py --online_blend). The reference is the seasonal naive x[i - 52] on Japan and the two-season climatology
on the US datasets, i.e. what TERN uses there. Writes results/<dataset>/<tag>_blend/h{h}_s{seed}.json.

    python scripts/posthoc_blend.py --tags DLinear,PatchTST,iTransformer,TimeMixer,TimesNet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {"japan": "naive", "region785": "climatology", "state360": "climatology"}
PERIOD = 52


def climatology(x: np.ndarray, i: int, seasons: int = 2, width: int = 2) -> np.ndarray:
    vals = [x[i - PERIOD * k - width:i - PERIOD * k + width + 1].mean(0)
            for k in range(1, seasons + 1) if i - PERIOD * k - width >= 0]
    return np.mean(vals, axis=0)


def blend(dataset: str, y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray, weights: np.ndarray, h: int,
          window: int) -> np.ndarray:
    """y_true, y_pred: (n_test, N) raw scale; x: (T, N) raw series; weights: (N,) region weights."""
    n = y_true.shape[0]
    test_idx = np.arange(x.shape[0] - n, x.shape[0])
    ref = np.array([x[i - PERIOD] if REFERENCE[dataset] == "naive" else climatology(x, i) for i in test_idx])
    out = np.empty_like(y_pred)
    for k in range(n):
        usable = list(range(max(0, k - h - window + 1), k - h + 1))  # targets observed at origin k - h
        if len(usable) < 4:
            out[k] = y_pred[k]
            continue
        losses = [np.mean(weights * ((a / 10 * ref[usable] + (1 - a / 10) * y_pred[usable] - y_true[usable]) ** 2))
                  for a in range(11)]
        a = int(np.argmin(losses)) / 10
        out[k] = a * ref[k] + (1 - a) * y_pred[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", default="DLinear,PatchTST,iTransformer,TimeMixer,TimesNet")
    ap.add_argument("--datasets", default="japan,region785,state360")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--window", type=int, default=12)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from utils import colagnn_metrics

    for ds in args.datasets.split(","):
        x = np.loadtxt(ROOT / "data" / "colagnn" / f"{ds}.txt", delimiter=",")
        for tag in args.tags.split(","):
            n_written = 0
            for h in (3, 5, 10, 15):
                for seed in [int(s) for s in args.seeds.split(",")]:
                    f = ROOT / "preds" / ds / tag / f"h{h}_s{seed}.npz"
                    if not f.exists():
                        continue
                    d = np.load(f)
                    lo, hi = d["min"], d["max"]
                    y_true, y_pred = d["y_true"] * (hi - lo) + lo, d["y_pred"] * (hi - lo) + lo
                    w = (hi - lo) ** 2
                    out = blend(ds, y_true, y_pred, x, w / w.mean(), h, args.window)
                    res = {"dataset": ds, "model": tag, "tag": f"{tag}_blend", "seed": seed, "horizon": h, "window": 20,
                           "metrics": colagnn_metrics(y_true, out), "note": f"post-hoc online blend, W={args.window}"}
                    path = ROOT / "results" / ds / f"{tag}_blend" / f"h{h}_s{seed}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(res, indent=1))
                    n_written += 1
            print(f"{ds} {tag}: {n_written} blended runs written")


if __name__ == "__main__":
    main()

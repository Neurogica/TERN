#!/usr/bin/env python
"""Run the official EpiGNN code (ext_repos/epignn) under the protocol and store its final test metrics.

    python scripts/run_official.py --dataset japan --horizon 5 --seed 0
Writes results/<dataset>/EpiGNN_rerun/h{h}_s{seed}.json with pooled RMSE/PCC and the per-region variants.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADJACENCY = {"japan": "japan-adj", "region785": "region-adj", "state360": "state-adj-49"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(ADJACENCY))
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cwd = ROOT / "ext_repos" / "epignn"
    if not cwd.exists():
        raise SystemExit("ext_repos/epignn not found; run scripts/clone_ext_repos.sh first")
    cmd = (f"PYTHONPATH=src {sys.executable} src/train.py --dataset {args.dataset} --sim_mat {ADJACENCY[args.dataset]} "
           f"--horizon {args.horizon} --seed {args.seed} --gpu 0")
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    log = p.stdout + "\n" + p.stderr
    lines = [ln for ln in log.splitlines() if ln.startswith("TEST")]
    if p.returncode != 0 or not lines:
        print(log[-3000:])
        raise SystemExit(f"official run failed (rc={p.returncode})")
    last = lines[-1]
    value = lambda key: float(re.search(rf"\b{key} ([-\d.]+)", last).group(1))  # noqa: E731
    metrics = {"rmse": value("RMSE"), "mae": value("MAE"), "pcc": value("PCC"),
               "rmse_states": value("RMSEs"), "pcc_states": value("PCCs")}
    out = ROOT / "results" / args.dataset / "EpiGNN_rerun" / f"h{args.horizon}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "model": "EpiGNN_rerun", "tag": "EpiGNN_rerun",
                               "seed": args.seed, "horizon": args.horizon, "window": 20, "metrics": metrics,
                               "official_line": last, "time_s": time.time() - t0}, indent=1))
    print(json.dumps({"dataset": args.dataset, "h": args.horizon, "seed": args.seed, **metrics}))


if __name__ == "__main__":
    main()

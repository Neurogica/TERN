#!/usr/bin/env python
"""Run a job file (one shell command per line) with N-way parallelism on a shared GPU.

Jobs whose result JSON already exists are skipped, so a queue can be re-run safely after an interruption.
    python scripts/run_queue.py jobs/tern_full_history.txt --parallel 8
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def result_path(cmd: str) -> Path | None:
    """Result JSON that a train.py command will write (mirrors the layout in src/train.py)."""
    toks = shlex.split(cmd)
    get = lambda key, default=None: toks[toks.index(key) + 1] if key in toks else default  # noqa: E731
    if "--out" in toks:
        return ROOT / get("--out")
    if "train.py" not in cmd:
        return None
    tag = get("--tag") or get("--model", "tern")
    name = f"h{get('--horizon', '5')}_s{get('--seed', '0')}.json"
    return ROOT / get("--results_dir", "results") / get("--dataset", "japan") / tag / name


def run_one(i: int, cmd: str, logdir: Path):
    t0 = time.time()
    with open(logdir / f"job{i:04d}.log", "w") as f:
        f.write(cmd + "\n\n")
        f.flush()
        rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    return i, rc, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobfile")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-run jobs whose result already exists")
    args = ap.parse_args()
    lines = [ln.strip() for ln in Path(args.jobfile).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    jobs = [ln for ln in lines if args.force or not ((p := result_path(ln)) is not None and p.exists())]
    logdir = ROOT / "logs" / (Path(args.jobfile).stem + time.strftime("_%m%d_%H%M"))
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs to run ({len(lines)} lines) | parallel={args.parallel} | logs -> {logdir}", flush=True)
    t0, fails = time.time(), 0
    with ThreadPoolExecutor(args.parallel) as ex:
        futures = [ex.submit(run_one, i, c, logdir) for i, c in enumerate(jobs)]
        for n, fut in enumerate(as_completed(futures), 1):
            i, rc, dt = fut.result()
            fails += rc != 0
            print(f"[{n}/{len(jobs)}] job{i:04d} rc={rc} {dt:7.1f}s  elapsed {time.time() - t0:8.1f}s", flush=True)
    print(f"done: {len(jobs) - fails} ok, {fails} failed, {time.time() - t0:.1f}s", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

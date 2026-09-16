#!/usr/bin/env python
"""Write job files (one train.py command per line) for the experiments of the paper.

    python scripts/make_jobs.py tern       --seeds 0,1,2,3,4    # both regimes, config/tern.json
    python scripts/make_jobs.py baselines  --seeds 0,1,2,3,4    # Time-Series-Library models, config/baselines.json
    python scripts/make_jobs.py ablation   --seeds 0,1,2        # one component removed at a time (full-history regime)
    python scripts/run_queue.py jobs/<file>.txt --parallel 8
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "python src/train.py"
DATASETS = ["japan", "region785", "state360"]
HORIZONS = [3, 5, 10, 15]


def command(dataset: str, horizon: int, seed: int, model: str, tag: str, kwargs: dict, extra: str = "") -> str:
    return (f"{PY} --dataset {dataset} --horizon {horizon} --seed {seed} --model {model} --tag {tag} "
            f"--model_kwargs '{json.dumps(kwargs)}' --quiet {extra}").strip()


def per_horizon(cfg: dict, horizon: int) -> tuple[dict, str]:
    kwargs = {**cfg["kwargs"], **cfg.get("kwargs_by_horizon", {}).get(str(horizon), {})}
    extra = f"{cfg.get('args', '')} {cfg.get('args_by_horizon', {}).get(str(horizon), '')}".strip()
    return kwargs, extra


def jobs_tern(seeds, save_pred: bool) -> dict[str, list[str]]:
    cfg = json.loads((ROOT / "config" / "tern.json").read_text())
    out = {}
    for regime, tag in (("window", "tern"), ("full_history", "tern_full")):
        lines = []
        for ds in DATASETS:
            for h in HORIZONS:
                kwargs, extra = per_horizon(cfg[regime][ds], h)
                extra += " --save_pred" if save_pred else ""
                lines += [command(ds, h, s, "tern", tag, kwargs, extra) for s in seeds]
        out[f"tern_{regime}"] = lines
    return out


def jobs_baselines(seeds, save_pred: bool) -> dict[str, list[str]]:
    cfg = json.loads((ROOT / "config" / "baselines.json").read_text())
    extra = cfg.get("args", "") + (" --save_pred" if save_pred else "")
    lines = [command(ds, h, s, m, m, kw, extra)
             for ds in DATASETS for h in HORIZONS for s in seeds for m, kw in cfg["models"].items()]
    return {"baselines": lines}


# ablation variants: name -> function(kwargs, extra) -> (kwargs, extra) or None when not applicable to the dataset
def _drop_flag(extra: str, pattern: str) -> str:
    return re.sub(pattern, "", extra).replace("  ", " ").strip()


ABLATIONS = {
    "rule_gdn2":    lambda kw, ex: ({**kw, "rule": "gdn2"}, ex),
    "rule_delta":   lambda kw, ex: ({**kw, "rule": "delta"}, ex),
    "rule_gla":     lambda kw, ex: ({**kw, "rule": "gla"}, ex),
    "decay_scalar": lambda kw, ex: ({**kw, "decay": "scalar"}, ex),
    "decay_none":   lambda kw, ex: ({**kw, "decay": "none"}, ex),
    "ts_uniform":   lambda kw, ex: ({**kw, "timescale_init": "uniform"}, ex),
    "ts_frozen":    lambda kw, ex: ({**kw, "learn_timescale": False}, ex),
    "no_phase":     lambda kw, ex: ({**kw, "phase_features": False}, ex),
    "no_region_attention": lambda kw, ex: ({**kw, "region_attention": False}, ex),
    "softmax_mixer": lambda kw, ex: ({**kw, "mixer_type": "attn"}, ex),
    "no_seasonal_reference": lambda kw, ex: (
        None if not (kw.get("season_embedding") or kw.get("climatology_seasons"))
        else ({**kw, "season_embedding": False, "climatology_seasons": 0, "climatology_residual": False}, ex)),
    "constant_shrinkage": lambda kw, ex: (None if "residual_scale" not in kw
                                          else ({**kw, "residual_scale": 0.3}, ex)),
    "unweighted_loss": lambda kw, ex: (None if "--scale_weighted_loss" not in ex
                                       else (kw, _drop_flag(ex, r"--scale_weighted_loss"))),
    "no_online_blend": lambda kw, ex: (None if "--online_blend" not in ex
                                       else (kw, _drop_flag(ex, r"--online_blend \d+"))),
    "no_online_refit": lambda kw, ex: (None if "--online_refit" not in ex
                                       else (kw, _drop_flag(ex, r"--online_refit \d+"))),
    "no_polyak":    lambda kw, ex: None if "--ema" not in ex else (kw, _drop_flag(ex, r"--ema [\d.]+")),
    "no_adjacency_bias": lambda kw, ex: (None if not kw.get("adjacency_bias")
                                         else ({**kw, "adjacency_bias": False}, ex)),
}


def jobs_ablation(seeds, save_pred: bool) -> dict[str, list[str]]:
    cfg = json.loads((ROOT / "config" / "tern.json").read_text())["full_history"]
    lines = []
    for ds in DATASETS:
        for h in HORIZONS:
            kwargs, extra = per_horizon(cfg[ds], h)
            for name, fn in ABLATIONS.items():
                res = fn(kwargs, extra)
                if res is None:
                    continue
                kw, ex = res
                lines += [command(ds, h, s, "tern", f"abl_{name}", kw, ex) for s in seeds]
    return {"ablation": lines}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=["tern", "baselines", "ablation"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--save_pred", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    files = {"tern": jobs_tern, "baselines": jobs_baselines, "ablation": jobs_ablation}[args.suite](seeds, args.save_pred)
    (ROOT / "jobs").mkdir(exist_ok=True)
    for name, lines in files.items():
        path = ROOT / "jobs" / f"{name}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"{len(lines):4d} jobs -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

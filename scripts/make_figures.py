#!/usr/bin/env python
"""Figures of the paper.

    python scripts/make_figures.py qualitative --dataset japan --horizon 5 --regions 18,40 --tags tern_full,PatchTST,DLinear
    python scripts/make_figures.py timescales  --tag tern_full
The qualitative figure needs preds/<dataset>/<tag>/h{h}_s{seed}.npz (train.py --save_pred); the time-scale figure
reads the learned time constants stored in results/<dataset>/<tag>/h*_s*.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
                            "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                            "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS_LABEL = {"japan": "Japan-Pref.", "region785": "US-Regions", "state360": "US-States"}
COLORS = {"tern_full": "#c0392b", "tern": "#c0392b", "PatchTST": "#2c7fb8", "DLinear": "#7fbf7b",
          "iTransformer": "#8856a7", "TimeMixer": "#e6ab02", "TimesNet": "#666666"}


def qualitative(args) -> None:
    tags = args.tags.split(",")
    regions = [int(r) for r in args.regions.split(",")]
    runs = {t: np.load(ROOT / "preds" / args.dataset / t / f"h{args.horizon}_s{args.seed}.npz") for t in tags}
    d0 = runs[tags[0]]
    lo, hi = d0["min"], d0["max"]
    denorm = lambda y: y * (hi - lo) + lo  # noqa: E731
    weeks = np.arange(d0["y_true"].shape[0])
    fig, axes = plt.subplots(2, len(regions), figsize=(3.3 * len(regions), 2.85), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for j, r in enumerate(regions):
        ax, ax_err = axes[0, j], axes[1, j]
        y_true = denorm(d0["y_true"])[:, r]
        ax.plot(weeks, y_true, color="k", lw=1.2, label="observed")
        for t, d in runs.items():
            y_pred = denorm(d["y_pred"])[:, r]
            ax.plot(weeks, y_pred, color=COLORS.get(t), lw=1.0, alpha=0.9, label="ours" if t == tags[0] else t)
            ax_err.plot(weeks, np.abs(y_pred - y_true), color=COLORS.get(t), lw=0.9, alpha=0.9)
        ax.set_title(f"region {r}, $h={args.horizon}$")
        ax_err.set_xlabel("test week")
        if j == 0:
            ax.set_ylabel("cases")
            ax.legend(frameon=False, loc="upper left", handlelength=1.4, labelspacing=0.25, borderaxespad=0.3)
            ax_err.set_ylabel("abs. error")
    fig.tight_layout(pad=0.4)
    out = ROOT / args.out if args.out else ROOT / "figures" / f"qualitative_{args.dataset}_h{args.horizon}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def timescales(args) -> None:
    by_ds: dict[str, list[np.ndarray]] = {}
    for f in sorted((ROOT / "results").glob(f"*/{args.tag}/h*_s*.json")):
        r = json.loads(f.read_text())
        if r.get("timescales"):
            taus = np.concatenate([np.asarray(t).reshape(-1) for t in r["timescales"]])
            by_ds.setdefault(r["dataset"], []).append(taus)
    if not by_ds:
        raise SystemExit("no learned time constants found")
    order = [d for d in DS_LABEL if d in by_ds]
    fig, axes = plt.subplots(1, len(order), figsize=(7.0, 1.55))
    bins = np.logspace(np.log10(0.8), np.log10(24), 26)
    for ax, ds in zip(np.atleast_1d(axes), order, strict=True):
        taus = np.concatenate(by_ds[ds])
        ax.axvspan(1.0, args.init_max, color="0.88", lw=0)
        ax.hist(taus, bins=bins, color="#4e62aa", alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 24)
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
        ax.set_title(DS_LABEL[ds], loc="left", fontsize=7.5, pad=1.5)
        ax.set_xlabel(r"time constant $\tau$ [weeks]")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("channels")
    fig.tight_layout(pad=0.3)
    out = ROOT / args.out if args.out else ROOT / "figures" / "timescales.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    medians = {d: round(float(np.median(np.concatenate(v))), 1) for d, v in by_ds.items()}
    print("wrote", out, "| median tau per dataset:", medians)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="figure", required=True)
    q = sub.add_parser("qualitative")
    q.add_argument("--dataset", default="japan")
    q.add_argument("--horizon", type=int, default=5)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--regions", default="18,40", help="column indices (18 and 40 are Tokyo and Osaka in japan.txt)")
    q.add_argument("--tags", default="tern_full,PatchTST,DLinear", help="first tag is the proposed model")
    q.add_argument("--out", default="")
    t = sub.add_parser("timescales")
    t.add_argument("--tag", default="tern_full")
    t.add_argument("--init_max", type=float, default=20.0, help="upper end of the initialisation range (weeks)")
    t.add_argument("--out", default="")
    args = ap.parse_args()
    (qualitative if args.figure == "qualitative" else timescales)(args)


if __name__ == "__main__":
    main()

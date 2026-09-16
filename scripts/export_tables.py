#!/usr/bin/env python
"""Export the two LaTeX tables of the paper from results/<dataset>/<tag>/h{h}_s{seed}.json.

    python scripts/export_tables.py --out tables
Table 1 (main): naive references, reported epidemic GNNs (data/published/epignn_table2.json, optional), the EpiGNN
re-run, the Time-Series-Library baselines, zero-shot foundation models and TERN in both regimes; RMSE / PCC pooled
and averaged over lead times {3, 5, 10, 15} and seeds. Table 2 (ablation): abl_* tags versus tern_full on seeds 0-2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["japan", "region785", "state360"]
DS_LABEL = {"japan": "Japan-Pref.", "region785": "US-Regions", "state360": "US-States"}
HORIZONS = [3, 5, 10, 15]
CONTAMINATED = ("region785", "state360")  # US series are in the pretraining corpus of the foundation models

MAIN_ROWS = [  # (group, label, context, tag or published name, ranked, median over seeds)
    ("Naive baselines", "Seasonal naive (52 wk)", "1 season", "SeasonalNaive52", True, False),
    ("Naive baselines", "Climatology (2 seasons)", "2 seasons", "Climatology2", True, False),
    ("Epidemic-specific (reported)", r"Cola-GNN~\cite{colagnn}", "20 wk", "published:Cola-GNN", True, False),
    ("Epidemic-specific (reported)", r"EpiGNN~\cite{epignn}", "20 wk", "published:EpiGNN", True, False),
    ("Epidemic-specific (reported)", r"EpiGNN~\cite{epignn} (official code, re-run, seed median)", "20 wk",
     "EpiGNN_rerun", True, True),
    ("General time-series (re-run)", r"DLinear~\cite{dlinear}", "20 wk", "DLinear", True, False),
    ("General time-series (re-run)", r"PatchTST~\cite{patchtst}", "20 wk", "PatchTST", True, False),
    ("General time-series (re-run)", r"iTransformer~\cite{itransformer}", "20 wk", "iTransformer", True, False),
    ("General time-series (re-run)", r"TimeMixer~\cite{timemixer}", "20 wk", "TimeMixer", True, False),
    ("General time-series (re-run)", r"TimesNet~\cite{timesnet}", "20 wk", "TimesNet", True, False),
    ("Foundation models (zero-shot)", r"Chronos-2~\cite{chronos} (zero-shot)", "full", "Chronos2_zs_full", False, False),
    ("Foundation models (zero-shot)", r"TimesFM-3~\cite{timesfm} (zero-shot)", "full", "TimesFM3_zs_full", False, False),
    ("Ours", r"\method{} (window regime)", "20 wk", "tern", True, False),
    ("Ours", r"\method{} (full-history regime)", "full", "tern_full", True, False),
]
ABLATION_ROWS = [  # (tag, label)
    ("abl_rule_gdn2", "GDN-2 rule"), ("abl_rule_delta", "no erase (KDA)"),
    ("abl_decay_none", "no decay"), ("abl_no_phase", "no phase feat."),
    ("abl_no_region_attention", "no region attn."), ("abl_softmax_mixer", "softmax mixer"),
    ("abl_no_seasonal_reference", "no seasonal ref."), ("abl_constant_shrinkage", r"constant $s_h$"),
    ("abl_unweighted_loss", "unweighted loss"), ("abl_no_online_blend", "no online blend"),
    ("abl_no_online_refit", "no online refit"), ("abl_no_polyak", "no Polyak avg."),
    ("abl_no_adjacency_bias", "no adjacency bias"),
]


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for f in results_dir.glob("*/*/h*_s*.json"):
        r = json.loads(f.read_text())
        rows.append({"dataset": r["dataset"], "tag": r["tag"], "seed": r["seed"], "horizon": r["horizon"],
                     "rmse": r["metrics"]["rmse"], "pcc": r["metrics"]["pcc"]})
    return pd.DataFrame(rows)


def horizon_average(df: pd.DataFrame, ds: str, tag: str, seeds=None, median: bool = False):
    """(rmse, pcc) averaged over the four lead times (mean or median over seeds first), or None."""
    d = df[(df.dataset == ds) & (df.tag == tag)]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    if d.empty or d.horizon.nunique() != len(HORIZONS):
        return None
    agg = d.groupby("horizon")[["rmse", "pcc"]].median() if median else d.groupby("horizon")[["rmse", "pcc"]].mean()
    return float(agg.rmse.mean()), float(agg.pcc.mean())


def published(path: Path, name: str, ds: str):
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get(ds, {}).get(name)
    return (sum(entry["rmse"]) / 4, sum(entry["pcc"]) / 4) if entry else None


def fmt(v: float | None, metric: str) -> str:
    return "--" if v is None else (f"{v:.0f}" if metric == "rmse" else f"{v:.3f}")


def mark(cells: dict[int, float | None], lower_better: bool):
    """indices of best / second-best rows (ties at printed precision share the mark)."""
    vals = sorted(((v, i) for i, v in cells.items() if v is not None), reverse=not lower_better)
    if not vals:
        return set(), set()
    metric = "rmse" if lower_better else "pcc"
    best = {i for v, i in vals if fmt(v, metric) == fmt(vals[0][0], metric)}
    rest = [(v, i) for v, i in vals if i not in best]
    second = {i for v, i in rest if rest and fmt(v, metric) == fmt(rest[0][0], metric)}
    return best, second


def main_table(df: pd.DataFrame, published_path: Path) -> str:
    values, ranked = [], []
    for group, label, ctx, key, is_ranked, median in MAIN_ROWS:
        v = {}
        for ds in DATASETS:
            res = published(published_path, key.split(":")[1], ds) if key.startswith("published:") else \
                horizon_average(df, ds, key, median=median)
            if res:
                v[ds] = res
        values.append((group, label, ctx, v))
        ranked.append(is_ranked)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
             r"\caption{Results on the three Cola-GNN influenza benchmarks, averaged over lead times "
             r"$h\in\{3,5,10,15\}$ (pooled RMSE\dn{} / PCC\up{}). Context: history read at forecast time. Best bold, "
             r"second underlined. $^\dagger$Pretraining corpus contains the CDC FluView series of the US datasets "
             r"(shown, not ranked).}", r"\label{tab:main}", r"\vspace{3pt}", r"\begin{tabular}{ll" + "cc" * len(DATASETS) + "}",
             r"\toprule", "Method & Context & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DS_LABEL[d]}}}" for d in DATASETS)
             + r" \\", " & & " + " & ".join(r"RMSE\dn & PCC\up" for _ in DATASETS) + r" \\", r"\midrule"]
    marks = {}
    for ds in DATASETS:
        for m, lower in (("rmse", True), ("pcc", False)):
            cells = {i: (v[ds][0 if m == "rmse" else 1] if ds in v else None)
                     for i, (_, _, _, v) in enumerate(values) if ranked[i]}
            marks[(ds, m)] = mark(cells, lower)
    last_group = None
    for i, (group, label, ctx, v) in enumerate(values):
        if not v:
            continue
        if group != last_group:
            lines.append(rf"\rowcolor{{gray!15}}\multicolumn{{{2 + 2 * len(DATASETS)}}}{{l}}{{\textit{{{group}}}}} \\")
            last_group = group
        cells = []
        for ds in DATASETS:
            for j, m in enumerate(("rmse", "pcc")):
                s = fmt(v[ds][j] if ds in v else None, m)
                if ranked[i] and ds in v:
                    best, second = marks[(ds, m)]
                    s = rf"\best{{{s}}}" if i in best else (rf"\second{{{s}}}" if i in second else s)
                elif not ranked[i] and ds in CONTAMINATED and ds in v:
                    s += r"$^\dagger$"
                cells.append(s)
        lines.append(f"{label} & {ctx} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def ablation_table(df: pd.DataFrame, seeds=(0, 1, 2)) -> str:
    rows = [(lab, {ds: horizon_average(df, ds, tag, seeds) for ds in DATASETS}) for tag, lab in ABLATION_ROWS]
    rows = [r for r in rows if any(r[1].values())]
    rows.append((r"\method{} (full)", {ds: horizon_average(df, ds, "tern_full", seeds) for ds in DATASETS}))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2pt}",
             r"\caption{Ablation study in the full-history regime, removing or replacing one component at a time "
             r"(seeds 0--2, averaged over the four lead times). Bold: best per column. Dashes mark components not used on "
             r"a dataset.}", r"\label{tab:ablation}", r"\vspace{3pt}",
             r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}", r"\toprule",
             "Variant & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DS_LABEL[d]}}}" for d in DATASETS) + r" \\",
             " & " + " & ".join(r"RMSE\dn & PCC\up" for _ in DATASETS) + r" \\", r"\midrule"]
    for k, (label, v) in enumerate(rows):
        if k == len(rows) - 1:
            lines.append(r"\midrule")
        cells = []
        for ds in DATASETS:
            for j, m in enumerate(("rmse", "pcc")):
                col = {i: (r[1][ds][j] if r[1].get(ds) else None) for i, r in enumerate(rows)}
                best, _ = mark(col, m == "rmse")
                s = fmt(v[ds][j] if v.get(ds) else None, m)
                cells.append(rf"\best{{{s}}}" if k in best else s)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--published", default=str(ROOT / "data" / "published" / "epignn_table2.json"),
                    help="optional JSON with the numbers of Table 2 of the EpiGNN paper")
    ap.add_argument("--out", default=str(ROOT / "tables"))
    args = ap.parse_args()
    df = load_results(Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.tex").write_text(main_table(df, Path(args.published)))
    (out / "ablation.tex").write_text(ablation_table(df))
    print(f"wrote {out / 'main.tex'} and {out / 'ablation.tex'} from {len(df)} runs")


if __name__ == "__main__":
    main()

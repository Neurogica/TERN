"""Train and evaluate one model on one Cola-GNN dataset and lead time; one run writes one JSON file.

Window regime (default): the model reads the 20-week window of the protocol.
    python src/train.py --dataset japan --horizon 5 --model tern --model_kwargs '{"d_model": 64}' --seed 0
Full-history regime: the model reads the entire causal history and forecasts at every step (TERN only).
    python src/train.py --dataset japan --horizon 5 --model tern --full_history --online_blend 12 \
        --model_kwargs '{"d_model": 32, "dropout": 0.5, "season_embedding": true}' --scale_weighted_loss

Results go to results/<dataset>/<tag>/h{horizon}_s{seed}.json; with --save_pred the test forecasts (and the gate
traces of TERN) go to preds/<dataset>/<tag>/h{horizon}_s{seed}.npz.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ColaGNNData  # noqa: E402
from models import TERN, build_model  # noqa: E402
from utils import colagnn_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_mse(out: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """Squared error, optionally weighted per region (w has mean 1)."""
    return F.mse_loss(out, y) if w is None else ((out - y) ** 2 * w).mean()


def region_weights(data: ColaGNNData, enabled: bool, device: str) -> torch.Tensor | None:
    """(N,) weights proportional to (max - min)^2, so that the normalised loss matches the pooled RMSE."""
    if not enabled:
        return None
    w = torch.tensor((data.max - data.min) ** 2, dtype=torch.float32, device=device)
    return w / w.mean()


def clip_and_step(model, opt, loss, clip: float) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()


def batches(X, Y, batch: int, shuffle: bool, gen=None, T0=None):
    idx = torch.randperm(len(X), generator=gen) if shuffle else torch.arange(len(X))
    for s in range(0, len(X), batch):
        j = idx[s:s + batch].to(X.device)
        yield X[j], Y[j], (None if T0 is None else T0[j])


def forward(model, x, t0=None):
    return model(x, None, t0) if isinstance(model, TERN) and t0 is not None else model(x)


def online_alpha(out_upto: torch.Tensor, x0: torch.Tensor, t_now: int, h: int, period: int, window: int,
                 w: torch.Tensor | None) -> float:
    """Convex weight alpha in {0, 0.1, ..., 1} for alpha * naive + (1 - alpha) * model, chosen on the last
    `window` origins whose targets are observed at t_now (naive = x[t + h - period])."""
    origins = torch.arange(max(period - h, t_now - h - window + 1), t_now - h + 1, device=x0.device)
    if len(origins) < 4:
        return 0.0
    tgt = origins + h
    model_f, naive_f, y = out_upto[origins], x0[tgt - period], x0[tgt]
    losses = [weighted_mse(a / 10 * naive_f + (1 - a / 10) * model_f, y, w).item() for a in range(11)]
    return int(np.argmin(losses)) / 10


@torch.no_grad()
def gate_traces(model: TERN, x: torch.Tensor, feats=None, batch: int = 64) -> dict:
    """Mean erase strength, write gate and decay of the first block, shape (n, L, N)."""
    model.set_record_gates(True)
    model.eval()
    out = {"gamma": [], "beta": [], "alpha": []}
    n, L, N = x.shape
    for s in range(0, n, batch):
        xb = x[s:s + batch]
        model(xb, None if feats is None else feats[s:s + batch])
        g = model.blocks[0].mixer.last_gates
        b = xb.shape[0]
        for k in ("gamma", "beta"):
            if g.get(k) is not None:
                out[k].append(g[k].mean(-1).view(b, N, L).permute(0, 2, 1).cpu().numpy())
        out["alpha"].append(g["alpha"].mean((-1, -2)).view(b, N, L).permute(0, 2, 1).cpu().numpy())
    model.set_record_gates(False)
    return {f"gate_{k}": np.concatenate(v) for k, v in out.items() if v}


def write_result(path: Path, args, name: str, model, data: ColaGNNData, y_true: np.ndarray, pred: np.ndarray,
                 best_epoch: int, epochs_run: int, best_val: float, test_loss: float, train_time: float,
                 extra: dict | None = None) -> dict:
    metrics = colagnn_metrics(data.denorm(y_true), data.denorm(pred))
    kwargs = {k: v for k, v in json.loads(args.model_kwargs).items()}
    res = {"dataset": args.dataset, "model": args.model, "tag": name, "seed": args.seed, "horizon": args.horizon,
           "window": -1 if args.full_history else args.window, "full_history": bool(args.full_history),
           "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "best_epoch": best_epoch, "epochs_run": epochs_run, "val_mse": best_val, "test_mse_norm": test_loss,
           "train_time_s": train_time, "metrics": metrics, "model_kwargs": kwargs,
           "train_args": {k: v for k, v in vars(args).items() if k != "model_kwargs"}, **(extra or {})}
    if isinstance(model, TERN):
        res["timescales"] = [t.cpu().tolist() for t in model.timescales()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({"out": str(path), "h": args.horizon, "best_epoch": best_epoch,
                      **{k: round(v, 4) for k, v in metrics.items()}, "time_s": round(train_time, 1)}))
    return res


def save_predictions(path: Path, y_true, pred, x_last, data: ColaGNNData, gates: dict, **more) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_true=y_true, y_pred=pred, x_last=x_last, min=data.min, max=data.max, **gates, **more)


# ---------------------------------------------------------------------------------------------------
# window regime
# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_windows(model, X, Y, batch: int, T0=None, w=None):
    model.eval()
    preds, loss, n = [], 0.0, 0
    for xb, yb, t0b in batches(X, Y, batch, shuffle=False, T0=T0):
        out = forward(model, xb, t0b)[:, -1, :]
        se = (out - yb) ** 2
        loss += (se * w).sum().item() if w is not None else se.sum().item()
        n += yb.numel()
        preds.append(out.float().cpu())
    return torch.cat(preds).numpy(), loss / n


def train_window(args, data: ColaGNNData, model, name: str, out_path: Path, run_id: str) -> None:
    Xtr, Ytr = data.train
    Xva, Yva = data.val
    Xte, Yte = data.test
    use_t0 = isinstance(model, TERN) and model.season is not None
    T0 = {s: (data.window_starts(s) if use_t0 else None) for s in ("train", "val", "test")}
    w = region_weights(data, args.scale_weighted_loss, args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator().manual_seed(args.seed)

    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for xb, yb, t0b in batches(Xtr, Ytr, args.batch, shuffle=True, gen=gen, T0=T0["train"]):
            loss = weighted_mse(forward(model, xb, t0b)[:, -1, :], yb, w)
            clip_and_step(model, opt, loss, args.clip)
        _, val_loss = predict_windows(model, Xva, Yva, args.batch, T0["val"], w)
        if val_loss < best_val - 1e-12:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if not args.quiet and epoch % 50 == 0:
            print(f"epoch {epoch:4d}  train {loss.item():.5f}  val {val_loss:.5f}  best {best_val:.5f}@{best_epoch}",
                  flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    if args.refit_trainval > 0:  # continue on train + val windows at a reduced learning rate, no more selection
        n_more = max(1, int(round(args.refit_trainval * (best_epoch + 1))))
        Xtv, Ytv = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])
        T0tv = torch.cat([T0["train"], T0["val"]]) if use_t0 else None
        opt2 = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3, weight_decay=args.wd)
        for _ in range(n_more):
            model.train()
            for xb, yb, t0b in batches(Xtv, Ytv, args.batch, shuffle=True, gen=gen, T0=T0tv):
                clip_and_step(model, opt2, weighted_mse(forward(model, xb, t0b)[:, -1, :], yb, w), args.clip)
    train_time = time.time() - t_start
    pred, test_loss = predict_windows(model, Xte, Yte, args.batch, T0["test"])
    y_true = Yte.float().cpu().numpy()
    write_result(out_path, args, name, model, data, y_true, pred, best_epoch, epoch + 1, best_val, test_loss,
                 train_time)
    if args.save_pred:
        gates = gate_traces(model, Xte) if isinstance(model, TERN) else {}
        save_predictions(ROOT / "preds" / args.dataset / name / f"{run_id}.npz", y_true, pred,
                         Xte[:, -1, :].cpu().numpy(), data, gates)


# ---------------------------------------------------------------------------------------------------
# full-history regime
# ---------------------------------------------------------------------------------------------------
def train_full_history(args, data: ColaGNNData, model: TERN, name: str, out_path: Path, run_id: str) -> None:
    h = args.horizon
    x_all, sets = data.stream(h)
    tr_pos, tr_tgt = sets["train"]
    va_pos, va_tgt = sets["val"]
    te_pos, te_tgt = sets["test"]
    n_train, n_val = data.n_train, data.n_val
    feats_all = model.climatology_features(x_all)
    feats = (lambda a, b: None) if feats_all is None else (lambda a, b: feats_all[:, a:b])
    w = region_weights(data, args.scale_weighted_loss, args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    ema_model = copy.deepcopy(model) if args.ema > 0 else None
    eval_model = ema_model if ema_model is not None else model

    def ema_update():
        with torch.no_grad():
            for pe, p in zip(ema_model.parameters(), model.parameters(), strict=True):
                pe.mul_(args.ema).add_(p.detach(), alpha=1 - args.ema)
            for be, b in zip(ema_model.buffers(), model.buffers(), strict=True):
                be.copy_(b)

    def evaluate(upto: int, pos, tgt):
        eval_model.eval()
        with torch.no_grad():
            pred = eval_model(x_all[:, :upto], feats(0, upto), 0)[0][pos]
            return pred, weighted_mse(pred, x_all[0, tgt], w).item()

    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        out = model(x_all[:, :n_train], feats(0, n_train), 0)[0]  # (n_train, N)
        loss = weighted_mse(out[tr_pos], x_all[0, tr_tgt], w)
        clip_and_step(model, opt, loss, args.clip)
        if ema_model is not None:
            ema_update()
        if epoch % max(1, args.val_every) and epoch != args.epochs - 1:
            bad += 1  # skipped validation epochs count towards patience
            continue
        _, val_loss = evaluate(n_val, va_pos, va_tgt)
        if val_loss < best_val - 1e-12:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
        else:
            bad += 1
        if not args.quiet and epoch % 50 == 0:
            print(f"epoch {epoch:4d}  train {loss.item():.5f}  val {val_loss:.5f}  best {best_val:.5f}@{best_epoch}",
                  flush=True)
        if bad >= args.patience:
            break
    eval_model.load_state_dict(best_state)
    model = eval_model
    if args.refit_trainval > 0:
        n_more = max(1, int(round(args.refit_trainval * (best_epoch + 1))))
        opt2 = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3, weight_decay=args.wd)
        tv_pos, tv_tgt = torch.cat([tr_pos, va_pos]), torch.cat([tr_tgt, va_tgt])
        for _ in range(n_more):
            model.train()
            out = model(x_all[:, :n_val], feats(0, n_val), 0)[0]
            clip_and_step(model, opt2, weighted_mse(out[tv_pos], x_all[0, tv_tgt], w), args.clip)

    alphas: list[float] = []
    if args.online_refit > 0:
        # rolling forecast: before every test origin take gradient steps on all targets observed so far
        opt3 = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3, weight_decay=args.wd)
        all_pos, all_tgt = torch.cat([tr_pos, va_pos, te_pos]), torch.cat([tr_tgt, va_tgt, te_tgt])
        preds = []
        for j in range(len(te_pos)):
            t_now = int(te_pos[j])
            observed = all_tgt <= t_now
            for _ in range(args.online_refit):
                model.train()
                out = model(x_all[:, :t_now + 1], feats(0, t_now + 1), 0)[0]
                clip_and_step(model, opt3, weighted_mse(out[all_pos[observed]], x_all[0, all_tgt[observed]], w),
                              args.clip)
            model.eval()
            with torch.no_grad():
                out = model(x_all[:, :t_now + 1], feats(0, t_now + 1), 0)[0]
                p_t = out[t_now]
                if args.online_blend > 0 and t_now + h - args.blend_period >= 0:
                    a = online_alpha(out, x_all[0], t_now, h, args.blend_period, args.online_blend, w)
                    p_t = a * x_all[0, t_now + h - args.blend_period] + (1 - a) * p_t
                    alphas.append(a)
                preds.append(p_t)
        pred = torch.stack(preds)
    else:
        pred, _ = evaluate(data.n, te_pos, te_tgt)
        if args.online_blend > 0:  # blend with the seasonal naive using only targets observed at the origin
            model.eval()
            with torch.no_grad():
                out_all = model(x_all, feats(0, data.n), 0)[0]
            blended = []
            for j in range(len(te_pos)):
                t_now, p_t = int(te_pos[j]), pred[j]
                if t_now + h - args.blend_period >= 0:
                    a = online_alpha(out_all[:t_now + 1], x_all[0], t_now, h, args.blend_period, args.online_blend, w)
                    p_t = a * x_all[0, t_now + h - args.blend_period] + (1 - a) * p_t
                    alphas.append(a)
                blended.append(p_t)
            pred = torch.stack(blended)
    train_time = time.time() - t_start
    test_loss = F.mse_loss(pred, x_all[0, te_tgt]).item()
    y_true, pred = x_all[0, te_tgt].cpu().numpy(), pred.cpu().numpy()
    extra = {"online_blend_alpha_mean": float(np.mean(alphas))} if alphas else {}
    write_result(out_path, args, name, model, data, y_true, pred, best_epoch, epoch + 1, best_val, test_loss,
                 train_time, extra)
    if args.save_pred:
        save_predictions(ROOT / "preds" / args.dataset / name / f"{run_id}.npz", y_true, pred,
                         x_all[0, te_pos].cpu().numpy(), data, gate_traces(model, x_all, feats_all),
                         test_pos=te_pos.cpu().numpy())


# ---------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="japan", choices=["japan", "region785", "state360"])
    ap.add_argument("--horizon", type=int, default=5, help="lead time h: the target is x[t+h]")
    ap.add_argument("--window", type=int, default=20, help="input window of the protocol")
    ap.add_argument("--full_history", action="store_true",
                    help="read the entire causal history and forecast at every step (TERN only)")
    ap.add_argument("--model", default="tern", help="tern or a Time-Series-Library model name")
    ap.add_argument("--model_kwargs", default="{}", help="JSON dict passed to the model constructor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clipping (0 = off)")
    ap.add_argument("--val_every", type=int, default=1, help="evaluate the validation loss every k epochs")
    ap.add_argument("--ema", type=float, default=0.0, help="Polyak averaging of the weights used for evaluation")
    ap.add_argument("--scale_weighted_loss", action="store_true",
                    help="weight each region's squared error by (max-min)^2 so the loss matches the pooled RMSE")
    ap.add_argument("--refit_trainval", type=float, default=0.0,
                    help="after early stopping, continue on train+val for this fraction of the best epoch count")
    ap.add_argument("--online_refit", type=int, default=0,
                    help="full history: gradient steps on all observed targets before every test origin")
    ap.add_argument("--online_blend", type=int, default=0,
                    help="full history: blend with the seasonal naive x[t+h-period]; the weight is chosen on the "
                         "last W = this many origins whose targets are observed (0 = off)")
    ap.add_argument("--blend_period", type=int, default=52)
    ap.add_argument("--tag", default="", help="result sub-directory (default: the model name)")
    ap.add_argument("--out", default="", help="explicit output JSON path")
    ap.add_argument("--save_pred", action="store_true", help="save test forecasts (and gate traces) as npz")
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "colagnn"))
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    kwargs = json.loads(args.model_kwargs)
    name = args.tag or args.model
    run_id = f"h{args.horizon}_s{args.seed}"
    out_path = Path(args.out) if args.out else Path(args.results_dir) / args.dataset / name / f"{run_id}.json"
    data = ColaGNNData(args.dataset, args.data_dir, window=args.window, horizon=args.horizon, device=args.device)
    if kwargs.get("adjacency_bias"):
        kwargs["adjacency"] = data.adj.tolist()
    if args.full_history:
        if args.model.lower() != "tern":
            raise SystemExit("--full_history is only implemented for TERN")
        kwargs.update(full_history=True, horizon=args.horizon)
    set_seed(args.seed)
    model = build_model(args.model, seq_len=args.window, pred_len=1, n_vars=data.m, **kwargs).to(args.device)
    if args.full_history:
        train_full_history(args, data, model, name, out_path, run_id)
    else:
        train_window(args, data, model, name, out_path, run_id)


if __name__ == "__main__":
    main()

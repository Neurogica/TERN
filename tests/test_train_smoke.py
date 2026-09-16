import json
import sys
from pathlib import Path

import numpy as np

import data.colagnn as colagnn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train  # noqa: E402


def _run(synthetic_dataset, tmp_path, extra, kwargs=None):
    colagnn.DATASETS["japan"] = ("japan.txt", "japan-adj.txt", 4)
    out = tmp_path / "results" / "japan" / "t" / "h5_s0.json"
    kwargs = {"d_model": 8, "n_layers": 1, "n_heads": 2, **(kwargs or {})}
    argv = ["--dataset", "japan", "--horizon", "5", "--seed", "0", "--epochs", "3", "--patience", "5",
            "--model", "tern", "--model_kwargs", json.dumps(kwargs),
            "--tag", "t", "--data_dir", str(synthetic_dataset), "--out", str(out), "--device", "cpu", "--quiet",
            "--save_pred"] + extra
    train.main(argv)
    return json.loads(out.read_text())


def test_window_regime_smoke(synthetic_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    res = _run(synthetic_dataset, tmp_path, ["--refit_trainval", "0.5"])
    assert set(res["metrics"]) >= {"rmse", "pcc"} and np.isfinite(res["metrics"]["rmse"])
    assert res["window"] == 20 and res["full_history"] is False and res["epochs_run"] == 3
    pred = np.load(tmp_path / "preds" / "japan" / "t" / "h5_s0.npz")
    assert pred["y_pred"].shape == (48, 4) and "gate_gamma" in pred.files


def test_full_history_regime_smoke(synthetic_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    res = _run(synthetic_dataset, tmp_path, ["--full_history", "--scale_weighted_loss", "--online_blend", "4",
                                             "--online_refit", "1", "--ema", "0.9", "--val_every", "2"],
               kwargs={"season_embedding": True, "climatology_seasons": 1, "climatology_residual": True,
                       "adjacency_bias": True})
    assert res["full_history"] is True and np.isfinite(res["metrics"]["rmse"])
    assert 0.0 <= res["online_blend_alpha_mean"] <= 1.0
    assert len(res["timescales"]) == 1
    pred = np.load(tmp_path / "preds" / "japan" / "t" / "h5_s0.npz")
    assert pred["y_pred"].shape == (48, 4) and pred["test_pos"].shape == (48,)

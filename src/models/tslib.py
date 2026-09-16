"""Uniform wrapper around Time-Series-Library models (cloned into ext_repos/tslib by scripts/clone_ext_repos.sh)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.nn as nn

TSLIB_MODELS = ["DLinear", "PatchTST", "iTransformer", "TimeMixer", "TimesNet"]

_TSLIB = Path(__file__).resolve().parents[2] / "ext_repos" / "tslib"
_CLASHING = ("models", "layers", "utils")  # top-level packages of the library that shadow ours
_CACHE: dict[str, type] = {}


def _import_tslib_model(name: str):
    """Import `models.<name>.Model` from the library in an isolated module namespace.

    The library uses the top-level packages `models`, `layers` and `utils`, which collide with the packages of this
    repository, so the corresponding `sys.modules` entries are swapped out during the import and restored afterwards."""
    if name in _CACHE:
        return _CACHE[name]
    if not _TSLIB.exists():
        raise FileNotFoundError(f"{_TSLIB} not found; run scripts/clone_ext_repos.sh first")
    ours = {k: v for k, v in sys.modules.items() if k.split(".")[0] in _CLASHING}
    for k in ours:
        del sys.modules[k]
    sys.path.insert(0, str(_TSLIB))
    try:
        model_cls = importlib.import_module(f"models.{name}").Model
    finally:
        sys.path.remove(str(_TSLIB))
        for k in [k for k in sys.modules if k.split(".")[0] in _CLASHING]:
            del sys.modules[k]
        sys.modules.update(ours)
    _CACHE[name] = model_cls
    return model_cls


class TSLibModel(nn.Module):
    """(B, L, N) -> (B, pred_len, N) for any long-term-forecasting model of the Time-Series-Library."""

    def __init__(self, name: str, seq_len: int, pred_len: int, n_vars: int, d_model: int = 64, n_heads: int = 4,
                 e_layers: int = 2, d_layers: int = 1, d_ff: int = 128, dropout: float = 0.1, factor: int = 3,
                 moving_avg: int = 25, top_k: int = 5, num_kernels: int = 6, patch_len: int = 4, stride: int = 2,
                 down_sampling_layers: int = 1, down_sampling_window: int = 2, seg_len: int = 4, **extra):
        super().__init__()
        cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=seq_len, label_len=0, pred_len=pred_len,
                              enc_in=n_vars, dec_in=n_vars, c_out=n_vars, d_model=d_model, n_heads=n_heads,
                              e_layers=e_layers, d_layers=d_layers, d_ff=d_ff, dropout=dropout, factor=factor,
                              activation="gelu", embed="timeF", freq="w", num_class=0, moving_avg=moving_avg,
                              top_k=top_k, num_kernels=num_kernels, channel_independence=1,
                              decomp_method="moving_avg", use_norm=1, down_sampling_layers=down_sampling_layers,
                              down_sampling_method="avg", down_sampling_window=down_sampling_window,
                              seg_len=seg_len, patch_len=patch_len, stride=stride, output_attention=False,
                              distil=True, p_hidden_dims=[128, 128], p_hidden_layers=2, d_conv=4, expand=2,
                              seasonal_patterns="Monthly", inverse=False, mask_rate=0.25, anomaly_ratio=0.25,
                              use_gpu=True, gpu=0, gpu_type="cuda", use_multi_gpu=False, devices="0",
                              num_workers=0, itr=1, train_epochs=10, batch_size=32, patience=3,
                              learning_rate=1e-3, des="test", loss="MSE", lradj="type1", use_amp=False,
                              use_dtw=False, augmentation_ratio=0, features="M", target="OT", model=name,
                              data="custom", root_path="", data_path="", checkpoints="", model_id="",
                              is_training=1, **extra)
        self.pred_len = pred_len
        self.model = _import_tslib_model(name)(cfg)

    def forward(self, x):
        return self.model(x, None, None, None)[:, -self.pred_len:, :]

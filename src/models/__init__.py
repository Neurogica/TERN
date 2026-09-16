from .tern import TERN, DeltaMemoryMixer, phase_features
from .tslib import TSLIB_MODELS, TSLibModel

__all__ = ["TERN", "DeltaMemoryMixer", "phase_features", "TSLibModel", "TSLIB_MODELS", "build_model"]


def build_model(name: str, seq_len: int, pred_len: int, n_vars: int, **kw):
    """`tern` -> TERN; any Time-Series-Library model name -> TSLibModel."""
    if name.lower() == "tern":
        return TERN(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars, **kw)
    if name in TSLIB_MODELS:
        kw = {k: v for k, v in kw.items() if k not in ("adjacency", "adjacency_bias")}
        return TSLibModel(name, seq_len=seq_len, pred_len=pred_len, n_vars=n_vars, **kw)
    raise ValueError(f"unknown model {name!r}; choose 'tern' or one of {TSLIB_MODELS}")

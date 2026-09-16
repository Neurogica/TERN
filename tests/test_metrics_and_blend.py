import importlib.util
import sys
from pathlib import Path

import numpy as np

from utils import colagnn_metrics, pearson

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_against_numpy():
    rng = np.random.default_rng(1)
    y = rng.normal(100, 30, (40, 6))
    p = y + rng.normal(0, 5, y.shape)
    m = colagnn_metrics(y, p)
    assert np.isclose(m["rmse"], np.sqrt(np.mean((p - y) ** 2)))
    assert np.isclose(m["mae"], np.mean(np.abs(p - y)))
    assert np.isclose(m["pcc"], np.corrcoef(y.ravel(), p.ravel())[0, 1])
    assert np.isclose(m["rmse_states"], np.mean(np.sqrt(np.mean((p - y) ** 2, axis=0))))
    assert 0.9 < m["pcc_states"] <= 1.0


def test_pearson_degenerate_is_nan():
    assert np.isnan(pearson(np.ones(5), np.arange(5.0)))


def _load_posthoc_blend():
    spec = importlib.util.spec_from_file_location("posthoc_blend", ROOT / "scripts" / "posthoc_blend.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["posthoc_blend"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_online_blend_prefers_the_better_forecast():
    pb = _load_posthoc_blend()
    rng = np.random.default_rng(0)
    T, N, h = 260, 3, 5
    x = 100 + 50 * np.sin(2 * np.pi * np.arange(T) / 52)[:, None] * np.ones((1, N))
    n_test = 78
    y_true = x[T - n_test:]
    bad_model = y_true + rng.normal(0, 40, y_true.shape)  # much worse than the (perfect) seasonal naive
    out = pb.blend("japan", y_true, bad_model, x, np.ones(N), h, window=12)
    early = out[:h + 3]  # not enough observed targets yet: unchanged model forecast
    assert np.allclose(early, bad_model[:h + 3])
    late_err = np.abs(out[30:] - y_true[30:]).mean()
    assert late_err < 0.5 * np.abs(bad_model[30:] - y_true[30:]).mean()

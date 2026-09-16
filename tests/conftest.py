import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A small dataset in the Cola-GNN file format: 160 weeks x 4 regions with a 52-week cycle, plus adjacency."""
    rng = np.random.default_rng(0)
    t = np.arange(160)
    base = 100 * (1 + np.sin(2 * np.pi * t / 52))[:, None] * np.array([1.0, 2.0, 0.5, 3.0])[None, :]
    x = np.clip(base + rng.normal(0, 5, base.shape), 0, None)
    np.savetxt(tmp_path / "japan.txt", x, delimiter=",", fmt="%.1f")
    adj = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    np.savetxt(tmp_path / "japan-adj.txt", adj, delimiter=",", fmt="%d")
    return tmp_path

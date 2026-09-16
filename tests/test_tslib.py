"""Time-Series-Library baselines load through the isolated import without shadowing our own packages."""
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "ext_repos" / "tslib").exists(),
                                reason="ext_repos/tslib not cloned (scripts/clone_ext_repos.sh)")


def test_tslib_and_tern_coexist():
    import data
    import models
    import utils
    from models import build_model

    dlinear = build_model("DLinear", seq_len=20, pred_len=1, n_vars=5)
    assert Path(models.__file__).parent == ROOT / "src" / "models"
    assert Path(utils.__file__).parent == ROOT / "src" / "utils"
    assert Path(data.__file__).parent == ROOT / "src" / "data"
    tern = build_model("tern", seq_len=20, pred_len=1, n_vars=5, d_model=16, n_layers=1, n_heads=2)
    x = torch.randn(2, 20, 5)
    assert dlinear(x).shape == tern(x).shape == (2, 1, 5)


@pytest.mark.parametrize("name", ["PatchTST", "iTransformer"])
def test_tslib_forward(name):
    from models import build_model

    model = build_model(name, seq_len=20, pred_len=1, n_vars=5)
    assert model(torch.randn(2, 20, 5)).shape == (2, 1, 5)

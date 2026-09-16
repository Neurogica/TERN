import pytest
import torch

from models import TERN, build_model, phase_features
from models.tern import DeltaMemoryMixer


@pytest.mark.parametrize("rule", ["eda", "delta", "gdn2", "gla"])
@pytest.mark.parametrize("decay", ["channel", "scalar", "none"])
def test_mixer_shapes_and_gates(rule, decay):
    torch.manual_seed(0)
    m = DeltaMemoryMixer(16, n_heads=4, rule=rule, decay=decay)
    m.record_gates = True
    u, phi = torch.randn(3, 12, 16), torch.randn(3, 12, 3)
    out = m(u, phi)
    assert out.shape == (3, 12, 16) and torch.isfinite(out).all()
    g = m.last_gates
    assert ((g["alpha"] > 0) & (g["alpha"] <= 1)).all()
    if rule in ("eda", "delta"):
        assert ((g["beta"] >= 0) & (g["beta"] <= 1)).all()
    if rule == "eda":
        assert ((g["gamma"] >= 0) & (g["gamma"] <= 1)).all()
    taus = m.timescales()
    if decay == "none":
        assert taus.numel() == 0
    else:
        assert taus.shape == (4, 4 if decay == "channel" else 1)
        assert taus.min() >= 0.99 and taus.max() <= 20.01  # initialised on [1, l_max] with l_max = 20


def test_mixer_is_causal():
    torch.manual_seed(0)
    m = DeltaMemoryMixer(8, n_heads=2).eval()
    u, phi = torch.randn(1, 10, 8), torch.randn(1, 10, 3)
    out = m(u, phi)
    u2, phi2 = u.clone(), phi.clone()
    u2[:, 7:] += 5.0  # perturb the future
    phi2[:, 7:] += 1.0
    out2 = m(u2, phi2)
    assert torch.allclose(out[:, :7], out2[:, :7], atol=1e-5)
    assert not torch.allclose(out[:, 7:], out2[:, 7:])


def test_phase_features():
    x = torch.tensor([[[1.0], [2.0], [4.0], [4.0]]])  # (B=1, L=4, N=1)
    phi = phase_features(x, eps=0.5)
    assert phi.shape == (1, 4, 1, 3)
    assert torch.allclose(phi[0, :, 0, 0], torch.tensor([0.0, 1.0, 2.0, 0.0]))  # first difference
    assert torch.allclose(phi[0, :, 0, 1], torch.tensor([0.0, 1.0, 1.0, -2.0]))  # second difference
    assert torch.allclose(phi[0, :, 0, 2], torch.tensor([0.0, 1 / 1.5, 2 / 2.5, 0.0]))  # relative growth


@pytest.mark.parametrize("head", ["flatten", "last"])
def test_window_regime_output(head):
    torch.manual_seed(0)
    model = TERN(seq_len=20, pred_len=1, n_vars=5, d_model=16, n_layers=1, n_heads=4, head=head)
    y = model(torch.rand(2, 20, 5))
    assert y.shape == (2, 1, 5) and torch.isfinite(y).all()


def test_full_history_regime_with_seasonal_components():
    torch.manual_seed(0)
    model = TERN(seq_len=20, pred_len=1, n_vars=3, d_model=16, n_layers=1, n_heads=4, full_history=True, horizon=5,
                 season_embedding=True, climatology_seasons=2, climatology_width=2, climatology_residual=True,
                 residual_scale=0.3)
    x = torch.rand(1, 130, 3)
    feats = model.climatology_features(x)
    assert feats.shape == (1, 130, 3, 3)  # two seasonal means + validity fraction
    # climatology channel for k=1 at position t uses x[t + h - 52 +- 2]: strictly past values
    t, lag = 120, 52 - 5
    expected = x[0, t - lag - 2:t - lag + 3].mean(0)
    assert torch.allclose(feats[0, t, :, 0], expected, atol=1e-6)
    assert feats[0, 10, :, 2].max() == 0.0 and feats[0, 129, :, 2].min() == 1.0  # validity fraction
    y = model(x, feats, 0)
    assert y.shape == (1, 130, 3)
    # zero-initialised head: the forecast starts exactly at the climatology
    base = feats[0, ..., :2].mean(-1)
    assert torch.allclose(y[0], base, atol=1e-5)


def test_softmax_mixer_and_registry():
    model = build_model("tern", seq_len=20, pred_len=1, n_vars=3, d_model=16, n_layers=1, n_heads=4,
                       mixer_type="attn", region_attention=False)
    assert model(torch.rand(2, 20, 3)).shape == (2, 1, 3)
    with pytest.raises(ValueError):
        build_model("unknown", seq_len=20, pred_len=1, n_vars=3)

"""TERN: a delta-rule fast-weight memory for epidemic surveillance series.

Every region is mixed over time by an erase-then-delta memory (``DeltaMemoryMixer``), regions are
coupled by one attention layer, and a linear head produces the forecast. Per step ``t`` and head::

    S_t = (I - beta_t k_t k_t^T) (I - gamma_t e_t e_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

* ``alpha_t = exp(-r * m_t)``: channel-wise decay with learned rates ``r = softplus(rho)`` (time constants
  initialised log-uniformly on ``[1, L]`` weeks) and an input-dependent modulation ``m_t`` (1 at init).
* ``e_t``: erase address (normalised, factorised through a small intermediate), ``gamma_t`` its strength.
* ``beta_t``: write strength of the delta rule.
* All gates and the erase address read ``z_t = [u_t; phi_t]``, the block input concatenated with three
  phase features of the normalised series (first difference, second difference, relative growth).

Ablation switches
    rule            'eda' (ours) | 'delta' (Kimi Delta Attention / gated delta rule) |
                    'gdn2' (Gated DeltaNet-2 erase/write gates) | 'gla' (gated linear attention)
    decay           'channel' | 'scalar' | 'none'
    timescale_init  'logspaced' | 'uniform';  learn_timescale: bool
    phase_features  bool;  mixer_type 'delta' | 'attn' (causal softmax attention in the same block)
    region_attention bool; adjacency_bias bool
    season_embedding bool (week-of-year embedding); climatology_seasons K > 0 with climatology_residual
    (the head predicts a correction to the K-season climatology, shrunk by residual_scale)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TERN", "DeltaMemoryMixer", "phase_features"]


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def phase_features(x: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """x: (B, L, N) normalised series -> (B, L, N, 3): [first difference, second difference, relative growth]."""
    d1 = torch.zeros_like(x)
    d1[:, 1:] = x[:, 1:] - x[:, :-1]
    d2 = torch.zeros_like(x)
    d2[:, 1:] = d1[:, 1:] - d1[:, :-1]
    prev = torch.zeros_like(x)
    prev[:, 1:] = x[:, :-1]
    rel = (d1 / (prev.abs() + eps)).clamp(-5.0, 5.0)
    return torch.stack([d1, d2, rel], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ShortConv(nn.Module):
    """Depthwise causal convolution over time (q, k, v paths, as in Mamba / Gated DeltaNet)."""

    def __init__(self, d: int, kernel: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel, groups=d, padding=kernel - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d)
        return self.conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)


# ---------------------------------------------------------------------------------------------------
# one memory step per rule.  S: (B, H, dk, dv); alpha: (B, H, dk) or (B, H, 1); e, k, q: (B, H, dk); v: (B, H, dv)
# ---------------------------------------------------------------------------------------------------
def _read(S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhkv,bhk->bhv", S, q)


def _delta_write(S: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    kS = torch.einsum("bhk,bhkv->bhv", k, S)
    bt = beta[:, :, None, None]
    return S - bt * k.unsqueeze(-1) * kS.unsqueeze(-2) + bt * k.unsqueeze(-1) * v.unsqueeze(-2)


def step_eda(S, alpha, e, gamma, k, v, beta, q):
    """Erase-then-delta: decay, erase along e_t, then the gated delta write at k_t."""
    S = S * alpha.unsqueeze(-1)
    eS = torch.einsum("bhk,bhkv->bhv", e, S)
    S = S - gamma[:, :, None, None] * e.unsqueeze(-1) * eS.unsqueeze(-2)
    S = _delta_write(S, k, v, beta)
    return S, _read(S, q)


def step_delta(S, alpha, k, v, beta, q):
    """Gated delta rule (Kimi Delta Attention with channel-wise decay; Gated DeltaNet with scalar decay)."""
    S = _delta_write(S * alpha.unsqueeze(-1), k, v, beta)
    return S, _read(S, q)


def step_gdn2(S, alpha, k, v, erase_gate, write_gate, q):
    """Gated DeltaNet-2: channel-wise erase and write gates on the key/value."""
    S = S * alpha.unsqueeze(-1)
    e = erase_gate * k
    eS = torch.einsum("bhk,bhkv->bhv", e, S)
    S = S - k.unsqueeze(-1) * eS.unsqueeze(-2) + k.unsqueeze(-1) * (write_gate * v).unsqueeze(-2)
    return S, _read(S, q)


def step_gla(S, alpha, k, v, q):
    """Gated linear attention: additive write, no delta correction."""
    S = S * alpha.unsqueeze(-1) + k.unsqueeze(-1) * v.unsqueeze(-2)
    return S, _read(S, q)


class DeltaMemoryMixer(nn.Module):
    """Erase-then-delta fast-weight memory over time, applied independently to every region."""

    def __init__(self, d_model: int, n_heads: int = 8, rule: str = "eda", decay: str = "channel",
                 phase_features: bool = True, timescale_init: str = "logspaced", learn_timescale: bool = True,
                 l_max: int = 20, tau_min: float = 1.0, tau_max: float | None = None, erase_dim: int = 16,
                 conv_kernel: int = 4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if rule not in ("eda", "delta", "gdn2", "gla"):
            raise ValueError(f"unknown rule {rule}")
        if decay not in ("channel", "scalar", "none"):
            raise ValueError(f"unknown decay {decay}")
        self.d, self.H, self.dk = d_model, n_heads, d_model // n_heads
        self.rule, self.decay, self.phase_features = rule, decay, phase_features
        self.record_gates = False
        self.last_gates: dict = {}
        d, H, dk = self.d, self.H, self.dk
        self.wq, self.wk, self.wv = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.cq, self.ck, self.cv = (ShortConv(d, conv_kernel) for _ in range(3))
        gin = d + (3 if phase_features else 0)  # gate input: block input (+ three phase features)
        # decay: learned per-channel rates r = softplus(rho), time constants 1/r initialised on [tau_min, tau_max]
        if decay != "none":
            self.n_dec = dk if decay == "channel" else 1
            tau_max = float(tau_max or l_max)
            if timescale_init == "logspaced" and self.n_dec > 1:
                taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), self.n_dec)
            else:
                taus = torch.full((self.n_dec,), math.sqrt(tau_min * tau_max))
            rho = torch.tensor([inv_softplus(1.0 / t) for t in taus.tolist()])
            self.rho = nn.Parameter(rho.unsqueeze(0).repeat(H, 1).clone(), requires_grad=learn_timescale)
            self.wm = nn.Linear(gin, H * self.n_dec)
            nn.init.zeros_(self.wm.weight)
            nn.init.constant_(self.wm.bias, inv_softplus(1.0))  # m_t = 1 at initialisation
        # write / erase controls
        if rule in ("eda", "delta"):
            self.wbeta = nn.Linear(gin, H)
        if rule == "eda":
            self.we1 = nn.Linear(gin, H * erase_dim, bias=False)
            self.we2 = nn.Parameter(torch.randn(H, erase_dim, dk) / math.sqrt(erase_dim))
            self.wgamma = nn.Linear(gin, H)
            nn.init.constant_(self.wgamma.bias, -2.0)
            self.erase_dim = erase_dim
        if rule == "gdn2":
            self.w_erase = nn.Linear(gin, H * dk)
            self.w_write = nn.Linear(gin, H * dk)
        # output
        self.onorm = RMSNorm(dk)
        self.wg = nn.Linear(d, d)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, u: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        B, L, d = u.shape
        H, dk = self.H, self.dk
        q = F.normalize(F.silu(self.cq(self.wq(u))).view(B, L, H, dk), dim=-1)
        k = F.normalize(F.silu(self.ck(self.wk(u))).view(B, L, H, dk), dim=-1)
        v = F.silu(self.cv(self.wv(u))).view(B, L, H, dk)
        z = torch.cat([u, phi], -1) if self.phase_features else u
        alpha = beta = gamma = e = eg = wg = None
        if self.decay != "none":
            r = F.softplus(self.rho)  # (H, n_dec)
            m = F.softplus(self.wm(z)).view(B, L, H, self.n_dec)
            alpha = torch.exp(-(r * m))  # (B, L, H, n_dec)
        else:
            alpha = u.new_ones(B, L, H, 1)
        if self.rule in ("eda", "delta"):
            beta = torch.sigmoid(self.wbeta(z))  # (B, L, H)
        if self.rule == "eda":
            e = self.we1(z).view(B, L, H, self.erase_dim)
            e = F.normalize(torch.einsum("blhr,hrk->blhk", e, self.we2), dim=-1)
            gamma = torch.sigmoid(self.wgamma(z))  # (B, L, H)
        if self.rule == "gdn2":
            eg = torch.sigmoid(self.w_erase(z)).view(B, L, H, dk)
            wg = torch.sigmoid(self.w_write(z)).view(B, L, H, dk)
        S = u.new_zeros(B, H, dk, dk)
        outs = []
        for t in range(L):
            kt, vt, qt, at = k[:, t], v[:, t], q[:, t], alpha[:, t]
            if self.rule == "eda":
                S, o_t = step_eda(S, at, e[:, t], gamma[:, t], kt, vt, beta[:, t], qt)
            elif self.rule == "delta":
                S, o_t = step_delta(S, at, kt, vt, beta[:, t], qt)
            elif self.rule == "gdn2":
                S, o_t = step_gdn2(S, at, kt, vt, eg[:, t], wg[:, t], qt)
            else:
                S, o_t = step_gla(S, at, kt, vt, qt)
            outs.append(o_t)
        o = torch.stack(outs, 1)  # (B, L, H, dk)
        if self.record_gates:
            self.last_gates = {"alpha": alpha.detach(), "beta": None if beta is None else beta.detach(),
                               "gamma": None if gamma is None else gamma.detach()}
        o = self.onorm(o) * F.silu(self.wg(u)).view(B, L, H, dk)
        return self.wo(o.reshape(B, L, d))

    def timescales(self) -> torch.Tensor:
        """Learned time constants tau = 1 / softplus(rho), shape (H, n_dec); empty if decay is off."""
        if not hasattr(self, "rho"):
            return torch.zeros(0)
        return 1.0 / F.softplus(self.rho.detach())


class CausalSelfAttention(nn.Module):
    """Softmax-attention mixer used as an ablation of the delta memory (same block otherwise)."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, u: torch.Tensor, phi: torch.Tensor | None = None) -> torch.Tensor:
        L = u.shape[1]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=u.device), 1)
        return self.attn(u, u, u, attn_mask=mask, need_weights=False)[0]


class Block(nn.Module):
    def __init__(self, d_model: int, mixer_type: str = "delta", mlp_ratio: int = 2, dropout: float = 0.1,
                 **mixer_kw):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        if mixer_type == "delta":
            self.mixer = DeltaMemoryMixer(d_model, **mixer_kw)
        elif mixer_type == "attn":
            self.mixer = CausalSelfAttention(d_model, n_heads=mixer_kw.get("n_heads", 8), dropout=dropout)
        else:
            raise ValueError(f"unknown mixer_type {mixer_type}")
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_ratio * d_model, d_model), nn.Dropout(dropout))

    def forward(self, u: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        u = u + self.drop(self.mixer(self.n1(u), phi))
        return u + self.mlp(self.n2(u))


class RegionAttention(nn.Module):
    """One multi-head attention layer across regions at every time step (optional adjacency bias)."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, adjacency=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        if adjacency is not None:  # additive attention bias: row-normalised adjacency with self loops
            a = torch.as_tensor(adjacency, dtype=torch.float32)
            a = a + torch.eye(a.shape[0])
            self.register_buffer("adj", a / a.sum(1, keepdim=True))
            self.adj_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.adj = None

    def forward(self, u: torch.Tensor, B: int, N: int) -> torch.Tensor:  # u: (B*N, L, d)
        L, d = u.shape[1], u.shape[2]
        z = u.view(B, N, L, d).permute(0, 2, 1, 3).reshape(B * L, N, d)
        zn = self.norm(z)
        mask = None if self.adj is None else self.adj_scale * self.adj
        z = z + self.drop(self.attn(zn, zn, zn, attn_mask=mask, need_weights=False)[0])
        return z.view(B, L, N, d).permute(0, 2, 1, 3).reshape(B * N, L, d)


class TERN(nn.Module):
    """Forecaster: per-region normalisation -> embedding -> B x {delta-memory mixer, MLP} -> region attention -> head.

    Window regime (``full_history=False``): input (B, L, N) window, output (B, pred_len, N) with an instance
    normalisation (``revin``) and a flatten or last-step head.
    Full-history regime (``full_history=True``): the whole causal series (1, T, N) is read once and a forecast
    is emitted at every step, output (1, T, N); the optional climatology correction and week-of-year embedding
    live here.
    """

    def __init__(self, seq_len: int, pred_len: int, n_vars: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 8, dropout: float = 0.1, head: str = "flatten", revin: bool = True,
                 full_history: bool = False, horizon: int = 1,
                 # memory
                 rule: str = "eda", decay: str = "channel", timescale_init: str = "logspaced",
                 learn_timescale: bool = True, tau_max: float | None = None, erase_dim: int = 16,
                 phase_features: bool = True, mixer_type: str = "delta", mlp_ratio: int = 2,
                 # coupling
                 region_attention: bool = True, adjacency_bias: bool = False, adjacency=None,
                 # seasonal reference (full-history regime)
                 season_embedding: bool = False, season_period: int = 52,
                 climatology_seasons: int = 0, climatology_width: int = 2, climatology_residual: bool = False,
                 residual_scale: float = 1.0):
        super().__init__()
        self.full_history = full_history
        self.L, self.pred_len, self.N = seq_len, pred_len, n_vars
        if full_history:  # causal per-step head, no window normalisation
            revin, head = False, "last"
        self.revin, self.head_type = revin, head
        self.season_period = season_period
        self.season = nn.Embedding(season_period, d_model) if season_embedding else None
        # climatology channels: for k = 1..K past seasons, the mean of x over +-width weeks around the position
        # one season before the target week (t + horizon - 52 k); all lags are strictly positive, hence causal.
        self.climatology_groups: list[list[int]] = []
        for k in range(1, climatology_seasons + 1):
            centre = season_period * k - horizon
            grp = [centre + o for o in range(-climatology_width, climatology_width + 1) if centre + o >= 1]
            if grp:
                self.climatology_groups.append(grp)
        if climatology_residual and not self.climatology_groups:
            raise ValueError("climatology_residual requires climatology_seasons > 0")
        self.climatology_residual, self.residual_scale = climatology_residual, residual_scale
        self.n_extra = len(self.climatology_groups) + (1 if self.climatology_groups else 0)  # + validity channel
        self.embed = nn.Linear(1 + self.n_extra, d_model)
        self.pos = None if full_history else nn.Parameter(torch.zeros(1, seq_len, d_model))
        if self.pos is not None:
            nn.init.normal_(self.pos, std=0.02)
        mixer_kw = dict(n_heads=n_heads, rule=rule, decay=decay, phase_features=phase_features,
                        timescale_init=timescale_init, learn_timescale=learn_timescale, l_max=seq_len,
                        tau_max=tau_max, erase_dim=erase_dim)
        self.blocks = nn.ModuleList([Block(d_model, mixer_type=mixer_type, mlp_ratio=mlp_ratio, dropout=dropout,
                                           **mixer_kw) for _ in range(n_layers)])
        self.region_attention = (RegionAttention(d_model, n_heads, dropout, adjacency if adjacency_bias else None)
                                 if region_attention else None)
        self.norm = nn.LayerNorm(d_model)
        if head == "flatten":
            self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(seq_len * d_model, pred_len))
        elif head == "last":
            self.head = nn.Linear(d_model, pred_len)
        else:
            raise ValueError(f"unknown head {head}")
        if climatology_residual:  # start exactly at the climatology
            last = [m for m in self.head.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    # ---------------------------------------------------------------------------------------------
    def climatology_features(self, x_full: torch.Tensor) -> torch.Tensor | None:
        """x_full: (B, T, N) -> (B, T, N, K + 1): K smoothed past-season means and the fraction of real inputs.

        Where a season is not yet available the current value is used (persistence fallback)."""
        if not self.climatology_groups:
            return None
        T = x_full.shape[1]

        def shifted(lag: int):
            f = x_full.clone()
            m = torch.zeros_like(x_full)
            if lag < T:
                f[:, lag:] = x_full[:, :-lag]
                m[:, lag:] = 1.0
            return f, m

        feats, valid = [], []
        for grp in self.climatology_groups:
            fs, ms = zip(*[shifted(lag) for lag in grp], strict=True)
            fs, ms = torch.stack(fs, 0), torch.stack(ms, 0)
            cnt = ms.sum(0)
            mean_valid = (fs * ms).sum(0) / cnt.clamp(min=1.0)
            feats.append(torch.where(cnt > 0, mean_valid, x_full))
            valid.append((cnt > 0).float())
        feats.append(torch.stack(valid, -1).mean(-1))
        return torch.stack(feats, -1)

    def forward(self, x: torch.Tensor, feats: torch.Tensor | None = None, t0=0) -> torch.Tensor:
        """x: (B, L, N); feats: optional precomputed climatology features aligned with x;
        t0: absolute index of x[:, 0] in the series (int or (B,) tensor), for the week-of-year embedding."""
        B, L, N = x.shape
        phi = phase_features(x).permute(0, 2, 1, 3).reshape(B * N, L, 3)
        if self.revin:
            mu = x.mean(1, keepdim=True)
            sd = x.std(1, keepdim=True, unbiased=False) + 1e-5
            x = (x - mu) / sd
        inp = x.unsqueeze(-1)
        if self.n_extra:
            if feats is None:
                feats = self.climatology_features(x)
            inp = torch.cat([inp, feats], -1)
        u = self.embed(inp.permute(0, 2, 1, 3).reshape(B * N, L, 1 + self.n_extra))
        if self.season is not None:
            ar = torch.arange(L, device=x.device)
            if torch.is_tensor(t0):
                idx = (t0.view(B, 1) + ar.view(1, L)) % self.season_period
                u = u + self.season(idx).unsqueeze(1).expand(B, N, L, -1).reshape(B * N, L, -1)
            else:
                u = u + self.season((t0 + ar) % self.season_period).unsqueeze(0)
        if self.pos is not None:
            u = u + self.pos
        for blk in self.blocks:
            u = blk(u, phi)
        if self.region_attention is not None:
            u = self.region_attention(u, B, N)
        u = self.norm(u)
        if self.full_history:
            y = self.head(u)  # (B*N, L, pred_len)
            if self.climatology_residual:
                n_k = len(self.climatology_groups)
                base = inp[..., 1:1 + n_k].mean(-1)  # (B, L, N)
                y = self.residual_scale * y + base.permute(0, 2, 1).reshape(B * N, L, 1)
            return y[..., 0].view(B, N, L).permute(0, 2, 1)  # (B, L, N)
        y = self.head(u if self.head_type == "flatten" else u[:, -1])  # (B*N, pred_len)
        y = y.view(B, N, self.pred_len).permute(0, 2, 1)
        if self.revin:
            y = y * sd + mu
        return y

    def timescales(self) -> list[torch.Tensor]:
        return [blk.mixer.timescales() for blk in self.blocks if hasattr(blk.mixer, "timescales")]

    def set_record_gates(self, flag: bool) -> None:
        for blk in self.blocks:
            if hasattr(blk.mixer, "record_gates"):
                blk.mixer.record_gates = flag

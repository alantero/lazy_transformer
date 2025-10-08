# utils/capacity.py
# Capacity signal from logits: c = 1 - H / log(V)
# - Stateless helpers: entropy_from_logits, capacity_from_logits
# - Broadcasting helpers to per-token / per-group / per-channel shapes
# - Optional temporal EMA smoothing (within the current sequence)
#
# No hard dependency on other repo files. Uses PyTorch if available.

from __future__ import annotations
from typing import Optional, Tuple, Literal

try:
    import torch
    import torch.nn.functional as F
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None      # type: ignore
    _HAVE_TORCH = False


def _require_torch() -> None:
    if not _HAVE_TORCH:
        raise ImportError("utils/capacity.py requires PyTorch for tensor ops.")


# ------------------------------- core metrics ---------------------------------

def entropy_from_logits(logits: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
    """
    Shannon entropy (nats) of categorical distributions given unnormalized logits.
    Args:
        logits: [..., V] tensor
        dim: axis of the class dimension (default: last)
    Returns:
        H: [...] entropy per item (same shape as logits without `dim`)
    """
    _require_torch()
    logp = F.log_softmax(logits, dim=dim)
    p = logp.exp()
    H = -(p * logp).sum(dim=dim)
    return H


def capacity_from_logits(
    logits: "torch.Tensor",
    *,
    clamp: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-12,
) -> "torch.Tensor":
    """
    Capacity proxy in [0,1] from logits: c = 1 - H / log(V), with H in nats.
    (Equivalent to bits formulation; ratio is base-invariant.)
    Args:
        logits: [B, T, V] (or any [..., V])
        clamp: lower/upper clamp for numerical stability
        eps: small epsilon for safe division
    Returns:
        c: logits.shape without the last dim (e.g., [B, T])
    """
    _require_torch()
    V = logits.size(-1)
    H = entropy_from_logits(logits, dim=-1)              # [...,]
    denom = float(torch.log(torch.tensor(float(V), device=logits.device)))
    denom = max(denom, eps)
    c = 1.0 - (H / denom)
    if clamp is not None:
        lo, hi = clamp
        c = c.clamp(min=float(lo), max=float(hi))
    return c


# ------------------------------- smoothing ------------------------------------

def temporal_ema(
    x_bt: "torch.Tensor",
    alpha: float = 0.9,
) -> "torch.Tensor":
    """
    Stateless temporal EMA along time axis for each batch independently:
        y[b, t] = alpha * y[b, t-1] + (1 - alpha) * x[b, t], y[b,0]=(1-alpha)*x[b,0]
    Args:
        x_bt: [B, T] or [B, T, K] tensor
        alpha: decay in [0,1)
    Returns:
        y: same shape as x_bt
    """
    _require_torch()
    if not (0.0 <= float(alpha) < 1.0):
        raise ValueError(f"alpha must be in [0,1), got {alpha}.")
    if x_bt.ndim not in (2, 3):
        raise ValueError(f"temporal_ema expects 2D or 3D [B,T,(K)] input, got {x_bt.shape}")

    B, T = x_bt.shape[0], x_bt.shape[1]
    rest = () if x_bt.ndim == 2 else (x_bt.shape[2],)
    y = torch.empty_like(x_bt)
    # t = 0
    y[:, 0] = (1.0 - float(alpha)) * x_bt[:, 0]
    # t > 0
    for t in range(1, T):
        y[:, t] = alpha * y[:, t - 1] + (1.0 - float(alpha)) * x_bt[:, t]
    return y


# ------------------------------ broadcasting ----------------------------------

BroadcastKind = Literal["token", "group", "channel"]

def broadcast_capacity(
    cap_bt: "torch.Tensor",
    *,
    kind: BroadcastKind = "token",
    groups: Optional[int] = None,
    channels: Optional[int] = None,
) -> "torch.Tensor":
    """
    Broadcast a [B,T] capacity to shapes used by modules:
      - "token"   → [B, T, 1]
      - "group"   → [B, T, G]     (requires `groups`)
      - "channel" → [B, T, D]     (requires `channels`)
    """
    _require_torch()
    if cap_bt.ndim != 2:
        raise ValueError(f"cap_bt must be [B,T], got {cap_bt.shape}")
    B, T = cap_bt.shape
    if kind == "token":
        return cap_bt.unsqueeze(-1)                          # [B,T,1]
    elif kind == "group":
        if not isinstance(groups, int) or groups < 1:
            raise ValueError("broadcast kind='group' requires a positive integer `groups`.")
        return cap_bt.unsqueeze(-1).expand(B, T, groups)     # [B,T,G]
    elif kind == "channel":
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("broadcast kind='channel' requires a positive integer `channels`.")
        return cap_bt.unsqueeze(-1).expand(B, T, channels)   # [B,T,D]
    else:
        raise ValueError(f"Unknown broadcast kind: {kind!r}")


# ------------------------------ high-level API --------------------------------

def capacity_signal(
    logits: "torch.Tensor",
    *,
    mask: Optional["torch.Tensor"] = None,         # [B,T] (True = valid)
    smooth_alpha: Optional[float] = None,          # if set, apply temporal EMA (per batch)
    kind: BroadcastKind = "token",
    groups: Optional[int] = None,
    channels: Optional[int] = None,
    detach: bool = True,
) -> "torch.Tensor":
    """
    Compute a capacity signal aligned to model shapes from logits:
      1) c_bt = 1 - H / log(V)  ∈ [0,1]
      2) optional temporal EMA along t (per batch)
      3) broadcast to [B,T,1] / [B,T,G] / [B,T,D]
      4) optional mask (invalid → 0)
    Args:
        logits: [B, T, V] tensor
        mask:   [B, T] bool (optional)
        smooth_alpha: if provided, apply temporal_ema to c_bt
        kind: one of {"token", "group", "channel"}
        groups/channels: required for kind="group"/"channel"
        detach: return a detached tensor (avoid backprop through capacity)
    Returns:
        cap: [B, T, 1] or [B, T, G] or [B, T, D]
    """
    _require_torch()
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,T,V], got {tuple(logits.shape)}")

    B, T, _ = logits.shape
    c_bt = capacity_from_logits(logits)            # [B,T]
    if smooth_alpha is not None:
        c_bt = temporal_ema(c_bt, alpha=float(smooth_alpha))  # [B,T]

    cap = broadcast_capacity(c_bt, kind=kind, groups=groups, channels=channels)  # [B,T,K]
    if mask is not None:
        if mask.shape != (B, T):
            raise ValueError(f"mask must be [B,T], got {tuple(mask.shape)}")
        cap = cap * mask.unsqueeze(-1).to(dtype=cap.dtype)

    return cap.detach() if detach else cap


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    print("[capacity] Running sanity tests...")
    _require_torch()
    torch.manual_seed(0)

    B, T, V = 2, 7, 12
    logits = torch.randn(B, T, V)

    # 1) Entropy/Capacity shapes
    H = entropy_from_logits(logits)                   # [B,T]
    assert H.shape == (B, T)
    cap_bt = capacity_from_logits(logits)             # [B,T]
    assert cap_bt.shape == (B, T)
    assert float(cap_bt.min()) >= -1e-6 and float(cap_bt.max()) <= 1.0 + 1e-6

    # 2) Broadcast
    cap_tok = broadcast_capacity(cap_bt, kind="token")
    cap_grp = broadcast_capacity(cap_bt, kind="group", groups=4)
    cap_chan = broadcast_capacity(cap_bt, kind="channel", channels=16)
    assert cap_tok.shape == (B, T, 1)
    assert cap_grp.shape == (B, T, 4)
    assert cap_chan.shape == (B, T, 16)

    # 3) Smoothing monotonic sanity: EMA reduces variance over time
    c = torch.rand(B, T)
    y = temporal_ema(c, alpha=0.8)
    var_c = float(c.var(dim=1).mean())
    var_y = float(y.var(dim=1).mean())
    assert var_y <= var_c + 1e-6

    # 4) Confidence monotonicity: sharper logits → higher capacity
    sharp = torch.zeros(B, T, V)
    sharp[..., 0] = 10.0  # highly confident on class 0
    cap_sharp = capacity_from_logits(sharp).mean().item()
    cap_rand = capacity_from_logits(torch.randn(B, T, V)).mean().item()
    assert cap_sharp > cap_rand, "Sharper distributions should have higher capacity."

    # 5) High-level signal
    mask = torch.ones(B, T, dtype=torch.bool)
    cap_sig = capacity_signal(logits, mask=mask, smooth_alpha=0.9, kind="group", groups=3)
    assert cap_sig.shape == (B, T, 3)

    print("[capacity] All good ✓")

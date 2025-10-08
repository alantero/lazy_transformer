# train/metrics.py
# Lightweight training/eval metrics (PyTorch-only, no repo-global deps).
# - Token accuracy (top-1 / top-k) with optional mask / ignore_index
# - ΔBKM-style state change (mean squared delta)
# - Overlap stitching metrics wrapper (uses modules.stitching if available)
# - Parameter/gradient norms and counts
# - Throughput helpers (tokens/sec)

from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ----------------------------- token-level metrics ----------------------------

@torch.no_grad()
def token_accuracy(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    topk: Sequence[int] = (1, 5),
) -> Dict[str, float]:
    """
    Compute token accuracy (top-1 and optional top-k).
      logits:  [B, T, V]
      targets: [B, T] (int64)
      mask:    [B, T] in {0,1} (1 means “count”); combined with ignore_index if given
    Returns dict with keys: 'acc', 'acc@K' ...
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,T,V], got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(f"targets must be [B,T], got {tuple(targets.shape)}")

    B, T, V = logits.shape
    device = logits.device

    # Build a boolean valid mask: where to evaluate accuracy
    if mask is not None:
        valid = mask.to(dtype=torch.bool, device=device)
        if valid.shape != (B, T):
            raise ValueError(f"mask must be [B,T], got {tuple(mask.shape)}")
    else:
        valid = torch.ones(B, T, dtype=torch.bool, device=device)

    if ignore_index is not None:
        valid = valid & (targets.to(device=device) != int(ignore_index))

    # Flatten
    flat_logits = logits.reshape(B * T, V)
    flat_targets = targets.to(device=device).reshape(B * T)
    flat_valid = valid.reshape(B * T)

    # If nothing is valid, return zeros (avoid div by zero)
    denom = int(flat_valid.sum().item())
    out: Dict[str, float] = {"acc": 0.0}
    if denom == 0:
        for k in topk:
            out[f"acc@{k}"] = 0.0
        return out

    # Top-1
    pred = flat_logits.argmax(dim=-1)
    acc = (pred.eq(flat_targets) & flat_valid).sum().item() / denom
    out["acc"] = float(acc)

    # Top-k
    if topk:
        maxk = max(topk)
        # torch.topk is cheaper than argsort; gather & compare
        topk_idx = flat_logits.topk(k=maxk, dim=-1).indices  # [N, maxk]
        # Broadcast targets and compare against top-k set
        tgt_exp = flat_targets.view(-1, 1).expand_as(topk_idx)
        match_any = (topk_idx.eq(tgt_exp)).any(dim=-1) & flat_valid
        # For smaller ks, we can recompute on the first k columns
        for k in topk:
            k_idx = topk_idx[:, :k]
            match_k = (k_idx.eq(tgt_exp[:, :k])).any(dim=-1) & flat_valid
            out[f"acc@{k}"] = float(match_k.sum().item() / denom)
    return out


# ------------------------------- state metrics --------------------------------

@torch.no_grad()
def delta_bkm(
    h_prev: Tensor,
    h_next: Tensor,
    *,
    mask: Optional[Tensor] = None,
    reduce: str = "mean",
) -> float:
    """
    ΔBKM-style state change: mean squared delta between two hidden states.
      h_prev/h_next: [..., T, D] or [B, T, D]; shapes must match
      mask: [B, T] or broadcastable to leading dims ending at T (optional)
      reduce: 'mean' | 'sum' (returned as float)
    """
    if h_prev.shape != h_next.shape:
        raise ValueError(f"Shapes must match: {tuple(h_prev.shape)} vs {tuple(h_next.shape)}")
    diff2 = (h_next - h_prev).pow(2)  # [..., T, D]

    if mask is not None:
        # Make mask broadcastable to diff2: expand to [..., T, 1]
        while mask.dim() < diff2.dim():
            mask = mask.unsqueeze(-1)
        diff2 = diff2 * mask.to(dtype=diff2.dtype, device=diff2.device)

    if reduce == "mean":
        denom = diff2.numel() if mask is None else max(int(mask.sum().item()) * diff2.shape[-1], 1)
        return float(diff2.sum().item() / denom)
    elif reduce == "sum":
        return float(diff2.sum().item())
    else:
        raise ValueError("reduce must be 'mean' or 'sum'.")


# ------------------------------ stitching metrics -----------------------------

# We keep this import optional to avoid hard dependency on modules/stitching in minimal setups.
try:
    from modules.stitching import stitching_loss as _stitching_loss  # type: ignore
    _HAVE_STITCH = True
except Exception:
    _HAVE_STITCH = False
    _stitching_loss = None  # type: ignore


@torch.no_grad()
def overlap_metrics(
    windows: Tensor,
    *,
    W: int,
    O: int,
    layout: str = "bnwd",
    align: str = "none",
    groups: Optional[int] = None,
) -> Dict[str, float]:
    """
    Wrapper to compute overlap consistency metrics on windowed sequences.
    If modules.stitching is available, delegates to it for robust stats; else falls back
    to a simple MSE over overlaps of adjacent windows (no alignment).
      windows: [B, n_win, W, D] (layout='bnwd') or [B, D, n_win, W] (layout='bdnw')
    Returns a dict with keys {stitch_mse, stitch_cos, bkm}.
    """
    if _HAVE_STITCH:
        loss, stats = _stitching_loss(  # type: ignore[misc]
            windows, W=W, O=O, layout=layout, align=align, groups=groups, reduction="mean"
        )
        # We ignore the loss value; return the stats only
        return {k: float(v) for k, v in stats.items()}

    # Fallback: minimal MSE without alignment
    x = windows
    if layout == "bdnw":
        x = x.permute(0, 2, 3, 1).contiguous()  # -> [B,n_win,W,D]
    if x.dim() != 4:
        raise ValueError(f"windows must be 4D, got {tuple(x.shape)}")
    B, nW, Wt, D = x.shape
    if Wt != W:
        raise ValueError("W mismatch between arg and tensor.")
    if O <= 0 or nW <= 1:
        return {"stitch_mse": 0.0, "stitch_cos": 1.0, "bkm": 0.0}

    mse_acc = 0.0
    cos_acc = 0.0
    bkm_acc = 0.0
    denom = 0
    for i in range(nW - 1):
        xi = x[:, i, W - O : W, :]   # [B,O,D]
        xj = x[:, i + 1, 0:O,  :, :] # [B,O,D]
        # MSE
        diff = xi - xj
        mse = float((diff.pow(2)).mean().item())
        mse_acc += mse
        # Cosine (flatten)
        xi_vec = xi.reshape(B, -1)
        xj_vec = xj.reshape(B, -1)
        num = (xi_vec * xj_vec).sum(dim=1)
        den = (xi_vec.norm(dim=1) * xj_vec.norm(dim=1)).clamp_min(1e-8)
        cos = float((num / den).mean().item())
        cos_acc += cos
        # BKM proxy
        bkm = float((diff.pow(2)).mean().item())
        bkm_acc += bkm
        denom += 1
    return {
        "stitch_mse": mse_acc / max(denom, 1),
        "stitch_cos": cos_acc / max(denom, 1),
        "bkm": bkm_acc / max(denom, 1),
    }


# --------------------------- parameter & grad metrics --------------------------

@torch.no_grad()
def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Total number of parameters (optionally only those with requires_grad=True)."""
    return int(sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)))


@torch.no_grad()
def parameter_norm(model: nn.Module, p: float = 2.0) -> float:
    """||θ||_p over all parameters (default p=2)."""
    total = 0.0
    for w in model.parameters():
        total += float(w.detach().abs().pow(p).sum().item())
    return total ** (1.0 / p)


@torch.no_grad()
def grad_norm(model: nn.Module, p: float = 2.0) -> float:
    """||∇θ||_p over all parameters with .grad (default p=2)."""
    total = 0.0
    for w in model.parameters():
        if w.grad is None:
            continue
        g = w.grad.detach()
        total += float(g.abs().pow(p).sum().item())
    return total ** (1.0 / p)


# ------------------------------ throughput helpers ----------------------------

def tokens_per_second(num_tokens: int, elapsed_seconds: float) -> float:
    """Simple throughput helper."""
    if elapsed_seconds <= 0:
        return 0.0
    return float(num_tokens) / float(elapsed_seconds)


def batch_num_tokens(mask: Optional[Tensor] = None, targets: Optional[Tensor] = None, *, pad_id: Optional[int] = None) -> int:
    """
    Count “valid” tokens in a batch, given either:
      - a binary mask [B,T] (1=valid), or
      - targets [B,T] with an ignore_index=pad_id to exclude padding.
    """
    if mask is not None:
        return int(mask.to(dtype=torch.int64).sum().item())
    if targets is not None and pad_id is not None:
        return int((targets != int(pad_id)).sum().item())
    if targets is not None:
        return int(targets.numel())
    return 0


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[metrics] Running sanity tests...")

    # Token accuracy
    B, T, V = 3, 7, 11
    logits = torch.randn(B, T, V)
    targets = torch.randint(low=0, high=V, size=(B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    accs = token_accuracy(logits, targets, mask=mask, topk=(1, 3, 5))
    print("  token_accuracy:", {k: f"{v:.3f}" for k, v in accs.items()})
    assert "acc" in accs and "acc@3" in accs and "acc@5" in accs

    # ΔBKM
    h0 = torch.randn(B, T, 16)
    h1 = h0 + 0.01 * torch.randn_like(h0)
    bkm = delta_bkm(h0, h1)
    print(f"  delta_bkm: {bkm:.3e}")
    assert bkm >= 0.0

    # Overlap metrics (fallback path works even if modules.stitching not present)
    W, O = 8, 4
    S = W - O
    nW = 1 + (T - W) // S if T >= W else 1
    # build windows [B,nW,W,D]
    D = 6
    x_full = torch.sin(torch.linspace(0, 2 * torch.pi, steps=T)).view(1, T, 1).repeat(B, 1, D)
    wins = torch.empty(B, nW, W, D)
    for i in range(nW):
        s = i * S
        wins[:, i] = x_full[:, s:s + W, :]
    st = overlap_metrics(wins, W=W, O=O, layout="bnwd", align="none")
    print("  overlap_metrics:", {k: f"{v:.3e}" for k, v in st.items()})
    assert "stitch_mse" in st and "stitch_cos" in st

    # Param/grad metrics
    m = nn.Linear(16, 16)
    y = m(h0)
    loss = (y ** 2).mean()
    loss.backward()
    n_params = count_parameters(m)
    pn = parameter_norm(m)
    gn = grad_norm(m)
    print(f"  params={n_params}, ||θ||={pn:.3e}, ||∇θ||={gn:.3e}")
    assert n_params > 0 and pn >= 0.0 and gn >= 0.0

    # Throughput helpers
    tok = batch_num_tokens(targets=targets, pad_id=0)
    tps = tokens_per_second(tok, 0.123)
    print(f"  tokens={tok}, tok/s≈{tps:.1f}")

    print("[metrics] All good ✓")

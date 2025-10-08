# losses/stitching_loss.py
# Stitching losses focused on overlaps between adjacent windows.
# - Symmetric KL on logits across overlaps (stability with log-softmax).
# - (Optional) low-D Procrustes MSE on hidden features across overlaps.
# Shapes follow the convention: windows are [B, n_win, W, ...] by default.

from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
_Layout = Literal["bnwd", "bdnw"]


# ------------------------------- shape helpers --------------------------------

def _to_bnwd(x: Tensor, layout: _Layout) -> Tensor:
    """Return windows as [B, n_win, W, D_or_V]."""
    if x.dim() != 4:
        raise ValueError(f"Expected a 4D tensor of windows, got shape {tuple(x.shape)}.")
    if layout == "bnwd":
        return x
    if layout == "bdnw":
        # [B, D, n_win, W] -> [B, n_win, W, D]
        return x.permute(0, 2, 3, 1).contiguous()
    raise ValueError("layout must be 'bnwd' or 'bdnw'.")


# ---------------------------- overlap bookkeeping -----------------------------

def overlap_pairs(n_win: int, W: int, O: int):
    """
    Describe overlaps between consecutive windows (i, i+1).
    Returns list of tuples (sl_i, sl_j, i, j) selecting tail/head overlap slices.
    """
    if O <= 0 or n_win <= 1:
        return []
    S = W - O
    O_eff = O if S > 0 else min(W, O)
    out = []
    for i in range(n_win - 1):
        j = i + 1
        out.append((slice(W - O_eff, W), slice(0, O_eff), i, j))
    return out


# ------------------------------ symmetric KL ----------------------------------

def symmetric_kl_overlaps(
    logits: Tensor,
    *,
    W: int,
    O: int,
    layout: _Layout = "bnwd",
    temperature: float = 1.0,
    mask: Optional[Tensor] = None,  # [B, n_win, W] or [B, n_win, W, 1]
    eps: float = 1e-12,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Symmetric KL across overlaps between consecutive windows of logits.

    Args:
      logits: [B, n_win, W, V] (or [B, V, n_win, W] with layout='bdnw')
      W, O: window length and overlap length
      temperature: softmax temperature for smoothing
      mask: optional validity mask per token in overlaps; same layout (without V)
    Returns:
      (loss, stats) with loss averaged over all overlapped tokens and batch.
    """
    x = _to_bnwd(logits, layout)  # [B, n_win, W, V]
    B, nW, W_, V = x.shape
    if W_ != W:
        raise ValueError(f"W mismatch: arg W={W}, tensor W={W_}.")

    if O <= 0 or nW <= 1:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, {"skl": 0.0}

    if mask is not None:
        m = _to_bnwd(mask, layout)
        if m.dim() == 4 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.shape != (B, nW, W):
            raise ValueError(f"mask shape must be [B,n_win,W] (or ..,1). Got {tuple(m.shape)}")
    else:
        m = None

    tau = float(temperature)
    lse = lambda z: F.log_softmax(z / tau, dim=-1)
    skl_total = x.new_tensor(0.0)
    denom = 0.0

    for sli, slj, i, j in overlap_pairs(nW, W, O):
        li = x[:, i, sli, :]            # [B, O, V]
        lj = x[:, j, slj, :]            # [B, O, V]
        logp = lse(li)
        logq = lse(lj)
        p = logp.exp()
        q = logq.exp()
        kl_pq = (p * (logp - logq)).sum(dim=-1)  # [B, O]
        kl_qp = (q * (logq - logp)).sum(dim=-1)
        skl = 0.5 * (kl_pq + kl_qp)              # [B, O]

        if m is not None:
            mi = m[:, i, sli]  # [B, O]
            mj = m[:, j, slj]
            mw = (mi > 0) & (mj > 0)
            # avoid empty mask
            if mw.any():
                skl_total = skl_total + skl[mw].mean()
                denom += 1.0
        else:
            skl_total = skl_total + skl.mean()
            denom += 1.0

    if denom == 0:
        loss = torch.zeros((), device=x.device, dtype=x.dtype)
    else:
        loss = skl_total / denom

    return loss, {"skl": float(loss.detach().cpu())}


# ------------------------- low-D Procrustes MSE (optional) --------------------

# We try importing the helper from modules/utils with a safe fallback.
try:
    from modules.stitching import procrustes_align_lowd as _procrustes_align_lowd  # type: ignore
    _HAVE_P_ALIGN = True
except Exception:
    _procrustes_align_lowd = None  # type: ignore
    _HAVE_P_ALIGN = False


def lowd_procrustes_overlaps(
    h_lowd: Tensor,
    *,
    W: int,
    O: int,
    layout: _Layout = "bnwd",
    rank: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Low-D Procrustes alignment MSE across overlaps.

    Args:
      h_lowd: hidden windows [B, n_win, W, d_low] (or [B, d_low, n_win, W] with layout='bdnw')
      rank: optional (ignored if external aligner preserves orientation); kept for API symmetry.
    Returns:
      (loss, stats) averaged over all overlaps.
    """
    if not _HAVE_P_ALIGN:
        # Graceful fallback: plain MSE without alignment
        x = _to_bnwd(h_lowd, layout)
        B, nW, W_, D = x.shape
        if O <= 0 or nW <= 1:
            z = torch.zeros((), device=x.device, dtype=x.dtype)
            return z, {"mse_lowd": 0.0}
        ms = 0.0
        denom = 0
        for sli, slj, i, j in overlap_pairs(nW, W_, O):
            xi = x[:, i, sli, :]
            xj = x[:, j, slj, :]
            ms += float(torch.mean((xi - xj).pow(2)).detach().cpu())
            denom += 1
        loss = torch.tensor(ms / max(1, denom), device=x.device, dtype=x.dtype)
        return loss, {"mse_lowd": float(loss.detach().cpu())}

    x = _to_bnwd(h_lowd, layout)  # [B, n_win, W, d_low]
    B, nW, W_, d = x.shape
    if W_ != W:
        raise ValueError(f"W mismatch: arg W={W}, tensor W={W_}.")
    if O <= 0 or nW <= 1:
        z = torch.zeros((), device=x.device, dtype=x.dtype)
        return z, {"mse_lowd": 0.0}

    loss_acc = x.new_tensor(0.0)
    denom = 0.0
    for sli, slj, i, j in overlap_pairs(nW, W, O):
        # Overlap chunks [B, O, d] -> bases [B, d, O]
        Ai = x[:, i, sli, :].transpose(1, 2).contiguous()  # [B, d, O]
        Bj = x[:, j, slj, :].transpose(1, 2).contiguous()  # [B, d, O]
        # Align columns of Ai to Bj
        A_al, R, stats = _procrustes_align_lowd(Ai, Bj, rank=rank)  # type: ignore
        # MSE after alignment (compare in the same orientation)
        # A_al, Bj: [B, d, O] -> compare column-wise (time)
        mse = torch.mean((A_al - Bj) ** 2)
        loss_acc = loss_acc + mse
        denom += 1.0

    loss = loss_acc / denom
    return loss, {"mse_lowd": float(loss.detach().cpu())}


# ------------------------------ combined module ------------------------------

class StitchingOverlapLoss(nn.Module):
    """
    Combined stitching loss on overlaps:
      - symmetric KL on logits (optional),
      - low-D Procrustes MSE on hidden features (optional).

    Use individual helpers if you only need one term.
    """

    def __init__(
        self,
        *,
        W: int,
        O: int,
        layout: _Layout = "bnwd",
        # logits KL
        use_skl: bool = True,
        skl_weight: float = 1.0,
        temperature: float = 1.0,
        # low-D Procrustes MSE
        use_lowd: bool = False,
        lowd_weight: float = 1.0,
        rank: Optional[int] = None,
    ):
        super().__init__()
        self.W = int(W)
        self.O = int(O)
        self.layout = layout
        self.use_skl = bool(use_skl)
        self.skl_weight = float(skl_weight)
        self.temperature = float(temperature)
        self.use_lowd = bool(use_lowd)
        self.lowd_weight = float(lowd_weight)
        self.rank = rank

    def forward(
        self,
        *,
        logits: Optional[Tensor] = None,   # [B, n_win, W, V] or [B, V, n_win, W] with layout='bdnw'
        h_lowd: Optional[Tensor] = None,   # [B, n_win, W, d_low] (or [B, d_low, n_win, W])
        mask: Optional[Tensor] = None,     # [B, n_win, W] (optional for SKL)
    ) -> Tuple[Tensor, Dict[str, float]]:
        terms = []
        stats: Dict[str, float] = {}

        if self.use_skl:
            if logits is None:
                raise ValueError("SKL requested but `logits` is None.")
            skl, st = symmetric_kl_overlaps(
                logits, W=self.W, O=self.O, layout=self.layout,
                temperature=self.temperature, mask=mask,
            )
            terms.append(self.skl_weight * skl)
            stats.update({f"skl": st["skl"], "skl_w": self.skl_weight})

        if self.use_lowd:
            if h_lowd is None:
                raise ValueError("low-D MSE requested but `h_lowd` is None.")
            lmse, st2 = lowd_procrustes_overlaps(
                h_lowd, W=self.W, O=self.O, layout=self.layout, rank=self.rank,
            )
            terms.append(self.lowd_weight * lmse)
            stats.update({f"mse_lowd": st2["mse_lowd"], "mse_lowd_w": self.lowd_weight})

        if not terms:
            zero = torch.zeros((), device=(logits or h_lowd).device, dtype=(logits or h_lowd).dtype)  # type: ignore[arg-type]
            return zero, {"skl": 0.0, "mse_lowd": 0.0}

        loss = sum(terms)
        stats["loss_stitch"] = float(loss.detach().cpu())
        return loss, stats


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    print("[stitching_loss] Running sanity tests...")
    torch.manual_seed(0)
    B, T, D, V = 2, 64, 12, 16
    W, O = 16, 8
    S = W - O
    nW = 1 + (T - W) // S
    assert T == W + (nW - 1) * S

    # Build smooth hidden and logits with small mismatch across overlaps
    t = torch.linspace(0, 1, T)
    base = torch.sin(2 * math.pi * 2 * t)
    h_full = base.view(1, T, 1).repeat(B, 1, D)
    logits_full = torch.randn(B, T, V) * 0.1
    logits_full[..., 0] += (3 * base).unsqueeze(0).repeat(B, 1)

    # Slice into windows [B, n_win, W, *]
    def slice_wins(x):
        wins = []
        for i in range(nW):
            s = i * S
            wins.append(x[:, s:s + W])
        return torch.stack(wins, dim=1)

    h_w = slice_wins(h_full)                 # [B, nW, W, D]
    log_w = slice_wins(logits_full)          # [B, nW, W, V]

    # Create a mild inconsistency: bias head overlap of window 1
    h_w2 = h_w.clone()
    h_w2[:, 1, :O, :] += 0.05
    log_w2 = log_w.clone()
    log_w2[:, 1, :O, :] += 0.2

    # 1) SKL symmetry and zero on identical logits
    skl0, st0 = symmetric_kl_overlaps(log_w, W=W, O=O)
    skl1, _ = symmetric_kl_overlaps(log_w2, W=W, O=O)
    skl2, _ = symmetric_kl_overlaps(log_w2.flip(1), W=W, O=O)  # just a different arrangement; nonzero generally
    print(f"  SKL identical={st0['skl']:.3e}, perturbed≈{float(skl1):.3e}")
    assert abs(st0["skl"]) < 1e-8
    assert float(skl1) > float(skl0) - 1e-9

    # 2) Low-D Procrustes reduces MSE across overlaps (if helper available)
    if _HAVE_P_ALIGN:
        mse_no, _ = lowd_procrustes_overlaps(h_w2, W=W, O=O)  # already aligned path
        # Force plain MSE by faking unavailability: compare direct overlap
        x = _to_bnwd(h_w2, "bnwd")
        ms_plain = 0.0
        den = 0
        for sli, slj, i, j in overlap_pairs(nW, W, O):
            xi = x[:, i, sli, :]
            xj = x[:, j, slj, :]
            ms_plain += float(torch.mean((xi - xj).pow(2)))
            den += 1
        mse_plain = ms_plain / max(1, den)
        print(f"  lowD MSE plain≈{mse_plain:.3e}, procrustes≈{float(mse_no):.3e}")
        assert float(mse_no) <= mse_plain + 1e-9

    # 3) Combined module wires both
    crit = StitchingOverlapLoss(W=W, O=O, use_skl=True, use_lowd=_HAVE_P_ALIGN, skl_weight=0.7, lowd_weight=0.3)
    loss, st = crit(logits=log_w2, h_lowd=h_w2)
    print(f"  combined loss={float(loss):.3e} | stats={st}")

    print("[stitching_loss] All good ✓")

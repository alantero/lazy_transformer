# modules/stitching.py
# Overlap stitching utilities for windowed sequences (PyTorch-only).
# Computes consistency losses/metrics across overlapping windows and (optionally)
# performs simple per-group Procrustes-style alignment (scale and/or affine).
#
# Expected (default) layout: windows shaped as [B, n_win, W, D].
# If you keep [B, D, n_win, W] elsewhere, pass layout="bdnw" and we’ll adapt.

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Optional Procrustes helpers (fallback to None if utils.procrustes not available)
try:
    from utils.procrustes import procrustes as _procrustes, orthobase_fit as _orthobase_fit  # type: ignore
    _HAVE_PROCR = True
except Exception:
    try:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from utils.procrustes import procrustes as _procrustes, orthobase_fit as _orthobase_fit  # type: ignore
        _HAVE_PROCR = True
    except Exception:
        _procrustes = None  # type: ignore
        _orthobase_fit = None  # type: ignore
        _HAVE_PROCR = False

Tensor = torch.Tensor
_Layout = Literal["bnwd", "bdnw"]
_Align = Literal["none", "scale", "affine"]


# ------------------------------- shape helpers --------------------------------

def _to_bnwd(x: Tensor, layout: _Layout) -> Tensor:
    """Return windows as [B, n_win, W, D]."""
    if x.dim() != 4:
        raise ValueError(f"Expected a 4D tensor of windows, got shape {tuple(x.shape)}.")
    if layout == "bnwd":
        return x
    elif layout == "bdnw":
        # [B, D, n_win, W] -> [B, n_win, W, D]
        return x.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError("layout must be 'bnwd' or 'bdnw'.")


def _from_bnwd(x: Tensor, layout: _Layout) -> Tensor:
    """Inverse of _to_bnwd for completeness (not used in loss)."""
    if layout == "bnwd":
        return x
    return x.permute(0, 3, 1, 2).contiguous()  # [B,n_win,W,D] -> [B,D,n_win,W]


# ---------------------------- overlap bookkeeping -----------------------------

def overlap_pairs(n_win: int, W: int, O: int) -> List[Tuple[slice, slice, int, int]]:
    """
    Describe overlaps between consecutive windows (i, i+1).
    Return a list of tuples: (sl_i, sl_j, i, j), where
      - sl_i selects the overlap tail in window i
      - sl_j selects the overlap head in window j
    Using the convention: stride S = W - O; overlap length = O.
    """
    if O < 0 or W <= 0 or n_win <= 0:
        return []
    if O == 0:
        return []
    S = W - O
    if S <= 0:
        # Degenerate: windows fully overlap or O>=W. We still compare last W of i vs first W of i+1.
        O_eff = min(W, O)
    else:
        O_eff = O

    pairs: List[Tuple[slice, slice, int, int]] = []
    for i in range(n_win - 1):
        j = i + 1
        sl_i = slice(W - O_eff, W)   # tail O samples
        sl_j = slice(0, O_eff)       # head O samples
        pairs.append((sl_i, sl_j, i, j))
    return pairs


# --------------------------- optional Procrustes fit ---------------------------

@torch.no_grad()
def _fit_groupwise_affine(x: Tensor, y: Tensor, groups: int, kind: _Align, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """
    Fit per-group scale (and optional bias) to map x ≈ s * x (+ a) ≈ y over [B, O, D].
    Returns (s, a) shaped [1,1,G,1] for easy broadcasting to [B,O,G,Cg].
    """
    if kind == "none":
        device, dtype = x.device, x.dtype
        one = torch.ones((), device=device, dtype=dtype)
        zero = torch.zeros((), device=device, dtype=dtype)
        return one.view(1, 1, 1, 1), zero.view(1, 1, 1, 1)

    B, O, D = x.shape
    if D % groups != 0:
        raise ValueError(f"'groups' must divide D. Got D={D}, groups={groups}.")
    Cg = D // groups

    # [B,O,D] -> [B,O,G,Cg]
    xg = x.view(B, O, groups, Cg)
    yg = y.view(B, O, groups, Cg)

    # Means over positions/channels; keep dims for broadcasting
    mx = xg.mean(dim=(1, 3), keepdim=True)  # [B,1,G,1]
    my = yg.mean(dim=(1, 3), keepdim=True)  # [B,1,G,1]
    vx = (xg - mx).pow(2).mean(dim=(1, 3), keepdim=True)  # [B,1,G,1]
    cov = ((xg - mx) * (yg - my)).mean(dim=(1, 3), keepdim=True)  # [B,1,G,1]

    s = cov / (vx + eps)  # [B,1,G,1]
    if kind == "scale":
        a = torch.zeros_like(s)
    else:  # "affine"
        a = my - s * mx

    # Average across batch for a cleaner/stable per-group fit
    s = s.mean(dim=0, keepdim=True)  # [1,1,G,1]
    a = a.mean(dim=0, keepdim=True)  # [1,1,G,1]
    return s, a


def _apply_groupwise_affine(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """x[B,O,D] with reshaped s,a [1,1,G,1] → aligned x."""
    B, O, D = x.shape
    G = s.shape[2]
    Cg = D // G
    xg = x.view(B, O, G, Cg)
    yg = s * xg + a
    return yg.view(B, O, D)


def procrustes_align_lowd(A: Tensor, B: Tensor, *, rank: int | None = None) -> tuple[Tensor, Tensor, dict] | None:
    """
    Align columns of A to B via rotation-only Procrustes in low-D (optional rank truncation).
    Expects [..., d, k] with k ≤ d. Returns (A_aligned, R, stats) or None if utils.procrustes unavailable.
    """
    if not _HAVE_PROCR:
        return None
    #A_in, B_in = A, B
    #if rank is not None:
    #    # Orthonormalize A only; keep B's column orientation so we align exactly to input B.
    #    A_in = _orthobase_fit(A_in, rank=rank)  # type: ignore[misc]
    #A_al, R, stats = _procrustes(A_in, B_in, allow_reflection=False, center=False, scale=False, return_stats=True)  # type: ignore[misc]
    #return A_al, R, {k: (float(v.mean().item()) if isinstance(v, Tensor) else float(v)) for k, v in stats.items()}  # type: ignore[union-attr]
    # Preserve column orientation; do not re-orthonormalize here.
    A_al, R, stats = _procrustes(
        A, B,
        allow_reflection=False,
        center=False,
        scale=False,
        return_stats=True,
    )  # type: ignore[misc]
    stats_out = {k: (float(v.mean().item()) if isinstance(v, Tensor) else float(v)) for k, v in stats.items()}  # type: ignore[union-attr]
    return A_al, R, stats_out

# ---------------------------- stitching loss/metrics --------------------------

def stitching_loss(
    windows: Tensor,
    *,
    W: int,
    O: int,
    layout: _Layout = "bnwd",
    align: _Align = "none",
    groups: Optional[int] = None,
    reduction: Literal["mean", "sum"] = "mean",
    eps: float = 1e-8,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Consistency loss across overlaps of adjacent windows.
    - windows: [B, n_win, W, D] (or [B, D, n_win, W] with layout='bdnw')
    - align: 'none' | 'scale' | 'affine' (per-group alignment before measuring)
    - groups: required if align != 'none' (D must be divisible by groups)

    Returns (loss, stats), where stats includes MSE and cosine similarity on overlaps.
    """
    x = _to_bnwd(windows, layout)  # [B, n_win, W, D]
    B, nW, W_, D = x.shape
    if W_ != W:
        raise ValueError(f"W mismatch: arg W={W}, tensor W={W_}.")
    if O < 0 or O > W:
        raise ValueError(f"O must be in [0, W]. Got O={O}, W={W}.")

    if O == 0 or nW <= 1:
        # No overlaps → zero loss but valid stats
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, {"stitch_mse": 0.0, "stitch_cos": 1.0, "bkm": 0.0}

    pair_list = overlap_pairs(nW, W, O)  # slices in window-local coordinates
    mse_acc = 0.0
    cos_acc = 0.0
    bkm_acc = 0.0
    denom = 0

    total_loss = None

    for sl_i, sl_j, i, j in pair_list:
        # Extract overlap tensors: [B, O, D]
        xi = x[:, i, sl_i, :]  # tail of window i
        xj = x[:, j, sl_j, :]  # head of window j

        # Optional per-group alignment of xi toward xj (or vice versa)
        if align != "none":
            if groups is None:
                raise ValueError("groups must be provided when align!='none'.")
            s, a = _fit_groupwise_affine(xi, xj, groups=groups, kind=align, eps=eps)
            xi_al = _apply_groupwise_affine(xi, s, a)
            # Uncomment to also align xj to xi and average, if desired.
            # s2, a2 = _fit_groupwise_affine(xj, xi, groups=groups, kind=align, eps=eps)
            # xj_al = _apply_groupwise_affine(xj, s2, a2)
            # diff = 0.5 * ((xi_al - xj) + (xi - xj_al))
            diff = xi_al - xj
        else:
            diff = xi - xj  # [B,O,D]

        # MSE over this overlap
        mse = torch.mean(diff.pow(2))
        total_loss = mse if total_loss is None else (total_loss + mse)

        # Cosine similarity (flatten over B,O,D)
        xi_vec = xi.reshape(B, -1)
        xj_vec = xj.reshape(B, -1)
        num = (xi_vec * xj_vec).sum(dim=1)
        den = (xi_vec.norm(dim=1) * xj_vec.norm(dim=1)).clamp_min(eps)
        cos = (num / den).mean()

        # ΔBKM-style scalar (same as we used in task_loss suggestion)
        bkm = torch.mean((xj - xi).pow(2))

        mse_acc += float(mse.detach().cpu())
        cos_acc += float(cos.detach().cpu())
        bkm_acc += float(bkm.detach().cpu())
        denom += 1

    if total_loss is None:
        total_loss = torch.zeros((), device=x.device, dtype=x.dtype)

    if reduction == "mean":
        total_loss = total_loss / max(denom, 1)
    elif reduction == "sum":
        pass
    else:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    stats = {
        "stitch_mse": mse_acc / max(denom, 1),
        "stitch_cos": cos_acc / max(denom, 1),
        "bkm": bkm_acc / max(denom, 1),
    }
    return total_loss, stats


# ------------------------------- simple wrapper -------------------------------

class StitchingLoss(nn.Module):
    """
    nn.Module wrapper for stitching_loss (useful to plug into a training step).
    """
    def __init__(
        self,
        W: int,
        O: int,
        *,
        layout: _Layout = "bnwd",
        align: _Align = "none",
        groups: Optional[int] = None,
        reduction: Literal["mean", "sum"] = "mean",
        eps: float = 1e-8,
        weight: float = 1.0,
    ):
        super().__init__()
        self.W = int(W)
        self.O = int(O)
        self.layout = layout
        self.align = align
        self.groups = groups
        self.reduction = reduction
        self.eps = float(eps)
        self.weight = float(weight)

    def forward(self, windows: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        loss, stats = stitching_loss(
            windows,
            W=self.W,
            O=self.O,
            layout=self.layout,
            align=self.align,
            groups=self.groups,
            reduction=self.reduction,
            eps=self.eps,
        )
        return self.weight * loss, {k: v for k, v in stats.items()}


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    # Sanity tests (no external deps)
    torch.manual_seed(0)
    print("[stitching] Running sanity tests...")

    B, T, D = 2, 64, 12
    W, O = 16, 8  # stride S = W - O = 8
    S = W - O
    n_win = 1 + (T - W) // S
    assert T == W + (n_win - 1) * S, "Choose T,W,O consistent so windows tile T."

    # Create a base smooth signal along T and copy into D channels
    t = torch.linspace(0, 1, T)
    base = torch.sin(2.0 * torch.pi * 3 * t)  # [T]
    h_full = base.view(1, T, 1).repeat(B, 1, D).contiguous()  # [B,T,D]

    # Slice into windows [B, n_win, W, D] (we avoid importing utils.windows here)
    wins = torch.empty(B, n_win, W, D)
    for i in range(n_win):
        s = i * S
        wins[:, i, :, :] = h_full[:, s:s + W, :]

    # 1) Perfect consistency → loss ~ 0
    loss0, stats0 = stitching_loss(wins, W=W, O=O, layout="bnwd", align="none")
    print(f"  perfect: loss={float(loss0):.3e}, stats={ {k: f'{v:.3e}' for k,v in stats0.items()} }")
    assert float(loss0) < 1e-12
    assert abs(stats0["stitch_cos"] - 1.0) < 1e-6

    # 2) Add a constant bias to window 1 head overlap → MSE>0, affine alignment fixes it
    w2 = wins.clone()
    w2[:, 1, :O, :] += 0.5  # bias on head of window 1 (overlap region with window 0)
    loss_bias, stats_bias = stitching_loss(w2, W=W, O=O, layout="bnwd", align="none")
    loss_aff, stats_aff = stitching_loss(w2, W=W, O=O, layout="bnwd", align="affine", groups=3)
    print(f"  bias:    no-align loss={float(loss_bias):.3e}, affine loss={float(loss_aff):.3e}")
    assert float(loss_bias) > 1e-6
    assert float(loss_aff) < float(loss_bias)

    # 3) Scale mismatch between windows → scale alignment helps
    w3 = wins.clone()
    w3[:, 2, :O, :] *= 1.3
    loss_scale, _ = stitching_loss(w3, W=W, O=O, layout="bnwd", align="none")
    loss_salign, _ = stitching_loss(w3, W=W, O=O, layout="bnwd", align="scale", groups=3)
    print(f"  scale:   no-align loss={float(loss_scale):.3e}, scale loss={float(loss_salign):.3e}")
    assert float(loss_salign) < float(loss_scale)

    # 4) Accept [B,D,n_win,W] layout too
    wins_bdnw = wins.permute(0, 3, 1, 2).contiguous()
    loss4, _ = stitching_loss(wins_bdnw, W=W, O=O, layout="bdnw", align="none")
    print(f"  layout bdnw: loss={float(loss4):.3e}")
    assert abs(float(loss4) - float(loss0)) < 1e-12

    # 5) (Optional) Toy Procrustes sanity: align two low-D bases
    if _HAVE_PROCR:
        k = 4
        # build a random orthonormal basis Bbasis: [B,d,k]
        X = torch.randn(B, D, k)
        # use torch QR to create a basis and then orthobase_fit to mimic pipeline
        Q, _ = torch.linalg.qr(X, mode="reduced")
        Bbasis = Q  # already orthonormal columns
        # Random rotation R_true with det +1
        G = torch.randn(k, k)
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        R_true = U @ Vh
        if torch.det(R_true) < 0:
            U[:, -1] *= -1.0
            R_true = U @ Vh
        A = Bbasis @ R_true  # rotated basis
        A_noisy = A + 1e-8 * torch.randn_like(A)
        # Align A→B and check MSE drops
        mse_before = float(torch.mean((A_noisy - Bbasis) ** 2))
        out = procrustes_align_lowd(A_noisy, Bbasis, rank=k)
        assert out is not None
        A_al, R_est, st = out
        mse_after = float(torch.mean((A_al - Bbasis) ** 2))
        print(f"  procrustes: mse_before={mse_before:.3e} → mse_after={mse_after:.3e}")
        assert mse_after < 1e-6

    print("[stitching] All good ✓")

# operators/lowrank.py
# Feature-space low-rank mixing operators.
# PyTorch-only, no external deps.
#
# Given x[B, T, D], apply a block-diagonal low-rank linear operator along the
# last (feature) dimension, independently for each time step:
#   - kind="general":       M_g = A_g @ B_g^T                          (rank ≤ r)
#   - kind="symmetric":     M_g = A_g @ A_g^T  (PSD)                   (rank ≤ r)
#   - kind="skew":          M_g = A_g @ B_g^T - B_g @ A_g^T  (skew)    (rank ≤ 2r)
#
# Grouping: D must be divisible by `groups` (G). Each group acts on a chunk of
# size Cg=D/G with its own factors A_g, B_g (shape [Cg, r]). The forward
# computes y = x + scale * (x @ M^T) (+ bias), i.e., residual right-multiply by the (groupwise) mixing.
#
# Notes
# - Trainable scale: a single scalar "scale" (default 0.0) multiplies the
#   operator output → lazy-minimal start (no effect until learned).
# - Optional bias on features (per-channel), applied after the mixing.
# - Efficient computation via two small GEMMs per group using einsum; no need
#   to form the full D×D matrix on forward. A helper builds the full matrix for
#   debugging/tests if desired.
# - Accepts optional mask [B,T] (bool) to gate updates/bias on padded positions,
#   preserving causal semantics (operator is per-time-step and does not look ahead).

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor
_Kind = Literal["general", "symmetric", "skew"]


class LowRankMix(nn.Module):
    """
    Block-diagonal low-rank mixing over features.

    Args:
      d_model:  total feature dimension D
      rank:     low-rank dimension r
      groups:   number of groups G (must divide D). Each group has size Cg=D/G.
      kind:     'general' | 'symmetric' | 'skew'
      bias:     add learnable bias over D features after mixing
      init_scale: std for A/B init (small); scale param init (multiplier) set to 0.0
      dtype:    torch dtype for parameters (default float32)

    Shapes:
      A: [G, Cg, r],    B: [G, Cg, r]
      x: [B, T, D]  →  y: [B, T, D]
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        *,
        groups: int = 1,
        kind: _Kind = "general",
        bias: bool = False,
        init_scale: float = 0.02,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        if d_model <= 0 or rank <= 0:
            raise ValueError("d_model and rank must be > 0.")
        if groups <= 0 or (d_model % groups) != 0:
            raise ValueError(f"'groups' must divide d_model. Got D={d_model}, groups={groups}.")
        if kind not in ("general", "symmetric", "skew"):
            raise ValueError("kind must be one of {'general','symmetric','skew'}.")

        self.d_model = int(d_model)
        self.rank = int(rank)
        self.groups = int(groups)
        self.cg = self.d_model // self.groups
        self.kind: _Kind = kind

        # Factors per group (A and, if needed, B)
        g, cg, r = self.groups, self.cg, self.rank
        factory_kwargs = {"dtype": dtype}

        self.A = nn.Parameter(torch.empty(g, cg, r, **factory_kwargs))
        if self.kind in ("general", "skew"):
            self.B = nn.Parameter(torch.empty(g, cg, r, **factory_kwargs))
        else:
            self.register_parameter("B", None)

        # Global scale (lazy-minimal start = 0.0 → initially identity if bias=False)
        self.scale = nn.Parameter(torch.zeros((), **factory_kwargs))

        # Optional bias per feature
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.d_model, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Init
        with torch.no_grad():
            self.A.normal_(mean=0.0, std=init_scale)
            if self.B is not None:
                self.B.normal_(mean=0.0, std=init_scale)

    def extra_repr(self) -> str:  # nice __repr__
        return (f"d_model={self.d_model}, rank={self.rank}, groups={self.groups}, "
                f"kind='{self.kind}', bias={'True' if self.bias is not None else 'False'}")

    def _apply_general(self, xg: Tensor, A: Tensor, B: Tensor) -> Tensor:
        # y = x @ (A B^T)^T = x @ (B A^T)
        # xg: [B,T,G,Cg], A/B: [G,Cg,r]
        tmp = torch.einsum("btgc,gcr->btgr", xg, B)             # [B,T,G,r]  = xg @ B
        yg  = torch.einsum("btgr,grc->btgc", tmp, A.transpose(-1, -2))  # @ A^T -> [B,T,G,Cg]
        return yg

    def _apply_symmetric(self, xg: Tensor, A: Tensor) -> Tensor:
        # y = x @ (A A^T) (symmetric PSD)
        tmp = torch.einsum("btgc,gcr->btgr", xg, A)             # xg @ A  -> [B,T,G,r]
        yg  = torch.einsum("btgr,grc->btgc", tmp, A.transpose(-1, -2))  # @ A^T -> [B,T,G,Cg]
        return yg

    def _apply_skew(self, xg: Tensor, A: Tensor, B: Tensor) -> Tensor:
        # y = x @ ( (A B^T - B A^T)^T ) = x @ (B A^T - A B^T)
        tmp1 = torch.einsum("btgc,gcr->btgr", xg, B)            # xg @ B -> [B,T,G,r]
        y1   = torch.einsum("btgr,grc->btgc", tmp1, A.transpose(-1, -2))  # @ A^T -> [B,T,G,Cg]
        tmp2 = torch.einsum("btgc,gcr->btgr", xg, A)            # xg @ A -> [B,T,G,r]
        y2   = torch.einsum("btgr,grc->btgc", tmp2, B.transpose(-1, -2))  # @ B^T -> [B,T,G,Cg]
        return y1 - y2

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, T, D] → y: [B, T, D]
        mask: Optional[bool Tensor] of shape [B, T] to gate updates/bias on padded positions.
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must be [B,T,{self.d_model}], got {tuple(x.shape)}")

        B, T, D = x.shape
        g, cg = self.groups, self.cg

        if mask is not None:
            if mask.shape != (B, T):
                raise ValueError(f"mask must have shape [B, T], got {tuple(mask.shape)}")
            if mask.dtype != torch.bool:
                mask = mask.bool()
            m4 = mask.view(B, T, 1, 1)  # for gating updates on [B,T,G,Cg]
            m3 = mask.view(B, T, 1)     # for gating bias on [B,T,D]
        else:
            m4 = None
            m3 = None

        # Reshape features into groups for block-diagonal mixing
        xg = x.view(B, T, g, cg)  # [B,T,G,Cg]

        if self.kind == "general":
            yg = self._apply_general(xg, self.A, self.B)  # type: ignore[arg-type]
        elif self.kind == "symmetric":
            yg = self._apply_symmetric(xg, self.A)
        else:  # "skew"
            yg = self._apply_skew(xg, self.A, self.B)     # type: ignore[arg-type]

        if m4 is not None:
            yg = yg * m4  # zero out updates at PAD positions

        # Scale (scalar) + optional bias
        # NOTE: einsum can produce a non-contiguous tensor; use reshape for safety.
        y = x + self.scale * yg.reshape(B, T, D)  # residual-style update
        if self.bias is not None:
            if m3 is None:
                y = y + self.bias.view(1, 1, D)
            else:
                y = y + m3.to(y.dtype) * self.bias.view(1, 1, D)

        if m3 is not None:
            # Defensive: force PAD positions to original x (mask gates updates and bias anyway)
            y = torch.where(m3, y, x)

        # This operator is time-local (no future-peeking), mask only gates PADs.
        return y

    @torch.no_grad()
    def full_matrix(self) -> Tensor:
        """
        Build the full [D,D] mixing matrix (block diagonal).
        Useful for debugging/tests; not used in forward.
        """
        g, cg, r = self.groups, self.cg, self.rank
        device = self.A.device
        dtype = self.A.dtype

        blocks: list[Tensor] = []
        for gi in range(g):
            A = self.A[gi]  # [Cg,r]
            if self.kind == "symmetric":
                M = A @ A.t()
            else:
                B = self.B[gi]  # type: ignore[index]
                if self.kind == "general":
                    M = A @ B.t()
                else:  # skew
                    M = A @ B.t() - B @ A.t()
            blocks.append(M)

        # Assemble block diagonal
        out = torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype)
        for gi, M in enumerate(blocks):
            s = gi * cg
            out[s:s+cg, s:s+cg] = M
        return out


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[lowrank] Running sanity tests...")

    B, T, D = 3, 7, 12
    G = 3
    Cg = D // G
    r = 4
    x = torch.randn(B, T, D)

    # --- General: compare forward vs explicit full matrix
    mod = LowRankMix(d_model=D, rank=r, groups=G, kind="general", bias=True, init_scale=0.01)
    with torch.no_grad():
        mod.scale.fill_(1.0)
        mod.bias.normal_(0.0, 0.01)  # type: ignore[union-attr]

    M = mod.full_matrix()  # [D,D]
    y_ref = x + (x @ M.t()) + (mod.bias.view(1, 1, D) if mod.bias is not None else 0.0)
    y = mod(x)
    err = float((y - y_ref).abs().max())
    print(f"  general: max|Δ| = {err:.2e}")
    assert err < 1e-6

    # Masked test for general
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -2:] = False
    y_mask = mod(x, mask=mask)
    assert (y_mask[:, -2:] - x[:, -2:]).abs().max() == 0

    # --- Symmetric: check PSD and match explicit product
    sym = LowRankMix(d_model=D, rank=r, groups=G, kind="symmetric", bias=False, init_scale=0.01)
    with torch.no_grad():
        sym.scale.fill_(1.0)
    Ms = sym.full_matrix()
    # PSD check: v^T Ms v ≥ 0
    v = torch.randn(D)
    quad = float(v @ (Ms @ v))
    print(f"  symmetric: v^T M v = {quad:.3e} (should be ≥ 0 up to num. noise)")
    assert quad > -1e-6
    y_ref = x + (x @ Ms.t())
    ys = sym(x)
    err2 = float((ys - y_ref).abs().max())
    print(f"  symmetric: max|Δ| = {err2:.2e}")
    assert err2 < 1e-6

    # Masked test for symmetric
    y_mask_sym = sym(x, mask=mask)
    assert (y_mask_sym[:, -2:] - x[:, -2:]).abs().max() == 0

    # --- Skew: check antisymmetry and match explicit product
    sk = LowRankMix(d_model=D, rank=r, groups=G, kind="skew", bias=False, init_scale=0.01)
    with torch.no_grad():
        sk.scale.fill_(1.0)
    Mk = sk.full_matrix()
    # Skew symmetry: M^T = -M
    skew_err = float((Mk.t() + Mk).abs().max())
    print(f"  skew: ||M^T + M||_∞ = {skew_err:.2e}")
    assert skew_err < 1e-6
    y_ref = x + (x @ Mk.t())
    yk = sk(x)
    err3 = float((yk - y_ref).abs().max())
    print(f"  skew: max|Δ| = {err3:.2e}")
    assert err3 < 1e-6

    print("[lowrank] All good ✓")

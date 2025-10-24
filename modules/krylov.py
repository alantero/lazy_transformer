# modules/krylov.py
# Tiny Krylov-space residual correction for lazy fixes (PyTorch-only, no repo-global deps).
# Goal: given an implicit linear map A (via apply(v)) and a residual r, find a small
#       correction δ in the Krylov space K_m(A, r) = span{r, A r, A^2 r, ...}
#       that minimizes ||A δ - r|| in least squares (normal equations).
#
# API:
#   - krylov_lazy_correction(apply, residual, m=2, ...) -> delta
#   - KrylovRefiner(m=2) with .refine(apply, residual) and nn.Module wrapper
#
# Notes
# - We never form A; we only call `apply(v)` to get A v and (for bases) A^k r.
# - Uses modified Gram–Schmidt to (re)ortho las bases y mejorar estabilidad.
# - Resuelve (W^T W) c = W^T r con W = [A v0, ..., A v_{m-1}], δ = V c.
# - Funciona batcheado: cualquier shape [..., N] vale; aplanamos a [B*, N].
# - Seguro numéricamente para m pequeño (1–4) como “lazy fix” tras ΔBKM alta.

from __future__ import annotations
from typing import Callable, Tuple, Optional, Literal

import torch
import torch.nn as nn

Tensor = torch.Tensor
_Ortho = Literal["mgs", "none"]


def _flatten_bt(x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
    """Flatten all leading dims into batch: x[*, N] -> X[B, N], return X and original prefix shape."""
    if x.dim() == 1:
        return x.unsqueeze(0), ()
    N = x.shape[-1]
    B = int(x.numel() // N)
    X = x.reshape(B, N)
    return X, x.shape[:-1]


def _unflatten_bt(X: Tensor, prefix: Tuple[int, ...]) -> Tensor:
    if not prefix:
        return X.squeeze(0)
    return X.reshape(*prefix, X.shape[-1])


def _norm2(X: Tensor, eps: float = 1e-12) -> Tensor:
    # L2 por fila (batched)
    return torch.sqrt(torch.clamp((X * X).sum(dim=-1, keepdim=True), min=eps))


def _mgs_orthonormalize(V: Tensor, eps: float = 1e-9) -> Tensor:
    """
    Modified Gram–Schmidt por filas-batch.
      V: [B, k, N] (k vectores por batch). Devuelve Q ortonormal aproximado.
    """
    B, k, N = V.shape
    Q = torch.zeros_like(V)
    for i in range(k):
        vi = V[:, i, :]  # [B,N]
        # resta proyecciones sobre previos
        if i > 0:
            Qi = Q[:, :i, :]                               # [B,i,N]
            proj = (vi.unsqueeze(1) @ Qi.transpose(1, 2)).squeeze(1)  # [B,i]
            vi = vi - (proj.unsqueeze(-1) * Qi).sum(dim=1)            # [B,N]
        # normaliza
        nrm = _norm2(vi, eps=eps)                          # [B,1]
        vi = vi / nrm
        Q[:, i, :] = vi
    return Q


def _flatten_mask(mask: Optional[Tensor], prefix_N: Tuple[int, ...], N: int) -> Optional[Tensor]:
    if mask is None:
        return None
    # Ensure mask has last dim N (broadcast from 1 if needed)
    if mask.shape[-1] != N:
        if mask.shape[-1] == 1:
            mask = mask.expand(*mask.shape[:-1], N)
        else:
            raise ValueError(f"mask last dim must be N={N}, got {tuple(mask.shape)}")
    M, _ = _flatten_bt(mask.to(dtype=torch.bool))
    return M.to(dtype=torch.float32)


@torch.no_grad()
def build_krylov_basis(
    apply: Callable[[Tensor], Tensor],
    residual: Tensor,
    m: int = 2,
    *,
    ortho: _Ortho = "mgs",
    eps: float = 1e-9,
    mask: Optional[Tensor] = None,
    freeze_masked: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Construye bases V=[v0..v_{m-1}] y W=[A v0..A v_{m-1}] por batch.
      residual r: shape [..., N] → aplanamos a [B,N].
      mask:       shape [..., N] (True=activo, False=congelado). Si freeze_masked=True, se fuerza 0 en esas posiciones.
    Devuelve:
      V: [B, m, N] (ortonormal si ortho='mgs'), W: [B, m, N] = A V
    """
    if m <= 0:
        raise ValueError("m must be >= 1")
    R, prefix = _flatten_bt(residual)      # [B,N]
    B, N = R.shape

    M = _flatten_mask(mask, prefix, N)  # [B,N] o None
    if M is not None and freeze_masked:
        R = R * M

    # v0 = r / ||r||
    nrmr = _norm2(R, eps=eps)              # [B,1]
    v0 = R / nrmr                          # [B,N]

    V = [v0]
    for _ in range(1, m):
        # A @ v_{k-1}
        ak = apply(V[-1]).reshape(B, N)    # [B,N]
        if M is not None and freeze_masked:
            ak = ak * M
        V.append(ak)

    Vt = torch.stack(V, dim=1)             # [B,m,N]

    # Ortonormaliza si procede
    if ortho == "mgs":
        Vt = _mgs_orthonormalize(Vt, eps=eps)

    # W = A V (enmascarado si procede)
    W_list = []
    for i in range(m):
        wi = apply(Vt[:, i, :]).reshape(B, N)
        if M is not None and freeze_masked:
            wi = wi * M
        W_list.append(wi)
    W = torch.stack(W_list, dim=1)  # [B,m,N]
    return Vt, W


def krylov_lazy_correction(
    apply: Callable[[Tensor], Tensor],
    residual: Tensor,
    *,
    m: int = 2,
    ortho: _Ortho = "mgs",
    eps: float = 1e-9,
    mask: Optional[Tensor] = None,
    freeze_masked: bool = True,
) -> Tensor:
    """
    Resuelve min_c || A (V c) - r ||_2 en subespacio de Krylov K_m(A, r).
      mask: posiciones congeladas (se proyecta base y W si freeze_masked=True, y la delta final se anula en esas posiciones).
    Devuelve:
      delta: corrección en el espacio original (mismo shape que r).
    """
    R, prefix = _flatten_bt(residual)  # [B,N]
    B, N = R.shape

    # Bases
    V, W = build_krylov_basis(
        apply, residual, m=m, ortho=ortho, eps=eps, mask=mask, freeze_masked=freeze_masked
    )  # [B,m,N] cada uno

    # (W^T W) c = W^T r
    G = torch.einsum("bmn,bkn->bmk", W, W)            # [B,m,m]
    b = torch.einsum("bmn,bn->bm", W, R)              # [B,m]

    lam = eps
    eye = torch.eye(m, device=G.device, dtype=G.dtype).unsqueeze(0).expand(B, m, m)
    G_reg = G + lam * eye

    try:
        c = torch.linalg.solve(G_reg, b.unsqueeze(-1)).squeeze(-1)  # [B,m]
    except RuntimeError:
        c = torch.linalg.lstsq(G_reg, b.unsqueeze(-1)).solution.squeeze(-1)

    # δ = V c
    delta = torch.einsum("bmn,bm->bn", V, c)          # [B,N]

    # Cierra con máscara si procede
    if mask is not None:
        M = _flatten_mask(mask, prefix, N)  # [B,N]
        delta = delta * M

    return _unflatten_bt(delta, prefix)               # [...,N]


class KrylovRefiner(nn.Module):
    """
    nn.Module wrapper para correcciones rápidas de residuo en subespacios pequeños.
    Uso típico:
        ref = KrylovRefiner(m=2)
        if drift_or_bkm_high:
            delta = ref.refine(apply, residual)   # mismo shape que residual
            h = h + delta
    """
    def __init__(self, m: int = 2, ortho: _Ortho = "mgs"):
        super().__init__()
        if m <= 0:
            raise ValueError("m must be >= 1")
        self.m = int(m)
        self.ortho: _Ortho = ortho

    @torch.no_grad()
    def refine(
        self,
        apply: Callable[[Tensor], Tensor],
        residual: Tensor,
        *,
        mask: Optional[Tensor] = None,
        freeze_masked: bool = True,
    ) -> Tensor:
        return krylov_lazy_correction(
            apply, residual, m=self.m, ortho=self.ortho, mask=mask, freeze_masked=freeze_masked
        )


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[krylov] Running sanity tests...")

    # Build a synthetic linear operator A on R^N via apply(x) = A x
    def make_spd(N: int, cond: float = 10.0) -> Tensor:
        U, _ = torch.linalg.qr(torch.randn(N, N))
        s = torch.linspace(1.0, cond, steps=N)
        A = U @ torch.diag(s) @ U.t()  # SPD
        return A

    def batch_apply(A: Tensor, v: Tensor) -> Tensor:
        # v: [..., N] -> [..., N]
        N = A.shape[0]
        V, prefix = _flatten_bt(v)
        Y = V @ A.t()                 # [B,N]
        return _unflatten_bt(Y, prefix)

    # Test 1: SPD operator, residual reduction
    N = 64
    A = make_spd(N, cond=20.0)
    # target system A δ = r ; start with δ=0 → residual = r
    r = torch.randn(3, N) * 0.5
    applyA = lambda x: batch_apply(A, x)

    # m=1 (steepest-like), m=2 (Krylov small)
    d1 = krylov_lazy_correction(applyA, r, m=1)
    d2 = krylov_lazy_correction(applyA, r, m=2)

    res0 = (applyA(torch.zeros_like(r)) - r).norm(dim=-1).mean()  # ||-r||
    res1 = (applyA(d1) - r).norm(dim=-1).mean()
    res2 = (applyA(d2) - r).norm(dim=-1).mean()

    print(f"  SPD: ||-r||={float(res0):.3e} → m=1:{float(res1):.3e}, m=2:{float(res2):.3e}")
    assert res1 < res0 and res2 <= res1 + 1e-8

    # Test 2: Non-symmetric A (skew + diag), still should reduce LS residual
    N = 48
    M = torch.randn(N, N)
    A2 = 0.1 * (M - M.t()) + 1.5 * torch.eye(N)  # well-conditioned, non-symmetric
    applyA2 = lambda x: batch_apply(A2, x)
    r2 = torch.randn(2, N)

    d2a = krylov_lazy_correction(applyA2, r2, m=2)
    res0b = (applyA2(torch.zeros_like(r2)) - r2).norm(dim=-1).mean()
    res2b = (applyA2(d2a) - r2).norm(dim=-1).mean()
    print(f"  NSym: ||-r||={float(res0b):.3e} → m=2:{float(res2b):.3e}")
    assert res2b < res0b

    print("[krylov] All good ✓")

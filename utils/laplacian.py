# utils/laplacian.py
# Laplacian utilities (NumPy/Torch). 1D/2D discrete Laplacian with Dirichlet/Neumann/Periodic BCs,
# and dense graph Laplacian (normalized/unnormalized). No project-wide deps.

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

# Optional backends
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

try:
    import torch as _torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _torch = None  # type: ignore
    _HAVE_TORCH = False


# ----------------------------- backend helpers ------------------------------

def _is_numpy(x: Any) -> bool:
    return (_np is not None) and isinstance(x, _np.ndarray)

def _is_torch(x: Any) -> bool:
    return (_HAVE_TORCH) and isinstance(x, _torch.Tensor)

def _zeros_like(x: Any) -> Any:
    if _is_numpy(x): return _np.zeros_like(x, dtype=float)
    return _torch.zeros_like(x, dtype=_torch.float32)

def _move_axis(x: Any, src: int, dst: int) -> Any:
    if _is_numpy(x): return _np.moveaxis(x, src, dst)
    return _torch.movedim(x, src, dst)

def _take(x: Any, sl: slice, axis: int) -> Any:
    if axis < 0: axis += x.ndim
    idx = [slice(None)] * x.ndim
    idx[axis] = sl
    return x[tuple(idx)]

def _pad1(x: Any, axis: int, left: int, right: int, bc: str, value: float = 0.0) -> Any:
    """Pad 1D along `axis` using given boundary condition."""
    if left == 0 and right == 0:
        return x
    if _is_numpy(x):
        if axis < 0: axis += x.ndim
        mode = {"dirichlet": "constant", "neumann": "edge", "periodic": "wrap"}[bc]
        pad = [(0, 0)] * x.ndim
        pad[axis] = (left, right)
        kwargs = {"mode": mode}
        if mode == "constant":
            kwargs["constant_values"] = value
        return _np.pad(x, pad, **kwargs)
    else:
        # Import lazily to avoid hard dependency if torch is absent.
        import torch.nn.functional as F  # type: ignore
        if axis < 0: axis += x.ndim
        y = _move_axis(x, axis, -1)  # [..., N]
        mode = {"dirichlet": "constant", "neumann": "replicate", "periodic": "circular"}[bc]
        if mode == "constant":
            y = F.pad(y, (left, right), mode=mode, value=float(value))
        else:
            y = F.pad(y, (left, right), mode=mode)
        return _move_axis(y, -1, axis)


# --------------------------- 1D / 2D Laplacians -----------------------------

def apply_laplacian_1d(x: Any, *, axis: int = -1, h: float = 1.0,
                       bc: str = "dirichlet") -> Any:
    """
    Discrete 1D Laplacian Δx ≈ (x[i-1] - 2x[i] + x[i+1]) / h^2 along `axis`.
    bc ∈ {"dirichlet","neumann","periodic"}.
    """
    if bc not in {"dirichlet", "neumann", "periodic"}:
        raise ValueError("bc must be 'dirichlet', 'neumann', or 'periodic'")
    xpad = _pad1(x, axis, 1, 1, bc, value=0.0)
    left  = _take(xpad, slice(0, -2), axis)
    mid   = _take(xpad, slice(1, -1), axis)
    right = _take(xpad, slice(2, None), axis)
    return (left - 2.0 * mid + right) / (h * h)

def apply_laplacian_2d(x: Any, *, axes: Tuple[int, int] = (-2, -1),
                       hx: float = 1.0, hy: float = 1.0,
                       bc: str = "dirichlet") -> Any:
    """2D Laplacian: Δx ≈ (∂²/∂ax² + ∂²/∂ay²) using separable 1D stencils."""
    ax0, ax1 = axes
    l0 = apply_laplacian_1d(x, axis=ax0, h=hx, bc=bc)
    l1 = apply_laplacian_1d(x, axis=ax1, h=hy, bc=bc)
    return l0 + l1


# ------------------------------ Graph Laplacian ------------------------------

def graph_laplacian(A: Any, *, normalized: bool = False,
                    add_self_loops: bool = False, eps: float = 1e-9) -> Any:
    """
    Dense graph Laplacian for adjacency matrix A (N×N).
    - Unnormalized: L = D - A
    - Normalized:   L = I - D^{-1/2} A D^{-1/2}
    If add_self_loops=True, use A+I for degree (as in many GNN recipes).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (N×N).")
    N = A.shape[0]

    if _is_numpy(A):
        I = _np.eye(N, dtype=A.dtype)
        A_eff = A + I if add_self_loops else A
        d = A_eff.sum(axis=1)
        if normalized:
            inv_sqrt = 1.0 / _np.sqrt(_np.maximum(d, eps))
            Dm12_A = A * inv_sqrt[:, None] * inv_sqrt[None, :]
            return I - Dm12_A
        else:
            D = _np.diag(d)
            return D - A
    else:
        device = A.device
        I = _torch.eye(N, dtype=A.dtype, device=device)
        A_eff = A + I if add_self_loops else A
        d = A_eff.sum(dim=1)
        if normalized:
            inv_sqrt = 1.0 / _torch.sqrt(_torch.clamp(d, min=eps))
            Dm12_A = A * inv_sqrt.view(-1, 1) * inv_sqrt.view(1, -1)
            return I - Dm12_A
        else:
            D = _torch.diag(d)
            return D - A


# ---------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    print("[laplacian] Running sanity tests...")

    # --- NumPy tests ---
    if _np is not None:
        # 1D: f(i)=i^2 -> Δf≈2 (interior)
        n = 16
        i = _np.arange(n, dtype=float)
        f = i ** 2
        Lf = apply_laplacian_1d(f, axis=0, h=1.0, bc="dirichlet")
        err_1d = float(_np.max(_np.abs(Lf[1:-1] - 2.0)))
        print(f"  1D (NumPy) max interior error: {err_1d:.3e}")
        assert err_1d < 1e-12

        # 2D: f(x,y)=x^2+y^2 -> Δf≈4 (interior)
        H, W = 20, 24
        yy = _np.arange(H, dtype=float).reshape(H, 1)
        xx = _np.arange(W, dtype=float).reshape(1, W)
        F = xx**2 + yy**2
        LF = apply_laplacian_2d(F, axes=(-2, -1), hx=1.0, hy=1.0, bc="dirichlet")
        err_2d = float(_np.max(_np.abs(LF[1:-1, 1:-1] - 4.0)))
        print(f"  2D (NumPy) max interior error: {err_2d:.3e}")
        assert err_2d < 1e-12

        # Graph: path graph, rows of L sum to 0
        A = _np.zeros((5, 5), dtype=float)
        for u in range(4):
            A[u, u+1] = A[u+1, u] = 1.0
        L = graph_laplacian(A, normalized=False, add_self_loops=False)
        row_sum = float(_np.max(_np.abs(L.sum(axis=1))))
        print(f"  Graph (NumPy) row-sum max: {row_sum:.3e}")
        assert row_sum < 1e-12

    # --- Torch tests ---
    if _HAVE_TORCH:
        # 1D
        n = 16
        i = _torch.arange(n, dtype=_torch.float32)
        f = i ** 2
        Lf = apply_laplacian_1d(f, axis=0, h=1.0, bc="dirichlet")
        err_1d_t = float(_torch.max(_torch.abs(Lf[1:-1] - 2.0)))
        print(f"  1D (Torch)  max interior error: {err_1d_t:.3e}")
        assert err_1d_t < 1e-6

        # 2D
        H, W = 20, 24
        yy_t = _torch.arange(H, dtype=_torch.float32).view(H, 1)
        xx_t = _torch.arange(W, dtype=_torch.float32).view(1, W)
        F_t = xx_t**2 + yy_t**2
        LF_t = apply_laplacian_2d(F_t, axes=(-2, -1), hx=1.0, hy=1.0, bc="dirichlet")
        err_2d_t = float(_torch.max(_torch.abs(LF_t[1:-1, 1:-1] - 4.0)))
        print(f"  2D (Torch)  max interior error: {err_2d_t:.3e}")
        assert err_2d_t < 1e-5

        # Graph
        A_t = _torch.zeros(5, 5, dtype=_torch.float32)
        for u in range(4):
            A_t[u, u+1] = A_t[u+1, u] = 1.0
        L_t = graph_laplacian(A_t, normalized=False)
        row_sum_t = float(_torch.max(_torch.abs(L_t.sum(dim=1))))
        print(f"  Graph (Torch) row-sum max: {row_sum_t:.3e}")
        assert row_sum_t < 1e-6

    print("[laplacian] All good ✓")

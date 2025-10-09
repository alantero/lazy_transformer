# utils/procrustes.py
# Lightweight (batched) orthonormalization and Procrustes alignment
# - Works with PyTorch tensors or NumPy arrays (auto-detects backend)
# - Rotation-only Procrustes (no translation, no scaling) by default
# - Shapes: [..., d, k] with k ≤ d (columns are basis vectors/features)

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

# Optional backends
try:  # PyTorch (preferred for GPU / autograd)
    import torch
    _HAVE_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAVE_TORCH = False

try:  # NumPy
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    np = None  # type: ignore
    _HAVE_NUMPY = False


# ------------------------------- small helpers -------------------------------

def _is_torch(x: Any) -> bool:
    return _HAVE_TORCH and isinstance(x, torch.Tensor)

def _is_numpy(x: Any) -> bool:
    return _HAVE_NUMPY and isinstance(x, np.ndarray)

def _assert_backend(x: Any) -> None:
    if not (_is_torch(x) or _is_numpy(x)):
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")

def _move_to_2d(x: Any) -> Tuple[Any, Tuple[int, ...]]:
    """
    Reshape [..., d, k] → [B, d, k] (B is product of leading dims; B≥1).
    Returns: (x2d, batch_shape)
    """
    _assert_backend(x)
    if x.ndim < 2:
        raise ValueError(f"Expected at least 2D array/tensor [..., d, k], got shape {tuple(x.shape)}")
    d, k = x.shape[-2], x.shape[-1]
    if _is_torch(x):
        B = int(x.numel() // (d * k))
        x2 = x.reshape(B, d, k)
        return x2, tuple(x.shape[:-2])
    else:
        B = int(x.size // (d * k))
        x2 = x.reshape(B, d, k)
        return x2, tuple(x.shape[:-2])

def _restore_batch(x2: Any, batch_shape: Tuple[int, ...]) -> Any:
    """Reshape [B, d, k] → [..., d, k] with the provided batch_shape."""
    _assert_backend(x2)
    if _is_torch(x2):
        return x2.reshape(*batch_shape, *x2.shape[-2:])
    else:
        return x2.reshape(*batch_shape, *x2.shape[-2:])

def _eye_like(n: int, ref: Any):
    if _is_torch(ref):
        return torch.eye(n, dtype=ref.dtype, device=ref.device)
    return np.eye(n, dtype=ref.dtype)

def _matmul(a: Any, b: Any) -> Any:
    if _is_torch(a):
        return a @ b
    return a @ b

def _transpose(a: Any) -> Any:
    if _is_torch(a):
        return a.transpose(-2, -1)
    return np.swapaxes(a, -2, -1)



"""
def _svd(a: Any) -> Tuple[Any, Any, Any]:
    # Returns U, S, Vt (so that a = U @ diag(S) @ Vt)
    if _is_torch(a):
        # torch.linalg.svd returns U, S, Vh
        U, S, Vh = torch.linalg.svd(a, full_matrices=False)
        return U, S, Vh
    else:
        U, S, Vt = np.linalg.svd(a, full_matrices=False)
        return U, S, Vt
"""

# utils/procrustes.py

#import torch
from contextlib import nullcontext

def _svd(a: torch.Tensor, full_matrices: bool = False):
    """
    SVD robusta:
      - Fuerza float32 (half/bfloat16 no siempre están soportados en gesvdj).
      - Desactiva autocast dentro de la op.
      - Si CUDA falla, cae a CPU y devuelve en el device original.
    """
    in_device = a.device
    try:
        # PyTorch ≥ 2.1
        autocast_ctx = torch.amp.autocast("cuda", enabled=False)
    except Exception:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        a32 = a.detach()
        if a32.dtype not in (torch.float32, torch.float64):
            a32 = a32.float()
        try:
            U, S, Vh = torch.linalg.svd(a32, full_matrices=full_matrices)
        except RuntimeError:
            # Fallback CPU
            U, S, Vh = torch.linalg.svd(a32.cpu(), full_matrices=full_matrices)
            U = U.to(in_device)
            S = S.to(in_device)
            Vh = Vh.to(in_device)
    return U, S, Vh


def _qr(a: Any) -> Tuple[Any, Any]:
    if _is_torch(a):
        Q, R = torch.linalg.qr(a, mode="reduced")
        return Q, R
    else:
        Q, R = np.linalg.qr(a, mode="reduced")
        return Q, R

def _norm_fro(a: Any) -> Any:
    if _is_torch(a):
        return torch.linalg.norm(a, ord="fro")
    else:
        return np.linalg.norm(a, ord="fro")


# ------------------------------ public functions ------------------------------

def orthobase_fit(X: Any, rank: Optional[int] = None, method: str = "qr") -> Any:
    """
    Orthonormalize the columns of X (last dimension), batched.
    Args:
        X: [..., d, k] tensor/array
        rank: if provided, truncate to first `rank` orthonormal columns
        method: "qr" (default) or "svd"
    Returns:
        Q: [..., d, r] with Q^T Q = I_r
    """
    _assert_backend(X)
    X2, bshape = _move_to_2d(X)             # [B, d, k]
    B, d, k = X2.shape
    r = min(rank if rank is not None else k, k)
    if r < 1:
        raise ValueError("rank must be ≥ 1.")

    if method not in ("qr", "svd"):
        raise ValueError("method must be 'qr' or 'svd'.")

    if method == "qr":
        Q_all = []
        for i in range(B):
            Qi, _ = _qr(X2[i])              # [d, k]
            Q_all.append(Qi[..., :r])       # truncate if needed
        Q2 = torch.stack(Q_all) if _is_torch(X2) else np.stack(Q_all, axis=0)  # [B, d, r]
    else:  # svd
        Q_all = []
        for i in range(B):
            U, S, Vt = _svd(X2[i])          # [d, k], [k], [k, k]
            Q_all.append(U[..., :r])
        Q2 = torch.stack(Q_all) if _is_torch(X2) else np.stack(Q_all, axis=0)

    return _restore_batch(Q2, bshape)


def procrustes(
    A: Any,
    B: Any,
    *,
    allow_reflection: bool = False,
    center: bool = False,
    scale: bool = False,
    return_stats: bool = True,
) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
    """
    Rotation-only Procrustes alignment (optionally: center/scale) with batching.

    We solve:  R* = argmin_R || A R - B ||_F
    with R orthogonal (det(R)=+1 unless `allow_reflection=True`).
    Shapes:
        A: [..., d, k]
        B: [..., d, k]
    Returns:
        A_aligned: [..., d, k]  (A @ R)
        R:         [..., k, k]
        stats:     {'mse': ..., 'det': ...}  (optional)
    Notes:
        - If center=True, columns are mean-centered before alignment.
        - If scale=True, a global scalar is fitted for A (after rotation).
    """
    _assert_backend(A); _assert_backend(B)
    if A.shape != B.shape:
        raise ValueError(f"Shapes must match, got A={tuple(A.shape)} vs B={tuple(B.shape)}.")

    # Flatten to [B, d, k]
    A2, bshape = _move_to_2d(A)
    B2, _bshapeB = _move_to_2d(B)
    if bshape != _bshapeB:
        raise ValueError("Batch shapes must match.")

    Bn, d, k = A2.shape
    if k > d:
        raise ValueError("Require k ≤ d.")

    # (Optional) center columns
    def _center_cols(X):
        if _is_torch(X):
            mu = X.mean(dim=-2, keepdim=True)  # mean over rows (d)
            return X - mu, mu
        else:
            mu = X.mean(axis=-2, keepdims=True)
            return X - mu, mu

    A_proc = A2
    B_proc = B2
    muA = muB = None

    if center:
        A_proc, muA = _center_cols(A2)
        B_proc, muB = _center_cols(B2)

    # Cross-covariance in feature space (k x k): M = A^T B
    # That aligns columns (bases/features) via Kabsch in k-dim
    def _kabsch(M):
        # Standard orthogonal Procrustes: M = A^T B = U S V^T, R* = U V^T
        if _is_torch(M):
            # run in float32 with autocast disabled to avoid Half CUDA limitations
            try:
                _ac = torch.amp.autocast("cuda", enabled=False)
            except Exception:
                _ac = nullcontext()
            with _ac:
                U, S, Vt = _svd(M)  # returns float32 tensors
                R = _matmul(U, Vt)  # [k, k]
                if not allow_reflection:
                    detR = torch.det(R)
                    if detR < 0:
                        U_adj = U.clone()
                        U_adj[..., :, -1] *= -1.0
                        R = _matmul(U_adj, Vt)
            return R
        else:
            U, S, Vt = _svd(M)  # numpy path
            R = _matmul(U, Vt)
            if not allow_reflection:
                detR = np.linalg.det(R)
                if detR < 0:
                    U[:, -1] *= -1.0
                    R = _matmul(U, Vt)
            return R

    # Batched Kabsch
    R_list, A_aligned_list, det_list = [], [], []
    scale_list = []

    for i in range(Bn):
        Mi = _matmul(_transpose(A_proc[i]), B_proc[i])  # [k, k]
        Ri = _kabsch(Mi)                                # [k, k]
        Ai_rot = _matmul(A_proc[i], Ri)                 # [d, k]

        si = None
        if scale:
            # Optimal scalar s = trace(B^T A R) / ||A||_F^2
            num = _matmul(_transpose(B_proc[i]), Ai_rot)   # [k,k]
            if _is_torch(num):
                num_trace = torch.trace(num)
                den = torch.sum(A_proc[i] * A_proc[i])
                si = (num_trace / (den + 1e-12)).item()
                Ai_rot = Ai_rot * si
            else:
                num_trace = np.trace(num)
                den = np.sum(A_proc[i] * A_proc[i])
                si = float(num_trace / (den + 1e-12))
                Ai_rot = Ai_rot * si

        # Undo centering if needed: A_aligned = Ai_rot + muB (to match B's location)
        if center and (muB is not None):
            Ai_rot = Ai_rot + muB[i]

        # Save
        if _is_torch(A2):
            R_list.append(Ri)
            A_aligned_list.append(Ai_rot)
            if _is_torch(Ri):
                try:
                    _ac = torch.amp.autocast("cuda", enabled=False)
                except Exception:
                    _ac = nullcontext()
                with _ac:
                    det_list.append(torch.det(Ri.to(torch.float32)).unsqueeze(0))
            else:  # numerical corner (should not occur when torch)
                det_list.append(torch.tensor(np.linalg.det(Ri)).unsqueeze(0))
            if scale:
                scale_list.append(torch.tensor(si).unsqueeze(0))
        else:
            R_list.append(Ri)
            A_aligned_list.append(Ai_rot)
            det_list.append(np.array([np.linalg.det(Ri)], dtype=Ai_rot.dtype))
            if scale:
                scale_list.append(np.array([si], dtype=Ai_rot.dtype))

    # Stack back
    if _is_torch(A2):
        Rb = torch.stack(R_list, dim=0)                  # [B, k, k]
        Ab = torch.stack(A_aligned_list, dim=0)          # [B, d, k]
        detb = torch.cat(det_list, dim=0)                # [B]
    else:
        Rb = np.stack(R_list, axis=0)
        Ab = np.stack(A_aligned_list, axis=0)
        detb = np.concatenate(det_list, axis=0)

    # Stats
    stats = None
    if return_stats:
        if _is_torch(A2):
            mse = torch.mean((Ab - B2) ** 2, dim=(-2, -1))  # [B]
            stats = {"mse": mse, "det": detb}
            if scale:
                stats["scale"] = torch.cat(scale_list, dim=0)
        else:
            mse = np.mean((Ab - B2) ** 2, axis=(-2, -1))     # [B]
            stats = {"mse": mse, "det": detb}
            if scale:
                stats["scale"] = np.concatenate(scale_list, axis=0)

    # Restore batch shape
    A_align = _restore_batch(Ab, bshape)
    R = _restore_batch(Rb, bshape)
    return A_align, R, stats


# --------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    print("[procrustes] Running sanity tests...")

    # Prefer torch for batched tests if available
    if _HAVE_TORCH:
        torch.manual_seed(0)
        B, d, k = 3, 16, 4
        # Build a random orthonormal basis B: [..., d, k]
        X = torch.randn(B, d, k)
        Bbasis = orthobase_fit(X, rank=k, method="qr")            # [B,d,k]

        # Build a random k×k rotation R_true (via SVD on random matrix)
        G = torch.randn(B, k, k)
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        R_true = U @ Vh
        # Enforce det(R_true)=+1 to be consistent with allow_reflection=False
        dets = torch.linalg.det(R_true)
        if (dets < 0).any():
            U_adj = U.clone()
            U_adj[dets < 0, :, -1] *= -1.0
            R_true = U_adj @ Vh
        A = Bbasis @ R_true                                       # [B,d,k]
        # Add small noise
        A_noisy = A + 1e-4 * torch.randn_like(A)

        A_aligned, R_est, stats = procrustes(A_noisy, Bbasis, allow_reflection=False, center=False, scale=False, return_stats=True)
        err = float(stats["mse"].mean()) if isinstance(stats["mse"], torch.Tensor) else float(stats["mse"].mean())
        det_mean = float(stats["det"].mean()) if isinstance(stats["det"], torch.Tensor) else float(stats["det"].mean())
        print(f"  Torch: mean MSE={err:.3e}, mean det(R)={det_mean:.3f}")
        assert err < 1e-6, "Alignment should be very accurate on near-rotated data."
        assert abs(det_mean - 1.0) < 1e-3, "det(R) should be close to +1."

        # Center+scale path (sanity): if we scale down B, solve back
        A_scaled = 0.5 * A_noisy
        A_align_s, R_s, st_s = procrustes(A_scaled, Bbasis, center=True, scale=True, return_stats=True)
        err_s = float(st_s["mse"].mean()) if _is_torch(st_s["mse"]) else float(st_s["mse"].mean())
        print(f"  Torch (center+scale): mean MSE={err_s:.3e}")
        assert err_s < 1e-5

    # NumPy path
    if _HAVE_NUMPY:
        rng = np.random.default_rng(0)
        Bn, d, k = 2, 12, 3
        X = rng.standard_normal((Bn, d, k))
        # Orthonormalization with NumPy
        # (apply per batch)
        B_list = []
        for i in range(Bn):
            Qi, _ = np.linalg.qr(X[i])
            B_list.append(Qi[:, :k])
        Bnp = np.stack(B_list, axis=0)

        G = rng.standard_normal((Bn, k, k))
        # SVD-based rotation: R_true = U @ V^T
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
        R_true = U @ Vt  # Vt is V^T in NumPy
        # Enforce det(R_true)=+1 to be consistent with allow_reflection=False
        dets = np.linalg.det(R_true)
        if np.any(dets < 0):
            U_adj = U.copy()
            if U_adj.ndim == 3:
                U_adj[dets < 0, :, -1] *= -1.0
            else:
                U_adj[:, -1] *= -1.0
            R_true = U_adj @ Vt
        A = Bnp @ R_true
        A_noisy = A + 1e-4 * rng.standard_normal(A.shape)

        A_aligned, R_est, stats = procrustes(A_noisy, Bnp, allow_reflection=False, center=False, scale=False, return_stats=True)
        err = float(stats["mse"].mean())
        print(f"  NumPy: mean MSE={err:.3e}")
        assert err < 1e-6

    print("[procrustes] All good ✓")

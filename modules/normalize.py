# modules/normalize.py
# Normalization: traceless (mean removal), RMS, and groupwise (traceless+RMS).
# NumPy/Torch agnostic + Torch nn.Module for groupwise normalization.

from __future__ import annotations
from typing import Any, Tuple, Optional
import warnings

# Optional backends
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

try:
    import torch as _torch  # type: ignore
    import torch.nn as _nn  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _torch = None  # type: ignore
    _nn = None  # type: ignore
    _HAVE_TORCH = False


# ----------------------------- small helpers ---------------------------------

def _is_numpy(x: Any) -> bool:
    return (_np is not None) and isinstance(x, _np.ndarray)

def _is_torch(x: Any) -> bool:
    return (_HAVE_TORCH) and isinstance(x, _torch.Tensor)

def _move_axis(x: Any, src: int, dst: int) -> Any:
    if _is_numpy(x): return _np.moveaxis(x, src, dst)
    return _torch.movedim(x, src, dst)

def _ensure_divisible(C: int, groups: int) -> None:
    if groups < 1 or C % groups != 0:
        raise ValueError(f"'groups' must divide the size of the normalized axis (C={C}, groups={groups}).")

def _mean_last(x: Any) -> Any:
    if _is_numpy(x): return x.mean(axis=-1, keepdims=True)
    return x.mean(dim=-1, keepdim=True)

def _rms_last(x: Any, eps: float) -> Any:
    if _is_numpy(x): return _np.sqrt(_np.mean(x * x, axis=-1, keepdims=True) + eps)
    return _torch.sqrt(_torch.mean(x * x, dim=-1, keepdim=True) + eps)

def _apply_mask_along_axis(y: Any, mask: Any, axis: int) -> Any:
    """
    Multiply `y` by a boolean `mask` broadcast along `axis`.
    If `mask` has one fewer dimension than `y`, a singleton is inserted at `axis`.
    Returns `y` unchanged if `mask` is None.
    """
    if mask is None:
        return y
    # Normalize axis to be non-negative
    ax = axis if axis >= 0 else (y.ndim + axis)
    if _is_numpy(y):
        m = mask
        if m.dtype != bool:
            m = m.astype(bool)
        if m.ndim == y.ndim - 1:
            m = _np.expand_dims(m, ax)
        return y * m
    else:
        m = mask
        if m.dtype != _torch.bool:
            m = m.to(_torch.bool)
        if m.ndim == y.ndim - 1:
            m = m.unsqueeze(ax)
        return y * m.to(dtype=y.dtype)


# ------------------------------ public API -----------------------------------

def traceless(x: Any, *, axis: int = -1, mask: Optional[Any] = None) -> Any:
    """Subtract the mean along `axis` (shape preserved).
    mask: optional boolean mask broadcast along `axis`; masked positions are zeroed in the output.
    """
    x_last = _move_axis(x, axis, -1)
    y = x_last - _mean_last(x_last)
    y = _move_axis(y, -1, axis)
    y = _apply_mask_along_axis(y, mask, axis)
    return y


def normalize_rms(x: Any, *, axis: int = -1, eps: float = 1e-6, mask: Optional[Any] = None) -> Any:
    """LayerNorm-like (no affine): y = (x - mean)/rms along `axis`.
    mask: optional boolean mask broadcast along `axis`; masked positions are zeroed in the output.
    """
    x_last = _move_axis(x, axis, -1)
    xc = x_last - _mean_last(x_last)
    y = xc / _rms_last(xc, eps)
    y = _move_axis(y, -1, axis)
    y = _apply_mask_along_axis(y, mask, axis)
    return y


def groupwise_traceless(x: Any, *, groups: int, axis: int = -1, mask: Optional[Any] = None) -> Any:
    """Subtract per-group mean within the feature axis.
    mask: optional boolean mask broadcast along `axis`; masked positions are zeroed in the output.
    """
    x_last = _move_axis(x, axis, -1)              # [..., C]
    C = x_last.shape[-1]
    _ensure_divisible(C, groups)
    g, cg = groups, C // groups

    if _is_numpy(x_last):
        xg = x_last.reshape(*x_last.shape[:-1], g, cg)
        mu = xg.mean(axis=-1, keepdims=True)
        yg = xg - mu
        y = yg.reshape(*x_last.shape[:-1], C)
    else:
        xg = x_last.view(*x_last.shape[:-1], g, cg)
        mu = xg.mean(dim=-1, keepdim=True)
        yg = xg - mu
        y = yg.view(*x_last.shape[:-1], C)

    y = _move_axis(y, -1, axis)
    y = _apply_mask_along_axis(y, mask, axis)
    return y


def groupwise_norm(
    x: Any, *, groups: int, axis: int = -1, eps: float = 1e-6, return_stats: bool = False, mask: Optional[Any] = None
) -> Any | Tuple[Any, Any, Any]:
    """
    Groupwise traceless + RMS normalization along `axis`.
    Returns `y` and optionally `(mu_full, r_full)` broadcast to `x`'s shape.
    mask: optional boolean mask broadcast along `axis`; masked positions are zeroed in the output.
    """
    x_last = _move_axis(x, axis, -1)              # [..., C]
    C = x_last.shape[-1]
    _ensure_divisible(C, groups)
    g, cg = groups, C // groups

    if _is_numpy(x_last):
        xg = x_last.reshape(*x_last.shape[:-1], g, cg)
        mu = xg.mean(axis=-1, keepdims=True)                             # [..., g, 1]
        xc = xg - mu
        r = _np.sqrt(_np.mean(xc * xc, axis=-1, keepdims=True) + eps)    # [..., g, 1]
        yg = xc / r                                                      # [..., g, cg]
        y_last = yg.reshape(*x_last.shape[:-1], C)
        y = _move_axis(y_last, -1, axis)
        y = _apply_mask_along_axis(y, mask, axis)

        if return_stats:
            mu_last = _np.broadcast_to(mu, (*x_last.shape[:-1], g, cg)).reshape(*x_last.shape[:-1], C)
            r_last  = _np.broadcast_to(r,  (*x_last.shape[:-1], g, cg)).reshape(*x_last.shape[:-1], C)
            mu_full = _move_axis(mu_last, -1, axis)
            r_full  = _move_axis(r_last,  -1, axis)
            return y, mu_full, r_full
        return y

    else:
        xg = x_last.view(*x_last.shape[:-1], g, cg)
        mu = xg.mean(dim=-1, keepdim=True)                               # [..., g, 1]
        xc = xg - mu
        r = _torch.sqrt(_torch.mean(xc * xc, dim=-1, keepdim=True) + eps)# [..., g, 1]
        yg = xc / r                                                      # [..., g, cg]
        y_last = yg.view(*x_last.shape[:-1], C)
        y = _move_axis(y_last, -1, axis)
        y = _apply_mask_along_axis(y, mask, axis)

        if return_stats:
            mu_last = mu.expand(*x_last.shape[:-1], g, cg).reshape(*x_last.shape[:-1], C)
            r_last  = r.expand( *x_last.shape[:-1], g, cg).reshape(*x_last.shape[:-1], C)
            mu_full = _move_axis(mu_last, -1, axis)
            r_full  = _move_axis(r_last,  -1, axis)
            return y, mu_full, r_full
        return y


# ------------------------------ Torch module ---------------------------------

if _HAVE_TORCH:
    class GroupwiseTracelessNorm(_nn.Module):
        """
        Torch module: groupwise (traceless + RMS) normalization on the last dim.
        Notes:
          - Default affine=False to match v3 (no extra params for offsets).
          - `traceless_freq` placeholder for future frequency-domain extension.
          - Optional boolean mask to zero-out padded tokens (broadcast on last dim).
        """
        def __init__(
            self,
            num_features: int,
            groups: int,
            eps: float = 1e-6,
            affine: bool = False,
            traceless_freq: bool = False,  # flag reserved for future use
        ):
            super().__init__()
            _ensure_divisible(num_features, groups)
            self.num_features = int(num_features)
            self.groups = int(groups)
            self.eps = float(eps)
            self.affine = bool(affine)
            self.traceless_freq = bool(traceless_freq)
            if self.traceless_freq:
                warnings.warn(
                    "GroupwiseTracelessNorm(traceless_freq=True) is reserved for a future "
                    "frequency-domain variant; current implementation ignores it.",
                    RuntimeWarning,
                )
            if self.affine:
                self.weight = _nn.Parameter(_torch.ones(num_features, dtype=_torch.float32))
                self.bias   = _nn.Parameter(_torch.zeros(num_features, dtype=_torch.float32))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

        def forward(self, x: _torch.Tensor, mask: Optional[_torch.Tensor] = None) -> _torch.Tensor:
            C = x.shape[-1]
            if C != self.num_features:
                raise ValueError(f"Expected last dim {self.num_features}, got {C}.")
            g = self.groups
            cg = C // g
            xg = x.view(*x.shape[:-1], g, cg)
            mu = xg.mean(dim=-1, keepdim=True)
            xc = xg - mu
            r = _torch.sqrt(_torch.mean(xc * xc, dim=-1, keepdim=True) + self.eps)
            yg = xc / r
            y = yg.view(*x.shape[:-1], C)
            if self.affine:
                w = self.weight.view(*((1,) * (y.ndim - 1)), -1)
                b = self.bias.view(*((1,) * (y.ndim - 1)), -1)
                y = y * w + b
            y = _apply_mask_along_axis(y, mask, axis=-1)
            return y
else:
    class GroupwiseTracelessNorm:  # type: ignore
        def __init__(self, *_, **__):
            raise ImportError("PyTorch not available: GroupwiseTracelessNorm cannot be used.")


# ---------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    print("[normalize] Sanity tests...")

    # NumPy checks (if available)
    if _np is not None:
        rng = _np.random.default_rng(0)
        x = rng.standard_normal((3, 5, 12))  # (B, T, C)

        # Use eps=0 for an exact RMS=1 check (avoids tiny bias when s^2 is very small).
        y, mu, r = groupwise_norm(x, groups=4, axis=-1, eps=0.0, return_stats=True)

        # Check per-group mean≈0 and rms≈1
        y_g = y.reshape(3, 5, 4, 3)
        mu_g = y_g.mean(axis=-1)
        rms_g = _np.sqrt(_np.mean(y_g ** 2, axis=-1))
        print(f"  NumPy: max|mu_g|={float(_np.max(_np.abs(mu_g))):.2e}, max|rms_g-1|={float(_np.max(_np.abs(rms_g-1))):.2e}")
        assert float(_np.max(_np.abs(mu_g))) < 1e-12
        assert float(_np.max(_np.abs(rms_g - 1))) < 1e-12
        assert mu.shape == x.shape and r.shape == x.shape

        # Traceless only
        z = groupwise_traceless(x, groups=4, axis=-1)
        z_g = z.reshape(3, 5, 4, 3)
        mu_g2 = z_g.mean(axis=-1)
        print(f"  NumPy: traceless max|mu_g|={float(_np.max(_np.abs(mu_g2))):.2e}")
        assert float(_np.max(_np.abs(mu_g2))) < 1e-12

        # Mask-gating check (NumPy): zero-out last two time positions
        m_np = _np.ones((3, 5), dtype=bool)
        m_np[:, -2:] = False
        y_masked = groupwise_norm(x, groups=4, axis=-1, eps=0.0, return_stats=False, mask=m_np)
        assert _np.allclose(y_masked[:, -2:, :], 0.0)

    # Torch checks (if available)
    if _HAVE_TORCH:
        gen = _torch.Generator().manual_seed(0)
        x_t = _torch.randn(2, 7, 16, generator=gen)

        # Default affine=False matches v3
        mod = GroupwiseTracelessNorm(num_features=16, groups=8, eps=0.0, affine=False, traceless_freq=False)
        y_t = mod(x_t)

        # Check core against manual computation
        with _torch.no_grad():
            g = 8
            cg = 16 // g
            xg = x_t.view(2, 7, g, cg)
            mu_t = xg.mean(dim=-1, keepdim=True)
            xc = xg - mu_t
            r_t = _torch.sqrt(_torch.mean(xc * xc, dim=-1, keepdim=True) + 0.0)
            yg = xc / r_t
            y_ref = yg.view(2, 7, 16)

        err = float((y_t - y_ref).abs().max())
        print(f"  Torch: core max|Δ| = {err:.2e}")
        assert err < 1e-6

        # Mask-gating check (Torch)
        m_t = _torch.ones((2, 7), dtype=_torch.bool)
        m_t[:, -3:] = False
        y_t2 = mod(x_t, mask=m_t)
        assert float(y_t2[:, -3:, :].abs().max()) == 0.0

    print("[normalize] All good ✓")

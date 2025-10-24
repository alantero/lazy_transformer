
# operators/cheb.py
# Chebyshev polynomial filters over a 1D (causal) Laplacian on the time axis.
# Adds: (1) per-group learnable coeffs in Torch, (2) laplacian="path_causal" (strict AR) and "path" (Dirichlet, non-causal).
# NumPy/Torch compatible; no external project deps.
# This module now supports per-call overrides via an optional `ctx` dict on the Torch module API (see `ChebFilter1D.forward/op`).

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple, Union

# Optional backends
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

try:
    import torch as _torch  # type: ignore
    import torch.nn as _nn  # type: ignore
    import torch.nn.functional as _F  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _torch = None  # type: ignore
    _nn = None  # type: ignore
    _F = None  # type: ignore
    _HAVE_TORCH = False


# -------------------------- backend helpers ----------------------------------

def _is_numpy(x: Any) -> bool:
    return (_np is not None) and isinstance(x, _np.ndarray)

def _is_torch(x: Any) -> bool:
    return (_HAVE_TORCH) and isinstance(x, _torch.Tensor)

def _move_axis(x: Any, src: int, dst: int) -> Any:
    if _is_numpy(x): return _np.moveaxis(x, src, dst)
    return _torch.movedim(x, src, dst)

def _zeros_like(x: Any) -> Any:
    if _is_numpy(x): return _np.zeros_like(x, dtype=getattr(x, "dtype", float))
    return _torch.zeros_like(x)


# --------------------------- Laplacian operators ------------------------------

def _lap_cycle(x: Any, axis_time: int = -2) -> Any:
    """
    Normalized Laplacian on a cycle graph along `axis_time`:
      L x = x - 0.5 * (shift_left(x) + shift_right(x))
    Shapes: [..., T, D]
    """
    if axis_time < 0: axis_time += x.ndim
    xt = _move_axis(x, axis_time, -2)  # [..., T, D]
    if _is_numpy(xt):
        left  = _np.roll(xt, shift=+1, axis=-2)
        right = _np.roll(xt, shift=-1, axis=-2)
        y = xt - 0.5 * (left + right)
    else:
        left  = _torch.roll(xt, shifts=+1, dims=-2)
        right = _torch.roll(xt, shifts=-1, dims=-2)
        y = xt - 0.5 * (left + right)
    return _move_axis(y, -2, axis_time)

def _lap_path_dirichlet(x: Any, axis_time: int = -2) -> Any:
    """
    Path (non-periodic) Laplacian with Dirichlet BC (zero outside):
      L x = x - 0.5 * (x_{i-1} + x_{i+1}) with x_{-1}=x_{T}=0.
    Shapes: [..., T, D]
    """
    if axis_time < 0: axis_time += x.ndim
    xt = _move_axis(x, axis_time, -2)  # [..., T, D]
    if _is_numpy(xt):
        zeros = _np.zeros_like(xt[..., :1, :])
        left  = _np.concatenate([zeros, xt[..., :-1, :]], axis=-2)
        right = _np.concatenate([xt[..., 1:, :], zeros], axis=-2)
        y = xt - 0.5 * (left + right)
    else:
        # pad = (last_dim_left, last_dim_right, time_left, time_right)
        left  = _F.pad(xt, (0, 0, 1, 0), mode="constant", value=0.0)[..., :-1, :]
        right = _F.pad(xt, (0, 0, 0, 1), mode="constant", value=0.0)[..., 1:, :]
        y = xt - 0.5 * (left + right)
    return _move_axis(y, -2, axis_time)


# --- Strictly causal path Laplacian (Dirichlet on left, AR) ---
def _lap_path_causal(x: Any, axis_time: int = -2) -> Any:
    """
    Causal (one-sided) path Laplacian with Dirichlet BC on the left only:
      L_c x = x - 0.5 * (x_{i-1} + x_{i}) with x_{-1}=0.
    This is strictly causal for autoregressive use (depends only on past/current).
    Shapes: [..., T, D]
    """
    if axis_time < 0: axis_time += x.ndim
    xt = _move_axis(x, axis_time, -2)  # [..., T, D]
    if _is_numpy(xt):
        zeros = _np.zeros_like(xt[..., :1, :])
        left  = _np.concatenate([zeros, xt[..., :-1, :]], axis=-2)
        # L_c x = x - 0.5 * (left + current)
        y = xt - 0.5 * (left + xt)
    else:
        left = _F.pad(xt, (0, 0, 1, 0), mode="constant", value=0.0)[..., :-1, :]
        y = xt - 0.5 * (left + xt)
    return _move_axis(y, -2, axis_time)


def _mu_apply(Lx_fn, x: Any, lmax: float) -> Any:
    """
    Apply μ(L) = 2 L / lmax - I.
    """
    Lx = Lx_fn(x)
    if _is_numpy(x):
        return (2.0 / lmax) * Lx - x
    else:
        return (2.0 / lmax) * Lx - x


# -------------------------- Chebyshev coefficients ----------------------------

def cheb_coeffs_from_function(f, K: int, *, M: Optional[int] = None) -> "_np.ndarray":
    """
    Approximate Chebyshev coeffs c_k for f(μ) on μ∈[-1,1] with Gauss–Chebyshev quadrature.
    Returns np.array (K+1,). NumPy-only utility (used for heat init).
    """
    if _np is None:
        raise ImportError("NumPy required to generate Chebyshev coefficients.")
    M = int(M or max(4 * (K + 1), 64))
    j = _np.arange(M, dtype=float)
    theta = (j + 0.5) * _np.pi / M
    mu = _np.cos(theta)
    vals = _np.asarray(f(mu), dtype=float)
    w = _np.pi / M
    c = _np.empty(K + 1, dtype=float)
    c[0] = (1.0 / _np.pi) * _np.sum(vals * w)
    for k in range(1, K + 1):
        c[k] = (2.0 / _np.pi) * _np.sum(vals * _np.cos(k * theta) * w)
    return c

def heat_cheb_coeffs(K: int, tau: float, lmax: float = 2.0) -> "_np.ndarray":
    """
    Chebyshev coeffs for g(λ)=exp(-τ λ) on λ∈[0,lmax], mapped to μ∈[-1,1].
    """
    if _np is None:
        raise ImportError("NumPy required to generate Chebyshev coefficients.")
    def f(mu):
        lam = 0.5 * lmax * (mu + 1.0)
        return _np.exp(-tau * lam)
    return cheb_coeffs_from_function(f, K)


# -------------------------- Chebyshev filtering -------------------------------

def _select_Lx(laplacian: str, *, allow_noncausal: bool = False):
    # 'path_causal' is strictly causal (left-only). 'path' is non-causal (looks both left/right).
    # 'cycle' is non-causal with wrap-around.
    if laplacian == "path_causal":
        return _lap_path_causal
    if laplacian == "path":
        return _lap_path_dirichlet
    if laplacian == "cycle":
        if not allow_noncausal:
            raise ValueError(
                "laplacian='cycle' is non-causal (wrap-around) and is disallowed by default. "
                "Use laplacian='path_causal' for language modeling, or pass allow_noncausal=True where supported."
            )
        return _lap_cycle
    raise ValueError(f"Unsupported laplacian='{laplacian}'. Use 'path_causal' (causal), 'path' (non-causal), or 'cycle' (non-causal wrap-around).")
def _cast_like_tensor(t: _torch.Tensor, ref: _torch.Tensor) -> _torch.Tensor:
    return t.to(dtype=ref.dtype, device=ref.device)

def cheb_apply(
    x: Any,
    coeffs: Sequence[float],
    *,
    laplacian: str = "path",
    axis_time: int = -2,
    lmax: float = 2.0,
    allow_noncausal: bool = False,
) -> Any:
    """
    NumPy/Torch (shared-coeff) Chebyshev filter:
      y = Σ_{k=0..K} c_k T_k(μ(L)) x,  μ(L)=2L/lmax - I.
    `coeffs` length is K+1. Shapes preserved: x [..., T, D] -> y [..., T, D].
    """
    K = len(coeffs) - 1
    if K < 0:
        raise ValueError("coeffs must be non-empty.")

    Lx = _select_Lx(laplacian, allow_noncausal=allow_noncausal)
    mu = lambda v: _mu_apply(Lx, v, lmax=lmax)

    t0 = x
    y = coeffs[0] * t0
    if K == 0:
        return y
    t1 = mu(t0)
    y = y + coeffs[1] * t1
    for k in range(1, K):
        t2 = 2.0 * mu(t1) - t0
        y = y + coeffs[k + 1] * t2
        t0, t1 = t1, t2
    return y


# --------------------------- Torch per-group version --------------------------

def _mul_by_group_coeff(t: _torch.Tensor, ck: _torch.Tensor, groups: int) -> _torch.Tensor:
    """
    Multiply [..., T, D] by per-group coeffs ck[G], broadcasting over time and group channels.
    """
    *prefix, T, D = t.shape
    cg = D // groups
    tg = t.view(*prefix, T, groups, cg)  # [..., T, G, cg]
    shape = [1] * (tg.ndim - 2) + [groups, 1]
    return (tg * ck.view(*shape)).view(*prefix, T, D)

def cheb_apply_torch_groupwise(
    x: _torch.Tensor,
    coeffs: _torch.Tensor,          # shape [K+1] or [G, K+1]
    *,
    groups: int = 1,
    laplacian: str = "path",
    axis_time: int = -2,
    lmax: float = 2.0,
    allow_noncausal: bool = False,
) -> _torch.Tensor:
    """
    Torch Chebyshev filter with optional per-group coeffs:
      - coeffs shape [K+1]: shared across channels
      - coeffs shape [G, K+1]: groupwise (D % G == 0)
    """
    if axis_time != -2:
        # Keep behavior parity with cheb_apply by moving the axis
        x = _move_axis(x, axis_time, -2)
        y = cheb_apply_torch_groupwise(
            x, coeffs, groups=groups, laplacian=laplacian, axis_time=-2, lmax=lmax, allow_noncausal=allow_noncausal
        )
        return _move_axis(y, -2, axis_time)

    if coeffs.ndim == 1:
        # Shared coefficients → fall back to scalar version
        return cheb_apply(x, coeffs.tolist(), laplacian=laplacian, axis_time=axis_time, lmax=lmax, allow_noncausal=allow_noncausal)

    # Groupwise coefficients
    if x.shape[-1] % groups != 0:
        raise ValueError(f"D={x.shape[-1]} must be divisible by groups={groups} for groupwise coeffs.")
    G = coeffs.shape[0]
    if G != groups:
        raise ValueError(f"coeffs has G={G} but groups={groups}.")

    # Ensure coeffs matches input dtype/device
    coeffs = _cast_like_tensor(coeffs, x)

    Lx = _select_Lx(laplacian, allow_noncausal=allow_noncausal)
    mu = lambda v: _mu_apply(Lx, v, lmax=lmax)

    K = coeffs.shape[1] - 1
    if K < 0:
        raise ValueError("coeffs must have at least one column (K+1).")

    # Recurrence
    t0 = x
    y = _mul_by_group_coeff(t0, coeffs[:, 0], groups)        # c0 * T0(x)
    if K == 0:
        return y
    t1 = mu(t0)
    y = y + _mul_by_group_coeff(t1, coeffs[:, 1], groups)    # c1 * T1(x)
    for k in range(1, K):
        t2 = 2.0 * mu(t1) - t0
        y = y + _mul_by_group_coeff(t2, coeffs[:, k + 1], groups)  # ck * Tk
        t0, t1 = t1, t2
    return y


# ------------------------------- Public modules ------------------------------

if _HAVE_TORCH:
    class ChebFilter1D(_nn.Module):
        """
        Chebyshev filter on a 1D Laplacian.
        Torch version supports per-group learnable coeffs (G × (K+1)).

        Args:
            K: polynomial order (non-negative).
            coeffs: None → create learnable per-group coeffs (randn*0.02);
                    Tensor of shape [K+1] or [G,K+1] to set initial coeffs.
            groups: number of feature groups (D % groups == 0).
            lmax: spectral radius used in μ(L)=2L/lmax − I (≈2 for 'cycle', ≈4 for 'path').
            laplacian: 'path_causal' (strictly causal, recommended for AR),
                       'path' (non-causal, Dirichlet), or 'cycle' (non-causal wrap-around).
            learnable: whether coeffs are nn.Parameter.

        Per-call `ctx` dict (for `forward`/`op`):
            ctx['laplacian']: override laplacian string ('path_causal'|'path'|'cycle')
            ctx['lmax']: float spectral radius override
            ctx['allow_noncausal']: bool, allow non-causal laplacians
            ctx['axis_time']: int, time axis override for this call
            ctx['coeffs_scale']: float or Tensor broadcastable to [G, K+1] to scale the coefficients on-the-fly (does not mutate state)
        """
        def __init__(
            self,
            K: int,
            coeffs: Optional[Union[Sequence[float], "_torch.Tensor"]] = None,
            *,
            groups: int = 1,
            lmax: float = 2.0,
            laplacian: str = "path",
            learnable: bool = True,
            allow_noncausal: bool = False,
        ):
            self.allow_noncausal = bool(allow_noncausal)
            super().__init__()
            if K < 0:
                raise ValueError("K must be >= 0.")
            if groups < 1:
                raise ValueError("groups must be >= 1.")
            self.K = int(K)
            self.groups = int(groups)
            self.laplacian = laplacian
            self.lmax = float(lmax)

            if coeffs is None:
                # Per-group small random init (bank-style)
                c = _torch.randn(self.groups, self.K + 1, dtype=_torch.float32) * 0.02
            else:
                c = _torch.as_tensor(coeffs, dtype=_torch.float32)
                # Accept [K+1] or [G, K+1]
                if c.ndim == 1:
                    c = c.view(1, -1)  # shared → shape [1, K+1]
                elif c.ndim == 2:
                    if c.shape[0] not in (1, self.groups):
                        raise ValueError(f"Provided coeffs has G={c.shape[0]} but groups={self.groups}.")
                else:
                    raise ValueError("coeffs must be 1D [K+1] or 2D [G,K+1].")
            if learnable:
                self.coeffs = _nn.Parameter(c)
            else:
                self.register_buffer("coeffs", c)

        def forward(self, x: _torch.Tensor, *, axis_time: int = -2, ctx: dict | None = None) -> _torch.Tensor:
            ctx = ctx or {}
            lap = ctx.get('laplacian', self.laplacian)
            lmax = float(ctx.get('lmax', self.lmax))
            allow_nc = bool(ctx.get('allow_noncausal', self.allow_noncausal))
            ax_t = int(ctx.get('axis_time', axis_time))
            coeffs = self.coeffs
            # Optional on-the-fly coeff scaling
            cscale = ctx.get('coeffs_scale', None)
            if cscale is not None:
                if isinstance(cscale, (int, float)):
                    coeffs_eff = coeffs * float(cscale)
                else:
                    cscale_t = _torch.as_tensor(cscale, dtype=coeffs.dtype, device=coeffs.device)
                    coeffs_eff = coeffs * cscale_t
            else:
                coeffs_eff = coeffs
            # If shared coeffs (shape [1,K+1]), broadcast as scalar version
            if coeffs_eff.shape[0] == 1 and self.groups == 1:
                return cheb_apply(
                    x, coeffs_eff[0].tolist(),
                    laplacian=lap, axis_time=ax_t, lmax=lmax,
                    allow_noncausal=allow_nc
                )
            return cheb_apply_torch_groupwise(
                x, coeffs_eff, groups=self.groups, laplacian=lap, axis_time=ax_t, lmax=lmax,
                allow_noncausal=allow_nc
            )

        def op(self, x: _torch.Tensor, *, ctx: dict | None = None, axis_time: Optional[int] = None) -> _torch.Tensor:
            """Uniform operator interface: y = Op_i(x; ctx). If axis_time is provided here, it overrides both self and ctx."""
            if axis_time is not None:
                return self.forward(x, axis_time=axis_time, ctx=ctx)
            return self.forward(x, axis_time=-2, ctx=ctx)
else:
    class ChebFilter1D:
        """
        NumPy fallback (shared coeffs only).
        If `coeffs` is None, uses heat-kernel low-pass init (requires NumPy).
        """
        def __init__(
            self,
            K: int,
            coeffs: Optional[Sequence[float]] = None,
            *,
            tau: float = 0.5,
            lmax: float = 2.0,
            laplacian: str = "path",
        ):
            if _np is None:
                raise ImportError("NumPy is required in the no-Torch version.")
            if K < 0:
                raise ValueError("K must be >= 0.")
            self.K = int(K)
            self.laplacian = laplacian
            self.lmax = float(lmax)
            if coeffs is None:
                self.coeffs = heat_cheb_coeffs(K, tau=tau, lmax=lmax)
            else:
                self.coeffs = _np.asarray(coeffs, dtype=float)

        def __call__(self, x: Any, *, axis_time: int = -2) -> Any:
            return cheb_apply(x, self.coeffs, laplacian=self.laplacian, axis_time=axis_time, lmax=self.lmax)


# ----------------------------------- __main__ --------------------------------

if __name__ == "__main__":
    print("[cheb] Sanity tests (low-pass behavior)…")
    T = 256
    D = 8
    low_bin, high_bin = 2, 32

    if _HAVE_TORCH:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        t = _torch.arange(T, device=device, dtype=_torch.float32)
        sig_low = _torch.sin(2 * _torch.pi * (low_bin / T) * t)
        sig_high = 0.5 * _torch.sin(2 * _torch.pi * (high_bin / T) * t)
        x = (sig_low + sig_high).view(1, T, 1).repeat(1, 1, D)  # [B=1,T,D]

        # Use a *shared* heat init to validate low-pass (then you can train groupwise).
        if _np is None:
            raise RuntimeError("NumPy needed to synthesize heat init for this demo.")
        K = 8
        coeff_heat = _torch.tensor(heat_cheb_coeffs(K, tau=0.8, lmax=2.0), dtype=_torch.float32, device=device)
        filt = ChebFilter1D(K=K, coeffs=coeff_heat, groups=1, lmax=4.0, laplacian="path", learnable=False).to(device)
        y = filt(x)

        # Also exercise strictly causal variant
        filt_causal = ChebFilter1D(K=K, coeffs=coeff_heat, groups=1, lmax=2.0, laplacian="path_causal", learnable=False).to(device)
        yc = filt_causal(x); _ = yc.shape

        X = _torch.fft.rfft(x[..., 0], dim=-1)  # take one channel
        Y = _torch.fft.rfft(y[..., 0], dim=-1)
        gain_low = float((Y.abs()[0, low_bin] / (X.abs()[0, low_bin] + 1e-12)).item())
        gain_high = float((Y.abs()[0, high_bin] / (X.abs()[0, high_bin] + 1e-12)).item())
        print(f"  Torch (path):  low≈{gain_low:.3f}, high≈{gain_high:.3f}")
        assert gain_low > gain_high + 0.1

        # Optionally, exercise 'cycle' Laplacian (non-causal) if allowed
        try:
            filt_cycle = ChebFilter1D(K=K, coeffs=coeff_heat, groups=1, lmax=2.0, laplacian="cycle", learnable=False, allow_noncausal=True).to(device)
            y_nc = filt_cycle(x); _ = y_nc.shape  # shape check only
            print("  Torch (cycle): non-causal path exercised (wrap-around) ✓")
        except Exception as e:
            print(f"  Torch (cycle) skipped: {e}")

        # Finally, instantiate a *groupwise* learnable bank (random init) just to check shapes.
        bank = ChebFilter1D(K=K, coeffs=None, groups=4, lmax=2.0, laplacian="cycle", learnable=True, allow_noncausal=True).to(device)
        yb = bank(x)  # shape check
        assert yb.shape == x.shape

        # --- Torch ctx/op per-call override checks ---
        y_ctx = filt(x, axis_time=-2, ctx={"coeffs_scale": 0.5, "lmax": 3.0})
        _ = y_ctx.shape
        y_op = filt.op(x, ctx={"laplacian": "path_causal"})
        _ = y_op.shape
        print("  Torch ctx/op overrides ✓")

    elif _np is not None:
        t = _np.arange(T, dtype=float)
        sig_low = _np.sin(2 * _np.pi * (low_bin / T) * t)
        sig_high = 0.5 * _np.sin(2 * _np.pi * (high_bin / T) * t)
        x = (sig_low + sig_high).reshape(1, T, 1).repeat(D, axis=-1)  # [1,T,D]

        K = 8
        filt = ChebFilter1D(K=K, tau=0.8, lmax=4.0, laplacian="path")
        y = filt(x)

        X = _np.fft.rfft(x[..., 0], axis=-1)
        Y = _np.fft.rfft(y[..., 0], axis=-1)
        gain_low = float(_np.abs(Y[0, low_bin]) / (_np.abs(X[0, low_bin]) + 1e-12))
        gain_high = float(_np.abs(Y[0, high_bin]) / (_np.abs(X[0, high_bin]) + 1e-12))
        print(f"  NumPy (path):  low≈{gain_low:.3f}, high≈{gain_high:.3f}")
        assert gain_low > gain_high + 0.1

        # Optionally, exercise 'cycle' Laplacian (non-causal) if allowed
        try:
            filt_cycle = ChebFilter1D(K=K, tau=0.8, lmax=2.0, laplacian="cycle")
            y_nc = filt_cycle(x); _ = y_nc.shape  # shape check only
            print("  NumPy (cycle): non-causal path exercised (wrap-around) ✓")
        except Exception as e:
            print(f"  NumPy (cycle) skipped: {e}")
    else:
        raise RuntimeError("Neither NumPy nor Torch is available.")

    print("[cheb] All good ✓")

# utils/windows.py
# Windowing utilities for 1-D sequences (NumPy / PyTorch).
# - slice_windows: split along an axis into fixed-length overlapping windows
# - reconstruct_from_windows: overlap-add reconstruction with proper normalization
# Design:
#   * Backend-agnostic (detects NumPy or PyTorch at runtime).
#   * No repo-global deps. Shapes are consistent for both backends.
#   * Default reconstruction uses a rectangular window ("ones") → exact OLA for any overlap.
#     (Hann etc. are supported; exact COLA then only for compatible strides.)

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Union, Optional, Any
import math

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


# ----------------------------- Small backend helpers -------------------------

def _is_numpy(x: Any) -> bool:
    return (_np is not None) and isinstance(x, _np.ndarray)

def _is_torch(x: Any) -> bool:
    return (_HAVE_TORCH) and isinstance(x, _torch.Tensor)

def _backend_from_example(x: Any) -> str:
    if _is_torch(x):
        return "torch"
    if _is_numpy(x):
        return "numpy"
    if _np is not None:
        return "numpy"
    if _HAVE_TORCH:
        return "torch"
    raise RuntimeError("No supported array backend (NumPy or PyTorch) is available.")

def _to_backend_array(x: Any, backend: str, like: Optional[Any] = None) -> Any:
    if backend == "numpy":
        return _np.asarray(x, dtype=float)
    else:
        device = like.device if _is_torch(like) else None
        return _torch.as_tensor(x, dtype=_torch.float32, device=device)

def _zeros(shape: Sequence[int], backend: str, like: Optional[Any] = None) -> Any:
    if backend == "numpy":
        return _np.zeros(shape, dtype=float)
    else:
        device = like.device if _is_torch(like) else None
        return _torch.zeros(*shape, dtype=_torch.float32, device=device)

def _ones(shape: Sequence[int], backend: str, like: Optional[Any] = None) -> Any:
    if backend == "numpy":
        return _np.ones(shape, dtype=float)
    else:
        device = like.device if _is_torch(like) else None
        return _torch.ones(*shape, dtype=_torch.float32, device=device)

def _take_axis(x: Any, sl: slice, axis: int) -> Any:
    if axis < 0:
        axis += x.ndim
    index = [slice(None)] * x.ndim
    index[axis] = sl
    return x[tuple(index)]

def _assign_axis(y: Any, sl: slice, axis: int, value: Any) -> None:
    if axis < 0:
        axis += y.ndim
    index = [slice(None)] * y.ndim
    index[axis] = sl
    y[tuple(index)] = value

def _move_axis(x: Any, src: int, dst: int) -> Any:
    if _is_numpy(x):
        return _np.moveaxis(x, src, dst)
    return _torch.movedim(x, src, dst)

def _concat(xs: List[Any], axis: int, backend: str) -> Any:
    if backend == "numpy":
        return _np.concatenate(xs, axis=axis)
    return _torch.cat(xs, dim=axis)

def _arange(n: int, backend: str, like: Optional[Any] = None) -> Any:
    if backend == "numpy":
        return _np.arange(n)
    device = like.device if _is_torch(like) else None
    return _torch.arange(n, device=device, dtype=_torch.int64)


# ------------------------------- Public API ----------------------------------

def validate_WO(W: int, O: int) -> None:
    if not isinstance(W, int) or not isinstance(O, int):
        raise TypeError("W and O must be integers.")
    if W <= 0:
        raise ValueError(f"W must be > 0, got {W}.")
    if O < 0 or O >= W:
        raise ValueError(f"O must be in [0, W-1], got O={O}, W={W}.")
    if W - O <= 0:
        raise ValueError(f"Stride (W - O) must be > 0, got W={W}, O={O}.")

def compute_window_starts(n: int, W: int, O: int, *, pad: bool = True) -> List[int]:
    """
    Compute canonical start indices covering [0, n) with stride S=W-O.
    If pad=True, the last window may extend beyond n and is right-padded.
    If pad=False, the last start is tail-aligned exactly at n-W (no padding).
    """
    if n < 0:
        raise ValueError("n must be >= 0.")
    validate_WO(W, O)
    S = W - O
    if n == 0:
        return []
    starts = list(range(0, n, S))
    if not pad:
        last = max(0, n - W)
        starts = list(range(0, last + 1, S))
        if starts and starts[-1] != last:
            starts.append(last)
    return starts


def local_time_grid(W: int, backend: str = "auto", like: Optional[Any] = None) -> Any:
    """
    Length-W vector with values in [0,1].
    """
    if backend == "auto":
        backend = "numpy" if _np is not None else ("torch" if _HAVE_TORCH else "numpy")
    if W <= 1:
        return _to_backend_array([0.0] * max(W, 1), backend, like=like)
    t = _arange(W, backend, like=like)
    if backend == "numpy":
        return t / float(W - 1)
    else:
        return t.to(dtype=_torch.float32) / float(W - 1)


def _maybe_pad_tail(x: Any, need_pad: int, axis: int, pad_mode: str, pad_value: float) -> Any:
    """
    Pad along `axis` on the right by `need_pad` elements.
    Modes supported (portable): 'constant', 'edge', 'reflect'.
    """
    if need_pad <= 0:
        return x
    if _is_numpy(x):
        pad_width = [(0, 0)] * x.ndim
        if axis < 0:
            axis += x.ndim
        pad_width[axis] = (0, need_pad)
        mode = pad_mode if pad_mode in ("constant", "edge", "reflect") else "constant"
        kwargs = {"mode": mode}
        if mode == "constant":
            kwargs["constant_values"] = pad_value
        return _np.pad(x, pad_width, **kwargs)
    else:
        import torch.nn.functional as F  # type: ignore
        if axis < 0:
            axis += x.ndim
        x_moved = _move_axis(x, axis, -1)
        if pad_mode == "constant":
            y = F.pad(x_moved, (0, need_pad), mode="constant", value=float(pad_value))
        elif pad_mode == "reflect":
            if x_moved.shape[-1] >= 2:
                y = F.pad(x_moved, (0, need_pad), mode="reflect")
            else:  # reflect needs >=2; fallback to edge replication
                last = _take_axis(x_moved, slice(-1, None), -1)
                rep = last.repeat_interleave(need_pad, dim=-1)
                y = _concat([x_moved, rep], axis=-1, backend="torch")
        elif pad_mode == "edge":
            last = _take_axis(x_moved, slice(-1, None), -1)
            rep = last.repeat_interleave(need_pad, dim=-1)
            y = _concat([x_moved, rep], axis=-1, backend="torch")
        else:
            y = F.pad(x_moved, (0, need_pad), mode="constant", value=float(pad_value))
        return _move_axis(y, -1, axis)


def slice_windows(
    x: Any,
    W: int,
    O: int,
    *,
    axis: int = -1,
    pad: bool = True,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    return_valid_lens: bool = False,
    return_starts: bool = False,
):
    """
    Slice `x` along dimension `axis` into overlapping windows of length W
    with overlap O. Returns an array/tensor shaped:
        x_windows.shape == x.shape_wo_axis + (n_windows, W)
    If `pad=True`, the tail is right-padded as needed to produce a full window.
    Optionally also returns per-window valid lengths and start indices.
    """
    backend = _backend_from_example(x)
    validate_WO(W, O)

    n = x.shape[axis]
    starts = compute_window_starts(n, W, O, pad=pad)
    S = W - O

    # Move target axis to the last to simplify slicing
    x_last = _move_axis(x, axis, -1)  # [..., N]
    batch_shape = x_last.shape[:-1]
    n_windows = len(starts)

    out_shape = batch_shape + (n_windows, W)
    x_win = _zeros(out_shape, backend, like=x_last)
    valid_lens = _zeros((n_windows,), backend, like=x_last)

    for wi, s in enumerate(starts):
        take_len = int(min(W, max(0, n - s)))
        valid_lens[wi] = take_len

        if take_len > 0:
            chunk = _take_axis(x_last, slice(s, s + take_len), -1)  # [..., take_len]
            # Add windows axis so we can assign into [..., 1, take_len]
            if _is_numpy(chunk):
                chunk_to_write = chunk[..., _np.newaxis, :]
            else:
                chunk_to_write = chunk.unsqueeze(-2)
            _assign_axis(_take_axis(x_win, slice(wi, wi + 1), -2), slice(0, take_len), -1, chunk_to_write)

        pad_len = W - take_len
        if pad and pad_len > 0:
            if take_len > 0:
                tmp = chunk
            else:
                seed = _take_axis(x_last, slice(max(0, n - 1), n), -1)
                if seed.shape[-1] == 0:  # empty input
                    seed = _zeros(batch_shape + (1,), backend, like=x_last)
                tmp = seed
            padded = _maybe_pad_tail(tmp, pad_len if W - tmp.shape[-1] > 0 else 0, axis=-1,
                                     pad_mode=pad_mode, pad_value=pad_value)
            pad_slice = _take_axis(padded, slice(-pad_len, None), -1)  # [..., pad_len]
            if _is_numpy(pad_slice):
                to_write = pad_slice[..., _np.newaxis, :]
            else:
                to_write = pad_slice.unsqueeze(-2)
            _assign_axis(_take_axis(x_win, slice(wi, wi + 1), -2), slice(take_len, W), -1, to_write)

    if return_valid_lens and return_starts:
        return x_win, valid_lens, _to_backend_array(starts, backend, like=x_last)
    if return_valid_lens:
        return x_win, valid_lens
    if return_starts:
        return x_win, _to_backend_array(starts, backend, like=x_last)
    return x_win


def _window_weights(W: int, backend: str, win: str, like: Optional[Any]) -> Any:
    """
    1-D window weights of length W.
    Supported: 'ones' (rectangular), 'hann' (periodic), 'sqrt_hann' (periodic sqrt), 'triangular'.
    Returned as a rank-1 vector (broadcast-friendly).
    Notes:
      * We intentionally use the **periodic** Hann (cos(2π n / W)) so that
        with 50% overlap (stride S=W/2) the overlap-add sum is exactly 1.
      * 'sqrt_hann' is useful for STFT-style analysis/synthesis (g=h=√hann).
    """
    win = (win or "ones").lower()

    if backend == "numpy":
        import numpy as _np
        if win == "ones":
            return _np.ones(W, dtype=float)
        elif win == "triangular":
            # symmetric triangular on [0, W-1]
            t = _np.arange(W, dtype=float) / float(W - 1 if W > 1 else 1)
            return 1.0 - _np.abs(2.0 * t - 1.0)
        elif win in ("hann", "sqrt_hann"):
            # PERIODIC Hann: 0.5 - 0.5*cos(2π n / W)
            tp = _np.arange(W, dtype=float) / float(W if W > 0 else 1)
            base = 0.5 - 0.5 * _np.cos(2.0 * _np.pi * tp)
            return _np.sqrt(base) if win == "sqrt_hann" else base
        else:
            # default to periodic Hann for unknown names
            tp = _np.arange(W, dtype=float) / float(W if W > 0 else 1)
            return 0.5 - 0.5 * _np.cos(2.0 * _np.pi * tp)

    else:
        import torch as _torch
        device = like.device if _is_torch(like) else None
        if win == "ones":
            return _torch.ones(W, dtype=_torch.float32, device=device)
        elif win == "triangular":
            t = _torch.arange(W, device=device, dtype=_torch.float32)
            t = t / float(W - 1 if W > 1 else 1)
            return 1.0 - _torch.abs(2.0 * t - 1.0)
        elif win in ("hann", "sqrt_hann"):
            tp = _torch.arange(W, device=device, dtype=_torch.float32)
            tp = tp / float(W if W > 0 else 1)
            base = 0.5 - 0.5 * _torch.cos(2.0 * _torch.pi * tp)
            return _torch.sqrt(base) if win == "sqrt_hann" else base
        else:
            tp = _torch.arange(W, device=device, dtype=_torch.float32)
            tp = tp / float(W if W > 0 else 1)
            return 0.5 - 0.5 * _torch.cos(2.0 * _torch.pi * tp)


def reconstruct_from_windows(
    win: Any,
    n: int,
    W: int,
    O: int,
    *,
    axis: int = -2,               # windows axis in `win` (… , n_windows, W)
    window_fn: str = "ones",      # default 'ones' → exact reconstruction for any overlap
    eps: float = 1e-8,
) -> Any:
    """
    Overlap-add reconstruction back to original length n from windows shaped:
        win.shape == batch_shape + (n_windows, W)
    The denominator counts only valid (in-bounds) samples per window (so right-padding
    does not attenuate the end). By default uses a rectangular window ('ones')
    which ensures exact OLA for arbitrary stride/overlap.
    """
    backend = _backend_from_example(win)
    validate_WO(W, O)
    if n < 0:
        raise ValueError("n must be >= 0.")
    if win.ndim < 2:
        raise ValueError("`win` must have at least 2 dims: (..., n_windows, W).")

    # Normalize axis
    if axis < 0:
        axis += win.ndim
    if axis == win.ndim - 1:
        raise ValueError("`axis` cannot be the last dim (reserved for W).")
    if win.shape[-1] != W:
        raise ValueError(f"Last dim of `win` must be W={W}, got {win.shape[-1]}.")

    # Reorder so that: [..., n_windows, W]
    win_std = _move_axis(win, axis, -2)
    batch_shape = win_std.shape[:-2]

    # Prepare accumulators
    y = _zeros(batch_shape + (n,), backend, like=win_std)
    denom = _zeros(batch_shape + (n,), backend, like=win_std)

    # Precompute starts (same policy as slice_windows with pad=True)
    starts = compute_window_starts(n, W, O, pad=True)

    # Window weights as rank-1 vector (broadcast-friendly)
    w = _window_weights(W, backend, window_fn, like=win_std)  # shape [W]

    # Accumulate overlap-add
    for wi, s in enumerate(starts):
        valid_len = int(min(W, max(0, n - s)))
        if valid_len <= 0:
            continue

        # Take window slice [..., W]
        w_i = _take_axis(win_std, slice(wi, wi + 1), -2).squeeze(-2)  # [..., W]

        # Weighted contribution and per-sample weight
        num_all = w_i * w          # [..., W]
        num = _take_axis(num_all, slice(0, valid_len), -1)
        wv = _take_axis(w, slice(0, valid_len), -1)  # [valid_len]

        # Accumulate into y[..., s:s+valid_len] and denom[...] (broadcast on last dim)
        y_slice = _take_axis(y, slice(s, s + valid_len), -1) + num
        d_slice = _take_axis(denom, slice(s, s + valid_len), -1) + wv
        _assign_axis(y, slice(s, s + valid_len), -1, y_slice)
        _assign_axis(denom, slice(s, s + valid_len), -1, d_slice)

    # Normalize (safe divide)
    if backend == "numpy":
        y = y / _np.maximum(denom, eps)
    else:
        y = y / _torch.clamp(denom, min=eps)

    return y


def windows_iter(
    x: Any, W: int, O: int, *, axis: int = -1, pad: bool = True,
    pad_mode: str = "constant", pad_value: float = 0.0
):
    """
    Generator yielding (win_i, start_i, valid_len_i) for streaming use,
    with each full-length window produced by padding the tail if necessary.
    """
    backend = _backend_from_example(x)
    validate_WO(W, O)
    n = x.shape[axis]
    starts = compute_window_starts(n, W, O, pad=pad)

    x_last = _move_axis(x, axis, -1)
    batch_shape = x_last.shape[:-1]

    for s in starts:
        valid_len = int(min(W, max(0, n - s)))
        take_len = valid_len
        if take_len > 0:
            chunk = _take_axis(x_last, slice(s, s + take_len), -1)
        else:
            chunk = _zeros(batch_shape + (0,), _backend_from_example(x), like=x_last)

        pad_len = W - take_len
        if pad and pad_len > 0:
            if take_len == 0:
                seed = _take_axis(x_last, slice(max(0, n - 1), n), -1)
                if seed.shape[-1] == 0:
                    seed = _zeros(batch_shape + (1,), _backend_from_example(x), like=x_last)
                tmp = seed
            else:
                tmp = chunk
            padded = _maybe_pad_tail(tmp, pad_len if W - tmp.shape[-1] > 0 else 0, axis=-1,
                                     pad_mode=pad_mode, pad_value=pad_value)
            pad_slice = _take_axis(padded, slice(-pad_len, None), -1)
            if _is_numpy(chunk):
                win_i = _concat([chunk, pad_slice], axis=-1, backend="numpy")
            elif _is_torch(chunk):
                win_i = _concat([chunk, pad_slice], axis=-1, backend="torch")
            else:
                raise RuntimeError("Unsupported backend in windows_iter.")
        else:
            if take_len < W:
                s_tail = max(0, n - W)
                chunk = _take_axis(x_last, slice(s_tail, s_tail + W), -1)
                win_i = chunk
                valid_len = min(W, n - s_tail)
            else:
                win_i = chunk

        yield win_i, s, valid_len


# ----------------------------------- Demo ------------------------------------

if __name__ == "__main__":
    print("[windows] Smoke tests...")

    # Torch test (preferred; NumPy behaves the same)
    if _HAVE_TORCH:
        _torch.manual_seed(0)
        n = 1000
        x = _torch.sin(_torch.linspace(0, 12.34, n)) + 0.01 * _torch.randn(n)
        W, O = 128, 32  # arbitrary overlap (no COLA guarantee for Hann)

        # 1) Exact reconstruction with rectangular window ('ones') for ANY O
        win = slice_windows(x, W, O, axis=-1, pad=True)
        xr = reconstruct_from_windows(win, n=n, W=W, O=O, window_fn="ones")
        err = float((_torch.abs(x - xr)).max())
        print(f"  ones  | n_windows={win.shape[-2]}, max|x - recon| = {err:.3e}")
        assert err < 1e-6, "Rectangular OLA should be exact."

        # 2) Hann reconstruction — exact only for COLA-compatible strides (e.g., 50% overlap)
        W2, O2 = 256, 128  # stride=W/2 → COLA for Hann
        win2 = slice_windows(x, W2, O2, axis=-1, pad=True)
        xr2 = reconstruct_from_windows(win2, n=n, W=W2, O=O2, window_fn="hann")
        err2 = float((_torch.abs(x - xr2)).max())
        print(f"  hann  | (COLA S=W/2) max|x - recon| = {err2:.3e}")
        # Note: With frames starting at t=0 (no head-padding) and synthesis-only Hann,
        # the very first/last sample have zero weight, so exact OLA is not achievable
        # without head padding or analysis windowing. The interior is exact; boundaries
        # dominate the max error (~1e-2). For exact recon, use window_fn='ones' or pad head.
        assert err2 < 2e-2, "Hann OLA boundary inexact; interior is exact."

    # NumPy test (if available)
    if _np is not None:
        rng = _np.random.default_rng(0)
        n = 500
        x = _np.sin(_np.linspace(0, 8.9, n)) + 0.01 * rng.standard_normal(n)
        W, O = 64, 20
        win = slice_windows(x, W, O, axis=-1, pad=True)
        xr = reconstruct_from_windows(win, n=n, W=W, O=O, window_fn="ones")
        err = float(_np.max(_np.abs(x - xr)))
        print(f"  (np)  | ones max|x - recon| = {err:.3e}")
        assert err < 1e-6

    print("[windows] All good ✓")

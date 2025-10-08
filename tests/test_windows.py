# tests/test_windows.py
# Exact OLA with ones window; shapes and indexing sanity.

import torch
import pytest

@pytest.mark.parametrize("B,D,T,W,O", [(2, 3, 301, 64, 32), (1, 1, 127, 32, 16)])
def test_ola_ones_exact(B, D, T, W, O):
    torch.manual_seed(0)
    from utils.windows import slice_windows, reconstruct_from_windows

    # x: [B, T, D]  (loop.py expects axis=1 == time)
    x = torch.randn(B, T, D, dtype=torch.float32)
    win = slice_windows(x, W, O, axis=1, pad=True)         # [B, D, n_win, W]
    x_rec = reconstruct_from_windows(win, W, O, axis=1, length=T)

    assert x_rec.shape == x.shape
    err = float((x - x_rec).abs().max())
    assert err < 1e-6, f"OLA ones should be exact, got {err:.3e}"

def test_windows_shapes_small():
    torch.manual_seed(0)
    from utils.windows import slice_windows, reconstruct_from_windows

    B, D, T, W, O = 2, 4, 50, 16, 8
    x = torch.randn(B, T, D)
    win = slice_windows(x, W, O, axis=1, pad=True)
    assert win.ndim == 4 and win.shape[0] == B and win.shape[1] == D and win.shape[-1] == W

    x_rec = reconstruct_from_windows(win, W, O, axis=1, length=T)
    assert x_rec.shape == (B, T, D)

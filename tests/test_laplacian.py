# tests/test_laplacian.py
# Normalized Laplacian: symmetry, row-sum≈0, spectrum in [0,2].

import torch

def test_norm_laplacian_cycle_props():
    from utils.laplacian import build_norm_laplacian
    T = 128
    L = build_norm_laplacian(T, mode="cycle")  # torch.Tensor [T,T]
    assert L.shape == (T, T)

    # symmetry
    sym_err = float((L - L.T).abs().max())
    assert sym_err < 1e-6

    # row sums ~ 0
    rowsum = L.sum(dim=-1)
    assert float(rowsum.abs().max()) < 1e-6

    # spectrum in [0, 2]
    evals = torch.linalg.eigvalsh(L)  # real for symmetric
    mn = float(evals.min())
    mx = float(evals.max())
    assert mn > -1e-6 and mx < 2.0 + 1e-6

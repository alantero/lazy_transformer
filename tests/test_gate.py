# tests/test_gate.py
# ResidualGate and MulGate: closed/open limits per-group/per-channel.

import torch

def test_residual_gate_limits():
    from modules.gate import ResidualGate
    torch.manual_seed(0)
    B, T, D, G = 3, 11, 12, 4
    x = torch.randn(B, T, D)
    u = torch.randn(B, T, D)
    cap0 = torch.zeros(B, T, 1)     # closed
    cap1 = torch.ones(B, T, 1)      # open

    g = ResidualGate(d=D, groups=G)
    y0 = g(x, u=u, cap=cap0)
    y1 = g(x, u=u, cap=cap1)

    assert float((y0 - x).abs().max()) < 1e-3
    assert float((y1 - (x + u)).abs().max()) < 1e-3

def test_mul_gate_limits():
    from modules.gate import MulGate
    torch.manual_seed(0)
    B, T, D = 3, 11, 12
    x = torch.randn(B, T, D)
    cap0 = torch.zeros(B, T, 1)
    cap1 = torch.ones(B, T, 1)

    g = MulGate(d=D, per_channel=True)
    y0 = g(x, cap=cap0)
    y1 = g(x, cap=cap1)

    # closed ≈ 0, open ≈ x
    assert float(y0.abs().max()) < 2e-3
    assert float((y1 - x).abs().max()) < 2e-3

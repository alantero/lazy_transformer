# tests/test_ledger.py
# Ledger: entropy/capacity shapes and bounds.

import pytest
import torch
import importlib

def test_ledger_entropy_capacity():
    led = importlib.import_module("modules.ledger")

    # try to find entropy function
    entropy = getattr(led, "entropy", None) or getattr(led, "entropy_from_logits", None)
    capacity = getattr(led, "capacity", None) or getattr(led, "capacity_from_logits", None)
    if entropy is None or capacity is None:
        pytest.skip("ledger entropy/capacity helpers not found")

    torch.manual_seed(0)
    B, T, V = 2, 7, 12
    logits = torch.randn(B, T, V)

    H = entropy(logits)   # expect [B, T]
    assert H.shape == (B, T)
    assert torch.isfinite(H).all()
    # capacity typically ~ 1 - H/log(V); ensure within [0,1] after clipping
    caps = capacity(logits)  # any of (B,T,1), (B,T,G), (B,T,V)
    assert caps.shape[:2] == (B, T)
    assert torch.isfinite(caps).all()
    assert float(caps.min()) >= -1e-5 and float(caps.max()) <= 1.0 + 1e-5

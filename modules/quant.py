# modules/quant.py
# Lightweight uniform affine quantization utilities (PyTorch-only, self-contained).
# Includes:
#   • Min/Max EMA observer (per-tensor or per-channel)
#   • FakeQuant module (QAT-friendly, straight-through estimator)
#   • QuantLinear: linear layer with weight/activation fake-quant
#   • Simple export helpers for int8 weights (scale/zero-point)
#   • Int4 path: per-group spectral coefficients (e.g., Chebyshev) via a small allocator
#
# Notes:
# - Per-channel quant typically uses axis=0 (out_features) for Linear weights.
# - Symmetric int8 uses range [-127, 127] (“narrow range” to avoid -128 asymmetry).
# - FakeQuant uses y = x + (qdq(x) - x).detach() → gradients flow as identity (STE).

from __future__ import annotations
from typing import Optional, Tuple, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
_Round = Literal["nearest"]  # placeholder for future (stochastic, etc.)


# ------------------------------- core math utils -------------------------------

def _qrange(bits: int, signed: bool = True, narrow: bool = True) -> Tuple[int, int]:
    """Return (qmin, qmax) for given bitwidth."""
    if bits <= 0 or bits > 16:
        raise ValueError("bits must be in [1,16].")
    if signed:
        if bits == 1:
            return (-1, 1)  # not typical; placeholder
        # narrow range removes the most-negative value to keep symmetric magnitude
        qmin = - (2 ** (bits - 1) - (1 if narrow else 0))
        qmax =   (2 ** (bits - 1) - 1)
        # For narrow=True at 8 bits → [-127, 127]
        return (qmin, qmax)
    else:
        return (0, 2 ** bits - 1)


def _calc_qparams(
    x_min: Tensor,
    x_max: Tensor,
    *,
    bits: int = 8,
    symmetric: bool = True,
    signed: bool = True,
    narrow: bool = True,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, int, int]:
    """
    Compute affine quantization params (scale, zero_point, qmin, qmax).
    x_min/x_max broadcastable scalars or per-channel tensors.
    """
    qmin, qmax = _qrange(bits, signed=signed, narrow=narrow and signed)
    if symmetric:
        # symmetric → zero_point = 0 (for signed), scale by max_abs / qmax
        max_abs = torch.maximum(x_max.abs(), x_min.abs())
        max_abs = torch.clamp(max_abs, min=eps)
        scale = max_abs / float(qmax)
        zero_point = torch.zeros_like(scale)
    else:
        # asymmetric affine
        span = (x_max - x_min).clamp_min(eps)
        scale = span / float(qmax - qmin)
        zp = qmin - torch.round(x_min / scale)
        zero_point = torch.clamp(zp, qmin, qmax)
    return scale, zero_point, qmin, qmax


def _fake_qdq(
    x: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    *,
    qmin: int,
    qmax: int,
) -> Tensor:
    """
    Quantize-dequantize with affine params. Broadcasting supported.
    The output is float but lives on the quantization grid.
    """
    # Ensure broadcastable shapes
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    y = (q - zero_point) * scale
    # Straight-through estimator: gradient like identity
    return x + (y - x).detach()


# --------------------------------- observers ----------------------------------

class EMAMinMaxObserver(nn.Module):
    """
    Exponential moving average min/max observer.
      - per_tensor: axis=None (track global min/max over all dims)
      - per_channel: set axis to the channel dimension (e.g., -1 for activations, 0 for Linear weight)
    """
    def __init__(
        self,
        *,
        axis: Optional[int] = None,
        momentum: float = 0.95,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.axis = axis
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("min_val", torch.tensor(0.0), persistent=True)
        self.register_buffer("max_val", torch.tensor(0.0), persistent=True)
        self._inited: bool = False

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        if self.axis is None:
            cur_min = torch.min(x).detach()
            cur_max = torch.max(x).detach()
        else:
            ax = self.axis if self.axis >= 0 else (x.dim() + self.axis)
            # Reduce over all dims except axis
            reduce_dims = [d for d in range(x.dim()) if d != ax]
            cur_min = torch.amin(x, dim=reduce_dims).detach()
            cur_max = torch.amax(x, dim=reduce_dims).detach()

        if not self._inited:
            self.min_val = cur_min.to(device=x.device, dtype=x.dtype)
            self.max_val = cur_max.to(device=x.device, dtype=x.dtype)
            self._inited = True
            return

        m = self.momentum
        # match shapes for EMA
        self.min_val = self.min_val.to(device=x.device, dtype=x.dtype)
        self.max_val = self.max_val.to(device=x.device, dtype=x.dtype)
        self.min_val.mul_(m).add_(cur_min, alpha=1.0 - m)
        self.max_val.mul_(m).add_(cur_max, alpha=1.0 - m)

        # ensure max >= min + eps
        span = (self.max_val - self.min_val).clamp_min(self.eps)
        self.max_val = self.min_val + span

    @torch.no_grad()
    def get_qparams(
        self,
        *,
        bits: int = 8,
        symmetric: bool = True,
        signed: bool = True,
        narrow: bool = True,
    ) -> Tuple[Tensor, Tensor, int, int]:
        if not self._inited:
            # safe defaults
            min_v = torch.as_tensor(-1.0, device=self.min_val.device, dtype=self.min_val.dtype)
            max_v = torch.as_tensor(1.0, device=self.max_val.device, dtype=self.max_val.dtype)
        else:
            min_v = self.min_val
            max_v = self.max_val
        return _calc_qparams(min_v, max_v, bits=bits, symmetric=symmetric, signed=signed, narrow=narrow, eps=self.eps)


# --------------------------------- fake quant ---------------------------------

class FakeQuant(nn.Module):
    """
    Fake-quantize a tensor using an EMA Min/Max observer.
    - axis=None  → per-tensor
    - axis=k     → per-channel along dimension k
    """
    def __init__(
        self,
        *,
        bits: int = 8,
        symmetric: bool = True,
        signed: bool = True,
        narrow: bool = True,
        axis: Optional[int] = None,
        momentum: float = 0.95,
        collect_stats: bool = True,
    ):
        super().__init__()
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.signed = bool(signed)
        self.narrow = bool(narrow)
        self.axis = axis
        self.collect_stats = bool(collect_stats)
        self.obs = EMAMinMaxObserver(axis=axis, momentum=momentum)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.collect_stats:
            self.obs.update(x)
        scale, zp, qmin, qmax = self.obs.get_qparams(bits=self.bits, symmetric=self.symmetric, signed=self.signed, narrow=self.narrow)

        # Reshape scale/zp for broadcasting if per-channel
        if self.axis is not None:
            # Insert singleton dims except at axis
            ax = self.axis if self.axis >= 0 else (x.dim() + self.axis)
            shape = [1] * x.dim()
            shape[ax] = -1
            scale = scale.view(*shape)
            zp = zp.view(*shape)

        return _fake_qdq(x, scale, zp, qmin=qmin, qmax=qmax)


# ----------------------------- int4/group allocator -----------------------------

class SpectralFakeQuant(nn.Module):
    """
    Convenience wrapper for spectral coefficients (e.g., Chebyshev/Taylor banks).
    Defaults to int4, symmetric, per-group along the provided axis (usually axis=0 for [G,K]).
    """
    def __init__(self, *, bits: int = 4, axis: int = 0, momentum: float = 0.95):
        super().__init__()
        if bits not in (2, 3, 4, 5, 6, 7, 8):
            raise ValueError("bits for spectral quant should be in [2..8].")
        self.bits = int(bits)
        self.axis = int(axis)
        self.fq = FakeQuant(bits=self.bits, symmetric=True, signed=True, narrow=True, axis=self.axis, momentum=momentum)

    def forward(self, coeffs: Tensor) -> Tensor:
        return self.fq(coeffs)


class QuantAllocator(nn.Module):
    """
    Small allocator for quantizers used in Phase 3:
      - spectral (int4 by default): per-group coefficients (axis=0)
      - mixture/intensity (int8): activations/logits mixing etc.
      - ledger (int8): capacity/ledger signals
    It centralizes the choices so callers don't hardcode bitwidths everywhere.
    """
    def __init__(
        self,
        *,
        bits_spectral: int = 4,
        bits_mixture: int = 8,
        bits_ledger: int = 8,
        spectral_axis: int = 0,
        act_axis: Optional[int] = None,
        momentum: float = 0.95,
    ):
        super().__init__()
        self.spectral = SpectralFakeQuant(bits=bits_spectral, axis=spectral_axis, momentum=momentum)
        self.mixture = FakeQuant(bits=bits_mixture, symmetric=True, signed=True, axis=act_axis, momentum=momentum)
        self.ledger = FakeQuant(bits=bits_ledger, symmetric=True, signed=True, axis=None, momentum=momentum)

    def quantize_spectral(self, coeffs: Tensor) -> Tensor:
        return self.spectral(coeffs)

    def quantize_mixture(self, x: Tensor) -> Tensor:
        return self.mixture(x)

    def quantize_ledger(self, x: Tensor) -> Tensor:
        return self.ledger(x)

    def set_collect(self, collect: bool = True) -> None:
        # Toggle stats collection on all sub-quantizers
        for mod in self.modules():
            if isinstance(mod, FakeQuant):
                mod.collect_stats = bool(collect)


# ----------------------------- calibration helpers -----------------------------

def toggle_fakequant_collect(module: nn.Module, collect: bool = True) -> None:
    """Recursively toggle .collect_stats on all FakeQuant modules in a model."""
    for m in module.modules():
        if isinstance(m, FakeQuant):
            m.collect_stats = bool(collect)

@torch.no_grad()
def calibrate_model(model: nn.Module, data_iter, *, num_batches: int = 32) -> None:
    """
    Brief post-pruning calibration: run a few batches forward so EMA observers
    collect realistic min/max and update qparams. Expects `data_iter` to yield
    either (args, kwargs) tuples or just input tensors that `model` accepts.
    """
    was_training = model.training
    model.train()  # observers update in training mode
    toggle_fakequant_collect(model, True)

    it = iter(data_iter)
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        if isinstance(batch, tuple):
            if len(batch) == 2 and isinstance(batch[1], dict):
                model(*batch[0], **batch[1])
            else:
                model(*batch)
        elif isinstance(batch, dict):
            model(**batch)
        else:
            model(batch)

    # After calibration, you may disable collection to freeze qparams
    toggle_fakequant_collect(model, False)
    if not was_training:
        model.eval()


# ------------------------------ quantized Linear -------------------------------

class QuantLinear(nn.Module):
    """
    Linear layer with weight/activation fake-quantization.
    - Weight per-channel quant along out_features axis (0).
    - Activation per-tensor by default (axis=None). Set axis to last dim for per-channel activations if needed.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        bits_w: int = 8,
        bits_a: int = 8,
        symmetric_w: bool = True,
        symmetric_a: bool = True,
        act_axis: Optional[int] = None,  # None → per-tensor; -1 → per-channel over last dim
        momentum: float = 0.95,
    ):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        # Per-channel on weights along axis=0 (out_features)
        self.wq = FakeQuant(bits=bits_w, symmetric=symmetric_w, signed=True, axis=0, momentum=momentum)
        # Activations fake-quant
        self.aq = FakeQuant(bits=bits_a, symmetric=symmetric_a, signed=True, axis=act_axis, momentum=momentum)

    def forward(self, x: Tensor) -> Tensor:
        x_q = self.aq(x)
        # Quantize-dequantize weights (QAT style)
        w_q = self.wq(self.lin.weight)
        return F.linear(x_q, w_q, self.lin.bias)

    # ------------------------------- export utils ------------------------------

    @torch.no_grad()
    def export_int8_weights(self) -> Dict[str, Tensor]:
        """
        Return a dict with:
          - 'int8_weight': int8 per-channel quantized weights [out, in]
          - 'scale': float32 per-channel scales [out]
          - 'zero_point': float32 per-channel zeros [out] (0 for symmetric)
        """
        w = self.lin.weight.detach()
        # Build qparams from the weight observer (per-channel axis=0)
        if not self.wq.obs._inited:
            self.wq.obs.update(w)
        scale, zp, qmin, qmax = self.wq.obs.get_qparams(bits=self.wq.bits, symmetric=self.wq.symmetric, signed=True, narrow=True)
        # Per-channel: reshape to broadcast on columns
        scale_vec = scale.view(-1)
        zp_vec = zp.view(-1)
        # Quantize to int8 grid
        q = torch.round(w / scale_vec.view(-1, 1) + zp_vec.view(-1, 1))
        q = torch.clamp(q, qmin, qmax).to(torch.int8)
        return {"int8_weight": q.cpu(), "scale": scale_vec.cpu(), "zero_point": zp_vec.cpu()}


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[quant] Running sanity tests...")

    # 1) Observer + FakeQuant (per-tensor)
    x = torch.randn(4, 8, 16) * 2.0
    fq = FakeQuant(bits=8, symmetric=True, axis=None, momentum=0.9)
    fq.train()
    y = fq(x)
    err = float((y - x).abs().mean())
    print(f"  FakeQuant per-tensor: mean|Δ|={err:.3e}")
    assert y.shape == x.shape and torch.isfinite(y).all()

    # 2) Observer + FakeQuant (per-channel on last dim)
    fq_c = FakeQuant(bits=8, symmetric=True, axis=-1, momentum=0.9)
    y2 = fq_c(x)
    err2 = float((y2 - x).abs().mean())
    print(f"  FakeQuant per-channel(last): mean|Δ|={err2:.3e}")
    assert y2.shape == x.shape

    # 2b) Spectral (int4) per-group on [G,K]
    G, K = 3, 6
    coeffs = torch.randn(G, K) * 0.2
    spec_q = SpectralFakeQuant(bits=4, axis=0, momentum=0.9)
    spec_q.train()
    coeffs_q = spec_q(coeffs)
    print(f"  Spectral int4: mean|Δ|={float((coeffs_q - coeffs).abs().mean()):.3e}")
    assert coeffs_q.shape == coeffs.shape

    # 2c) Allocator quick path
    alloc = QuantAllocator(bits_spectral=4, bits_mixture=8, bits_ledger=8, spectral_axis=0, act_axis=None)
    _ = alloc.quantize_spectral(coeffs)
    _ = alloc.quantize_mixture(torch.randn(4, 8, 16))
    _ = alloc.quantize_ledger(torch.randn(4, 8))

    # 3) QuantLinear vs float Linear (relative output error should be modest)
    B, Din, Dout = 32, 24, 16
    qlin = QuantLinear(Din, Dout, bits_w=8, bits_a=8, symmetric_w=True, symmetric_a=True, act_axis=None)
    qlin.train()
    xb = torch.randn(B, Din)
    with torch.no_grad():
        y_f = qlin.lin(xb)               # float path
    y_q = qlin(xb)                       # fake-quant path
    rel = float((y_q - y_f).abs().mean() / (y_f.abs().mean().clamp_min(1e-6)))
    print(f"  QuantLinear: rel. L1 err={rel:.3f}")
    assert rel < 0.35  # QAT noise allowance

    # 4) Export weights (int8 weights path; spectral int4 stays as fake-quant runtime)
    pkg = qlin.export_int8_weights()
    print(f"  export: int8_weight={tuple(pkg['int8_weight'].shape)}, scale={tuple(pkg['scale'].shape)}")
    assert pkg["int8_weight"].dtype == torch.int8 and pkg["scale"].ndim == 1

    print("[quant] All good ✓")

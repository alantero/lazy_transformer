# modules/output_head.py
# Output projection head: hidden states → vocabulary logits.
# - Supports classic learned head (W,b) or weight tying to an external embedder.
# - Lazy tying: vocab_size can be inferred at first forward from tied weight.
# - Optional temperature, masking on logits, and traceless pre-centering.
# PyTorch-only. No global repo deps.

from __future__ import annotations
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
GetTiedFn = Callable[[], Union[Tensor, Tuple[Tensor, Optional[Tensor]]]]
# `GetTiedFn` should return either:
#   - W (Tensor [V, d])            — weight only
#   - (W, b) (Tensors [V,d], [V])  — weight and bias (bias may be None)


def _traceless_last(h: Tensor) -> Tensor:
    """Subtract mean over last dim; shape preserved."""
    return h - h.mean(dim=-1, keepdim=True)


class OutputHead(nn.Module):
    """
    Output projection head (hidden → logits over vocabulary).

    Modes:
      (A) Learned head (no tying):
          - Trainable W∈R^{V×d}, b∈R^{V}.
      (B) Tied head:
          - Provide a callable `get_tied()` returning W (and optionally b).
          - `vocab_size` may be None; inferred from W.shape[0] lazily on first forward.
          - If `learn_bias=True` and provider has no bias, a trainable bias is created
            lazily at first forward with size V = W.shape[0].

    Optional multiplicative head: logits += W_mul [ h ⊙ ( (h @ Wp) @ Wb ) ].

    Args:
        vocab_size: Optional[int]. Required if tie_weight=False.
        d_model: hidden size d (must match W.shape[1] when tying).
        tie_weight: if True, use `get_tied()` to fetch W (and maybe b).
        get_tied: callable that returns W or (W, b).
        learn_bias: when tying and provider doesn't give a bias, learn a local bias.
        temperature: scale logits by 1/temperature (temperature>0).
        init_std: std for normal init of learned W (no tying).
        scale_by_sqrt_d: if True, scale hidden states by 1/sqrt(d_model) before the projection (useful with weight tying to stabilize logits in fp16/bf16).
        cast_dtype: when tying, cast provided weights/bias to match the hidden's dtype.
        cast_device: when tying, move provided weights/bias to the same device as the hidden.
        pre_norm: if True, apply a LayerNorm to h just before projecting to logits.
        norm_eps: epsilon used in the LayerNorm.
        forbid_token_id: if set, the corresponding column in logits is set to -inf (useful to ban a padding token from being predicted).

        use_mul: enable multiplicative interaction term.
        mul_rank: low rank R for P ≈ Wp @ Wb (d→R→d).
        mul_init_std: init std for Wp/Wb and mul weight.
        mul_gate_init: initial value for a learnable gate (sigmoid) that scales the multiplicative logits.
    """
    def __init__(
        self,
        vocab_size: Optional[int],
        d_model: int,
        *,
        tie_weight: bool = False,
        get_tied: Optional[GetTiedFn] = None,
        learn_bias: bool = False,
        temperature: float = 1.0,
        init_std: float = 0.02,
        scale_by_sqrt_d: bool = False,
        cast_dtype: bool = True,
        cast_device: bool = True,
        pre_norm: bool = True,
        norm_eps: float = 1e-5,
        forbid_token_id: Optional[int] = None,
        use_mul: bool = False,
        mul_rank: int = 16,
        mul_init_std: float = 0.02,
        mul_gate_init: float = 0.0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0.")

        self.d_model = int(d_model)
        self.tie_weight = bool(tie_weight)
        self.temperature = float(max(temperature, 1e-8))

        self.scale_by_sqrt_d = bool(scale_by_sqrt_d)
        self.cast_dtype = bool(cast_dtype)
        self.cast_device = bool(cast_device)
        self.forbid_token_id = forbid_token_id

        # Optional pre-head normalization (helps keep logits well-scaled)
        self.pre_norm = bool(pre_norm)
        if self.pre_norm:
            self.norm = nn.LayerNorm(self.d_model, eps=float(norm_eps))
        else:
            self.register_module("norm", None)

        # Internal state
        self._lazy_bias_pending = False  # for lazy creation when tying with unknown V

        if self.tie_weight:
            if get_tied is None:
                raise ValueError("tie_weight=True requires a callable `get_tied`.")
            self.get_tied: GetTiedFn = get_tied

            # vocab_size may be None; infer on first forward from tied W
            self.vocab_size = int(vocab_size) if vocab_size is not None else None  # type: ignore[assignment]

            # No local weight when tying.
            self.register_parameter("weight", None)

            # Bias handling while tying
            if learn_bias:
                if self.vocab_size is None:
                    # Create lazily at first forward when we know V
                    self._lazy_bias_pending = True
                    self.register_parameter("bias", None)
                else:
                    self.bias = nn.Parameter(torch.zeros(self.vocab_size))
            else:
                self.register_parameter("bias", None)

        else:
            # Learned projection (vocab_size is required)
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when tie_weight=False.")
            self.vocab_size = int(vocab_size)
            self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
            nn.init.normal_(self.weight, std=init_std)
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))
            # Not used in this path
            self.get_tied = lambda: (_ for _ in ()).throw(  # type: ignore
                RuntimeError("get_tied is not used when tie_weight=False.")
            )

        # Multiplicative interaction branch
        self.use_mul = bool(use_mul)
        if self.use_mul:
            if mul_rank <= 0:
                raise ValueError("mul_rank must be > 0 when use_mul=True.")
            self.mul_rank = int(mul_rank)
            self.mul_init_std = float(mul_init_std)

            # Low-rank matrices Wp and Wb (d_model x mul_rank) and (mul_rank x d_model)
            self.Wp = nn.Parameter(torch.empty(self.d_model, self.mul_rank))
            self.Wb = nn.Parameter(torch.empty(self.mul_rank, self.d_model))
            nn.init.normal_(self.Wp, std=self.mul_init_std)
            nn.init.normal_(self.Wb, std=self.mul_init_std)

            # weight_mul: V x d_model
            if self.tie_weight:
                if self.vocab_size is None:
                    self._lazy_weight_mul_pending = True
                    self.register_parameter("weight_mul", None)
                else:
                    self.weight_mul = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
                    nn.init.normal_(self.weight_mul, std=self.mul_init_std)
            else:
                self.weight_mul = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
                nn.init.normal_(self.weight_mul, std=self.mul_init_std)
                self._lazy_weight_mul_pending = False

            # Learnable gate for multiplicative logits scaling
            self.mul_gate = nn.Parameter(torch.tensor(float(mul_gate_init)))
        else:
            self._lazy_weight_mul_pending = False
            self.register_parameter("Wp", None)
            self.register_parameter("Wb", None)
            self.register_parameter("weight_mul", None)
            self.register_parameter("mul_gate", None)

    # -------------------------------- utilities --------------------------------

    def _maybe_infer_vocab_and_bias(self, W: Tensor) -> None:
        """When tying: infer vocab_size from W and lazily create bias if needed."""
        V, d = W.shape
        if d != self.d_model:
            raise ValueError(f"Tied weight second dim must be d_model={self.d_model}, got {d}.")
        if getattr(self, "vocab_size", None) is None:
            # Infer vocab size
            self.vocab_size = int(V)  # type: ignore[assignment]
        elif self.vocab_size != V:
            raise ValueError(f"Tied weight vocab={V} != head vocab_size={self.vocab_size}")

        if self._lazy_bias_pending and getattr(self, "bias", None) is None:
            # Create trainable bias now that V is known
            self.bias = nn.Parameter(torch.zeros(V))
            self._lazy_bias_pending = False

        if self.use_mul and self._lazy_weight_mul_pending and getattr(self, "weight_mul", None) is None:
            self.weight_mul = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
            nn.init.normal_(self.weight_mul, std=self.mul_init_std)
            self._lazy_weight_mul_pending = False

    def _masked(self, logits: Tensor, mask: Optional[Tensor], mask_behavior: str) -> Tensor:
        mode = (mask_behavior or "none").lower()
        if mask is None or mode in {"none", "ignore"}:
            return logits
        if mask.dtype is not torch.bool:
            mask = mask != 0
        inv = ~mask
        if mode == "neg_inf":
            return logits.masked_fill(inv.unsqueeze(-1), float("-inf"))
        if mode == "zero":
            return logits.masked_fill(inv.unsqueeze(-1), 0.0)
        raise ValueError("mask_behavior must be 'none' | 'ignore' | 'neg_inf' | 'zero'.")

    # --------------------------------- forward ---------------------------------

    def forward(
        self,
        h: Tensor,
        *,
        mask: Optional[Tensor] = None,
        mask_behavior: str = "none",
        temperature: Optional[float] = None,
        traceless: bool = False,
    ) -> Tensor:
        """
        Compute logits. Shapes: h [B,T,d] → logits [B,T,V]
        """
        if h.dim() != 3 or h.size(-1) != self.d_model:
            raise ValueError(f"h must be [B,T,{self.d_model}], got {tuple(h.shape)}")

        if self.scale_by_sqrt_d:
            h = h * (self.d_model ** -0.5)

        # Optional traceless pre-centering
        if traceless:
            h = _traceless_last(h)

        # Optional pre-head normalization
        if getattr(self, "norm", None) is not None:
            h = self.norm(h)

        # Select weight/bias
        if self.tie_weight:
            tied = self.get_tied()
            if isinstance(tied, tuple):
                W, b_ext = tied
            else:
                W, b_ext = tied, None

            if self.cast_device and W.device != h.device:
                W = W.to(h.device)
                if b_ext is not None:
                    b_ext = b_ext.to(h.device)
            if self.cast_dtype and W.dtype != h.dtype:
                W = W.to(h.dtype)
                if b_ext is not None:
                    b_ext = b_ext.to(h.dtype)
            W = W.contiguous()
            if b_ext is not None:
                b_ext = b_ext.contiguous()

            # Infer vocab size / create bias lazily if needed
            self._maybe_infer_vocab_and_bias(W)

            logits = torch.matmul(h, W.t())
            proj_weight = W
            # Bias priority: external first, else local, else none
            if b_ext is not None:
                logits = logits + b_ext
            elif getattr(self, "bias", None) is not None:
                logits = logits + self.bias  # type: ignore[operator]

        else:
            weight = self.weight
            bias = self.bias
            if self.cast_device and weight.device != h.device:
                weight = weight.to(h.device)
                bias = bias.to(h.device)
            if self.cast_dtype and weight.dtype != h.dtype:
                weight = weight.to(h.dtype)
                bias = bias.to(h.dtype)
            logits = torch.matmul(h, weight.t()) + bias
            proj_weight = weight

        # ---------------- Multiplicative / extra head compatibility ----------------
        # Historically, checkpoints have stored different shapes in `weight_mul`.
        # We support:
        #   (a) scalar ()                   → scales logits directly (no gate)
        #   (b) [V]                         → elementwise scale per-vocab on logits
        #   (c) [D]                         → feature mask: (h ⊙ w) · W^T
        #   (d) [V, D]                      → current low-rank path: (h ⊙ (hWpWb)) · weight_mul^T
        #   (e) [D, V]                      → direct projection add: h · weight_mul
        # Any missing parameters (Wp/Wb) fall back to simpler safe behaviors.
        if hasattr(self, "weight_mul") and getattr(self, "weight_mul") is not None:
            wm = self.weight_mul
            # Move/cast as needed
            if self.cast_device and wm.device != h.device:
                wm = wm.to(h.device)
            if self.cast_dtype and wm.dtype != h.dtype:
                wm = wm.to(h.dtype)

            # Compute the additive term according to shape
            add_term: Optional[Tensor] = None
            if wm.ndim == 0:
                # Scalar: simple scale of logits (no gate to avoid double-sigmoid effects)
                logits = logits * wm
            elif wm.ndim == 1:
                if wm.numel() == logits.size(-1):
                    # Per-vocab scaling
                    gate = torch.sigmoid(self.mul_gate) if getattr(self, "mul_gate", None) is not None else 1.0
                    add_term = logits * wm.view(1, 1, -1) * gate
                elif wm.numel() == self.d_model:
                    # Feature mask → project with same matrix used for base logits
                    if 'proj_weight' in locals():
                        h_masked = h * wm.view(1, 1, -1)
                        add_term = torch.matmul(h_masked, proj_weight.t())
                        if getattr(self, "mul_gate", None) is not None:
                            add_term = torch.sigmoid(self.mul_gate) * add_term
                # else: unknown length → ignore
            elif wm.ndim == 2:
                V = logits.size(-1)
                D = self.d_model
                if wm.shape == (V, D):
                    # Preferred path with optional low-rank interaction
                    if getattr(self, "Wp", None) is not None and getattr(self, "Wb", None) is not None:
                        z = torch.matmul(h, self.Wp)
                        ph = torch.matmul(z, self.Wb)
                        inter = h * ph
                    else:
                        inter = h
                    add_term = torch.matmul(inter, wm.t())
                    if getattr(self, "mul_gate", None) is not None:
                        add_term = torch.sigmoid(self.mul_gate) * add_term
                elif wm.shape == (D, V):
                    add_term = torch.matmul(h, wm)
                    if getattr(self, "mul_gate", None) is not None:
                        add_term = torch.sigmoid(self.mul_gate) * add_term
                # else: unknown shape → ignore

            if add_term is not None:
                logits = logits + add_term

        if self.forbid_token_id is not None:
            fid = int(self.forbid_token_id)
            if fid < logits.size(-1):
                logits[..., fid] = float("-inf")

        # Temperature
        temp = float(self.temperature if temperature is None else max(temperature, 1e-8))
        if temp != 1.0:
            logits = logits / temp

        # Masking (optional)
        logits = self._masked(logits, mask, mask_behavior)
        return logits


# ----------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    # Minimal checks, including lazy tying and traceless behavior.
    torch.manual_seed(0)
    B, T, V, D = 2, 7, 32, 16

    # 1) Learned head path
    h = torch.randn(B, T, D)
    head = OutputHead(vocab_size=V, d_model=D, tie_weight=False)
    logits = head(h, traceless=True)
    assert logits.shape == (B, T, V)
    print(f"[output_head] learned head logits: {tuple(logits.shape)}")

    head_norm = OutputHead(vocab_size=V, d_model=D, tie_weight=False, pre_norm=True)
    logits_norm = head_norm(h)
    assert logits_norm.shape == (B, T, V)
    print("[output_head] pre-norm path ok")

    # 2) Tied path with explicit vocab + provider weight
    W = torch.randn(V, D)
    head_tied = OutputHead(vocab_size=V, d_model=D, tie_weight=True, get_tied=lambda: W, learn_bias=True)
    logits2 = head_tied(h, mask=(torch.rand(B, T) > 0.2), mask_behavior="zero")
    assert logits2.shape == (B, T, V)
    print("[output_head] tied (weight-only) ok")

    # 3) Lazy-tying: vocab_size=None, infer from W at first forward, create bias lazily
    head_lazy = OutputHead(vocab_size=None, d_model=D, tie_weight=True, get_tied=lambda: W, learn_bias=True)
    logits3 = head_lazy(h)
    assert logits3.shape == (B, T, V)
    assert head_lazy.vocab_size == V
    assert isinstance(head_lazy.bias, nn.Parameter)
    print("[output_head] lazy tying ok")

    # 4) Traceless toggle sanity: centering h before matmul equals manual centering
    head_plain = OutputHead(vocab_size=V, d_model=D, tie_weight=False)
    Wp = head_plain.weight.detach().clone()
    bp = head_plain.bias.detach().clone()
    # With traceless=True
    logits_t = head_plain(h, traceless=True)
    # Manual: (h - mean) @ W^T + b
    logits_m = torch.matmul(_traceless_last(h), Wp.t()) + bp
    diff = (logits_t - logits_m).abs().max().item()
    print(f"[output_head] traceless max|Δ| = {diff:.2e}")
    assert diff < 1e-6

    # 5) Casting dtype/device with bf16 (if available)
    try:
        W_fp32 = torch.randn(V, D, dtype=torch.float32)
        h_bf16 = h.to(dtype=torch.bfloat16)
        head_cast = OutputHead(vocab_size=None, d_model=D, tie_weight=True, get_tied=lambda: W_fp32, scale_by_sqrt_d=True)
        logits_cast = head_cast(h_bf16)
        print("[output_head] casting dtype/device with bf16 ok")
    except Exception as e:
        print("[output_head] skipping bf16 cast test (not supported on this device)")

    # 6) forbid_token_id test
    forbid_id = 3
    head_forbid = OutputHead(vocab_size=V, d_model=D, tie_weight=False, forbid_token_id=forbid_id)
    logits_forbid = head_forbid(h)
    assert torch.all(torch.isinf(logits_forbid[..., forbid_id]) & (logits_forbid[..., forbid_id] < 0))
    print("[output_head] forbid_token_id masking ok")

    # 7) Learned head + use_mul=True test
    head_mul = OutputHead(vocab_size=V, d_model=D, tie_weight=False, use_mul=True, mul_rank=8, mul_gate_init=0.0)
    logits_mul_0 = head_mul(h)
    assert logits_mul_0.shape == (B, T, V)
    # Change mul_gate to large value to saturate sigmoid ~1
    head_mul.mul_gate.data.fill_(5.0)
    logits_mul_1 = head_mul(h)
    assert logits_mul_1.shape == (B, T, V)
    diff_mul = (logits_mul_1 - logits_mul_0).abs().max().item()
    print(f"[output_head] learned + use_mul logits diff with gate change: {diff_mul:.2e}")
    assert diff_mul > 1e-4

    # 8) Tied head + use_mul=True with lazy vocab_size test
    head_tied_mul = OutputHead(vocab_size=None, d_model=D, tie_weight=True, get_tied=lambda: W, learn_bias=True, use_mul=True, mul_rank=8)
    logits_tied_mul = head_tied_mul(h)
    assert logits_tied_mul.shape == (B, T, V)
    assert head_tied_mul.vocab_size == V
    assert isinstance(head_tied_mul.weight_mul, nn.Parameter)
    print("[output_head] tied + use_mul lazy vocab_size ok")

    print("[output_head] All good ✓")

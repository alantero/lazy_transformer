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

    Args:
        vocab_size: Optional[int]. Required if tie_weight=False.
        d_model: hidden size d (must match W.shape[1] when tying).
        tie_weight: if True, use `get_tied()` to fetch W (and maybe b).
        get_tied: callable that returns W or (W, b).
        learn_bias: when tying and provider doesn't give a bias, learn a local bias.
        temperature: scale logits by 1/temperature (temperature>0).
        init_std: std for normal init of learned W (no tying).

    Forward:
        h: [B, T, d] → logits: [B, T, V]
        mask (optional): [B, T] bool. If provided:
            - mask_behavior='neg_inf' → logits[~mask] = -inf
            - mask_behavior='zero'    → logits[~mask] = 0.0
            - mask_behavior='ignore'  → return logits unchanged (handle in loss via ignore_index)
            - mask_behavior='none'    → alias of 'ignore'
        traceless (optional): if True, center h along last dim before matmul (logits centered).
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
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0.")

        self.d_model = int(d_model)
        self.tie_weight = bool(tie_weight)
        self.temperature = float(max(temperature, 1e-8))

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

        # Optional traceless pre-centering
        if traceless:
            h = _traceless_last(h)

        # Select weight/bias
        if self.tie_weight:
            tied = self.get_tied()
            if isinstance(tied, tuple):
                W, b_ext = tied
            else:
                W, b_ext = tied, None

            # Infer vocab size / create bias lazily if needed
            self._maybe_infer_vocab_and_bias(W)

            logits = torch.matmul(h, W.t())
            # Bias priority: external first, else local, else none
            if b_ext is not None:
                logits = logits + b_ext
            elif getattr(self, "bias", None) is not None:
                logits = logits + self.bias  # type: ignore[operator]

        else:
            logits = torch.matmul(h, self.weight.t()) + self.bias  # type: ignore[attr-defined]

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

    print("[output_head] All good ✓")

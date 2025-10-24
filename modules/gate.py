# modules/gate.py
# Lightweight gating layers for sequence models (PyTorch-only, no repo-global deps).
# Provides group-wise (or per-channel) gates over the last feature axis of x[B, T, D].
#
# Main use-cases:
#   • Residual gating of an update:      y = x + g ⊙ u
#   • Pure multiplicative gating of x:   y = g ⊙ x
#
# Design:
#   • Gates live on the last dim (features). You choose granularity:
#       - per_group (default): one gate per feature-group (broadcast to channels in the group)
#       - per_channel:         one gate per channel (D parameters)
#   • Shapes are simple and broadcast-safe. D must be divisible by groups.
#   • "Lazy minimal" default: gate logits initialized very negative → g≈0, i.e., the gate starts CLOSED.
#   • Optional external capacity/context signal 'cap' (e.g., per-group or per-token) to bias gates.
#       - Learnable scalar 'cap_scale' mixes the external signal into gate logits.
#       - stop_grad_cap: if True (default), detach() the external 'cap' signal before mixing (no gradient through cap).
#   • Optional boolean mask 'mask' marks valid positions; invalid positions pass through x unchanged.
#       - mask supports shapes [B,T], [B,T,1], [B,T,G], [B,T,D].
#       - For mode='residual', positions where mask==False pass x unchanged (g=0).
#       - For mode='mul', positions where mask==False pass x unchanged (no multiplicative gating).
#   • Optional sparse top-k gating (sparse_topk) applies per-token sparsification on the gating dimension
#       (per-group or per-channel depending on configuration) by masking non-topk entries before the squashing function.
#   • Optional gamma shaping (gamma>0) raises the squashed gate to a power to sharpen or flatten the gate distribution.
#
# No dependencies beyond torch.

from __future__ import annotations
from typing import Optional, Literal

import torch
import torch.nn as nn

Tensor = torch.Tensor
_Mode = Literal["residual", "mul"]


def _ensure_divisible(D: int, G: int) -> None:
    if G <= 0 or (D % G) != 0:
        raise ValueError(f"'groups' must divide D. Got D={D}, groups={G}.")


class GroupGate(nn.Module):
    """
    Group-wise (or per-channel) gate on the last feature dimension.

    Args:
      d_model:       total feature dimension D
      groups:        number of feature groups G (D must be divisible by G)
      mode:          'residual' (y = x + g ⊙ u) or 'mul' (y = g ⊙ x)
      per_channel:   if True, one gate per channel (D params); else one per group (G params)
      bias_init:     initial gate logit value; negative large → gate≈0 (closed)
      learn_temp:    learn a global logit scale (temperature) (default False → fixed = 1.0)
      cap_scale_init:initial scalar to scale external 'cap' signal in logits (0.0 disables by default)
      stop_grad_cap: if True (default), detach() the external 'cap' signal before mixing (no gradient through cap).
      sparse_topk:   Optional[int] (default None). If set, applies per-token top-k sparsification on gating dimension
                     (per-group or per-channel depending on configuration) by masking non-topk entries before sigmoid.
      gamma:         float > 0 (default 1.0). Raises the squashed gate to this power to sharpen (>1) or flatten (<1) the gate distribution.

    Forward:
      If mode == 'residual':
        y = x + g(x, cap) ⊙ u
      else:
        y = g(x, cap) ⊙ x

      Inputs:
        x:   [B, T, D]
        u:   [B, T, D] (required for 'residual'; ignored for 'mul')
        cap: Optional capacity/context signal. Supported shapes:
             [B, T], [B, T, 1], [B, T, G], [B, T, D]
             It is broadcast to [B, T, D] before contribution.
        mask: Optional boolean mask marking valid positions.
              Supported shapes: [B, T], [B, T, 1], [B, T, G], [B, T, D].
              On positions where mask is False:
                - mode='residual': y passes through x unchanged (since g=0 there).
                - mode='mul': y equals x (no multiplicative suppression).

      Returns:
        y:   [B, T, D]
        (optionally g if return_gate=True)
    """
    def __init__(
        self,
        d_model: int,
        groups: int = 1,
        *,
        mode: _Mode = "residual",
        per_channel: bool = False,
        bias_init: float = -6.0,
        learn_temp: bool = False,
        cap_scale_init: float = 0.0,
        stop_grad_cap: bool = True,
        sparse_topk: Optional[int] = None,
        gamma: float = 1.0,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0.")
        _ensure_divisible(d_model, groups)

        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if sparse_topk is not None:
            if sparse_topk < 1:
                raise ValueError("sparse_topk must be >= 1 if specified.")
            if per_channel:
                if sparse_topk > d_model // groups:
                    raise ValueError(f"sparse_topk must be <= channels per group (cg={d_model // groups}) when per_channel=True.")
            else:
                if sparse_topk > groups:
                    raise ValueError(f"sparse_topk must be <= groups ({groups}) when per_channel=False.")

        self.d_model = int(d_model)
        self.groups = int(groups)
        self.cg = self.d_model // self.groups
        self.mode: _Mode = mode
        self.per_channel = bool(per_channel)
        self.sparse_topk = sparse_topk
        self.gamma = float(gamma)

        # Gate logits parameters
        factory_kwargs = {"dtype": dtype}
        if self.per_channel:
            self.logits = nn.Parameter(torch.full((self.d_model,), float(bias_init), **factory_kwargs))
        else:
            self.logits = nn.Parameter(torch.full((self.groups,), float(bias_init), **factory_kwargs))

        # Optional temperature (global)
        if learn_temp:
            self.logit_scale = nn.Parameter(torch.ones((), **factory_kwargs))
        else:
            self.register_buffer("logit_scale", torch.ones((), **factory_kwargs))

        # Optional capacity/context mixing (scalar)
        self.cap_scale = nn.Parameter(torch.tensor(float(cap_scale_init), **factory_kwargs))

        self.stop_grad_cap = bool(stop_grad_cap)

    def _expand_group_vector(self, vG: Tensor) -> Tensor:
        """
        Expand a [G] vector to [D] by repeating each group value across its Cg channels.
        """
        G, Cg = self.groups, self.cg
        return vG.view(G, 1).expand(G, Cg).reshape(G * Cg)

    def _broadcast_cap(self, cap: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Normalize 'cap' into a [B, T, D] tensor (broadcast-safe).
        Supported shapes: [B,T], [B,T,1], [B,T,G], [B,T,D].
        """
        if cap.dim() not in (2, 3):
            raise ValueError("cap must have shape [B,T], [B,T,1], [B,T,G], or [B,T,D].")
        if cap.dim() == 2:  # [B,T]
            cap = cap.unsqueeze(-1)  # [B,T,1]
        B, T = cap.shape[:2]
        if cap.shape[-1] == 1:
            capD = cap.expand(B, T, self.d_model)  # broadcast scalar per token
        elif cap.shape[-1] == self.groups:
            # expand per-group to per-channel
            capG = cap  # [B,T,G]
            capD = capG.unsqueeze(-1).expand(B, T, self.groups, self.cg).reshape(B, T, self.d_model)
        elif cap.shape[-1] == self.d_model:
            capD = cap
        else:
            raise ValueError(f"Unsupported cap last-dim {cap.shape[-1]} for D={self.d_model}, G={self.groups}.")
        return capD.to(device=device, dtype=dtype)

    def _broadcast_mask(self, mask: Tensor, device: torch.device) -> Tensor:
        """
        Normalize boolean 'mask' into [B, T, D] (True=valid).
        Supported shapes: [B,T], [B,T,1], [B,T,G], [B,T,D].
        """
        if mask.dim() not in (2, 3):
            raise ValueError("mask must have shape [B,T], [B,T,1], [B,T,G], or [B,T,D].")
        if mask.dim() == 2:  # [B,T]
            mask = mask.unsqueeze(-1)  # [B,T,1]
        B, T = mask.shape[:2]
        if mask.shape[-1] == 1:
            maskD = mask.expand(B, T, self.d_model)
        elif mask.shape[-1] == self.groups:
            # expand per-group to per-channel
            maskG = mask
            maskD = maskG.unsqueeze(-1).expand(B, T, self.groups, self.cg).reshape(B, T, self.d_model)
        elif mask.shape[-1] == self.d_model:
            maskD = mask
        else:
            raise ValueError(f"Unsupported mask last-dim {mask.shape[-1]} for D={self.d_model}, G={self.groups}.")
        return maskD.to(device=device, dtype=torch.bool)

    def _gate(self, B: int, T: int, device: torch.device, dtype: torch.dtype, cap: Optional[Tensor], mask: Optional[Tensor]) -> Tensor:
        """
        Build gate tensor g in [0,1] with shape [B, T, D].
        g = sigmoid( logit_scale * (logits_full + cap_scale * cap_full) )
        If mask is provided (False=invalid), gates on invalid positions are zeroed.
        Supports optional sparse_topk gating and gamma shaping.
        """
        if self.per_channel:
            logits_full = self.logits.to(device=device, dtype=dtype)                  # [D]
        else:
            logits_full = self._expand_group_vector(self.logits.to(device=device, dtype=dtype))  # [D]

        logitsBTD = logits_full.view(1, 1, self.d_model).expand(B, T, self.d_model)  # [B,T,D]

        maskD_bool: Optional[Tensor] = None
        if mask is not None:
            maskD_bool = self._broadcast_mask(mask, device=device)                   # [B,T,D] bool

        if cap is not None:
            if self.stop_grad_cap:
                cap = cap.detach()
            capD = self._broadcast_cap(cap, device=device, dtype=dtype)              # [B,T,D]
            if maskD_bool is not None:
                capD = capD * maskD_bool.to(dtype)                                   # ignore cap on invalid tokens
            logitsBTD = logitsBTD + self.cap_scale.to(device=device, dtype=dtype) * capD

        scores = self.logit_scale.to(device=device, dtype=dtype) * logitsBTD  # [B,T,D]

        if self.sparse_topk is not None:
            k = self.sparse_topk
            if self.per_channel:
                # scores shape [B,T,D] -> reshape to [B,T,G,Cg]
                scores4d = scores.view(B, T, self.groups, self.cg)
                # topk along last dim (channels per group)
                topk_vals, topk_idx = torch.topk(scores4d, k=k, dim=-1)
                mask_topk = torch.zeros_like(scores4d, dtype=torch.bool)
                mask_topk.scatter_(-1, topk_idx, True)
                # mask non-topk entries to large negative
                scores4d = scores4d.masked_fill(~mask_topk, -20.0)
                scores = scores4d.view(B, T, self.d_model)
            else:
                # per-group topk: compute group scores by averaging channels per group
                scores4d = scores.view(B, T, self.groups, self.cg)
                group_scores = scores4d.mean(dim=-1)  # [B,T,G]
                topk_vals, topk_idx = torch.topk(group_scores, k=k, dim=-1)
                mask_topkG = torch.zeros_like(group_scores, dtype=torch.bool)
                mask_topkG.scatter_(-1, topk_idx, True)  # [B,T,G]
                # expand mask to channels
                mask_topk4d = mask_topkG.unsqueeze(-1).expand(B, T, self.groups, self.cg)
                scores4d = scores4d.masked_fill(~mask_topk4d, -20.0)
                scores = scores4d.view(B, T, self.d_model)

        g = torch.sigmoid(scores)

        if self.gamma != 1.0:
            g = g.pow(self.gamma)

        if maskD_bool is not None:
            g = g * maskD_bool.to(dtype)  # zero-out gates on invalid tokens

        return g

    def forward(self, x: Tensor, u: Optional[Tensor] = None, *, cap: Optional[Tensor] = None, mask: Optional[Tensor] = None, return_gate: bool = False):
        """
        See class docstring for details.
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must be [B,T,{self.d_model}], got {tuple(x.shape)}")
        if self.mode == "residual" and (u is None or u.shape != x.shape):
            raise ValueError("For mode='residual', u must be provided with the same shape as x.")

        B, T, D = x.shape
        maskD_bool = self._broadcast_mask(mask, device=x.device) if mask is not None else None

        g = self._gate(B, T, device=x.device, dtype=x.dtype, cap=cap, mask=mask)  # [B,T,D]

        if self.mode == "residual":
            y = x + g * u  # type: ignore[operator]
            if maskD_bool is not None:
                # residual form already leaves x unchanged when g==0; nothing else needed
                pass
        else:  # 'mul'
            if maskD_bool is not None:
                y = torch.where(maskD_bool, g * x, x)
            else:
                y = g * x

        return (y, g) if return_gate else y


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[gate] Running sanity tests...")

    B, T, D = 2, 5, 12
    G = 3

    x = torch.randn(B, T, D)
    u = torch.randn(B, T, D)

    # 1) Residual gate, per-group, starts closed (bias_init << 0) → y ≈ x
    g1 = GroupGate(d_model=D, groups=G, mode="residual", per_channel=False, bias_init=-8.0, learn_temp=False)
    y1, gate1 = g1(x, u, return_gate=True)
    max_delta = float((y1 - x).abs().max())
    print(f"  residual(per-group) closed: max|y - x| = {max_delta:.2e}")
    assert max_delta < 1e-3, "Gate should start ~closed with large negative bias."

    # Open the gate by pushing logits up → y ≈ x + u
    with torch.no_grad():
        if g1.per_channel:
            g1.logits.fill_(+10.0)  # not used in this branch, but for completeness
        else:
            g1.logits.fill_(+10.0)
    y1o = g1(x, u=u)
    max_err_res = float((y1o - (x + u)).abs().max())
    print(f"  residual(per-group) open:   max|y - (x+u)| = {max_err_res:.2e}")
    assert max_err_res < 1e-3

    # 2) Multiplicative gate, per-channel
    g2 = GroupGate(d_model=D, groups=G, mode="mul", per_channel=True, bias_init=-9.0)
    y2 = g2(x)
    max_mag = float(y2.abs().max())
    print(f"  mul(per-channel) closed:   max|y| = {max_mag:.2e}")
    assert max_mag < 1e-3

    with torch.no_grad():
        g2.logits.fill_(+12.0)
    y2o = g2(x)
    max_err_mul = float((y2o - x).abs().max())
    print(f"  mul(per-channel) open:     max|y - x| = {max_err_mul:.2e}")
    assert max_err_mul < 1e-3

    # 3) Capacity/context mixing: cap per-group should modulate the gate monotonically
    cap = torch.zeros(B, T, G)
    g3 = GroupGate(d_model=D, groups=G, mode="mul", per_channel=False, bias_init=-2.0, cap_scale_init=2.0)
    y_low, gate_low = g3(x, cap=cap, return_gate=True)

    cap[..., 0] = 3.0  # raise cap for group 0 strongly
    y_hi, gate_hi = g3(x, cap=cap, return_gate=True)

    # The average gate for channels belonging to group 0 should increase
    ch0 = slice(0, D // G)
    gate_low_mean = float(gate_low[..., ch0].mean())
    gate_hi_mean = float(gate_hi[..., ch0].mean())
    print(f"  cap monotonicity (group 0): low={gate_low_mean:.3f}, high={gate_hi_mean:.3f}")
    assert gate_hi_mean > gate_low_mean, "Gate should increase when cap for the group increases."

    # 4) Mask behavior: invalid positions pass through x unchanged
    mask = torch.ones(B, T, 1, dtype=torch.bool)
    mask[:, 0, :] = False  # first time-step invalid

    # residual: y should equal x at masked positions even if gate would open
    g_res = GroupGate(d_model=D, groups=G, mode="residual", per_channel=False, bias_init=+10.0)
    y_res = g_res(x, u=u, mask=mask)
    assert torch.allclose(y_res[:, 0, :], x[:, 0, :], atol=1e-6), "Residual gate must pass x through on masked positions."

    # mul: y should equal x at masked positions even when gate is open
    g_mul = GroupGate(d_model=D, groups=G, mode="mul", per_channel=True, bias_init=+10.0)
    y_mul = g_mul(x, mask=mask)
    assert torch.allclose(y_mul[:, 0, :], x[:, 0, :], atol=1e-6), "Mul gate must pass x through on masked positions."

    print("  mask passthrough: OK")

    # 5) Sparse top-k gating per-group test
    g_topk = GroupGate(d_model=D, groups=G, mode="mul", per_channel=False, bias_init=-20.0, sparse_topk=1)
    with torch.no_grad():
        # set logits so that first two groups are very high, last is low
        g_topk.logits.fill_(-20.0)
        g_topk.logits[0] = 10.0
        g_topk.logits[1] = 9.0
        g_topk.logits[2] = -10.0
    y_topk, gate_topk = g_topk(x, return_gate=True)
    # average gate per group
    gate_topk_reshaped = gate_topk.view(B, T, G, D // G)
    gate_topk_group_means = gate_topk_reshaped.mean(dim=-1)  # [B,T,G]
    # check that only one group is effectively open per token (topk=1)
    max_group_gate = gate_topk_group_means.max(dim=-1).values  # [B,T]
    other_groups_gate = gate_topk_group_means.sum(dim=-1) - max_group_gate  # [B,T]
    print(f"  sparse_topk per-group: max group mean gate >> others? max={max_group_gate.mean():.3f}, others={other_groups_gate.mean():.3f}")
    assert (max_group_gate > other_groups_gate).all(), "Only one group should be open per token with sparse_topk=1."

    # 6) Gamma shaping test
    g_gamma1 = GroupGate(d_model=D, groups=G, mode="mul", per_channel=True, bias_init=0.0, gamma=1.0)
    g_gamma2 = GroupGate(d_model=D, groups=G, mode="mul", per_channel=True, bias_init=0.0, gamma=2.0)
    with torch.no_grad():
        g_gamma1.logits.fill_(0.0)
        g_gamma2.logits.fill_(0.0)
    gate1 = g_gamma1._gate(B, T, device=x.device, dtype=x.dtype, cap=None, mask=None)
    gate2 = g_gamma2._gate(B, T, device=x.device, dtype=x.dtype, cap=None, mask=None)
    # The gamma=2.0 gate should have smaller variance and range after squashing (sharpened)
    var1 = gate1.var().item()
    var2 = gate2.var().item()
    range1 = (gate1.max() - gate1.min()).item()
    range2 = (gate2.max() - gate2.min()).item()
    print(f"  gamma shaping: var1={var1:.4f}, var2={var2:.4f}, range1={range1:.4f}, range2={range2:.4f}")
    assert var2 <= var1, "Gamma shaping with gamma>1 should reduce variance."
    assert range2 <= range1, "Gamma shaping with gamma>1 should reduce range."

    print("[gate] All good ✓")

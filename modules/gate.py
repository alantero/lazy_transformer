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
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0.")
        _ensure_divisible(d_model, groups)

        self.d_model = int(d_model)
        self.groups = int(groups)
        self.cg = self.d_model // self.groups
        self.mode: _Mode = mode
        self.per_channel = bool(per_channel)

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

    def _gate(self, B: int, T: int, device: torch.device, dtype: torch.dtype, cap: Optional[Tensor]) -> Tensor:
        """
        Build gate tensor g in [0,1] with shape [B, T, D].
        g = sigmoid( logit_scale * (logits_full + cap_scale * cap_full) )
        """
        if self.per_channel:
            logits_full = self.logits.to(device=device, dtype=dtype)                  # [D]
        else:
            logits_full = self._expand_group_vector(self.logits.to(device=device, dtype=dtype))  # [D]

        logitsBTD = logits_full.view(1, 1, self.d_model).expand(B, T, self.d_model)  # [B,T,D]

        if cap is not None:
            if self.stop_grad_cap:
                cap = cap.detach()
            capD = self._broadcast_cap(cap, device=device, dtype=dtype)              # [B,T,D]
            logitsBTD = logitsBTD + self.cap_scale.to(device=device, dtype=dtype) * capD

        g = torch.sigmoid(self.logit_scale.to(device=device, dtype=dtype) * logitsBTD)
        return g

    def forward(self, x: Tensor, u: Optional[Tensor] = None, *, cap: Optional[Tensor] = None, return_gate: bool = False):
        """
        See class docstring for details.
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must be [B,T,{self.d_model}], got {tuple(x.shape)}")
        if self.mode == "residual" and (u is None or u.shape != x.shape):
            raise ValueError("For mode='residual', u must be provided with the same shape as x.")

        B, T, D = x.shape
        g = self._gate(B, T, device=x.device, dtype=x.dtype, cap=cap)  # [B,T,D]

        if self.mode == "residual":
            y = x + g * u  # type: ignore[operator]
        else:  # 'mul'
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

    print("[gate] All good ✓")

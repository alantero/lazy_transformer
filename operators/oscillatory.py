# operators/oscillatory.py
# 1D oscillatory (harmonic/wave-like) operator along the sequence/time axis.
# PyTorch-only, no external repo deps.
#
# What it does
# ------------
# Given x[B,T,D], it returns
#     y = (c_g^2) * ∂_t^2 x  -  (ω_g^2) * x
# per feature group g, where ∂_t^2 is the 1D second derivative along time (axis=1).
# This is the standard linear “oscillatory” operator: wave/oscillator with
# groupwise wave-speed c_g and natural frequency ω_g (all learnable if desired).
#
# Features
# --------
# - Finite-difference second derivative (central) with:
#     • 'cycle' (periodic) BCs via torch.roll
#     • 'path'  (Dirichlet-style) BCs using one-sided stencils at boundaries
# - Groupwise parameters: c_g and ω_g (shape [groups])
# - Stable defaults: c=0, ω=0 → zero operator (lazy minimal start)
# - Optional causal/backward 2nd-derivative stencil (no future lookahead)
# - Per-token mask support to gate PAD tokens
#
# Shapes
# ------
#  x: [B, T, D]   y: [B, T, D]
#  D must be divisible by groups; we broadcast per-group scalars to channels.

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn

Tensor = torch.Tensor
_BC = Literal["cycle", "path"]


def _diff2_central(x: Tensor, *, bc: _BC, dx: float) -> Tensor:
    """
    Second derivative along axis=1 (time).
    x: [B,T,...] -> same shape.
    For 'cycle', wrap indices (periodic). For 'path', use central interior and
    simple one-sided at boundaries.
    """
    if x.dim() < 2:
        raise ValueError("x must be at least [B,T,...].")
    if dx <= 0:
        raise ValueError("dx must be > 0.")

    if bc == "cycle":
        xp = torch.roll(x, shifts=-1, dims=1)
        xm = torch.roll(x, shifts=+1, dims=1)
        return (xp - 2.0 * x + xm) / (dx * dx)

    elif bc == "path":
        B, T = x.shape[:2]
        y = torch.empty_like(x)
        if T >= 3:
            # interior central
            y[:, 1:-1, ...] = (x[:, 2:, ...] - 2.0 * x[:, 1:-1, ...] + x[:, :-2, ...]) / (dx * dx)
            # one-sided second differences near boundaries (first-order)
            # forward at t=0:   x0, x1, x2
            y[:, 0, ...] = (x[:, 2, ...] - 2.0 * x[:, 1, ...] + x[:, 0, ...]) / (dx * dx)
            # backward at t=T-1:  x_{T-3}, x_{T-2}, x_{T-1}
            y[:, -1, ...] = (x[:, -1, ...] - 2.0 * x[:, -2, ...] + x[:, -3, ...]) / (dx * dx)
        elif T == 2:
            # minimal length: symmetric mirror to avoid index errors
            y[:, 0, ...] = (x[:, 1, ...] - 2.0 * x[:, 0, ...] + x[:, 1, ...]) / (dx * dx)
            y[:, 1, ...] = (x[:, 0, ...] - 2.0 * x[:, 1, ...] + x[:, 0, ...]) / (dx * dx)
        else:  # T == 1
            y.zero_()
        return y

    else:
        raise ValueError("bc must be 'cycle' or 'path'.")


def _diff2_backward(x: Tensor, *, dx: float) -> Tensor:
    """
    Causal/backward second derivative along axis=1 (time):
    uses x[t] - 2*x[t-1] + x[t-2] over dx^2.
    For t<2, returns 0 (no future lookahead, strict causality).
    Shape preserved.
    """
    if x.dim() < 2:
        raise ValueError("x must be at least [B,T,...].")
    if dx <= 0:
        raise ValueError("dx must be > 0.")
    B, T = x.shape[:2]
    y = torch.zeros_like(x)
    if T >= 3:
        y[:, 2:, ...] = (x[:, 2:, ...] - 2.0 * x[:, 1:-1, ...] + x[:, :-2, ...]) / (dx * dx)
    # t=0,1 remain zero → strictly causal
    return y


class Oscillatory1D(nn.Module):
    """
    y = (c^2) * ∂_t^2 x  -  (ω^2) * x   (per group, broadcast to channels).

    Args:
      d_model: feature dimension D
      groups: number of feature groups (D divisible by groups)
      bc: 'cycle' (periodic) | 'path' (Dirichlet-style)
      dx: grid spacing (float)
      learn_c/learn_omega: learnable per-group speeds c_g and frequencies ω_g
      c_init/omega_init: scalar initial values for all groups

    Forward:
      x [B,T,D] → y [B,T,D]
      mask (optional) gates updates per token; causal=True uses backward stencil (no future).

    Additional Parameters:
      mask: Optional tensor [B,T] gating updates per token (e.g. PAD tokens)
      causal: bool, if True uses backward stencil (no future lookahead)
    """
    def __init__(
        self,
        d_model: int,
        groups: int = 1,
        *,
        bc: _BC = "cycle",
        dx: float = 1.0,
        learn_c: bool = True,
        learn_omega: bool = True,
        c_init: float = 0.0,
        omega_init: float = 0.0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0.")
        if groups <= 0 or (d_model % groups) != 0:
            raise ValueError(f"'groups' must divide d_model. Got D={d_model}, groups={groups}.")
        if dx <= 0:
            raise ValueError("dx must be > 0.")

        self.d_model = int(d_model)
        self.groups = int(groups)
        self.cg = self.d_model // self.groups
        self.bc = bc
        self.dx = float(dx)

        # c_g (wave-speed) and ω_g (natural frequency). Unconstrained here.
        if learn_c:
            self.c = nn.Parameter(torch.full((self.groups,), float(c_init), dtype=torch.float32))
        else:
            self.register_buffer("c", torch.full((self.groups,), float(c_init), dtype=torch.float32))

        if learn_omega:
            self.omega = nn.Parameter(torch.full((self.groups,), float(omega_init), dtype=torch.float32))
        else:
            self.register_buffer("omega", torch.full((self.groups,), float(omega_init), dtype=torch.float32))

    @torch.no_grad()
    def set_c(self, c: Tensor | float) -> None:
        """Set group speeds c_g (scalar or [G])."""
        if isinstance(c, (int, float)):
            vec = torch.full((self.groups,), float(c), dtype=torch.float32, device=self.c.device)
        else:
            if c.numel() == 1:
                vec = torch.full((self.groups,), float(c.item()), dtype=torch.float32, device=self.c.device)
            else:
                if c.shape != (self.groups,):
                    raise ValueError(f"c must be scalar or shape [{self.groups}], got {tuple(c.shape)}")
                vec = c.to(dtype=torch.float32, device=self.c.device)
        if isinstance(self.c, nn.Parameter):
            self.c.data.copy_(vec)
        else:
            self.c.copy_(vec)

    @torch.no_grad()
    def set_omega(self, omega: Tensor | float) -> None:
        """Set group frequencies ω_g (scalar or [G])."""
        if isinstance(omega, (int, float)):
            vec = torch.full((self.groups,), float(omega), dtype=torch.float32, device=self.omega.device)
        else:
            if omega.numel() == 1:
                vec = torch.full((self.groups,), float(omega.item()), dtype=torch.float32, device=self.omega.device)
            else:
                if omega.shape != (self.groups,):
                    raise ValueError(f"omega must be scalar or shape [{self.groups}], got {tuple(omega.shape)}")
                vec = omega.to(dtype=torch.float32, device=self.omega.device)
        if isinstance(self.omega, nn.Parameter):
            self.omega.data.copy_(vec)
        else:
            self.omega.copy_(vec)

    def forward(self, x: Tensor, mask: torch.Tensor | None = None, *, causal: bool = False) -> Tensor:
        """
        Apply oscillatory operator. x: [B,T,D] -> y: [B,T,D]
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must be [B,T,{self.d_model}], got {tuple(x.shape)}")

        B, T, D = x.shape
        g, cg = self.groups, self.cg

        # reshape to [B,T,G,Cg] for groupwise scalars
        xg = x.view(B, T, g, cg)

        if causal:
            lap = _diff2_backward(xg, dx=self.dx)
        else:
            lap = _diff2_central(xg, bc=self.bc, dx=self.dx)

        c2 = (self.c.view(1, 1, g, 1).to(x) ** 2)                 # [1,1,G,1]
        w2 = (self.omega.view(1, 1, g, 1).to(x) ** 2)             # [1,1,G,1]

        yg = c2 * lap - w2 * xg
        y = yg.view(B, T, D)

        if mask is not None:
            if mask.dim() != 2 or mask.shape[0] != B or mask.shape[1] != T:
                raise ValueError(f"mask must be [B,T], got {tuple(mask.shape)}")
            m = mask.to(dtype=y.dtype, device=y.device).view(B, T, 1)
            y = y * m  # zero update on PAD tokens → identity over PAD downstream

        return y


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    # Sanity tests (relative-error based; absolute values are large ~O((2πf)^2))
    torch.manual_seed(0)
    print("[oscillatory] Running sanity tests...")

    device = torch.device("cpu")
    B, D = 2, 12
    G = 3
    cg = D // G

    # --- Periodic: use truly periodic grid t in [0,1) with N points and dx=1/N
    N = 1024
    dt = 1.0 / N
    t = torch.arange(N, dtype=torch.float32, device=device) * dt  # [0,1)
    f = 5  # frequency (cycles over [0,1))

    x_base = torch.sin(2.0 * torch.pi * f * t)  # [N]
    x = x_base.view(1, N, 1).repeat(B, 1, D).contiguous()

    # Ground truth: ∂^2_t sin(2π f t) = -(2π f)^2 sin(2π f t)
    lap_gt = - (2.0 * torch.pi * f) ** 2 * x

    osc = Oscillatory1D(d_model=D, groups=G, bc="cycle", dx=dt, learn_c=False, learn_omega=False, c_init=1.0, omega_init=0.0)
    y = osc(x)
    abs_err = (y - lap_gt).abs().max().item()
    scale = lap_gt.abs().max().item() + 1e-12
    rel_err = abs_err / scale
    print(f"  periodic: max|Δ lap| = {abs_err:.2e}  (rel={rel_err:.3e})")
    assert rel_err < 1e-2, "Relative error should be <1% on a fine periodic grid."

    # --- Add ω (natural frequency): expect y = lap - ω^2 x
    with torch.no_grad():
        osc.set_omega(3.0)  # same ω for all groups
    y2 = osc(x)
    abs_err2 = (y2 - (lap_gt - (3.0 ** 2) * x)).abs().max().item()
    rel_err2 = abs_err2 / scale
    print(f"  add ω:    max|Δ total| = {abs_err2:.2e} (rel={rel_err2:.3e})")
    assert rel_err2 < 1e-2

    # --- Dirichlet/path: check interior (boundaries are one-sided)
    osc_path = Oscillatory1D(d_model=D, groups=G, bc="path", dx=dt, learn_c=False, learn_omega=False, c_init=1.0, omega_init=0.0)
    y_path = osc_path(x)
    abs_int = (y_path[:, 1:-1, :] - lap_gt[:, 1:-1, :]).abs().max().item()
    scale_int = lap_gt[:, 1:-1, :].abs().max().item() + 1e-12
    rel_int = abs_int / scale_int
    print(f"  path:     max interior |Δ lap| = {abs_int:.2e} (rel={rel_int:.3e})")
    assert rel_int < 2e-2, "Interior relative error should be <2%."

    # --- Causal/backward: first two positions must be zero
    y_causal = Oscillatory1D(d_model=D, groups=G, bc="cycle", dx=dt, learn_c=False, learn_omega=False, c_init=1.0, omega_init=0.0)(x, causal=True)
    assert torch.allclose(y_causal[:, :2, :], torch.zeros_like(y_causal[:, :2, :])), "Causal stencil must not use future (t<2 zero)."

    # --- Mask gating: last quarter masked → zero update there
    mask = torch.ones((B, N), dtype=torch.bool)
    mask[:, -N//4:] = False
    osc_mask = Oscillatory1D(d_model=D, groups=G, bc="cycle", dx=dt, learn_c=False, learn_omega=False, c_init=1.0, omega_init=0.0)
    y_masked = osc_mask(x, mask=mask)
    assert (y_masked[:, -N//4:, :].abs().max() < 1e-8), "Masked positions must have zero update."

    print("[oscillatory] All good ✓")

# operators/advective.py
# 1D advective (convection) operator along the sequence/time axis.
# PyTorch-only, no external repo deps. Supports:
#   - Central (2nd order) or upwind (1st order) finite differences
#   - Periodic ('cycle') or non-periodic ('path', Dirichlet style) boundaries
#   - Groupwise speeds: a learnable scalar speed per group (broadcasted over channels)
#
# Shapes:
#   Input  x: [B, T, D]  (batch, time/sequence, features)
#   Output y: [B, T, D]
#
# Notas:
#   - 'cycle': usamos roll (índices circulares). Para que el test periódico sea “limpio”,
#     el mallado debe ser realmente periódico: t_k = k/T (no incluir el punto t=1).
#   - 'path': diferencias interiores centrales y unilaterales en los bordes.
#   - Agrupación: D divisible por `groups`; una velocidad por grupo.
#   - Inicialización: velocidades a 0.0 (lazy minimal), sin efecto hasta que se aprendan.

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn

Tensor = torch.Tensor
_BC = Literal["cycle", "path"]
_Scheme = Literal["central", "upwind"]


class Advective1D(nn.Module):
    """
    Advective (convection) operator along time/sequence (axis=1).

    Args:
      d_model: feature dimension D
      groups: number of feature groups (D must be divisible by groups)
      scheme: 'central' (2nd order) or 'upwind' (1st order adaptive to sign of speed)
      bc:     'cycle' (periodic) or 'path' (Dirichlet-style, no wrap)
      dx:     grid spacing (scalar float)
      learn_speed: if True, speed per group is a trainable Parameter; else it's a buffer
      speed_init: initial speed value (applied to all groups)

    Forward:
      x [B,T,D] → y [B,T,D], with groupwise speeds v_g:
        y_g = v_g * ∂_t x_g
      where ∂_t is approximated by the chosen FD scheme & boundary mode.
    """
    def __init__(
        self,
        d_model: int,
        groups: int = 1,
        *,
        scheme: _Scheme = "central",
        bc: _BC = "cycle",
        dx: float = 1.0,
        learn_speed: bool = True,
        speed_init: float = 0.0,
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
        self.scheme = scheme
        self.bc = bc
        self.dx = float(dx)

        # Groupwise speeds
        if learn_speed:
            self.speed = nn.Parameter(torch.full((self.groups,), float(speed_init), dtype=torch.float32))
        else:
            self.register_buffer("speed", torch.full((self.groups,), float(speed_init), dtype=torch.float32))

    @torch.no_grad()
    def set_speed(self, speed: Tensor | float) -> None:
        """Set group speeds from a tensor [G] or scalar."""
        if isinstance(speed, (int, float)):
            vec = torch.full((self.groups,), float(speed), dtype=torch.float32, device=self.speed.device)
        else:
            if speed.numel() == 1:
                vec = torch.full((self.groups,), float(speed.item()), dtype=torch.float32, device=self.speed.device)
            else:
                if speed.shape != (self.groups,):
                    raise ValueError(f"speed must be scalar or shape [{self.groups}], got {tuple(speed.shape)}")
                vec = speed.to(dtype=torch.float32, device=self.speed.device)
        if isinstance(self.speed, nn.Parameter):
            self.speed.data.copy_(vec)
        else:
            self.speed.copy_(vec)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply advective operator. x: [B,T,D] -> y: [B,T,D]
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must be [B,T,{self.d_model}], got {tuple(x.shape)}")

        B, T, D = x.shape
        g, cg = self.groups, self.cg

        # Reshape to [B,T,G,Cg] to broadcast group speeds over channels in the group
        xg = x.view(B, T, g, cg)

        if self.scheme == "central":
            if self.bc == "cycle":
                right = torch.roll(xg, shifts=-1, dims=1)
                left  = torch.roll(xg, shifts=+1, dims=1)
                diff = (right - left) / (2.0 * self.dx)
            else:  # 'path'
                diff = torch.empty_like(xg)
                diff[:, 1:-1, ...] = (xg[:, 2:, ...] - xg[:, :-2, ...]) / (2.0 * self.dx)
                diff[:, 0,    ...] = (xg[:, 1,    ...] - xg[:, 0,     ...]) / self.dx
                diff[:, -1,   ...] = (xg[:, -1,   ...] - xg[:, -2,    ...]) / self.dx

        elif self.scheme == "upwind":
            v = self.speed.view(1, 1, g, 1).to(dtype=x.dtype, device=x.device)
            if self.bc == "cycle":
                back = xg - torch.roll(xg, shifts=+1, dims=1)     # backward diff
                fwd  = torch.roll(xg, shifts=-1, dims=1) - xg     # forward diff
            else:  # 'path'
                back = torch.empty_like(xg)
                back[:, 1:, ...] = xg[:, 1:, ...] - xg[:, :-1, ...]
                back[:, 0,  ...] = 0.0
                fwd  = torch.empty_like(xg)
                fwd[:, :-1, ...] = xg[:, 1:, ...] - xg[:, :-1, ...]
                fwd[:, -1,  ...] = 0.0
            diff = torch.where(v >= 0, back, fwd) / self.dx
        else:
            raise ValueError("scheme must be 'central' or 'upwind'.")

        # Multiply by group speeds (broadcast [1,1,G,1])
        v = self.speed.view(1, 1, g, 1).to(dtype=x.dtype, device=x.device)
        yg = v * diff

        # Restore to [B,T,D]
        y = yg.view(B, T, D)
        return y


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    # Sanity tests
    torch.manual_seed(0)
    print("[advective] Running sanity tests...")

    B, T, D = 2, 1024 + 1, 12  # usamos T-1 en periodo; ver abajo
    groups = 3
    cg = D // groups
    device = torch.device("cpu")

    # --- Periódico (cycle): mallado realmente periódico
    # Para tests periódicos, trabajamos con N=T-1 muestras en [0,1) con Δ=1/N,
    # y construimos x con N puntos. Para el operador (que espera [B,T,D]),
    # usaremos ese N como T_eff.
    N = T - 1
    dt = 1.0 / N
    t = torch.arange(N, dtype=torch.float32, device=device) * dt  # [0, 1)
    freq = 5  # ciclos sobre [0,1)

    x_base = torch.sin(2.0 * torch.pi * freq * t)  # [N]
    x = x_base.view(1, N, 1).repeat(B, 1, D).contiguous()

    # Ground truth derivative for v=1: d/dt sin(2π f t) = (2π f) cos(2π f t)
    gt = (2.0 * torch.pi * freq) * torch.cos(2.0 * torch.pi * freq * t)  # [N]
    gt = gt.view(1, N, 1).repeat(B, 1, D)

    # Central difference, periodic BCs, v=1 per group
    adv = Advective1D(d_model=D, groups=groups, scheme="central", bc="cycle", dx=dt, learn_speed=True, speed_init=1.0)
    y = adv(x)
    err = (y - gt).abs().max().item()
    print(f"  central/cycle: max|Δ| = {err:.2e}")
    # Con N=1024 y f=5, error ~ O(Δ²) ≈ 0.005 → pasa 1e-2
    assert err < 1e-2, "Central periodic derivative should be accurate on smooth sinusoid with fine grid."

    # Upwind, periodic, mixed speeds per group (solo smoke: que corra y no sea cero)
    adv_up = Advective1D(d_model=D, groups=groups, scheme="upwind", bc="cycle", dx=dt, learn_speed=True, speed_init=0.0)
    with torch.no_grad():
        adv_up.set_speed(torch.tensor([+1.0, -0.5, +0.25], dtype=torch.float32))
    y_up = adv_up(x)
    mag = float(y_up.abs().mean().item())
    print(f"  upwind/cycle: mean|y| = {mag:.2e}")
    assert mag > 0.0

    # --- Dirichlet (path): test de precisión interior (bordes unilaterales)
    # Reusamos el mismo N y dt; el mallado no es periódico.
    x_path = x
    gt_path = gt
    adv_path = Advective1D(d_model=D, groups=groups, scheme="central", bc="path", dx=dt, learn_speed=False, speed_init=1.0)
    y_path = adv_path(x_path)
    interior_err = (y_path[:, 1:-1, :] - gt_path[:, 1:-1, :]).abs().max().item()
    print(f"  central/path: max interior |Δ| = {interior_err:.2e}")
    assert interior_err < 2e-2, "Central interior derivative should be reasonably accurate."

    print("[advective] All good ✓")

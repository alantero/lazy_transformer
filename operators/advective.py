# operators/advective.py
# 1D advective (convection) operator along the sequence/time axis.
# PyTorch-only, no external repo deps. Supports:
#   - Central (2nd order) or upwind (1st order) finite differences
#   - Periodic ('cycle') or non-periodic ('path', Dirichlet style) boundaries
#   - Groupwise speeds: a learnable scalar speed per group (broadcasted over channels)
#   - Optional attention/pad mask to zero out pads and avoid using them in differences
#
# Shapes:
#   Input  x: [B, T, D]  (batch, time/sequence, features)
#   Input  mask: [B, T] (bool, optional) pads are zeroed in outputs and never used to form differences
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
import torch.nn.functional as F

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
      causal: if True, enforce strict causality (no access to x[:, t+1:]) by using a left-padded depthwise conv that implements a backward finite difference. Defaults to False to preserve existing numerical tests (periodic central differences).

    Forward Args:
      x: input tensor of shape [B, T, D]
      mask: optional attention/pad mask of shape [B, T] (bool). Pads are zeroed in outputs and never used to form differences.
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
        causal: bool = False,
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
        self.causal = bool(causal)

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

    def _canon_mask(self, mask: Tensor | None, B: int, T: int, device: torch.device) -> Tensor | None:
        """
        Convert mask to canonical bool tensor of shape [B, T, 1, 1] for broadcasting.
        Accepts None, [B,T], [B,T,1], or [B,T,1,1].
        Returns None if input is None.
        """
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            raise TypeError("mask must be a tensor or None")
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.device != device:
            mask = mask.to(device)
        if mask.dim() == 2:
            # [B,T]
            return mask.view(B, T, 1, 1)
        elif mask.dim() == 3:
            # [B,T,1]
            if mask.shape[2] != 1:
                raise ValueError(f"mask with 3 dims must have shape [B,T,1], got {tuple(mask.shape)}")
            return mask.view(B, T, 1, 1)
        elif mask.dim() == 4:
            # [B,T,1,1]
            if mask.shape[2] != 1 or mask.shape[3] != 1:
                raise ValueError(f"mask with 4 dims must have shape [B,T,1,1], got {tuple(mask.shape)}")
            return mask
        else:
            raise ValueError(f"mask must have 2,3 or 4 dims, got {mask.dim()}")

    def _causal_backward_diff(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Compute causal (backward) finite difference using a depthwise 1D conv
        implemented with left padding. Does NOT look at future indices.
        x: [B, T, D]  -> returns diff: [B, T, D]
        mask: optional [B, T] bool mask, pads zeroed in output and not used in differences
        """
        B, T, D = x.shape
        # Conv1d expects [B, C, T]
        xc = x.permute(0, 2, 1).contiguous()  # [B, D, T]
        # Kernel for backward diff: [-1, 1] / dx, depthwise over D channels
        weight = x.new_zeros((D, 1, 2))
        weight[:, 0, 0] = -1.0 / self.dx
        weight[:, 0, 1] = +1.0 / self.dx
        bias = None
        # Left pad by K-1 (=1) so that y[:, :, t] depends on x[:, :, :t]
        xc_pad = F.pad(xc, (1, 0))  # (left, right)
        # Depthwise conv: groups=D
        diff = F.conv1d(xc_pad, weight, bias=bias, stride=1, padding=0, dilation=1, groups=D)
        # back to [B, T, D]
        diff = diff.permute(0, 2, 1).contiguous()
        # No past sample at t=0
        diff[:, 0, :] = 0.0
        if mask is not None:
            # mask: [B, T] bool
            diff = diff * mask.unsqueeze(-1).to(diff.dtype)
        return diff

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Apply advective operator. x: [B,T,D] -> y: [B,T,D]

        Args:
          x: input tensor [B, T, D]
          mask: optional attention/pad mask [B, T] bool, pads zeroed in output and never used to form differences
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must be [B,T,{self.d_model}], got {tuple(x.shape)}")

        B, T, D = x.shape
        g, cg = self.groups, self.cg

        m = self._canon_mask(mask, B, T, x.device)  # [B,T,1,1] or None

        if self.causal:
            # Use strictly backward finite differences (depthwise conv) along time
            diff_full = self._causal_backward_diff(x, mask=m.squeeze(-1).squeeze(-1) if m is not None else None)  # [B, T, D]
            diff = diff_full.view(B, T, g, cg)
        else:
            # Reshape to [B,T,G,Cg] to broadcast group speeds over channels in the group
            xg = x.view(B, T, g, cg)

            if m is not None:
                mb = m  # [B,T,1,1]
                # neighbor masks
                m_left = torch.zeros_like(mb)
                m_left[:, 1:, ...] = mb[:, :-1, ...]
                m_right = torch.zeros_like(mb)
                m_right[:, :-1, ...] = mb[:, 1:, ...]

            if self.scheme == "central":
                if self.bc == "cycle":
                    right = torch.roll(xg, shifts=-1, dims=1)
                    left  = torch.roll(xg, shifts=+1, dims=1)
                    if m is None:
                        diff = (right - left) / (2.0 * self.dx)
                    else:
                        mb_right = torch.roll(mb, shifts=-1, dims=1)
                        mb_left = torch.roll(mb, shifts=+1, dims=1)
                        valid_central = mb & mb_left & mb_right
                        diff = torch.zeros_like(xg)
                        # central where all three valid
                        diff[valid_central] = ((right - left) / (2.0 * self.dx))[valid_central]
                        # fallback to one-sided diffs where possible
                        valid_back = mb & m_left & (~valid_central)
                        diff[valid_back] = ((xg - left) / self.dx)[valid_back]
                        valid_fwd = mb & m_right & (~valid_central) & (~valid_back)
                        diff[valid_fwd] = ((right - xg) / self.dx)[valid_fwd]
                        # else zero
                else:  # 'path'
                    diff = torch.zeros_like(xg)
                    x_left = torch.empty_like(xg)
                    x_left[:, 1:, ...] = xg[:, :-1, ...]
                    x_left[:, 0, ...] = 0.0
                    x_right = torch.empty_like(xg)
                    x_right[:, :-1, ...] = xg[:, 1:, ...]
                    x_right[:, -1, ...] = 0.0
                    if m is None:
                        diff[:, 1:-1, ...] = (x_right[:, 1:-1, ...] - x_left[:, 1:-1, ...]) / (2.0 * self.dx)
                        diff[:, 0,    ...] = (xg[:, 1,    ...] - xg[:, 0,     ...]) / self.dx
                        diff[:, -1,   ...] = (xg[:, -1,   ...] - xg[:, -2,    ...]) / self.dx
                    else:
                        valid_central = mb & m_left & m_right
                        valid_back = mb & m_left & (~m_right)
                        valid_fwd = mb & (~m_left) & m_right
                        # central interior
                        diff[valid_central] = ((x_right - x_left) / (2.0 * self.dx))[valid_central]
                        # backward at left border or where no right neighbor
                        diff[valid_back] = ((xg - x_left) / self.dx)[valid_back]
                        # forward at right border or where no left neighbor
                        diff[valid_fwd] = ((x_right - xg) / self.dx)[valid_fwd]
                        # else zero (pads or invalid)
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
                if m is not None:
                    valid_back = mb & m_left
                    valid_fwd = mb & m_right
                    back = torch.where(valid_back, back, torch.zeros_like(back))
                    fwd = torch.where(valid_fwd, fwd, torch.zeros_like(fwd))
                diff = torch.where(v >= 0, back, fwd) / self.dx
            else:
                raise ValueError("scheme must be 'central' or 'upwind'.")

        # Multiply by group speeds (broadcast [1,1,G,1])
        v = self.speed.view(1, 1, g, 1).to(dtype=x.dtype, device=x.device)
        yg = v * diff

        # Restore to [B,T,D]
        if m is not None:
            y = (yg.view(B, T, D)) * (m.squeeze(-1).squeeze(-1).to(yg.dtype))
        else:
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

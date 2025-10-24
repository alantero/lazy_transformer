# modules/gauge.py
# Capacity-driven gauge: per-group scale s(t)∈[smin,smax] from a capacity signal.
# Adds optional pre-conv EMA smoothing and an optional stop-grad on the scale.

from __future__ import annotations
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise ImportError("modules/gauge.py requires PyTorch.") from e

# Safe-import EMA whether running as a script or a package
try:
    from utils.ema import EMA  # type: ignore
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.ema import EMA  # type: ignore

# Safe-import capacity_signal whether running as a script or a package
try:
    from utils.capacity import capacity_signal  # type: ignore
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.capacity import capacity_signal  # type: ignore


def _inv_sigmoid(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp(1e-6, 1 - 1e-6)
    return torch.log(y) - torch.log1p(-y)


class WeylGauge(nn.Module):
    """
    Map a (smoothed) capacity signal to a per-group scale and apply it to features.

    Args:
        d: feature dimension (last dim of h).
        groups: number of feature groups; d % groups == 0. One scale per group.
        ksize: kernel size for depthwise Conv1d over time.
        smin, smax: clamp range for stability (typical 0.5..2.0).
        init_identity: initialize so that output scale ≈ 1.
        use_ema: apply a slow EMA to `cap` before the conv.
        ema_alpha: EMA coefficient (closer to 1.0 = slower).
        stopgrad_scale: if True, detach the computed scale (no grads into gauge).
        use_groupnorm: if True, apply GroupNorm pre-normalization to h before fallback capacity.
        gn_groups: number of groups for GroupNorm; defaults to `groups`.
        gamma: exponent for gamma-shaped squashing of the scale after sigmoid.
            gamma=1.0 means no shaping.
    
    Fallback capacity signal:
        If neither `cap` nor `logits` are provided, capacity is computed directly from `h` as an energy proxy.

    Shapes:
        h:   [B, T, D]
        cap: [B, T], [B, T, 1], or [B, T, groups]
        out: [B, T, D]
        mask: [B, T], [B, T, 1], [B, T, groups], or [B, T, D] (bool or 0/1). False = invalid/pad/future.
    """
    def __init__(
        self,
        d: int,
        groups: int = 1,
        ksize: int = 3,
        smin: float = 0.5,
        smax: float = 2.0,
        init_identity: bool = True,
        use_ema: bool = True,
        ema_alpha: float = 0.9,
        stopgrad_scale: bool = False,
        use_groupnorm: bool = True,
        gn_groups: Optional[int] = None,
        gamma: float = 1.0,
    ):
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be > 0, got {d}")
        if groups < 1 or d % groups != 0:
            raise ValueError(f"'groups' must divide d. Got d={d}, groups={groups}.")
        if not (0.0 < smin < smax):
            raise ValueError(f"Require 0 < smin < smax. Got smin={smin}, smax={smax}.")
        if ksize < 1 or ksize % 1 != 0:
            raise ValueError("ksize must be a positive integer.")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in [0,1].")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0.")

        self.d = int(d)
        self.groups = int(groups)
        self.group_size = d // groups
        self.smin = float(smin)
        self.smax = float(smax)

        self.use_ema = bool(use_ema)
        self.ema_alpha = float(ema_alpha)
        self.stopgrad_scale = bool(stopgrad_scale)

        self.cap_ema = EMA(alpha=self.ema_alpha, debias=False) if self.use_ema else None

        pad = ksize // 2
        # Depthwise over `groups` channels (not D); we later expand to D.
        self.dw = nn.Conv1d(
            in_channels=groups,
            out_channels=groups,
            kernel_size=ksize,
            padding=pad,
            groups=groups,
            bias=True,
        )

        # Initialize as identity (scale=1): weights=0, bias to inverse mapping of 1
        target = (1.0 - self.smin) / (self.smax - self.smin)  # σ(z0)
        z0 = float(_inv_sigmoid(torch.tensor(target)))
        if init_identity:
            nn.init.constant_(self.dw.weight, 0.0)
            nn.init.constant_(self.dw.bias, z0)
        else:
            nn.init.normal_(self.dw.weight, std=1e-3)
            nn.init.constant_(self.dw.bias, z0)

        self.gamma = float(gamma)

        self.use_groupnorm = use_groupnorm
        if self.use_groupnorm:
            gn_groups = gn_groups or groups
            if d % gn_groups != 0:
                raise ValueError(f"gn_groups must divide d. Got d={d}, gn_groups={gn_groups}.")
            self.gn_groups = gn_groups
            self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=d, affine=False)
        else:
            self.gn = None
            self.gn_groups = None

    @torch.no_grad()
    def reset_to_identity(self) -> None:
        """Reset parameters so the gauge outputs scale≈1."""
        nn.init.constant_(self.dw.weight, 0.0)
        target = (1.0 - self.smin) / (self.smax - self.smin)
        z0 = float(_inv_sigmoid(torch.tensor(target)))
        nn.init.constant_(self.dw.bias, z0)

    def _capacity_to_groups(self, cap: torch.Tensor, T: int, B: int) -> torch.Tensor:
        """
        Normalize capacity input to shape [B, groups, T].
        Accepts:
            [B, T]           -> broadcast to 1 -> tile to groups
            [B, T, 1]        -> squeeze -> tile
            [B, T, groups]   -> permute to [B, groups, T]
            [B, T, D]        -> reshape to [B, T, groups, group_size] and mean over channels
        """
        if cap.dim() == 2:
            cap = cap.unsqueeze(-1)  # [B, T, 1]
        if cap.shape[0] != B or cap.shape[1] != T:
            raise ValueError(f"cap shape must start with [B,T], got {tuple(cap.shape)}; expected B={B}, T={T}.")

        C = cap.shape[-1]
        if C == 1:
            cap_g = cap.expand(B, T, self.groups)          # [B, T, groups]
            return cap_g.permute(0, 2, 1).contiguous()     # [B, groups, T]
        if C == self.groups:
            return cap.permute(0, 2, 1).contiguous()       # [B, groups, T]
        if C == self.d:
            # Aggregate channels into groups by mean
            cap4 = cap.view(B, T, self.groups, self.group_size)
            cap_g = cap4.mean(dim=-1)                      # [B, T, groups]
            return cap_g.permute(0, 2, 1).contiguous()     # [B, groups, T]
        raise ValueError(f"cap last dim must be 1, groups={self.groups}, or D={self.d}. Got {C}.")

    def _mask_to_groups(self, mask: Optional[torch.Tensor], T: int, B: int) -> Optional[torch.Tensor]:
        """
        Normalize a mask to shape [B, groups, T] (bool).
        Accepts:
            [B, T]            -> expand to [B, T, groups]
            [B, T, 1]         -> squeeze -> expand to groups
            [B, T, groups]    -> permute to [B, groups, T]
            [B, T, D]         -> reshape to [B, T, groups, group_size] and all() over channels
        Returns None if mask is None.
        """
        if mask is None:
            return None
        if mask.dim() not in (2, 3):
            raise ValueError("mask must have shape [B,T] or [B,T,C].")
        if mask.dtype != torch.bool:
            mask = mask != 0
        if mask.shape[0] != B or mask.shape[1] != T:
            raise ValueError(f"mask shape must start with [B,T], got {tuple(mask.shape)}; expected B={B}, T={T}.")
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # [B, T, 1]
        C = mask.shape[-1]
        if C == 1:
            m_g = mask.expand(B, T, self.groups)              # [B, T, groups]
            return m_g.permute(0, 2, 1).contiguous()          # [B, groups, T]
        if C == self.groups:
            return mask.permute(0, 2, 1).contiguous()         # [B, groups, T]
        if C == self.d:
            m4 = mask.view(B, T, self.groups, self.group_size)
            m_g = m4.all(dim=-1)                              # [B, T, groups]
            return m_g.permute(0, 2, 1).contiguous()          # [B, groups, T]
        raise ValueError(f"mask last dim must be 1, groups={self.groups}, or D={self.d}. Got {C}.")

    def _fallback_capacity_from_h(self, h: torch.Tensor, mask: Optional[torch.Tensor], kind: str = "group") -> torch.Tensor:
        """
        Compute fallback capacity from h as an energy proxy.
        Returns a tensor shaped like accepted `cap` inputs (prefer [B,T,groups]).
        """
        B, T, D = h.shape
        e = (h * h).mean(dim=-1, keepdim=True)  # [B, T, 1]
        if mask is not None:
            e = e * mask.to(e.dtype)
        # Normalize per batch to approx [0,1]
        mean_e = e.mean(dim=1, keepdim=True) * 2.0 + 1e-6
        e = e / mean_e
        e = e.clamp(0.0, 1.0)

        if kind == "group":
            # Aggregate/summarize to [B, T, groups] by averaging within groups
            e4 = e.expand(B, T, self.d).view(B, T, self.groups, self.group_size)
            cap_g = e4.mean(dim=-1)  # [B, T, groups]
            return cap_g
        elif kind == "channel":
            return e.expand(B, T, self.d)
        else:  # token
            return e  # [B, T, 1]

    def compute_scale(self, cap: torch.Tensor, *, mask: Optional[torch.Tensor] = None, detach: Optional[bool] = None) -> torch.Tensor:
        """
        Turn capacity signal into per-feature scale s ∈ [smin, smax].
        Returns: s_full with shape [B, T, D].

        detach:
            - None -> use self.stopgrad_scale
            - True -> detach the scale (no grads into conv/bias)
            - False -> allow grads (default behavior)
        """
        if cap.dim() not in (2, 3):
            raise ValueError("cap must have shape [B,T] or [B,T,C].")
        B, T = cap.shape[0], cap.shape[1]

        # Normalize to per-group shape first to keep EMA shape stable across calls
        cap_g = self._capacity_to_groups(cap, T=T, B=B)     # [B, groups, T]

        mask_g = self._mask_to_groups(mask, T=T, B=B)  # [B, groups, T] or None
        if mask_g is not None:
            # Zero-out invalid positions pre-EMA/conv so they don't leak through time via the kernel.
            cap_g = cap_g * mask_g.to(cap_g.dtype)

        # Optional slow EMA across steps (stateful). This smooths capacity between calls.
        if self.cap_ema is not None:
            cap_g = self.cap_ema.update(cap_g.detach())

        # Depthwise temporal smoothing over groups
        z = self.dw(cap_g)                                   # [B, groups, T]
        sigma = torch.sigmoid(z)                             # [B, groups, T]
        if self.gamma != 1.0:
            sigma = sigma.pow(self.gamma)
        s_g = self.smin + (self.smax - self.smin) * sigma    # [B, groups, T]

        if mask_g is not None:
            # Force identity scale on invalid tokens
            one_g = torch.ones_like(s_g)
            s_g = torch.where(mask_g, s_g, one_g)

        # Optional stop-grad on scale
        use_detach = self.stopgrad_scale if detach is None else bool(detach)
        if use_detach:
            s_g = s_g.detach()

        # Expand per-group to per-channel
        s_full_ch = s_g.repeat_interleave(self.group_size, dim=1)  # [B, D, T]
        s_full = s_full_ch.transpose(1, 2).contiguous()            # [B, T, D]
        return s_full

    def forward(self, h: torch.Tensor, cap: Optional[torch.Tensor] = None, *, logits: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, cap_kind: str = "group", smooth_alpha: Optional[float] = None, detach_scale: Optional[bool] = None) -> torch.Tensor:
        """
        Apply gauge: y = h * s(cap).

        h:   [B, T, D]
        cap: [B, T], [B, T, 1], or [B, T, groups]
        """
        if h.dim() != 3:
            raise ValueError(f"h must be [B,T,D], got shape {tuple(h.shape)}")
        B, T, D = h.shape
        if D != self.d:
            raise ValueError(f"Expected D={self.d}, got {D}")

        if self.use_groupnorm:
            h_norm = self.gn(h.transpose(1, 2)).transpose(1, 2)
        else:
            h_norm = h

        # Build capacity if not provided
        if cap is None:
            if logits is not None:
                # Default to group-wise capacity to match gauge grouping
                kind = cap_kind if cap_kind in ("token", "group", "channel") else "group"
                if kind == "group":
                    cap = capacity_signal(logits, mask=mask, smooth_alpha=smooth_alpha, kind="group", groups=self.groups, detach=True)
                elif kind == "channel":
                    cap = capacity_signal(logits, mask=mask, smooth_alpha=smooth_alpha, kind="channel", channels=self.d, detach=True)
                else:  # token
                    cap = capacity_signal(logits, mask=mask, smooth_alpha=smooth_alpha, kind="token", detach=True)
            else:
                # Fallback: compute capacity from h energy proxy
                cap = self._fallback_capacity_from_h(h_norm, mask, kind=cap_kind)

        s = self.compute_scale(cap, mask=mask, detach=detach_scale)  # [B, T, D]
        return h * s


# ---------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    # Smoke tests
    torch.manual_seed(0)
    B, T, D, G = 2, 64, 16, 4

    # Identity behavior (EMA shouldn't matter with identity init)
    gauge = WeylGauge(d=D, groups=G, ksize=5, smin=0.5, smax=2.0, init_identity=True, use_ema=True, ema_alpha=0.9, use_groupnorm=True, gamma=1.3)
    h = torch.randn(B, T, D)
    cap = torch.rand(B, T, 1)
    y = gauge(h, cap)

    # Mask behavior: last 8 tokens invalid -> identity scaling there
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -8:] = False
    cap_masked = torch.rand(B, T, 1)
    y_masked = gauge(h, cap_masked, mask=mask)
    assert torch.allclose(y_masked[:, -8:, :], h[:, -8:, :], atol=1e-6)

    diff = (y - h).abs().max().item()
    print(f"[gauge] identity init max|y-h| = {diff:.2e}")
    assert diff < 1e-5

    # Bounds with EMA
    s = gauge.compute_scale(cap)
    smin, smax = s.min().item(), s.max().item()
    print(f"[gauge] scale range = [{smin:.3f}, {smax:.3f}]")
    assert smin >= gauge.smin - 1e-5 and smax <= gauge.smax + 1e-5

    s_masked = gauge.compute_scale(cap, mask=mask)
    # Scales at masked tail should be exactly 1
    tail = s_masked[:, -8:, :].reshape(-1)
    assert torch.allclose(tail, torch.ones_like(tail), atol=1e-6)

    # Stop-grad test: make the mapping non-trivial and ensure params don't get grads
    gauge2 = WeylGauge(d=D, groups=G, ksize=3, smin=0.5, smax=2.0, init_identity=False, use_ema=True, ema_alpha=0.9, stopgrad_scale=True)
    h2 = torch.randn(B, T, D, requires_grad=True)
    cap2 = torch.rand(B, T, G)  # per-group capacity
    y2 = gauge2(h2, cap2)       # stop-grad active by default
    loss = (y2 ** 2).mean()
    loss.backward()
    max_grad = 0.0
    for p in gauge2.parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.abs().max().item())
    print(f"[gauge] stop-grad scale -> max|param grad| = {max_grad:.2e}")
    assert max_grad < 1e-12

    # Capacity-from-logits sanity: sharper logits -> higher capacity
    V = 32
    logits_sharp = torch.zeros(B, T, V)
    logits_sharp[..., 0] = 8.0  # high confidence on class 0
    logits_rand = torch.randn(B, T, V)

    cap_sharp = capacity_signal(logits_sharp, kind="group", groups=G)
    cap_rand = capacity_signal(logits_rand,  kind="group", groups=G)
    mean_sharp = float(cap_sharp.mean())
    mean_rand  = float(cap_rand.mean())
    print(f"[gauge] capacity(sharp)={mean_sharp:.3f} > capacity(rand)={mean_rand:.3f}")
    assert mean_sharp > mean_rand

    # Forward using internal capacity path (no explicit `cap`)
    y3 = gauge(h, logits=logits_rand, cap_kind="group", smooth_alpha=0.0)
    diff3 = (y3 - h).abs().max().item()
    print(f"[gauge] internal-cap path max|y-h| = {diff3:.2e}")
    assert diff3 < 1e-5

    y3m = gauge(h, logits=logits_rand, cap_kind="group", smooth_alpha=0.0, mask=mask)
    assert torch.allclose(y3m[:, -8:, :], h[:, -8:, :], atol=1e-6)

    # Test gamma shaping effect on scale
    gauge_gamma1 = WeylGauge(d=D, groups=G, ksize=3, smin=0.5, smax=2.0, init_identity=False, use_ema=False, gamma=1.0)
    gauge_gamma2 = WeylGauge(d=D, groups=G, ksize=3, smin=0.5, smax=2.0, init_identity=False, use_ema=False, gamma=2.0)
    # Create a dummy capacity input near sigmoid midpoint
    cap_dummy = torch.full((B, T, G), 0.5)
    s1 = gauge_gamma1.compute_scale(cap_dummy)
    s2 = gauge_gamma2.compute_scale(cap_dummy)
    # The ranges should differ due to gamma shaping
    range1 = (s1.max() - s1.min()).item()
    range2 = (s2.max() - s2.min()).item()
    print(f"[gauge] gamma=1.0 scale range: {range1:.4f}, gamma=2.0 scale range: {range2:.4f}")
    assert abs(range1 - range2) > 1e-5

    # Test fallback capacity from h
    gauge_fallback = WeylGauge(d=D, groups=G, ksize=3, smin=0.5, smax=2.0, use_ema=False, use_groupnorm=True)
    y_fallback = gauge_fallback(h)
    assert y_fallback.shape == h.shape
    # Mask last 8 tokens invalid; output should be identity there
    mask2 = torch.ones(B, T, dtype=torch.bool)
    mask2[:, -8:] = False
    y_fallback_masked = gauge_fallback(h, mask=mask2)
    assert torch.allclose(y_fallback_masked[:, -8:, :], h[:, -8:, :], atol=1e-6)

    print("[gauge] All good ✓")

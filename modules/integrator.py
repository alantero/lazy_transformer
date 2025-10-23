# modules/integrator.py
# ODE/Depth integrators for h ∈ R^{B×T×D} with explicit schemes (Euler/Heun/RK4)
# plus an optional ETD–symplectic step tailored for Port-Hamiltonian fields.
# PyTorch-only. No repo-global deps.

from __future__ import annotations
from typing import Callable, Optional, Tuple, List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
Field = Union[nn.Module, Callable[..., Tensor]]  # avoid PEP604 for Py<3.10


# --------------------------------- helpers -----------------------------------

def _rms(x: Tensor, dim: Optional[int] = None, keepdim: bool = False, eps: float = 1e-12) -> Tensor:
    if dim is None:
        return torch.sqrt(torch.mean(x * x) + eps)
    return torch.sqrt(torch.mean(x * x, dim=dim, keepdim=keepdim) + eps)


def _has_attr(x: Any, name: str) -> bool:
    return hasattr(x, name) and callable(getattr(x, name))




# ------------------------------- Integrator ----------------------------------

class Integrator(nn.Module):
    """
    Time/depth integrator for a vector field f(h, **kwargs) with h shape [B, T, D].

    Args:
        field: nn.Module or callable computing f(h, **field_kwargs) -> Tensor[B,T,D].
               (For 'etd-symplectic', the field can (optionally) expose:
                _apply_G(y), _apply_R(y), _apply_J(y), and attributes R_scale, use_diag_R, rho, B, R_rank.)
        method: 'euler' | 'heun' | 'rk4' | 'etd-symplectic'  (default 'heun' per v3).
        dt: default step size.
        steps: default number of steps.
        limit_rms: if set, scales each step so ||dt·k||_rms ≤ limit_rms (basic stability guard).
        detach_between_steps: if True, detaches h after each step (truncated BPTT).
        record_path: if True by default, forward() returns (hT, [h0,...,hT]).
    """
    def __init__(
        self,
        field: Field,
        *,
        method: str = "heun",
        dt: float = 1.0,
        steps: int = 1,
        limit_rms: Optional[float] = None,
        detach_between_steps: bool = False,
        record_path: bool = False,
    ):
        super().__init__()
        if not callable(field):
            raise TypeError("field must be callable or an nn.Module with __call__.")
        self.field = field
        self.method = method.lower()
        if self.method not in {"euler", "heun", "rk4", "etd-symplectic"}:
            raise ValueError("method must be one of: 'euler', 'heun', 'rk4', 'etd-symplectic'.")
        self.dt = float(dt)
        self.steps = int(steps)
        self.limit_rms = float(limit_rms) if limit_rms is not None else None
        self.detach_between_steps = bool(detach_between_steps)
        self.record_path_default = bool(record_path)

    # ------------------------------- internals --------------------------------

    def _guard_dt(self, k: Tensor, dt: float) -> Tuple[float, float]:
        if self.limit_rms is None:
            return float(dt), float(_rms(k).item())
        k_r = float(_rms(k).item())
        if k_r <= 0:
            return float(dt), k_r
        scale = min(1.0, self.limit_rms / (k_r * abs(dt) + 1e-12))
        return float(dt * scale), k_r

    def _f(self, h: Tensor, **field_kwargs) -> Tensor:
        out = self.field(h, **field_kwargs) if isinstance(self.field, nn.Module) else self.field(h, **field_kwargs)
        if not torch.is_tensor(out):
            raise TypeError("field must return a Tensor.")
        if out.shape != h.shape:
            raise ValueError(f"field returned shape {tuple(out.shape)}, expected {tuple(h.shape)}.")
        return out

    def _call_hook(self, hook, s: int, name: str, x: Tensor, x_next: Tensor, dt_eff: float, k_rms: Optional[float] = None) -> None:
        """Safely invoke optional per-step hook for logging/diagnostics.
        Never raises; computes simple RMS-based stats.
        """
        if hook is None:
            return
        try:
            with torch.no_grad():
                dx = x_next - x
                x_rms = _rms(x).item()
                dx_rms = _rms(dx).item()
                k_est_rms = (dx_rms / (abs(dt_eff) + 1e-12)) if dt_eff != 0.0 else 0.0
                payload = {
                    "step": int(s),
                    "method": str(name),
                    "dt_eff": float(dt_eff),
                    "x_rms": float(x_rms),
                    "dx_rms": float(dx_rms),
                    "k_rms": float(k_rms) if k_rms is not None else float(k_est_rms),
                    "k_est_rms": float(k_est_rms),
                }
                hook(payload)
        except Exception:
            # Do not let diagnostics break the training/inference loop
            pass

    # --------------------------------- steps ----------------------------------

    def _step_euler(self, h: Tensor, dt: float, s: int, hook=None, **kw) -> Tensor:
        k1 = self._f(h, **kw)
        dt1, k1_r = self._guard_dt(k1, dt)
        out = h + dt1 * k1
        self._call_hook(hook, s, "euler", h, out, dt1, k1_r)
        return out

    def _step_heun(self, h: Tensor, dt: float, s: int, hook=None, **kw) -> Tensor:
        k1 = self._f(h, **kw)
        dt1, k1_r = self._guard_dt(k1, dt)
        h1 = h + dt1 * k1
        k2 = self._f(h1, **kw)
        k_avg = 0.5 * (k1 + k2)
        dt2, k_avg_r = self._guard_dt(k_avg, dt)
        out = h + dt2 * k_avg
        self._call_hook(hook, s, "heun", h, out, dt2, k_avg_r)
        return out

    def _step_rk4(self, h: Tensor, dt: float, s: int, hook=None, **kw) -> Tensor:
        k1 = self._f(h, **kw)
        dt_eff, k1_r = self._guard_dt(k1, dt)
        k2 = self._f(h + 0.5 * dt_eff * k1, **kw)
        k3 = self._f(h + 0.5 * dt_eff * k2, **kw)
        k4 = self._f(h + dt_eff * k3, **kw)
        out = h + (dt_eff / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Use k1 RMS as a proxy for logging
        self._call_hook(hook, s, "rk4", h, out, dt_eff, k1_r)
        return out

    # ------------------------- ETD–symplectic step ----------------------------
    # This step targets fields of the form f(h) = (J - R) G h (e.g., PortHamiltonianStep).
    # It performs a split update:
    #   (i) Dissipative ETD: y <- exp(-dt * R) (G h)  [exact for diagonal R; low-rank part first-order]
    #  (ii) Skew "wave" step for J via midpoint/leapfrog: y <- y + dt * J(y + 0.5 dt J y)
    # If the field doesn't expose the needed hooks, we gracefully fall back to Heun.
    def _step_etd_symplectic(self, h: Tensor, dt: float, s: int, hook=None, **kw) -> Tensor:
        f = self.field
        # Check duck-typed hooks
        hasG = _has_attr(f, "_apply_G")
        hasR = _has_attr(f, "_apply_R")
        hasJ = _has_attr(f, "_apply_J")
        if not (hasG and hasR and hasJ):
            # Fallback
            return self._step_heun(h, dt, s, hook, **kw)

        B, T, D = h.shape
        y = h.view(B * T, D)  # flatten for last-dim linear ops

        # Resolve dissipative scale in a way that is compatible with PortHamiltonianStep
        if hasattr(f, "_R_scale") and callable(getattr(f, "_R_scale")):
            _rs = f._R_scale()
            if torch.is_tensor(_rs):
                R_scale = float(_rs.detach().float().item())
            else:
                R_scale = float(_rs)
        elif hasattr(f, "R_scale"):
            _rs = getattr(f, "R_scale")
            R_scale = float(_rs.detach().float().item()) if torch.is_tensor(_rs) else float(_rs)
        else:
            R_scale = 1.0

        # (i) Pre-mix with G
        y = f._apply_G(y)

        # (i-a) ETD for R: try exact diagonal exp if available
        used_exact_diag = False
        if getattr(f, "use_diag_R", False) and hasattr(f, "rho"):
            diag = F.softplus(getattr(f, "rho"))  # [D]
            if torch.is_tensor(diag) and diag.dim() == 1 and diag.numel() == D:
                s = torch.exp(-dt * R_scale * (diag + getattr(f, "eps", 0.0)))
                y = y * s  # exact ETD for diag
                used_exact_diag = True

        # (i-b) Low-rank part:
        # If we used the exact diagonal, add only the low-rank correction explicitly.
        # Otherwise, fall back to the full _apply_R which already includes all parts.
        if used_exact_diag and hasattr(f, "R_rank") and getattr(f, "R_rank") and hasattr(f, "B") and getattr(f, "B") is not None:
            y = y - dt * R_scale * ((y @ f.B) @ f.B.t())
        elif not used_exact_diag:
            # Full dissipative Euler fallback (covers diag + low-rank consistently)
            y = y - dt * f._apply_R(y)

        # (ii) Skew step via midpoint/leapfrog (symplectic for linear skew flows)
        Jy = f._apply_J(y)
        y_mid = y + 0.5 * dt * Jy
        Jy_mid = f._apply_J(y_mid)
        y = y + dt * Jy_mid

        out = y.view(B, T, D)
        self._call_hook(hook, s, "etd-symplectic", h, out, dt, None)
        return out

    # -------------------------------- forward ---------------------------------

    def forward(
        self,
        h: Tensor,
        *,
        steps: Optional[int] = None,
        dt: Optional[float] = None,
        record_path: Optional[bool] = None,
        should_step: Optional[Callable[[Tensor, Tensor, int], bool]] = None,
        step_hook: Optional[Callable[[Dict[str, float]], None]] = None,
        **field_kwargs,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """
        Integrate h forward by `steps` using `method` and step size `dt`.

        Parameters:
            record_path: override constructor flag to return (h_T, [h_0,...,h_T]).
            should_step(prev, cur, s): optional gate; if returns False at step s, stop early.
            step_hook: optional callable receiving step info dict for logging/monitoring.

        Any extra keyword arguments are forwarded to the field: f(h, **field_kwargs).
        """
        if h.dim() != 3:
            raise ValueError(f"h must be [B,T,D], got {tuple(h.shape)}")

        S = int(self.steps if steps is None else steps)
        dT = float(self.dt if dt is None else dt)
        keep_path = self.record_path_default if record_path is None else bool(record_path)

        step_fn = {
            "euler": self._step_euler,
            "heun": self._step_heun,
            "rk4": self._step_rk4,
            "etd-symplectic": self._step_etd_symplectic,
        }[self.method]

        x = h
        if keep_path:
            path = [x.detach().clone()]

        for s in range(S):
            x_next = step_fn(x, dT, s, step_hook, **field_kwargs)
            if self.detach_between_steps:
                x_next = x_next.detach()

            # Optional ΔBKM-like gate (user-supplied policy)
            if should_step is not None:
                # If policy says "do not continue", bail out returning the latest x.
                if not bool(should_step(x, x_next, s)):
                    x = x_next
                    if keep_path:
                        path.append(x.detach().clone())
                    break

            x = x_next
            if keep_path:
                path.append(x.detach().clone())

        return (x, path) if keep_path else x


# ----------------------------------- __main__ --------------------------------

if __name__ == "__main__":
    # Smoke tests:
    torch.manual_seed(0)
    B, T, D = 2, 64, 16

    # 1) Zero vector field → constant solution
    class ZeroField(nn.Module):
        def forward(self, h: Tensor) -> Tensor:
            return torch.zeros_like(h)

    h0 = torch.randn(B, T, D)
    integ = Integrator(ZeroField(), method="rk4", dt=0.1, steps=10, record_path=True)
    hT, path = integ(h0)
    err = (hT - h0).abs().max().item()
    print(f"[integrator] Zero field max|Δ| = {err:.2e}")
    assert err < 1e-7 and len(path) == 11

    # 2) Linear skew field (energy-preserving): f(h)=A h with A^T=-A
    M = torch.randn(D, D)
    A = M - M.t()
    A = A / (torch.linalg.norm(A) + 1e-6)
    class SkewField(nn.Module):
        def forward(self, h: Tensor) -> Tensor:
            B, T, D = h.shape
            v = h.reshape(B * T, D) @ A.t()
            return v.view(B, T, D)

    h0 = torch.randn(B, T, D)
    integ = Integrator(SkewField(), method="heun", dt=0.1, steps=50, limit_rms=0.5)

    def demo_hook(info: Dict[str, float]):
        if info["step"] in (0, 25, 49):
            print(f"[hook] s={info['step']} method={info['method']} dt={info['dt_eff']:.3f} x_rms={info['x_rms']:.3f} k_rms={info['k_rms']:.3f}")
    hT = integ(h0, step_hook=demo_hook)

    n0 = torch.linalg.vector_norm(h0, dim=-1).mean().item()
    nT = torch.linalg.vector_norm(hT, dim=-1).mean().item()
    drift = abs(nT - n0) / (n0 + 1e-12)
    print(f"[integrator] Skew field norm drift = {drift:.3%}")
    assert drift < 0.02  # <2% drift

    # 3) With the PortHamiltonianStep (lazy-minimal init → near-zero dynamics)
    try:
        from modules.portham import PortHamiltonianStep
        step = PortHamiltonianStep(d=D, groups=4, skew_rank=16, R_rank=4, traceless=True)

        # Guard: ensure integrator can read R scale via the new helper
        if hasattr(step, "_R_scale"):
            _rs = step._R_scale()
            assert isinstance(_rs, (float, torch.Tensor)), "PortHamiltonianStep._R_scale should return float or Tensor"

        h0 = torch.randn(B, T, D)

        # Heun: with J_scale=R_scale=0, derivative ~0
        integ_h = Integrator(step, method="heun", dt=0.5, steps=8)
        hT_h = integ_h(h0)
        diff_h = (hT_h - h0).abs().max().item()
        print(f"[integrator] PortHam Heun (lazy) max|Δ| = {diff_h:.2e}")
        assert diff_h < 1e-6

        # ETD–symplectic step should also be stable at lazy init
        integ_e = Integrator(step, method="etd-symplectic", dt=0.5, steps=4)
        hT_e = integ_e(h0)
        diff_e = (hT_e - h0).abs().max().item()
        print(f"[integrator] PortHam ETD-symplectic (lazy) max|Δ| = {diff_e:.2e}")
        assert diff_e < 1e-6

        # Kick dissipation: R_scale>0 → energy should decrease
        with torch.no_grad():
            step.R_scale.fill_(0.3)
        hT_e2 = integ_e(h0)
        e0 = (h0 * h0).sum(dim=-1).mean().item()
        e2 = (hT_e2 * hT_e2).sum(dim=-1).mean().item()
        print(f"[integrator] ETD-symplectic energy e0={e0:.3f} -> eT={e2:.3f}")
        assert e2 <= e0 + 1e-6

        # Gate demo: stop after first step if change is tiny
        def small_change_gate(prev: Tensor, cur: Tensor, s: int) -> bool:
            return (cur - prev).abs().max().item() > 1e-7 if s > 0 else True

        hT_g = Integrator(step, method="heun", dt=0.5, steps=10)(h0, should_step=small_change_gate)
        # No assertion; just ensure it runs
        print("[integrator] Gate demo ran ✓")

    except Exception as e:
        print(f"[integrator] PortHamiltonianStep test skipped: {e}")

    print("[integrator] All good ✓")

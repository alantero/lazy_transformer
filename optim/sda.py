# optim/sda.py
# Simple dual-ascent / stochastic dual averaging (SDA) for inequality constraints.
# Self-contained (PyTorch-only). No cross-file deps.
#
# Use-case in this repo (Phase 3):
#   - Regulate a metric, e.g. bits-per-token, capacity, ΔBKM, etc.
#   - Define constraints like  metric[name]  <= target[name]
#   - Add dual penalty λ·(metric-target) to the loss.
#   - Update λ with a (projected) dual step: λ ← [λ + η·(metric-target)]₊.
#
# API (high level):
#   sda = DualSDA({"bpp": 1.5}, lr=1e-2)        # target bpp ≤ 1.5
#   loss_aux, pen_stats = sda.penalty({"bpp": current_bpp})
#   total_loss = task_loss + loss_aux
#   total_loss.backward(); optimizer.step()
#   sda.update({"bpp": current_bpp})            # after step (or before next)
#
# Notes
# - Supports multiple constraints (with independent λ_i and per-constraint sense).
# - Optional EMA smoothing of observed metrics to stabilize updates.
# - “SDA” here is a light dual averaging variant (Robbins–Monro style); for most
#   training loops a simple projected dual ascent is enough and robust.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple, Any

import math
import torch
import torch.nn as nn

Tensor = torch.Tensor
_Sense = Literal["<=", ">="]  # inequality direction


@dataclass
class DualConstraint:
    name: str
    target: float
    sense: _Sense = "<="  # interpret as metric <= target (default)
    init_lambda: float = 0.0
    min_lambda: float = 0.0
    max_lambda: float = 1e6
    lr: float = 1e-2                # dual step size (can be annealed externally)
    ema_alpha: float = 0.9          # smoothing for the observed metric
    use_sda: bool = True            # if True, use 1/sqrt(t) scaling; else fixed lr


class DualSDA(nn.Module):
    """
    Manages a set of dual variables λ_i for inequality constraints over metrics.

    Constraint convention:
        if sense == "<=": encourage metric[name] <= target
        if sense == ">=": encourage metric[name] >= target

    Penalty added to the loss:
        L_dual = Σ_i  λ_i * g_i(metric)
      where g_i(metric) = (metric - target) for "<=" and (target - metric) for ">=".
      (So g_i(metric) > 0 means violation; λ_i ascends when violated.)

    Update rule (projected ascent):
        λ_i ← clip( λ_i + η_i(t) * g_i( m̂_i ), [min_lambda, max_lambda] )
      where m̂_i is the EMA-smoothed metric.

    Typical usage:
        sda = DualSDA({"bpp": 1.5})
        dual_loss, pen = sda.penalty({"bpp": bpp})     # add to task loss
        ...
        sda.update({"bpp": bpp})                       # adjust λ
    """
    def __init__(
        self,
        targets: Dict[str, float],
        *,
        senses: Optional[Dict[str, _Sense]] = None,
        lrs: Optional[Dict[str, float]] = None,
        init_lambdas: Optional[Dict[str, float]] = None,
        min_lambda: float = 0.0,
        max_lambda: float = 1e6,
        ema_alpha: float = 0.9,
        use_sda: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._device = device
        self._dtype = dtype

        # Buffers (so they move with .to(device))
        self.register_buffer("_t", torch.zeros((), dtype=torch.long, device=device), persistent=True)

        # Book-keeping per constraint
        self.names = list(targets.keys())
        self.targets = {k: float(v) for k, v in targets.items()}
        self.senses = {k: (senses[k] if senses and k in senses else "<=") for k in self.names}
        self.ema_alpha = float(ema_alpha)
        self.use_sda = bool(use_sda)

        # Dual variables λ (as parameters? we do not want grads through them → keep as buffers)
        lambdas = {}
        meters = {}
        steps = {}
        lrs_map = {}
        for k in self.names:
            init_lam = float(init_lambdas[k]) if (init_lambdas and k in init_lambdas) else 0.0
            lambdas[k] = torch.tensor(init_lam, device=device, dtype=dtype)
            meters[k] = torch.tensor(self.targets[k], device=device, dtype=dtype)  # init meter to target
            steps[k] = torch.tensor(0, device=device, dtype=torch.long)
            lrs_map[k] = float(lrs[k]) if (lrs and k in lrs) else float(1e-2)

        self._lambda = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in lambdas.items()})
        self._meter = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in meters.items()})
        self._stepi = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in steps.items()})
        self._lr = {k: lrs_map[k] for k in self.names}

        # Shared bounds
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)

    # ------------------------------- utilities --------------------------------

    @torch.no_grad()
    def _g(self, name: str, metric: Tensor | float) -> Tensor:
        """
        Constraint function g(metric):
          "<=" : metric - target
          ">=" : target - metric
        Positive → violation.
        """
        t = torch.as_tensor(self.targets[name], device=self._lambda[name].device, dtype=self._lambda[name].dtype)
        m = torch.as_tensor(metric, device=t.device, dtype=t.dtype)
        if self.senses[name] == "<=":
            return m - t
        else:  # ">="
            return t - m

    @torch.no_grad()
    def _dual_lr(self, name: str) -> float:
        """SDA-style 1/sqrt(t) scaling or fixed lr."""
        base = self._lr[name]
        if not self.use_sda:
            return base
        t = int(self._stepi[name].item()) + 1
        return base / math.sqrt(max(t, 1))

    # ---------------------------- public interface -----------------------------

    def penalty(self, metrics: Dict[str, Tensor | float]) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute dual penalty Σ λ_i * g_i(metric_i).
        Returns (penalty_tensor, stats), where stats include each term and λ.
        """
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device("cpu")
        dtype = self._dtype

        total = torch.zeros((), device=device, dtype=dtype)
        stats: Dict[str, float] = {}
        for name in self.names:
            if name not in metrics:
                continue
            g = self._g(name, metrics[name])  # scalar tensor or broadcastable
            lam = self._lambda[name]
            # Reduce g to a scalar (mean over batch/time if needed)
            g_scalar = g.mean()
            total = total + lam * g_scalar
            stats[f"dual/{name}/g"] = float(g_scalar.detach().cpu())
            stats[f"dual/{name}/lambda"] = float(lam.detach().cpu())
            stats[f"dual/{name}/term"] = float((lam * g_scalar).detach().cpu())
        return total, stats

    @torch.no_grad()
    def update(self, metrics: Dict[str, Tensor | float]) -> Dict[str, float]:
        """
        Update λ via projected dual ascent using EMA-smoothed metrics.
        Returns a dict of updated λ for logging.
        """
        out: Dict[str, float] = {}
        for name in self.names:
            if name not in metrics:
                continue
            # Update EMA meter
            m_prev = self._meter[name]
            obs = torch.as_tensor(metrics[name], device=m_prev.device, dtype=m_prev.dtype)
            obs_scalar = obs.mean()  # reduce if batched
            alpha = self.ema_alpha
            m_new = alpha * m_prev + (1.0 - alpha) * obs_scalar
            self._meter[name].copy_(m_new)

            # Dual step
            g = self._g(name, m_new)                # scalar tensor
            lr = self._dual_lr(name)
            lam = self._lambda[name] + lr * g
            lam = lam.clamp_(min=self.min_lambda, max=self.max_lambda)
            self._lambda[name].copy_(lam)

            # Per-constraint time index
            self._stepi[name].add_(1)

            out[f"dual/{name}/meter"] = float(m_new.detach().cpu())
            out[f"dual/{name}/g"] = float(g.detach().cpu())
            out[f"dual/{name}/lambda"] = float(lam.detach().cpu())

        # Global time
        self._t.add_(1)
        return out

    # ----------------------------- extra payload (optional) ----------------------------

    def to_payload(self) -> Dict[str, Any]:
        """
        Optional extra metadata snapshot (JSON-serializable). This does NOT replace
        PyTorch's state_dict(). Use this only if you want to save/restore non-tensor
        configs (targets, senses, lrs, etc.).
        """
        payload: Dict[str, Any] = {
            "names": self.names,
            "targets": self.targets,
            "senses": self.senses,
            "min_lambda": self.min_lambda,
            "max_lambda": self.max_lambda,
            "ema_alpha": self.ema_alpha,
            "use_sda": self.use_sda,
            "lr": self._lr,
            "t": int(self._t.item()),
            "lambda": {k: float(v.detach().cpu()) for k, v in self._lambda.items()},
            "meter": {k: float(v.detach().cpu()) for k, v in self._meter.items()},
            "stepi": {k: int(v.detach().cpu()) for k, v in self._stepi.items()},
            "dtype": str(self._dtype),
        }
        return payload

    @torch.no_grad()
    def load_payload(self, state: Dict[str, Any]) -> None:
        """
        Restore metadata saved by to_payload(). This does not modify parameters/buffers
        handled by the regular load_state_dict(). Missing keys are ignored.
        """
        # Basic structure (ignore if not present)
        self.names = list(state.get("names", getattr(self, "names", [])))
        self.targets = {k: float(v) for k, v in state.get("targets", getattr(self, "targets", {})).items()}
        self.senses = dict(state.get("senses", getattr(self, "senses", {})))
        self.min_lambda = float(state.get("min_lambda", getattr(self, "min_lambda", 0.0)))
        self.max_lambda = float(state.get("max_lambda", getattr(self, "max_lambda", 1e6)))
        self.ema_alpha = float(state.get("ema_alpha", getattr(self, "ema_alpha", 0.9)))
        self.use_sda = bool(state.get("use_sda", getattr(self, "use_sda", True)))
        self._lr = {k: float(v) for k, v in state.get("lr", getattr(self, "_lr", {})).items()}

        # Restore tensors where keys exist
        if self.names:
            for k in self.names:
                if ("lambda" in state) and (k in state["lambda"]) and (k in self._lambda):
                    self._lambda[k].data.copy_(torch.tensor(state["lambda"][k], device=self._lambda[k].device, dtype=self._lambda[k].dtype))
                if ("meter" in state) and (k in state["meter"]) and (k in self._meter):
                    self._meter[k].data.copy_(torch.tensor(state["meter"][k], device=self._meter[k].device, dtype=self._meter[k].dtype))
                if ("stepi" in state) and (k in state["stepi"]) and (k in self._stepi):
                    self._stepi[k].data.copy_(torch.tensor(state["stepi"][k], device=self._stepi[k].device, dtype=torch.long))
        if "t" in state:
            self._t.data.copy_(torch.tensor(state["t"], device=self._t.device, dtype=torch.long))


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[sda] Running sanity tests...")

    # Toy scenario: regulate a scalar metric m (e.g., bpp) to be <= 1.5
    sda = DualSDA({"m": 1.5}, lrs={"m": 5e-2}, init_lambdas={"m": 0.0}, ema_alpha=0.7, use_sda=True)

    # Simulate an observed metric decaying from 2.5 toward 1.2 over time
    vals = torch.linspace(2.5, 1.2, steps=150)
    lam_hist = []
    g_hist = []
    m_hist = []

    for v in vals:
        pen, _ = sda.penalty({"m": v})
        # emulate an optimizer step here if needed (not required for dual-only test)
        stats = sda.update({"m": v})
        lam_hist.append(stats["dual/m/lambda"])
        g_hist.append(stats["dual/m/g"])
        m_hist.append(stats["dual/m/meter"])

    print(f"  final λ≈{lam_hist[-1]:.3f}, g≈{g_hist[-1]:.3e}, meter≈{m_hist[-1]:.3f}")
    # Expectation: λ grows while m > target, then stabilizes/decreases as m falls below target.
    assert lam_hist[10] > lam_hist[0] and abs(g_hist[-1]) < 0.3

    # Multi-constraint quick check
    sda2 = DualSDA({"bpp": 1.4, "cap": 0.6}, senses={"bpp": "<=", "cap": ">="}, lrs={"bpp": 1e-2, "cap": 2e-2})
    for _ in range(50):
        # bpp slightly high, cap slightly low
        sda2.update({"bpp": 1.8, "cap": 0.4})
    lam_bpp = sda2._lambda["bpp"].item()
    lam_cap = sda2._lambda["cap"].item()
    print(f"  multi: λ_bpp={lam_bpp:.3f}, λ_cap={lam_cap:.3f}")
    assert lam_bpp > 0.0 and lam_cap > 0.0

    print("[sda] All good ✓")

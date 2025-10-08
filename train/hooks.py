# train/hooks.py
# Generic training hooks (PyTorch-only, self-contained).
# Utilities:
#   • Grad clipping (by norm or value)
#   • EMA (Polyak) of parameters (save/restore/apply-shadow)
#   • Cosine LR with linear warmup
#   • Lightweight checkpointing (model/opt/scheduler/ema/state)
#   • Drift/refresh detector (e.g., bulk vs. collar mean mismatch)
#   • ΔBKM gate (trigger action only when collar ΔBKM exceeds a threshold)
#   • Metric averager + logging helper
#
# This file has **no** repo-global deps; safe to import anywhere.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any

import math
import os
import time
import json
import logging

import torch
import torch.nn as nn


Tensor = torch.Tensor


# ------------------------------- metric helpers -------------------------------

class MetricAverager:
    """Running averages for scalar dicts."""
    def __init__(self):
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, scalars: Dict[str, float], n: int = 1) -> None:
        for k, v in scalars.items():
            self._totals[k] = self._totals.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0) + n

    def averages(self) -> Dict[str, float]:
        out = {}
        for k, s in self._totals.items():
            c = max(self._counts.get(k, 0), 1)
            out[k] = s / c
        return out

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()


def log_stats(step: int, stats: Dict[str, float], *, prefix: str = "", logger: Optional[logging.Logger] = None) -> None:
    """Pretty one-liner logging for scalar stats."""
    lg = logger if logger is not None else logging.getLogger(__name__)
    parts = [f"step {step:06d}"]
    if prefix:
        parts.append(f"[{prefix}]")
    for k, v in stats.items():
        parts.append(f"{k}={v:.6g}")
    lg.info(" | ".join(parts))


# -------------------------------- grad clipping --------------------------------

def clip_gradients(
    model: nn.Module,
    *,
    max_norm: Optional[float] = None,
    clip_value: Optional[float] = None,
) -> float:
    """
    Clip gradients by global L2 norm and/or elementwise value.
    Returns the (pre-clip) global grad norm (0 if no grads).
    """
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0

    # Compute global grad norm
    device = params[0].grad.device
    total = torch.zeros((), device=device)
    for p in params:
        total = total + p.grad.detach().float().pow(2).sum()
    grad_norm = float(torch.sqrt(total).item())

    if clip_value is not None:
        for p in params:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(params, max_norm)

    return grad_norm


# ----------------------------------- EMA --------------------------------------

class ParamsEMA:
    """
    Polyak EMA over parameters (optionally buffers).
    Typical usage:
        ema = ParamsEMA(model, decay=0.999)
        ...
        loss.backward(); optimizer.step()
        ema.update(model)   # after each optimizer step
        ...
        ema.apply_shadow(model)   # for eval/export
        ...
        ema.restore(model)        # back to live params
    """
    def __init__(self, model: nn.Module, *, decay: float = 0.999, include_buffers: bool = False):
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0,1).")
        self.decay = float(decay)
        self.include_buffers = bool(include_buffers)
        self.shadow: Dict[str, Tensor] = {}
        self._backup: Dict[str, Tensor] = {}
        self.register(model)

    @torch.no_grad()
    def register(self, model: nn.Module) -> None:
        self.shadow.clear()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
        if self.include_buffers:
            for name, b in model.named_buffers():
                self.shadow[f"__buf__:{name}"] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow.get(name, None)
            if s is None:
                self.shadow[name] = p.detach().clone()
            else:
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        if self.include_buffers:
            for name, b in model.named_buffers():
                key = f"__buf__:{name}"
                s = self.shadow.get(key, None)
                if s is None:
                    self.shadow[key] = b.detach().clone()
                else:
                    s.mul_(d).add_(b.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        """Swap in EMA weights (store live in backup)."""
        self._backup.clear()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name])
        if self.include_buffers:
            for name, b in model.named_buffers():
                key = f"__buf__:{name}"
                self._backup[key] = b.detach().clone()
                b.data.copy_(self.shadow[key])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore live weights (undo apply_shadow)."""
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self._backup:
                p.data.copy_(self._backup[name])
        if self.include_buffers:
            for name, b in model.named_buffers():
                key = f"__buf__:{name}"
                if key in self._backup:
                    b.data.copy_(self._backup[key])
        self._backup.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "include_buffers": self.include_buffers, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, Any], *, device: Optional[torch.device] = None) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.include_buffers = bool(state.get("include_buffers", self.include_buffers))
        shadow = state.get("shadow", {})
        dev = device if device is not None else torch.device("cpu")
        self.shadow = {k: torch.as_tensor(v, device=dev) for k, v in shadow.items()}


# ------------------------- cosine LR with linear warmup ------------------------

@dataclass
class CosineWarmupConfig:
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr: float = 0.0

class CosineWithWarmup:
    """
    Step-based cosine scheduler with linear warmup.
    Call .step(optimizer, step) to set lr for all param groups.
    """
    def __init__(self, cfg: CosineWarmupConfig):
        if cfg.total_steps <= 0:
            raise ValueError("total_steps must be > 0.")
        self.cfg = cfg

    def lr_at(self, step: int) -> float:
        w, T = self.cfg.warmup_steps, self.cfg.total_steps
        if step < w and w > 0:
            return self.cfg.base_lr * (step + 1) / w
        # cosine over [w, T]
        t = min(max(step - w, 0), max(T - w, 1))
        frac = t / max(T - w, 1)
        cos_term = 0.5 * (1.0 + math.cos(math.pi * frac))
        return self.cfg.min_lr + (self.cfg.base_lr - self.cfg.min_lr) * cos_term

    def step(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        lr = self.lr_at(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.cfg = CosineWarmupConfig(**state)


# --------------------------------- checkpointing -------------------------------

class Checkpointer:
    """
    Minimal checkpoint manager. Saves:
      - model.state_dict()
      - optimizer.state_dict()
      - scheduler state (if provided)
      - ema.state_dict() (if provided)
      - step + extra_stats (user scalars)
    """
    def __init__(self, dirpath: str, *, keep_last: int = 3):
        self.dir = dirpath
        self.keep_last = int(keep_last)
        os.makedirs(self.dir, exist_ok=True)

    def _ckpt_path(self, step: int) -> str:
        return os.path.join(self.dir, f"ckpt_step{step:06d}.pt")

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Optional[CosineWithWarmup] = None,
        ema: Optional[ParamsEMA] = None,
        extra_stats: Optional[Dict[str, float]] = None,
    ) -> str:
        path = self._ckpt_path(step)
        payload: Dict[str, Any] = {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "time": time.time(),
            "extra_stats": extra_stats or {},
        }
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if ema is not None:
            payload["ema"] = ema.state_dict()
        torch.save(payload, path)

        # cleanup old checkpoints
        self._cleanup()
        return path

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        scheduler: Optional[CosineWithWarmup] = None,
        ema: Optional[ParamsEMA] = None,
        map_location: Optional[str | torch.device] = None,
    ) -> Dict[str, Any]:
        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if ema is not None and "ema" in payload:
            ema.load_state_dict(payload["ema"], device=next(model.parameters()).device)
        return payload

    def _cleanup(self) -> None:
        # Keep at most keep_last newest checkpoints
        files = sorted([f for f in os.listdir(self.dir) if f.startswith("ckpt_step") and f.endswith(".pt")])
        while len(files) > self.keep_last:
            oldest = files.pop(0)
            try:
                os.remove(os.path.join(self.dir, oldest))
            except OSError:
                pass


# ----------------------------- drift/refresh hook -----------------------------

# ----------------------------- drift/refresh hook -----------------------------

class DriftRefresher:
    """
    Detects drift between a “bulk” region vs. a “collar” (overlap) region.
    Expects per-step calls to .measure(h_bulk, h_collar) and you query .should_refresh().
    Criterion: abs(mean(h_bulk) - mean(h_collar)) > threshold.
    """
    def __init__(self, *, threshold: float = 1e-2):
        self.threshold = float(threshold)
        self.last_drift: float = 0.0
        self._flag: bool = False

    @torch.no_grad()
    def measure(self, h_bulk: Tensor, h_collar: Tensor) -> float:
        mb = float(h_bulk.mean().item())
        mc = float(h_collar.mean().item())
        self.last_drift = abs(mb - mc)
        self._flag = bool(self.last_drift > self.threshold)
        return self.last_drift

    def should_refresh(self) -> bool:
        return self._flag

    def state(self) -> Dict[str, float]:
        return {"drift": self.last_drift, "threshold": self.threshold, "refresh": float(self._flag)}


# --------------------- ΔBKM collar gate (one-shot, non-accum) -----------------
class BKMCollarGate:
    """
    One-shot gate that triggers when ΔBKM (e.g., collar) exceeds a threshold.
    Non-accumulating: it returns a boolean per update, with optional cooldown
    steps to avoid re-triggering continuously.
    """
    def __init__(self, threshold: float = 1e-3, cooldown: int = 0):
        self.threshold = float(threshold)
        self.cooldown = int(cooldown)
        self._cool = 0
        self.last_bkm: float = 0.0

    @torch.no_grad()
    def update(self, bkm: float | Tensor) -> bool:
        val = float(bkm.detach().item() if isinstance(bkm, torch.Tensor) else bkm)
        self.last_bkm = val
        if self._cool > 0:
            self._cool -= 1
            return False
        trigger = bool(val > self.threshold)
        if trigger and self.cooldown > 0:
            self._cool = self.cooldown
        return trigger

    def state(self) -> Dict[str, float]:
        return {"bkm": self.last_bkm, "threshold": self.threshold, "cooldown": float(self.cooldown), "cool": float(self._cool)}


# -------------------------------- master hook API -----------------------------

class StepHooks:
    """
    Orchestrates common per-step actions:
      - LR scheduling (cosine+warmup)
      - Grad clipping (norm/value)
      - EMA update
      - Checkpointing
      - Optional drift detection stats
    Flexible: call the pieces you need from your loop.

    Typical usage:
        hooks = StepHooks(scheduler=sched, ema=ema, checkpointer=ckpt, max_grad_norm=1.0)
        for step, batch in enumerate(loader):
            hooks.before_step(optimizer, step)
            loss = ...; loss.backward()
            gnorm = hooks.after_backward(model)
            optimizer.step()
            hooks.after_step(model, optimizer, step, stats={"loss": float(loss.item()), "gnorm": gnorm})
    """
    def __init__(
        self,
        *,
        scheduler: Optional[CosineWithWarmup] = None,
        ema: Optional[ParamsEMA] = None,
        checkpointer: Optional[Checkpointer] = None,
        max_grad_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
        save_every: int = 0,
        logger: Optional[logging.Logger] = None,
        bkm_threshold: Optional[float] = None,
        bkm_cooldown: int = 0
    ):
        self.scheduler = scheduler
        self.ema = ema
        self.checkpointer = checkpointer
        self.max_grad_norm = max_grad_norm
        self.grad_clip_value = grad_clip_value
        self.save_every = int(save_every)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.bkm_gate: Optional[BKMCollarGate] = None
        if bkm_threshold is not None:
            self.bkm_gate = BKMCollarGate(threshold=float(bkm_threshold), cooldown=int(bkm_cooldown))

    def before_step(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        """Apply scheduler (if any). Returns lr (or -1.0 if none)."""
        if self.scheduler is None:
            return -1.0
        lr = self.scheduler.step(optimizer, step)
        return lr

    def after_backward(self, model: nn.Module) -> float:
        """Clip grads (if configured) and return pre-clip grad norm."""
        return clip_gradients(model, max_norm=self.max_grad_norm, clip_value=self.grad_clip_value)

    def should_activate_krylov(self, bkm_value: float) -> bool:
        """Return True if ΔBKM exceeds the threshold (one-shot, non-accumulating)."""
        if self.bkm_gate is None:
            return False
        return self.bkm_gate.update(bkm_value)

    def after_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        *,
        stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update EMA, maybe checkpoint, and log."""
        if self.ema is not None:
            self.ema.update(model)

        if self.checkpointer is not None and self.save_every > 0 and (step % self.save_every == 0) and step > 0:
            extra = stats or {}
            path = self.checkpointer.save(step, model, optimizer, scheduler=self.scheduler, ema=self.ema, extra_stats=extra)
            self.logger.info(f"[ckpt] saved → {path}")

        if stats:
            # Optional ΔBKM gate driven by provided stats (non-accumulating trigger)
            if self.bkm_gate is not None and ("bkm" in stats):
                try:
                    if self.bkm_gate.update(stats["bkm"]):
                        self.logger.info(f"[bkm] trigger=1 (ΔBKM {stats['bkm']:.6g} > {self.bkm_gate.threshold:.6g})")
                except Exception:
                    # Be robust to any type/format issues in user-provided stats
                    pass
            log_stats(step, stats, logger=self.logger)


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    """
    Minimal smoke test: fit a toy linear regression with the hooks turned on.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Toy data: y = X w + noise
    N, Din, Dout = 256, 8, 8
    X = torch.randn(N, Din, device=device)
    true_W = torch.randn(Din, Dout, device=device)
    y = X @ true_W + 0.05 * torch.randn(N, Dout, device=device)

    model = nn.Sequential(nn.Linear(Din, 32), nn.ReLU(), nn.Linear(32, Dout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Hooks: cosine lr, EMA, ckpt, grad clip
    sched = CosineWithWarmup(CosineWarmupConfig(base_lr=3e-3, warmup_steps=50, total_steps=400, min_lr=3e-4))
    ema = ParamsEMA(model, decay=0.99)
    ckpt = Checkpointer("./_tmp_ckpts", keep_last=2)
    hooks = StepHooks(scheduler=sched, ema=ema, checkpointer=ckpt, max_grad_norm=1.0, save_every=200)

    # Train
    B = 32
    steps = 400
    for step in range(steps):
        idx = torch.randint(0, N, (B,), device=device)
        xb, yb = X[idx], y[idx]

        hooks.before_step(opt, step)

        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = torch.nn.functional.mse_loss(pred, yb)
        loss.backward()

        gnorm = hooks.after_backward(model)
        opt.step()

        hooks.after_step(model, opt, step, stats={"loss": float(loss.item()), "gnorm": gnorm})

    # Quick EMA check (should be close in MSE to true W mapping)
    with torch.no_grad():
        ema.apply_shadow(model)
        mse = torch.nn.functional.mse_loss(model(X), y).item()
        ema.restore(model)
    logging.info(f"[EMA] full-data MSE after training (EMA-weights): {mse:.4e}")
    print("[hooks] All good ✓")

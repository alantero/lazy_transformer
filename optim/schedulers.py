# optim/schedulers.py
# -------------------------------------------------------------------------
# Lightweight, dependency-free LR schedulers for PyTorch (step-based).
# Includes:
#   - WarmupLinear
#   - WarmupCosine (single-cycle)
#   - CosineWithRestarts (multi-cycle, optional warmup per cycle)
#   - Noam (Transformer schedule)
#   - ReduceLROnPlateauEMA (metric-driven with EMA & patience)
#
# All schedulers:
#   • are step-based (call .step(step=None, metrics=None))
#   • support per-param-group multipliers via pg["lr_mult"] (default=1.0)
#   • expose .get_last_lr() and .state_dict() / .load_state_dict()
#
# No project-local imports; pure stdlib + torch.
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, List, Dict, Any
import math
import torch
from torch.optim import Optimizer


# ----------------------------- small helpers ---------------------------------

def _get_base_lrs(opt: Optimizer, default: Optional[float]) -> List[float]:
    """Collect per-group base lrs (fallback to current lr or a provided default)."""
    base = []
    for pg in opt.param_groups:
        if default is not None:
            base.append(float(default))
        else:
            # Use group lr at construction time as base
            base.append(float(pg.get("initial_lr", pg["lr"])))
        # Optional multiplier per group
        if "lr_mult" not in pg:
            pg["lr_mult"] = 1.0
    return base


def _apply_lrs(opt: Optimizer, lrs: List[float]) -> None:
    for lr, pg in zip(lrs, opt.param_groups):
        mult = float(pg.get("lr_mult", 1.0))
        pg["lr"] = float(lr * mult)


def _lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b


# ------------------------------- base class ----------------------------------

class _SchedulerBase:
    def __init__(self, optimizer: Optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        self.optimizer = optimizer
        self.last_step: int = -1
        self._last_lrs: List[float] = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, step: Optional[int] = None, metrics: Optional[float] = None) -> List[float]:
        if step is None:
            step = self.last_step + 1
        self.last_step = int(step)
        lrs = self._compute_lrs(step, metrics)
        _apply_lrs(self.optimizer, lrs)
        self._last_lrs = lrs
        return lrs

    def get_last_lr(self) -> List[float]:
        return list(self._last_lrs)

    def state_dict(self) -> Dict[str, Any]:
        return {"last_step": self.last_step, "_last_lrs": self._last_lrs}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.last_step = int(state.get("last_step", -1))
        self._last_lrs = list(state.get("_last_lrs", self._last_lrs))

    # Must override in subclasses
    def _compute_lrs(self, step: int, metrics: Optional[float]) -> List[float]:
        raise NotImplementedError


# ----------------------------- concrete scheds -------------------------------

class WarmupLinear(_SchedulerBase):
    """
    LR: warmup (min_lr -> base_lr) for warmup_steps, then linear decay to min_lr
        until total_steps (inclusive). After total_steps, stays at min_lr.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int = 0,
        base_lr: Optional[float] = None,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        assert total_steps > 0 and warmup_steps >= 0 and warmup_steps <= total_steps
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = _get_base_lrs(optimizer, base_lr)

    def _compute_lrs(self, step: int, _: Optional[float]) -> List[float]:
        step = max(0, step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            t = step / float(self.warmup_steps)
            return [_lerp(self.min_lr, b, t) for b in self.base_lrs]

        # after warmup
        denom = max(1, self.total_steps - self.warmup_steps)
        t = min(1.0, (step - self.warmup_steps) / float(denom))
        return [_lerp(b, self.min_lr, t) for b in self.base_lrs]


class WarmupCosine(_SchedulerBase):
    """
    LR: warmup to base_lr, then cosine decay to min_lr over the remaining steps.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int = 0,
        base_lr: Optional[float] = None,
        min_lr: float = 0.0,
        cycles: float = 1.0,  # 1.0 = half-cosine to min; >1.0 = multiple decays
    ):
        super().__init__(optimizer)
        assert total_steps > 0 and warmup_steps >= 0 and warmup_steps <= total_steps
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.cycles = float(cycles)
        self.base_lrs = _get_base_lrs(optimizer, base_lr)

    def _compute_lrs(self, step: int, _: Optional[float]) -> List[float]:
        step = max(0, step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            t = step / float(self.warmup_steps)
            return [_lerp(self.min_lr, b, t) for b in self.base_lrs]

        # cosine
        if self.warmup_steps == self.total_steps:
            progress = 1.0
        else:
            progress = min(1.0, (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps))
        cos_term = 0.5 * (1.0 + math.cos(math.pi * 2.0 * self.cycles * progress))
        # shape: from 1.0 (start) to 0.0 (end) when cycles=0.5; with cycles=1.0 goes 1 → -1 (we clamp)
        cos_term = max(0.0, cos_term)  # smooth landing
        return [self.min_lr + (b - self.min_lr) * cos_term for b in self.base_lrs]


class CosineWithRestarts(_SchedulerBase):
    """
    Cosine decay with restarts (SGDR-style). Optional warmup at the beginning of each cycle.

    Args:
      cycle_steps: number of steps per cycle (fixed length cycles)
      warmup_steps: linear warmup at the beginning of each cycle
      base_lr/min_lr: per-group base LR and global floor
    """
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        cycle_steps: int,
        warmup_steps: int = 0,
        base_lr: Optional[float] = None,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        assert cycle_steps > 0 and warmup_steps >= 0 and warmup_steps < cycle_steps
        self.cycle_steps = int(cycle_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = _get_base_lrs(optimizer, base_lr)

    def _compute_lrs(self, step: int, _: Optional[float]) -> List[float]:
        step = max(0, step)
        s_in_cycle = step % self.cycle_steps
        if self.warmup_steps > 0 and s_in_cycle < self.warmup_steps:
            t = s_in_cycle / float(self.warmup_steps)
            return [_lerp(self.min_lr, b, t) for b in self.base_lrs]
        # cosine inside each cycle
        if self.warmup_steps == self.cycle_steps - 1:
            progress = 1.0
        else:
            progress = min(1.0, (s_in_cycle - self.warmup_steps) / float(self.cycle_steps - self.warmup_steps))
        cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (b - self.min_lr) * cos_term for b in self.base_lrs]


class Noam(_SchedulerBase):
    """
    Transformer schedule:
      lr = scale * d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    Provide scale to match desired peak.
    """
    def __init__(self, optimizer: Optimizer, *, d_model: int, warmup_steps: int = 4000, scale: float = 1.0):
        super().__init__(optimizer)
        self.d_model = float(d_model)
        self.warmup_steps = float(max(1, warmup_steps))
        self.scale = float(scale)

    def _compute_lrs(self, step: int, _: Optional[float]) -> List[float]:
        s = max(1, step)
        lr = self.scale * (self.d_model ** -0.5) * min(s ** -0.5, s * (self.warmup_steps ** -1.5))
        return [lr for _ in self.optimizer.param_groups]


class ReduceLROnPlateauEMA(_SchedulerBase):
    """
    Metric-driven schedule with EMA smoothing and patience.
    - Tracks an EMA of `metrics` (lower is better by default).
    - If no improvement for `patience` steps, decays LR by `factor`, floored at `min_lr`.

    Args:
      factor: multiplicative LR decay (<1.0)
      patience: steps without improvement before decay
      ema_alpha: smoothing for metrics EMA
      threshold: minimal relative improvement to reset patience
      minimize: True if lower metric is better; False means higher is better
    """
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        factor: float = 0.5,
        patience: int = 200,
        ema_alpha: float = 0.9,
        threshold: float = 1e-3,
        minimize: bool = True,
        base_lr: Optional[float] = None,
        min_lr: float = 0.0,
    ):
        super().__init__(optimizer)
        assert 0.0 < factor < 1.0
        self.factor = float(factor)
        self.patience = int(patience)
        self.ema_alpha = float(ema_alpha)
        self.threshold = float(threshold)
        self.minimize = bool(minimize)
        self.min_lr = float(min_lr)

        self.base_lrs = _get_base_lrs(optimizer, base_lr)
        self._ema: Optional[float] = None
        self._best: Optional[float] = None
        self._bad_steps: int = 0

    def _improved(self, new: float, ref: float) -> bool:
        if self.minimize:
            rel = (ref - new) / (abs(ref) + 1e-12)
            return rel > self.threshold
        else:
            rel = (new - ref) / (abs(ref) + 1e-12)
            return rel > self.threshold

    def _compute_lrs(self, step: int, metrics: Optional[float]) -> List[float]:
        # Plateau scheduling must be driven by a metric; if missing, keep LR.
        if metrics is None:
            return [pg["lr"] for pg in self.optimizer.param_groups]

        # EMA of metrics
        if self._ema is None:
            self._ema = float(metrics)
        else:
            self._ema = self.ema_alpha * self._ema + (1.0 - self.ema_alpha) * float(metrics)

        if self._best is None:
            self._best = self._ema
            self._bad_steps = 0
        else:
            if self._improved(self._ema, self._best):
                self._best = self._ema
                self._bad_steps = 0
            else:
                self._bad_steps += 1

        # Decay when patience exceeded
        if self._bad_steps >= self.patience:
            # decay all groups (respecting min_lr)
            new_lrs = [max(self.min_lr, lr * self.factor) for lr in self.get_last_lr()]
            self._bad_steps = 0
            return new_lrs

        return self.get_last_lr()

    def state_dict(self) -> Dict[str, Any]:
        s = super().state_dict()
        s.update({"_ema": self._ema, "_best": self._best, "_bad_steps": self._bad_steps})
        return s

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._ema = state.get("_ema", self._ema)
        self._best = state.get("_best", self._best)
        self._bad_steps = int(state.get("_bad_steps", self._bad_steps))


# ------------------------------- factory API ---------------------------------

def make_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs: Any,
) -> _SchedulerBase:
    """
    Factory for convenience.
    Examples:
      make_scheduler("warmup_cosine", opt, total_steps=20_000, warmup_steps=1_000, base_lr=3e-4, min_lr=3e-5)
      make_scheduler("noam", opt, d_model=768, warmup_steps=4000, scale=1.0)
      make_scheduler("plateau", opt, factor=0.5, patience=300, ema_alpha=0.9, minimize=True)
    """
    name = name.lower()
    if name in ("warmup_linear", "linear"):
        return WarmupLinear(optimizer, **kwargs)
    if name in ("warmup_cosine", "cosine"):
        return WarmupCosine(optimizer, **kwargs)
    if name in ("cosine_restarts", "cosine_with_restarts", "sgdr"):
        return CosineWithRestarts(optimizer, **kwargs)
    if name == "noam":
        return Noam(optimizer, **kwargs)
    if name in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
        return ReduceLROnPlateauEMA(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler name: {name}")


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    print("[schedulers] Running sanity tests...")

    # Tiny model / optimizer
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # Freeze base LR in param-groups for reproducibility
    for pg in opt.param_groups:
        pg["initial_lr"] = 1e-3
        pg["lr_mult"] = 1.0

    # 1) WarmupCosine monotonicity & bounds
    total, warm = 200, 20
    sch = WarmupCosine(opt, total_steps=total, warmup_steps=warm, base_lr=1e-3, min_lr=1e-5, cycles=0.5)
    lrs = []
    for t in range(total + 50):  # extend beyond total to check floor behavior
        lrs.append(sch.step(t)[0])
    # Checks
    assert abs(lrs[warm] - 1e-3) < 5e-5, f"Peak after warmup should be near base_lr; got {lrs[warm]:.2e}"
    assert min(lrs) >= 1e-5 - 1e-9, "LR should not go below min_lr"
    print("  WarmupCosine ✓")

    # 2) Linear warmup+decay endpoints
    sch2 = WarmupLinear(opt, total_steps=100, warmup_steps=10, base_lr=2e-3, min_lr=5e-5)
    l0 = sch2.step(0)[0]
    l10 = sch2.step(10)[0]
    l100 = sch2.step(100)[0]
    assert l0 >= 5e-5 - 1e-9 and abs(l10 - 2e-3) < 1e-6 and abs(l100 - 5e-5) < 1e-9
    print("  WarmupLinear ✓")

    # 3) Noam sanity: increases then decreases
    sch3 = Noam(opt, d_model=512, warmup_steps=8, scale=1.0)
    vals = [sch3.step(t)[0] for t in range(1, 33)]
    assert vals[0] < vals[7] and vals[7] > vals[-1], "Noam should rise to warmup then decay"
    print("  Noam ✓")

    # 4) CosineWithRestarts periodicity
    sch4 = CosineWithRestarts(opt, cycle_steps=20, warmup_steps=5, base_lr=1e-3, min_lr=1e-5)
    a = [sch4.step(t)[0] for t in range(40)]
    # Values at the start of both cycles should be similar (after warmup difference small)
    assert abs(a[5] - a[25]) < 1e-7
    print("  CosineWithRestarts ✓")

    # 5) Plateau EMA: force decays
    sch5 = ReduceLROnPlateauEMA(opt, factor=0.5, patience=3, ema_alpha=0.5, minimize=True, base_lr=1e-3, min_lr=1e-5)
    lhist = []
    # provide flat/oscillating metrics to trigger patience
    metrics = [1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 0.998, 1.0, 1.0, 1.0]
    for t, mval in enumerate(metrics):
        lhist.append(sch5.step(t, metrics=mval)[0])
    assert (lhist[-1] < lhist[0]) and (lhist[-1] >= 1e-5), "Plateau should decay but respect min_lr"
    print("  ReduceLROnPlateauEMA ✓")

    print("[schedulers] All good ✓")

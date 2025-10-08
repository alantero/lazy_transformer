# utils/ema.py
# Exponential Moving Average (EMA) utilities.
# - Standalone (no repo-local deps).
# - Works with Python floats and (optionally) PyTorch tensors.
# - Optional bias-correction (debiased estimate), reset/save/load helpers.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Optional PyTorch backend
try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAVE_TORCH = False


def _is_tensor(x: Any) -> bool:
    return _HAVE_TORCH and isinstance(x, torch.Tensor)  # type: ignore[truthy-bool]


def ema_update(prev: Any, x: Any, alpha: float) -> Any:
    """
    Stateless EMA update:
        y_t = alpha * y_{t-1} + (1 - alpha) * x_t
    If prev is None, initializes from zero: returns (1 - alpha) * x (copied for tensors).

    Args:
        prev: previous EMA value (float or torch.Tensor or None)
        x: new sample (same type as prev)
        alpha: decay in [0, 1). Larger -> smoother.

    Returns:
        Updated EMA value (same type as x).
    """
    if not (0.0 <= float(alpha) < 1.0):
        raise ValueError(f"alpha must be in [0,1), got {alpha}.")
    if prev is None:
        # Initialize from zero so that debiasing (division by 1 - alpha^t) is correct.
        if _is_tensor(x):
            with torch.no_grad():  # type: ignore[attr-defined]
                x0 = x.detach().clone()
                return x0.mul(1.0 - float(alpha))
        return (1.0 - float(alpha)) * float(x)
    if _is_tensor(prev) or _is_tensor(x):
        if not _is_tensor(prev) or not _is_tensor(x):
            raise TypeError("Mixed types for EMA update (tensor vs non-tensor).")
        # compute without building a graph
        with torch.no_grad():  # type: ignore[attr-defined]
            # Move x to prev's dtype/device if needed (safe broadcast)
            x_cast = x.to(device=prev.device, dtype=prev.dtype)  # type: ignore[union-attr]
            out = prev.mul(alpha).add_(x_cast, alpha=(1.0 - alpha))
        return out
    # Python floats
    return float(alpha) * float(prev) + (1.0 - float(alpha)) * float(x)


@dataclass
class EMAConfig:
    alpha: float = 0.9          # decay in [0,1)
    debias: bool = True         # return debiased estimate if requested
    clip: Optional[Tuple[float, float]] = None  # optional clamp on inputs before update


class EMA:
    """
    Stateful EMA tracker.

    Typical usage:
        ema = EMA(alpha=0.99, debias=True)
        for x in stream:
            ema.update(x)
            y = ema.value()           # debiased by default if debias=True

    Notes:
    - For tensors, updates are done under no_grad and values are kept detached.
    - Debiasing: returns y_t / (1 - alpha^t) so the expectation matches the sample mean
      when initialized from zero / None. For constant signals, debiased → true value faster.
    """

    def __init__(self, alpha: float = 0.9, debias: bool = True, clip: Optional[Tuple[float, float]] = None):
        if not (0.0 <= float(alpha) < 1.0):
            raise ValueError(f"alpha must be in [0,1), got {alpha}.")
        self.cfg = EMAConfig(alpha=float(alpha), debias=bool(debias), clip=clip)
        self._t: int = 0
        self._value: Optional[Any] = None  # float or torch.Tensor

    # --------------------------- core API ---------------------------

    def reset(self, init: Optional[Any] = None) -> None:
        """Reset the EMA (optionally set an initial value)."""
        self._t = 0
        if init is None:
            self._value = None
        else:
            self._value = init.detach().clone() if _is_tensor(init) else float(init)

    def update(self, x: Any) -> Any:
        """
        Incorporate a new sample x into the EMA.
        x can be a float or a torch.Tensor. If clip=(a,b) was set, clamp x before update.
        Returns the (possibly debiased) current EMA.
        """
        if self.cfg.clip is not None:
            lo, hi = self.cfg.clip
            if _is_tensor(x):
                x = x.clamp(min=lo, max=hi)
            else:
                x = float(min(max(float(x), lo), hi))

        self._value = ema_update(self._value, x, self.cfg.alpha)
        self._t += 1
        return self.value()  # return current estimate

    def value(self, *, debiased: Optional[bool] = None) -> Any:
        """
        Get the current EMA value. If debiased is None, uses self.cfg.debias.
        For tensors, returns a detached tensor (no grad).
        """
        if self._value is None:
            return None
        use_debias = self.cfg.debias if debiased is None else bool(debiased)
        if not use_debias:
            return self._value.detach().clone() if _is_tensor(self._value) else float(self._value)
        # bias correction factor: 1 - alpha^t
        denom = 1.0 - (self.cfg.alpha ** max(self._t, 1))
        denom = max(denom, 1e-12)
        if _is_tensor(self._value):
            with torch.no_grad():  # type: ignore[attr-defined]
                return (self._value / denom).detach().clone()
        return float(self._value) / denom

    # --------------------------- utilities --------------------------

    def bias_correction(self) -> float:
        """Return the scalar bias-correction factor 1 - alpha^t."""
        return 1.0 - (self.cfg.alpha ** max(self._t, 1))

    def to(self, *, device: Optional[str] = None, dtype: Optional[Any] = None) -> "EMA":
        """
        Move tensor value to a given device/dtype (no-op for floats).
        Returns self for chaining.
        """
        if _is_tensor(self._value):
            with torch.no_grad():  # type: ignore[attr-defined]
                if device is not None:
                    self._value = self._value.to(device=device)
                if dtype is not None:
                    self._value = self._value.to(dtype=dtype)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Serialize EMA state."""
        state: Dict[str, Any] = {
            "alpha": self.cfg.alpha,
            "debias": self.cfg.debias,
            "clip": self.cfg.clip,
            "t": int(self._t),
        }
        if _is_tensor(self._value):
            state["is_tensor"] = True
            # Save a CPU tensor to keep checkpoints device-agnostic
            state["value"] = self._value.detach().cpu()
            state["dtype"] = str(self._value.dtype)
        else:
            state["is_tensor"] = False
            state["value"] = None if self._value is None else float(self._value)
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore EMA state saved by state_dict()."""
        self.cfg.alpha = float(state.get("alpha", self.cfg.alpha))
        self.cfg.debias = bool(state.get("debias", self.cfg.debias))
        self.cfg.clip = state.get("clip", self.cfg.clip)
        self._t = int(state.get("t", 0))
        is_tensor = bool(state.get("is_tensor", False))
        val = state.get("value", None)
        if is_tensor and _HAVE_TORCH:
            self._value = val.clone().detach() if val is not None else None
        else:
            self._value = None if val is None else float(val)

    def __repr__(self) -> str:  # pragma: no cover
        typ = "tensor" if _is_tensor(self._value) else "float"
        return f"EMA(alpha={self.cfg.alpha}, debias={self.cfg.debias}, t={self._t}, type={typ})"


# ---------------------------------- __main__ ----------------------------------
# Tiny sanity checks (run: `python utils/ema.py`)
if __name__ == "__main__":
    print("[ema] Running sanity tests...")

    # 1) Float sequence (constant -> EMA should approach the constant; debiased == constant exactly)
    ema_f = EMA(alpha=0.9, debias=True)
    for _ in range(20):
        ema_f.update(1.0)
    val_f = ema_f.value()  # debiased
    assert abs(val_f - 1.0) < 1e-6, f"Float debiased EMA should be ~1.0, got {val_f:.6f}"

    # 2) Float sequence (no debias -> close to constant but with small bias)
    ema_f2 = EMA(alpha=0.9, debias=False)
    for _ in range(5):
        ema_f2.update(1.0)
    val_f2 = ema_f2.value()
    expected = 1.0 - (0.9 ** 5)
    assert abs(val_f2 - expected) < 1e-6, f"Undebiased EMA after 5 steps should be ~{expected:.6f}, got {val_f2:.6f}"

    # 3) Tensor sequence (random), compare with manual EMA
    if _HAVE_TORCH:
        torch.manual_seed(0)
        x = torch.randn(8, 4)  # single tensor sample (we'll feed multiple times)
        ema_t = EMA(alpha=0.8, debias=False)
        ref = torch.zeros_like(x)
        for _ in range(10):
            ema_t.update(x)
            ref = 0.8 * ref + 0.2 * x
        err = float((ema_t.value() - ref).abs().max())
        assert err < 1e-6, f"Torch EMA mismatch: {err:.3e}"

        # Debias check on tensors: constant ones → exactly ones when debiased
        ones = torch.ones(3, 2)
        ema_ones = EMA(alpha=0.95, debias=True)
        for _ in range(25):
            ema_ones.update(ones)
        val_ones = ema_ones.value()
        err2 = float((val_ones - 1.0).abs().max())
        assert err2 < 1e-6, f"Debiased tensor EMA should be ~1, got max|err|={err2:.3e}"

    print("[ema] All good ✓")

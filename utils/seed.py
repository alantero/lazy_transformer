# utils/seed.py
# -----------------------------------------------------------------------------
# LazyCLO v3 — Reproducibility & seeding utilities for Python/NumPy/Torch.
# -----------------------------------------------------------------------------
# Features:
#   - seed_everything(seed, deterministic=True): sets seeds for random, numpy,
#     torch (CPU/CUDA), and toggles deterministic flags (when available).
#   - worker_init_fn: DataLoader-compatible worker seeding.
#   - rank-aware: offsets per (rank, worker_id) for DDP safety.
#   - Resilient demo: works even if NumPy or Torch aren't installed.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import random
import time
from typing import Optional, Callable, Any

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False


# ------------------------------ Rank helpers ---------------------------------

def get_rank() -> int:
    for k in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                continue
    return 0

def get_world_size() -> int:
    for k in ("WORLD_SIZE",):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                continue
    return 1


# ------------------------------ Seed helpers ---------------------------------

def _int32(x: int) -> int:
    return int(x & 0xFFFFFFFF)

def choose_seed(seed: Optional[int] = None) -> int:
    """Pick a non-zero 32-bit seed. If None, derive from time/pid/rank."""
    if seed is None or seed == 0:
        seed = int(time.time_ns() ^ os.getpid() ^ (get_rank() << 16))
    seed = _int32(seed)
    return seed if seed != 0 else 42

def seed_everything(seed: Optional[int] = 1337, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and Torch (CPU/CUDA). Returns the resolved seed."""
    s = choose_seed(seed)
    random.seed(s)

    if np is not None:
        try:
            np.random.seed(s)
        except Exception:
            pass

    if _HAVE_TORCH:
        try:
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass
        # Determinism knobs (guard existence)
        try:
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = bool(deterministic)
                torch.backends.cudnn.benchmark = not bool(deterministic)
        except Exception:
            pass
        try:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(bool(deterministic))
        except Exception:
            # Some ops may not support full deterministic mode; ignore.
            pass
    return s

def reset_seed_with_offset(base_seed: int, offset: int) -> int:
    """Reset global RNGs using base_seed ⊕ offset (handy for epochs/steps)."""
    return seed_everything(_int32(base_seed ^ offset))

def worker_init_fn(seed: int) -> Callable[[int], None]:
    """
    Return a function suitable for DataLoader(worker_init_fn=...).
    Each worker receives a distinct seed derived from (seed, rank, worker_id).
    """
    base = choose_seed(seed)
    rank = get_rank()

    def _init(worker_id: int) -> None:
        s = _int32(base ^ (rank << 8) ^ worker_id)
        random.seed(s)
        if np is not None:
            try:
                np.random.seed(s)
            except Exception:
                pass
        if _HAVE_TORCH:
            try:
                torch.manual_seed(s)
            except Exception:
                pass
    return _init


# ------------------------------ Demo (__main__) ------------------------------

def _almost_equal_lists(u: Any, v: Any, tol: float = 1e-9) -> bool:
    """
    Tolerant equality:
      - If both are numeric lists -> compare with tolerance.
      - Otherwise -> fallback to plain equality.
    """
    if isinstance(u, list) and isinstance(v, list):
        if len(u) != len(v):
            return False
        def _is_num(x: Any) -> bool:
            return isinstance(x, (int, float))
        if all(_is_num(x) and _is_num(y) for x, y in zip(u, v)):
            return all(abs(float(x) - float(y)) < tol for x, y in zip(u, v))
        return u == v
    return u == v


if __name__ == "__main__":
    # Simple reproducibility smoke test that survives missing deps.
    s = seed_everything(12345, deterministic=True)
    print(f"[Seed] Global seed set to: {s}")

    # Python random
    a = [random.random() for _ in range(3)]

    # NumPy
    if np is not None:
        b = np.random.randn(3).tolist()
    else:
        b = ["numpy-not-installed"]

    # Torch
    if _HAVE_TORCH:
        try:
            x_cpu = [float(v) for v in torch.randn(3).tolist()]
        except Exception:
            x_cpu = ["torch-cpu-error"]
        x_cuda: list = []
        if _HAVE_TORCH and getattr(torch.cuda, "is_available", lambda: False)():
            try:
                x_cuda = [float(v) for v in torch.randn(3, device="cuda").cpu().tolist()]
            except Exception:
                x_cuda = ["torch-cuda-error"]
    else:
        x_cpu, x_cuda = ["torch-not-installed"], []

    # Re-seed and re-sample (should match)
    seed_everything(12345, deterministic=True)
    a2 = [random.random() for _ in range(3)]
    if np is not None:
        b2 = np.random.randn(3).tolist()
    else:
        b2 = ["numpy-not-installed"]
    if _HAVE_TORCH:
        try:
            x_cpu2 = [float(v) for v in torch.randn(3).tolist()]
        except Exception:
            x_cpu2 = ["torch-cpu-error"]
        x_cuda2: list = []
        if _HAVE_TORCH and getattr(torch.cuda, "is_available", lambda: False)():
            try:
                x_cuda2 = [float(v) for v in torch.randn(3, device="cuda").cpu().tolist()]
            except Exception:
                x_cuda2 = ["torch-cuda-error"]
    else:
        x_cpu2, x_cuda2 = ["torch-not-installed"], []

    print(f"Python random reproducible: {_almost_equal_lists(a, a2)}")
    print(f"NumPy random reproducible:  {_almost_equal_lists(b, b2)}")
    print(f"Torch CPU reproducible:    {_almost_equal_lists(x_cpu, x_cpu2)}")
    if x_cuda and x_cuda2:
        print(f"Torch CUDA reproducible:   {_almost_equal_lists(x_cuda, x_cuda2)}")
    else:
        print("Torch CUDA reproducible:   (CUDA not available or Torch missing)")

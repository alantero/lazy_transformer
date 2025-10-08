# utils/profile.py
# ---------------------------------------------------------------------
# Small, dependency-light profiling helpers for training/inference:
#   - Timer / timeit(): precise wall-clock timing (with CUDA sync).
#   - NVTX/record_function ranges (safe no-op if unavailable).
#   - ThroughputMeter: items-per-second tracking.
#   - Parameter/memory helpers: count_parameters(), gpu_mem(), PeakGPU.
#   - Quick micro-bench harness for callables.
#
# Pure stdlib + optional torch/psutil. No project-local imports.
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List
import time
import statistics
import contextlib
import os

# Optional deps
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAVE_TORCH = False

try:
    import psutil  # for CPU memory (RSS)
    _HAVE_PSUTIL = True
except Exception:
    psutil = None  # type: ignore
    _HAVE_PSUTIL = False


# ------------------------------- utilities ------------------------------------

def _sync_cuda() -> None:
    if _HAVE_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Simple wall-clock timer (perf_counter)."""
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = bool(sync_cuda)
        self._t0: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        if self.sync_cuda:
            _sync_cuda()
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        if self.sync_cuda:
            _sync_cuda()
        dt = time.perf_counter() - self._t0
        self._elapsed += dt
        self._t0 = None
        return dt

    def reset(self) -> None:
        self._t0 = None
        self._elapsed = 0.0

    @property
    def elapsed(self) -> float:
        """Accumulated seconds."""
        return float(self._elapsed)

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


@contextlib.contextmanager
def nvtx_range(name: str, color: Optional[int] = None):
    """
    NVTX range (visible in Nsight Systems/Compute). No-op if unavailable.
    """
    pushed = False
    if _HAVE_TORCH and torch.cuda.is_available():
        nvtx = getattr(torch.cuda, "nvtx", None)
        if nvtx is not None and hasattr(nvtx, "range_push"):
            try:
                if color is None:
                    nvtx.range_push(name)
                else:
                    nvtx.range_push(name, color=color)  # type: ignore[arg-type]
                pushed = True
            except TypeError:
                nvtx.range_push(name)  # older versions
                pushed = True
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()  # type: ignore[attr-defined]
            except Exception:
                pass


@contextlib.contextmanager
def record_function(name: str):
    """
    Autograd profiler record_function (shows in torch profiler). No-op if unavailable.
    """
    if _HAVE_TORCH and hasattr(torch.autograd.profiler, "record_function"):
        with torch.autograd.profiler.record_function(name):  # type: ignore[attr-defined]
            yield
    else:
        yield


class ThroughputMeter:
    """Accumulate items and wall time to get items/sec."""
    def __init__(self) -> None:
        self.items: int = 0
        self.time_s: float = 0.0

    def update(self, n_items: int, dt_s: float) -> None:
        self.items += int(n_items)
        self.time_s += float(dt_s)

    @property
    def ips(self) -> float:
        return (self.items / self.time_s) if self.time_s > 0 else 0.0

    def reset(self) -> None:
        self.items = 0
        self.time_s = 0.0


def count_parameters(module: Any, trainable_only: bool = True) -> int:
    """Count parameters of a torch.nn.Module (0 if torch not available)."""
    if not _HAVE_TORCH:
        return 0
    if not hasattr(module, "parameters"):
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def gpu_mem(device: Optional[Any] = None) -> Dict[str, float]:
    """
    Return GPU memory stats (MB). Keys: allocated, reserved, peak_allocated, peak_reserved.
    """
    if not (_HAVE_TORCH and torch.cuda.is_available()):
        return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0, "peak_reserved": 0.0}
    dev = device if device is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev) / (1024**2)
    reserved = torch.cuda.memory_reserved(dev) / (1024**2)
    peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
    peak_res = torch.cuda.max_memory_reserved(dev) / (1024**2)
    return {
        "allocated": float(allocated),
        "reserved": float(reserved),
        "peak_allocated": float(peak_alloc),
        "peak_reserved": float(peak_res),
    }


@contextlib.contextmanager
def PeakGPU(device: Optional[Any] = None, reset_before: bool = True):
    """
    Context to measure peak GPU memory usage inside the block.
    Usage:
        with PeakGPU() as peak_mb:
            ... work ...
        print(peak_mb["peak_allocated"], peak_mb["peak_reserved"])
    """
    stats = {"peak_allocated": 0.0, "peak_reserved": 0.0}
    if not (_HAVE_TORCH and torch.cuda.is_available()):
        yield stats
        return
    dev = device if device is not None else torch.cuda.current_device()
    if reset_before:
        torch.cuda.reset_peak_memory_stats(dev)
    try:
        yield stats
    finally:
        s = gpu_mem(dev)
        stats.update({"peak_allocated": s["peak_allocated"], "peak_reserved": s["peak_reserved"]})


def cpu_mem_mb() -> float:
    """Return current process RSS in MB (requires psutil; else 0)."""
    if not _HAVE_PSUTIL:
        return 0.0
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss) / (1024**2)


# ------------------------------- timing API -----------------------------------

def timeit(
    fn: Callable[..., Any],
    *args: Any,
    iters: int = 20,
    warmup: int = 5,
    sync_cuda: bool = True,
    items_per_call: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Micro-benchmark a callable.
    Returns dict with mean/p50/p90/p99 (ms) and items/sec if items_per_call is given.
    """
    # warmup
    for _ in range(max(0, warmup)):
        _ = fn(*args, **kwargs)
    times: List[float] = []
    for _ in range(max(1, iters)):
        if sync_cuda:
            _sync_cuda()
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        if sync_cuda:
            _sync_cuda()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    mean_ms = statistics.fmean(times)
    p50 = statistics.median(times)
    p90 = _percentile(times, 0.90)
    p99 = _percentile(times, 0.99)
    out = {
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
    }
    if items_per_call:
        total_items = items_per_call * iters
        total_time_s = sum(times) / 1000.0
        out["items_per_s"] = total_items / total_time_s if total_time_s > 0 else 0.0
    return out


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = min(len(xs_sorted) - 1, max(0, int(round(q * (len(xs_sorted) - 1)))))
    return float(xs_sorted[idx])


# ------------------------------- bench harness --------------------------------

def benchmark_step(
    step_fn: Callable[[Any], Any],
    batch: Any,
    *,
    warmup: int = 3,
    iters: int = 10,
    items_in_batch: Optional[int] = None,
    sync_cuda: bool = True,
) -> Dict[str, float]:
    """
    Benchmark a typical training/inference step `step_fn(batch)`.
    Example:
        stats = benchmark_step(lambda b: model(b), batch, items_in_batch=b["tokens"].numel())
    """
    # Warmup
    for _ in range(max(0, warmup)):
        step_fn(batch)
    # Timed
    times: List[float] = []
    for _ in range(max(1, iters)):
        if sync_cuda:
            _sync_cuda()
        t0 = time.perf_counter()
        step_fn(batch)
        if sync_cuda:
            _sync_cuda()
        times.append((time.perf_counter() - t0) * 1000.0)
    # Aggregate
    mean_ms = statistics.fmean(times)
    out = {
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(times),
        "p90_ms": _percentile(times, 0.90),
        "p99_ms": _percentile(times, 0.99),
    }
    if items_in_batch is not None:
        total_items = items_in_batch * iters
        total_time_s = sum(times) / 1000.0
        out["items_per_s"] = total_items / total_time_s if total_time_s > 0 else 0.0
    return out


# --------------------------------- __main__ -----------------------------------

if __name__ == "__main__":
    print("[profile] Running sanity tests...")

    # Timer sanity
    with Timer(sync_cuda=False) as t:
        time.sleep(0.05)
    assert 40.0 <= t.elapsed * 1000.0 <= 120.0, f"Timer off? {t.elapsed*1000:.2f} ms"
    print(f"  timer: ~{t.elapsed*1000:.1f} ms ✓")

    # NVTX / record_function no-op safety
    with nvtx_range("demo-range"), record_function("demo-record"):
        pass
    print("  ranges: NVTX/record_function ok ✓")

    # Torch micro-bench (if torch available)
    if _HAVE_TORCH:
        device = "cuda" if (torch.cuda.is_available()) else "cpu"
        x = torch.randn(1024, 1024, device=device)
        w = torch.randn(1024, 1024, device=device)

        def _matmul():
            return x @ w

        stats = timeit(_matmul, iters=10, warmup=3, items_per_call=x.shape[0], sync_cuda=True)
        print(f"  matmul[{device}]: mean={stats['mean_ms']:.2f} ms, p90={stats['p90_ms']:.2f} ms, ips≈{stats.get('items_per_s', 0):.1f}")

        # Memory helpers
        if torch.cuda.is_available():
            with PeakGPU() as peak:
                y = _matmul()
                del y
            m = gpu_mem()
            print(f"  gpu mem: alloc={m['allocated']:.1f}MB, peak_alloc={peak['peak_allocated']:.1f}MB")
        else:
            print(f"  cpu mem: rss≈{cpu_mem_mb():.1f}MB")

        # Param counter
        model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 128))
        n_params = count_parameters(model)
        assert n_params > 0
        print(f"  params: {n_params} ✓")

        # Step bench
        batch = torch.randn(64, 256, device=device)
        step_stats = benchmark_step(lambda b: model(b), batch, warmup=2, iters=8, items_in_batch=batch.shape[0])
        print(f"  step: mean={step_stats['mean_ms']:.2f} ms, ips≈{step_stats.get('items_per_s', 0):.1f} ✓")

    print("[profile] All good ✓")

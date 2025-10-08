# configs/presets.py
# Minimal Python presets for ablations (Phase 12).
# Composable helpers to build LoopConfig / DistillConfig variants.

from __future__ import annotations
from dataclasses import replace
from typing import Callable, Dict, Any, Optional

import math
import torch

# Safe imports from the repo
try:
    from train.loop import LoopConfig
except Exception as e:
    raise ImportError(f"Cannot import LoopConfig from train.loop: {e}")

try:
    from train.distill import DistillConfig
except Exception:
    # Distillation presets are optional; only needed if you run Phase 10.1
    DistillConfig = None  # type: ignore


# ---------------------------- base builders -----------------------------------

def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def base_loop_cfg(
    *,
    vocab_size: int,
    pad_id: int,
    device: Optional[str] = None,
) -> LoopConfig:
    """Reasonable base for small experiments/ablations."""
    dev = device or _default_device()
    return LoopConfig(
        # windows
        W=256, O=32,
        # model dims
        vocab_size=vocab_size, d_model=256, groups=16,
        # spectral bank
        cheb_deg=4, cheb_laplacian="cycle",
        # port-hamiltonian (low-rank)
        skew_rank=16, R_rank=8,
        # integrator
        steps=2, dt=0.5, method="heun",
        # head / embeddings
        tie_softmax=True, factor_rank=64, pos_kind="sinusoidal",
        # misc
        pad_id=pad_id, device=dev,
        # gauge on por defecto (v3)
        use_gauge=True,
        # dual/rate (puedes desactivar en las ablaciones)
        use_dual_rate=True,
        target_bpp=None,      # por defecto 0.6*log2(V)
        dual_lr=1e-2, dual_ema=0.9,
        dual_use_log=True, dual_var_aware=True,
        # quant calibración breve activada
        quant_calibrate_after_prune=True,
        quant_calib_batches=64,
        freeze_qparams_after_calib=True,
    )

def base_distill_cfg(
    *,
    W: int,
    O: int,
    pad_id: int,
    device: Optional[str] = None,
    alpha: float = 0.5,
    temperature: float = 2.0,
    proj_dim: Optional[int] = 64,
) -> DistillConfig:
    """Base KD config (collar-only) — requires train/distill.py."""
    if DistillConfig is None:
        raise ImportError("DistillConfig not available (train/distill.py not importable).")
    dev = device or _default_device()
    return DistillConfig(
        W=W, O=O,
        alpha=alpha, temperature=temperature,
        proj_dim=proj_dim, proj_seed=0, collar_only=True,
        teacher_ckpt=None,
        lr=3e-4, weight_decay=0.0,
        device=dev, pad_id=pad_id,
    )


# ------------------------------ loop presets ----------------------------------

# Banco 3 vs 4 (aquí variamos cheb_deg, que actúa como orden del filtro Chebyshev)
def with_bank3(cfg: LoopConfig) -> LoopConfig:
    return replace(cfg, cheb_deg=3)

def with_bank4(cfg: LoopConfig) -> LoopConfig:
    return replace(cfg, cheb_deg=4)

# r = 1/2/3 (ajuste de rangos en el núcleo port-hamiltoniano)
# Puedes adaptar estas fórmulas si quieres otro escalado.
def with_rank_r1(cfg: LoopConfig) -> LoopConfig:
    return replace(cfg, skew_rank=8, R_rank=4)

def with_rank_r2(cfg: LoopConfig) -> LoopConfig:
    return replace(cfg, skew_rank=16, R_rank=8)

def with_rank_r3(cfg: LoopConfig) -> LoopConfig:
    return replace(cfg, skew_rank=24, R_rank=12)

# Rate on/off y SDA vs EMA (dual)
def with_rate_off(cfg: LoopConfig) -> LoopConfig:
    return replace(cfg, use_dual_rate=False)

def with_rate_on_sda(cfg: LoopConfig) -> LoopConfig:
    return replace(cfg, use_dual_rate=True, dual_use_log=True)  # SDA por defecto en nuestro loop

def with_rate_on_ema(cfg: LoopConfig) -> LoopConfig:
    # “EMA” aquí significa no usar la dinámica SDA (en nuestro loop, equivale a dual apagado
    # o a usar sólo estadísticos EMA sin penalización; lo modelamos como rate_on pero fácil de comparar)
    return replace(cfg, use_dual_rate=True, dual_use_log=True, dual_var_aware=False, dual_lr=5e-3, dual_ema=0.95)


# ---------------------------- distill presets ---------------------------------

def with_lowD32(cfg: DistillConfig) -> DistillConfig:
    return replace(cfg, proj_dim=32)

def with_lowD48(cfg: DistillConfig) -> DistillConfig:
    return replace(cfg, proj_dim=48)

def with_lowD64(cfg: DistillConfig) -> DistillConfig:
    return replace(cfg, proj_dim=64)


# ------------------------------ composition -----------------------------------

def compose_loop(
    *,
    vocab_size: int,
    pad_id: int,
    device: Optional[str] = None,
    transforms: Optional[list[Callable[[LoopConfig], LoopConfig]]] = None,
) -> LoopConfig:
    """base_loop_cfg(...) |> transforms (e.g., [with_bank3, with_rank_r2, with_rate_off])"""
    cfg = base_loop_cfg(vocab_size=vocab_size, pad_id=pad_id, device=device)
    for t in transforms or []:
        cfg = t(cfg)
    return cfg

def compose_distill(
    *,
    W: int,
    O: int,
    pad_id: int,
    device: Optional[str] = None,
    transforms: Optional[list[Callable[[DistillConfig], DistillConfig]]] = None,
) -> DistillConfig:
    """base_distill_cfg(...) |> transforms (e.g., [with_lowD32])"""
    cfg = base_distill_cfg(W=W, O=O, pad_id=pad_id, device=device)
    for t in transforms or []:
        cfg = t(cfg)
    return cfg


# ------------------------------- registries -----------------------------------

PRESET_LOOPS: Dict[str, Callable[..., LoopConfig]] = {
    # banco
    "bank3": lambda **kw: compose_loop(transforms=[with_bank3], **kw),
    "bank4": lambda **kw: compose_loop(transforms=[with_bank4], **kw),
    # rank
    "rank_r1": lambda **kw: compose_loop(transforms=[with_rank_r1], **kw),
    "rank_r2": lambda **kw: compose_loop(transforms=[with_rank_r2], **kw),
    "rank_r3": lambda **kw: compose_loop(transforms=[with_rank_r3], **kw),
    # rate
    "rate_off": lambda **kw: compose_loop(transforms=[with_rate_off], **kw),
    "rate_on_sda": lambda **kw: compose_loop(transforms=[with_rate_on_sda], **kw),
    "rate_on_ema": lambda **kw: compose_loop(transforms=[with_rate_on_ema], **kw),
    # combos comunes
    "bank3_r2_rate_off": lambda **kw: compose_loop(transforms=[with_bank3, with_rank_r2, with_rate_off], **kw),
    "bank4_r2_rate_on_sda": lambda **kw: compose_loop(transforms=[with_bank4, with_rank_r2, with_rate_on_sda], **kw),
    "bank4_r3_rate_on_ema": lambda **kw: compose_loop(transforms=[with_bank4, with_rank_r3, with_rate_on_ema], **kw),
}

PRESET_DISTILLS: Dict[str, Callable[..., DistillConfig]] = {}
if DistillConfig is not None:
    PRESET_DISTILLS.update({
        "lowD32": lambda **kw: compose_distill(transforms=[with_lowD32], **kw),
        "lowD48": lambda **kw: compose_distill(transforms=[with_lowD48], **kw),
        "lowD64": lambda **kw: compose_distill(transforms=[with_lowD64], **kw),
    })


# ------------------------------- __main__ -------------------------------------

if __name__ == "__main__":
    # Quick dry-run to show how to obtain configs
    import pprint
    V = 128
    pad = 0
    dev = _default_device()

    print("[configs] Loop examples:")
    names = ["bank3", "bank4", "rank_r1", "rank_r2", "rank_r3", "rate_off", "rate_on_sda", "rate_on_ema", "bank4_r2_rate_on_sda"]
    for n in names:
        cfg = PRESET_LOOPS[n](vocab_size=V, pad_id=pad, device=dev)
        print(f"  - {n}: cheb_deg={cfg.cheb_deg}, skew_rank={cfg.skew_rank}, R_rank={cfg.R_rank}, rate={cfg.use_dual_rate}")

    if DistillConfig is not None:
        print("\n[configs] Distill examples:")
        for n in ["lowD32", "lowD48", "lowD64"]:
            kd = PRESET_DISTILLS[n](W=256, O=32, pad_id=pad, device=dev)
            print(f"  - {n}: proj_dim={kd.proj_dim}, alpha={kd.alpha}, T={kd.temperature}")

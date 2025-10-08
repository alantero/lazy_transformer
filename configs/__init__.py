# configs/__init__.py
# Re-exports for convenience.
from .presets import (
    base_loop_cfg,
    base_distill_cfg,
    with_bank3, with_bank4,
    with_lowD32, with_lowD48, with_lowD64,
    with_rank_r1, with_rank_r2, with_rank_r3,
    with_rate_off, with_rate_on_sda, with_rate_on_ema,
    compose_loop, compose_distill, PRESET_LOOPS, PRESET_DISTILLS,
)

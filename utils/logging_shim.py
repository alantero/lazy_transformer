# utils/logging.py
# -----------------------------------------------------------------------------
# Compatibility shim for logging utilities.
# Re-exports from utils/logging_utils.py to avoid clashing with stdlib 'logging'.
# -----------------------------------------------------------------------------
from __future__ import annotations

import warnings as _warnings

# Add missing imports for fallback sys.path injection
import os, sys

# Import stdlib logging under a private alias (avoid name collision)
import logging as _stdlib_logging

# Try relative import first (package mode), then absolute, then sys.path-injected fallback
try:
    from .logging_utils import (  # type: ignore
        setup_logger,
        get_logger,
        silence_libs,
        JsonFormatter,
        get_rank,
        is_rank_zero,
    )
    _BACKEND = "utils.logging_utils (relative)"
except Exception:
    try:
        from utils.logging_utils import (  # type: ignore
            setup_logger,
            get_logger,
            silence_libs,
            JsonFormatter,
            get_rank,
            is_rank_zero,
        )
        _BACKEND = "utils.logging_utils (absolute)"
    except Exception:
        # Final fallback: inject repo root into sys.path so `utils` is importable
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from utils.logging_utils import (  # type: ignore
            setup_logger,
            get_logger,
            silence_libs,
            JsonFormatter,
            get_rank,
            is_rank_zero,
        )
        _BACKEND = "utils.logging_utils (sys.path injected)"

# Public surface re-exported
__all__ = [
    "setup_logger",
    "get_logger",
    "silence_libs",
    "JsonFormatter",
    "get_rank",
    "is_rank_zero",
]

# Emit a one-time compatibility note
_warnings.warn(
    f"[utils.logging] compatibility shim active; backend={_BACKEND}. "
    "Prefer `from utils.logging_utils import ...` to avoid confusion with stdlib `logging`.",
    category=UserWarning,
    stacklevel=2,
)

# --------------------------------- __main__ ---------------------------------
if __name__ == "__main__":
    # Minimal smoke test (run from repo root):
    #   python utils/logging_shim.py
    logger = setup_logger(
        name="lazyclo",
        config={
            "level": "DEBUG",
            "log_dir": "logs",
            "file_name": "compat_demo.log",
            "json": False,
            "color": True,
            "rank_zero_only": True,
            "rotate_when": "D",
            "rotate_bytes": 0,
            "backup_count": 2,
        },
    )
    silence_libs("ERROR")
    logger.debug("[compat] debug line")
    logger.info("[compat] info line; backend = %s", _BACKEND)
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("[compat] example exception")

    print("Compat shim OK. Wrote logs to ./logs/compat_demo.log")

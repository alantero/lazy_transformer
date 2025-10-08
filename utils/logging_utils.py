# utils/logging_utils.py
# -----------------------------------------------------------------------------
# LazyCLO v3 — Logging utilities (rank-aware, console + rotating file)
# -----------------------------------------------------------------------------
# Why this filename? To avoid clashing with Python's stdlib 'logging' package.
#
# Typical use:
#   from config import load_config
#   from utils.logging_utils import setup_logger, get_logger, silence_libs
#   cfg = load_config(...)
#   logger = setup_logger(cfg.logging.name, cfg.logging)
#   logger.info("Hello from LazyCLO!")
#
# Features:
# - Rank-aware (DDP): rank_zero_only=True silences non-zero ranks.
# - Console handler (Rich if available) + time/size-rotating file handlers.
# - Optional JSON formatter for log aggregation.
# - Helper to silence noisy third-party libs.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional, Iterable, Union, Any, Dict

# Optional rich console
try:
    from rich.logging import RichHandler  # type: ignore
    _HAVE_RICH = True
except Exception:
    _HAVE_RICH = False


# ----------------------------- Rank detection --------------------------------

def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def get_rank() -> int:
    # Works for PyTorch DDP/SLURM if env vars are set
    for k in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                continue
    return 0

def is_rank_zero() -> bool:
    return get_rank() == 0


# --------------------------- JSON log formatter ------------------------------

class JsonFormatter(logging.Formatter):
    """Minimal JSON-line formatter; safe for log aggregation."""
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "rank": get_rank(),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        import json as _json
        return _json.dumps(payload, ensure_ascii=False)


# ---------------------------- Core setup logic -------------------------------

def _extract_cfg(config: Optional[Union[Dict[str, Any], Any]]) -> Dict[str, Any]:
    """Accept a plain dict or a dataclass-like object (e.g., cfg.logging)."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    # dataclass / simple object with attributes
    return {
        k: getattr(config, k)
        for k in dir(config)
        if not k.startswith("_") and not callable(getattr(config, k))
    }

def setup_logger(
    name: str = "lazyclo",
    config: Optional[Union[Dict[str, Any], "LoggingConfig"]] = None,
    *,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    json_format: Optional[bool] = None,
    color: Optional[bool] = None,
    rank_zero_only: Optional[bool] = None,
    rotate_when: Optional[str] = None,
    rotate_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Args:
        name: logger name.
        config: dict/dataclass with fields like in config.LoggingConfig.
        Other kwargs override what's in `config`.
    """
    cfg = _extract_cfg(config)
    L = (level or cfg.get("level", "INFO")).upper()
    D = log_dir or cfg.get("log_dir", "logs")
    F = file_name or cfg.get("file_name", "run.log")
    J = cfg.get("json", False) if json_format is None else json_format
    C = cfg.get("color", True) if color is None else color
    R0 = cfg.get("rank_zero_only", True) if rank_zero_only is None else rank_zero_only
    ROT_W = rotate_when or cfg.get("rotate_when", "D")
    ROT_B = rotate_bytes if rotate_bytes is not None else int(cfg.get("rotate_bytes", 0))
    BCOUNT = backup_count if backup_count is not None else int(cfg.get("backup_count", 7))

    # Respect rank zero only
    if R0 and not is_rank_zero():
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        return logger

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(getattr(logging, L, logging.INFO))
        return logger

    logger.setLevel(getattr(logging, L, logging.INFO))
    logger.propagate = False

    # Console handler
    ch: logging.Handler
    if _HAVE_RICH and C and not J:
        ch = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        fmt_console = logging.Formatter("%(message)s")
    else:
        ch = logging.StreamHandler()
        fmt_console = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | r%(rank)d | %(name)s: %(message)s"
        )
        ch.addFilter(lambda rec: setattr(rec, "rank", get_rank()) or True)

    ch.setLevel(getattr(logging, L, logging.INFO))
    ch.setFormatter(fmt_console if not J else JsonFormatter())
    logger.addHandler(ch)

    # File handlers
    Path(D).mkdir(parents=True, exist_ok=True)
    log_path = str(Path(D) / F)

    # Time-based rotation
    if ROT_W:
        fh_time = logging.handlers.TimedRotatingFileHandler(
            log_path, when=ROT_W, interval=1, backupCount=BCOUNT, encoding="utf-8"
        )
        fh_time.setLevel(getattr(logging, L, logging.INFO))
        fh_time.setFormatter(JsonFormatter() if J else logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | r%(rank)d | %(name)s: %(message)s"
        ))
        fh_time.addFilter(lambda rec: setattr(rec, "rank", get_rank()) or True)
        logger.addHandler(fh_time)

    # Size-based rotation (optional)
    if ROT_B and ROT_B > 0:
        fh_size = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=ROT_B, backupCount=BCOUNT, encoding="utf-8"
        )
        fh_size.setLevel(getattr(logging, L, logging.INFO))
        fh_size.setFormatter(JsonFormatter() if J else logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | r%(rank)d | %(name)s: %(message)s"
        ))
        fh_size.addFilter(lambda rec: setattr(rec, "rank", get_rank()) or True)
        logger.addHandler(fh_size)

    return logger


def get_logger(name: str = "lazyclo") -> logging.Logger:
    """Return a logger (assumes setup_logger was called earlier)."""
    return logging.getLogger(name)


def silence_libs(level: str = "WARNING", libs: Optional[Iterable[str]] = None) -> None:
    """Reduce verbosity of third-party libraries."""
    libs = libs or ("urllib3", "numba", "matplotlib", "transformers", "torch")
    lvl = getattr(logging, level.upper(), logging.WARNING)
    for lib in libs:
        logging.getLogger(lib).setLevel(lvl)


# --------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    # Minimal demonstration (run from repo root):
    #   python -m utils.logging_utils
    cfg = {
        "name": "lazyclo",
        "level": "DEBUG",
        "log_dir": "logs",
        "file_name": "demo.log",
        "json": False,
        "color": True,
        "rank_zero_only": True,
        "rotate_when": "D",
        "rotate_bytes": 0,
        "backup_count": 3,
    }
    logger = setup_logger("lazyclo", cfg)
    silence_libs("ERROR")

    logger.debug("Debug message (sanity check).")
    logger.info("Info message: logging is initialized.")
    logger.warning("Warning message: example.")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("This is an example exception with traceback.")

    print("Wrote logs to ./logs/demo.log (and stdout).")

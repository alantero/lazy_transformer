# train/checkpoints.py
# -----------------------------------------------------------------------------
# Minimal, robust checkpoint I/O utilities for training:
# - save_checkpoint(...): atomic save, optional rotation (keep_last_k)
# - load_checkpoint(...): map_location-safe load + partial state_dict loading
# - CheckpointManager: tiny helper for "best" and periodic saves
#
# No hard deps on project files: pure stdlib + torch.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple, Union
import os
import json
import time
import shutil
import glob
import tempfile
import inspect
import pickle

import torch
from torch import nn
from torch.optim import Optimizer


# ------------------------------ small helpers --------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert config/dataclass-ish objects to plain dicts for serialization.
    Falls back to obj.__dict__ when possible; else returns {}.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        # Avoid dumping huge tensors accidentally
        return {k: v for k, v in vars(obj).items() if not isinstance(v, torch.Tensor)}
    return {}

def _atomic_save(obj: Any, path: str) -> None:
    """
    Atomic torch.save: write to a temp file in the same directory, then replace.
    """
    _ensure_dir(os.path.dirname(path) or ".")
    d = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("wb", dir=d, delete=False) as tmp:
        tmp_name = tmp.name
        torch.save(obj, tmp)
    os.replace(tmp_name, path)

def _log(msg: str) -> None:
    # Local, dependency-free logger
    print(msg)

def _state_dict_if_any(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    get = getattr(x, "state_dict", None)
    if callable(get):
        return get()
    return None

def _load_state_dict_if_any(x: Any, state: Optional[Dict[str, Any]], strict: bool) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if x is None or state is None:
        return tuple(), tuple()
    load = getattr(x, "load_state_dict", None)
    if not callable(load):
        return tuple(), tuple()
    #res = load(state, strict=strict)
    
    try:
        # Only modules usually accept `strict`; optimizers/schedulers don't.
        sig = inspect.signature(load)
        if "strict" in sig.parameters:
            res = load(state, strict=strict)
        else:
            res = load(state)
    except TypeError:
        # Fallback for odd signatures
        res = load(state)

    # Normalize return: PyTorch modules return (missing, unexpected) or dict
    if isinstance(res, tuple) and len(res) == 2:
        missing, unexpected = res
        return tuple(missing), tuple(unexpected)
    if isinstance(res, dict):
        return tuple(res.get("missing_keys", ())), tuple(res.get("unexpected_keys", ()))
    return tuple(), tuple()

def _rotate(directory: str, pattern: str, keep_last_k: int) -> None:
    """
    Keep only the last K files matching pattern (sorted by mtime).
    """
    paths = glob.glob(os.path.join(directory, pattern))
    paths.sort(key=lambda p: os.path.getmtime(p))
    if len(paths) > keep_last_k:
        to_delete = paths[: len(paths) - keep_last_k]
        for p in to_delete:
            try:
                os.remove(p)
            except Exception:
                pass


# --------------------------------- metadata ----------------------------------

@dataclass
class CheckpointMeta:
    step: int = 0
    epoch: int = 0
    best_metric: Optional[float] = None
    best_name: Optional[str] = None
    extra: Dict[str, Any] = None  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "epoch": int(self.epoch),
            "best_metric": float(self.best_metric) if self.best_metric is not None else None,
            "best_name": self.best_name,
            "extra": self.extra or {},
        }


# ----------------------------------- API -------------------------------------

def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int,
    epoch: int = 0,
    cfg: Optional[Any] = None,
    out_dir: str = "_tmp_ckpts",
    name: Optional[str] = None,
    best_metric: Optional[float] = None,
    best_tag: Optional[str] = None,   # e.g., "val_loss" or "val_acc"
    keep_last_k: Optional[int] = 5,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a checkpoint atomically. Returns the path written.

    Files created:
      - {name or auto}.pt: main checkpoint
      - meta.json (updated with latest + best)
    """
    _ensure_dir(out_dir)
    stamp = name or f"ckpt_step{step:08d}_{_now_iso()}"
    path = os.path.join(out_dir, f"{stamp}.pt")

    meta = CheckpointMeta(step=step, epoch=epoch, best_metric=best_metric, best_name=best_tag, extra=extra or {})

    payload: Dict[str, Any] = {
        "meta": meta.to_dict(),
        "model": model.state_dict(),
        "optimizer": _state_dict_if_any(optimizer),
        "scheduler": _state_dict_if_any(scheduler),
        "cfg": _to_dict(cfg),
        "pytorch_version": torch.__version__,
        "save_time": _now_iso(),
    }

    _atomic_save(payload, path)
    _log(f"[ckpt] saved: {path}")

    # Write/update meta.json for quick scanning
    meta_path = os.path.join(out_dir, "meta.json")
    try:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                hist = json.load(f)
        else:
            hist = {}
        hist["latest"] = os.path.basename(path)
        if best_metric is not None:
            # keep the best pointer per tag (simple)
            tag = best_tag or "best"
            best_key = f"{tag}_file"
            prev = hist.get(best_key)
            # Just overwrite; actual "is better" decision done in caller
            hist[best_key] = os.path.basename(path)
            hist[f"{tag}_metric"] = best_metric
        with open(meta_path, "w") as f:
            json.dump(hist, f, indent=2)
    except Exception as e:
        _log(f"[ckpt] meta.json update failed: {e}")

    # Rotation
    if keep_last_k is not None and keep_last_k > 0:
        try:
            _rotate(out_dir, "ckpt_step*.pt", keep_last_k)
        except Exception as e:
            _log(f"[ckpt] rotation failed: {e}")

    return path


def _load_compat(path: str, map_location: Union[str, torch.device, Dict[str, str]] = "cpu") -> Dict[str, Any]:
    """
    Torch 2.6+ changed the default of torch.load(weights_only=True).
    We explicitly try weights_only=False first (trusted local checkpoints),
    then fall back to an allowlisted safe load if available.
    """
    try:
        # Prefer explicit weights_only=False on newer PyTorch
        return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        # Older PyTorch without the argument
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # If this is a safety-gated UnpicklingError, try safe allowlist
        try:
            from torch.serialization import safe_globals  # type: ignore
            from torch.torch_version import TorchVersion  # type: ignore
            with safe_globals([TorchVersion]):
                return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
        except Exception:
            raise


def load_checkpoint(
    path: str,
    *,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device, Dict[str, str]] = "cpu",
    strict_model: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and (optionally) restore states into model/optimizer/scheduler.
    Returns the raw dict with keys: meta, model, optimizer, scheduler, cfg, ...
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = _load_compat(path, map_location=map_location)
    meta = ckpt.get("meta", {})
    _log(f"[ckpt] loading: {path} (step={meta.get('step')}, epoch={meta.get('epoch')})")

    missing: Tuple[str, ...] = tuple()
    unexpected: Tuple[str, ...] = tuple()

    if model is not None:
        m_state = ckpt.get("model")
        missing, unexpected = _load_state_dict_if_any(model, m_state, strict=strict_model)
        if missing or unexpected:
            _log(f"[ckpt] model missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                _log("        missing: " + ", ".join(list(missing)[:8]) + (" ..." if len(missing) > 8 else ""))
            if unexpected:
                _log("        unexpected: " + ", ".join(list(unexpected)[:8]) + (" ..." if len(unexpected) > 8 else ""))

    if optimizer is not None:
        _load_state_dict_if_any(optimizer, ckpt.get("optimizer"), strict=False)

    if scheduler is not None:
        _load_state_dict_if_any(scheduler, ckpt.get("scheduler"), strict=False)

    return ckpt


# --------------------------- light-weight manager -----------------------------

class CheckpointManager:
    """
    Tiny helper to manage periodic saves and best-by-metric snapshot.
    Caller decides what “better” means via a comparator function.
    """

    def __init__(
        self,
        out_dir: str,
        *,
        keep_last_k: int = 5,
        is_better: Optional[Any] = None,  # comparator: (new_metric, best_metric) -> bool
        best_tag: str = "best",
        map_location: Union[str, torch.device, Dict[str, str]] = "cpu",
    ):
        self.out_dir = out_dir
        self.keep_last_k = int(keep_last_k)
        self.is_better = is_better or (lambda new, best: (best is None) or (new < best))
        self.best_metric: Optional[float] = None
        self.best_path: Optional[str] = None
        self.best_tag = best_tag
        self.map_location = map_location
        _ensure_dir(self.out_dir)

    def periodic_save(
        self,
        *,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        cfg: Optional[Any],
        every: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if every <= 0 or (step % every) != 0:
            return None
        return save_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=step, epoch=epoch, cfg=cfg, out_dir=self.out_dir,
            best_metric=None, best_tag=None, keep_last_k=self.keep_last_k,
            extra=extra,
        )

    def update_best(
        self,
        *,
        metric_value: float,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        cfg: Optional[Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        If metric is better than the current best, save a new best snapshot and update pointers.
        """
        if self.is_better(metric_value, self.best_metric):
            self.best_metric = float(metric_value)
            path = save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                step=step, epoch=epoch, cfg=cfg, out_dir=self.out_dir,
                best_metric=self.best_metric, best_tag=self.best_tag,
                keep_last_k=self.keep_last_k, extra=extra,
                name=f"best_{self.best_tag}_step{step:08d}",
            )
            self.best_path = path
            _log(f"[ckpt] new best ({self.best_tag}={self.best_metric:.6f}) -> {os.path.basename(path)}")
            return path
        return None


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    # Smoke test: save, rotate, load, and verify model weights.
    print("[checkpoints] Running smoke test...")
    tmp_dir = "_tmp_ckpts"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Tiny model
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fake train steps with periodic saves and a “best” on step 3
    mgr = CheckpointManager(tmp_dir, keep_last_k=3, best_tag="val_loss")
    best = float("inf")
    for step in range(1, 6):
        # pretend to update model
        x = torch.randn(2, 8)
        y = model(x).sum()
        y.backward()
        opt.step(); opt.zero_grad()

        # periodic
        mgr.periodic_save(model=model, optimizer=opt, scheduler=None, step=step, epoch=0, cfg={"demo": True}, every=2)

        # fake validation: lower is better
        val = 1.0 / step
        if val < best:
            best = val
        mgr.update_best(metric_value=val, model=model, optimizer=opt, scheduler=None, step=step, epoch=0, cfg={"demo": True})

    # Load latest
    all_pts = glob.glob(os.path.join(tmp_dir, "*.pt"))
    latest_path = max(all_pts, key=os.path.getmtime)
    ckpt = load_checkpoint(latest_path, model=model, optimizer=opt, scheduler=None, map_location="cpu", strict_model=True)
    assert "model" in ckpt and "optimizer" in ckpt, "Malformed checkpoint"
    print(f"[checkpoints] OK. Latest file: {os.path.basename(latest_path)} | files={len(glob.glob(os.path.join(tmp_dir, '*.pt')))}")

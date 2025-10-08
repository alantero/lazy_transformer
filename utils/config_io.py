# utils/hf_config.py
# -----------------------------------------------------------------------------
# HF-like config I/O sin depender de `transformers`.
# - save_hf_config(cfg, save_dir)
# - load_hf_config(path_or_dir) -> LoopConfig
# - loopcfg_to_hf_dict / hf_dict_to_loopcfg (mapeo explícito)
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import asdict, fields
from typing import Any, Dict, Union
import json, os

try:
    # Para tipado; no importamos train.loop dentro de funciones para evitar ciclos
    from train.loop import LoopConfig  # type: ignore
    _HAVE_LOOPCFG = True
except Exception:
    LoopConfig = None  # type: ignore
    _HAVE_LOOPCFG = False


# --- helpers ------------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)

def _config_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        return os.path.join(path_or_dir, "config.json")
    return path_or_dir


# --- mapping: LoopConfig <-> HF dict -----------------------------------------

def loopcfg_to_hf_dict(cfg: "LoopConfig") -> Dict[str, Any]:
    d = asdict(cfg)
    # HF-ish header
    d_hf = {
        "model_type": "lazy-transformer",
        "architectures": ["ContinuousLM"],
        "torch_dtype": "float32",
    }
    d_hf.update(d)
    return d_hf

def hf_dict_to_loopcfg(d: Dict[str, Any]) -> "LoopConfig":
    if not _HAVE_LOOPCFG:
        raise ImportError("train.loop.LoopConfig no está disponible.")
    # Filtra solo campos de LoopConfig
    allowed = {f.name for f in fields(LoopConfig)}
    filtered = {k: v for k, v in d.items() if k in allowed}
    # Rellenar defaults vía constructor
    return LoopConfig(**filtered)  # type: ignore[arg-type]


# --- public API ----------------------------------------------------------------

def save_hf_config(cfg: "LoopConfig", save_dir: str) -> str:
    """
    Escribe save_dir/config.json con los campos de LoopConfig + cabecera HF-like.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "config.json")
    _write_json(path, loopcfg_to_hf_dict(cfg))
    return path

def load_hf_config(path_or_dir: str) -> "LoopConfig":
    """
    Carga un config.json estilo HF y lo convierte a LoopConfig.
    """
    path = _config_path(path_or_dir)
    d = _read_json(path)
    return hf_dict_to_loopcfg(d)

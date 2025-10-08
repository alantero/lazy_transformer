# config.py
# -----------------------------------------------------------------------------
# LazyCLO v3 — Configuration system (dataclasses + YAML/OmegaConf)
# -----------------------------------------------------------------------------
# This module centralizes all configuration for model, data, training, logging,
# and reproducibility. It supports:
#   1) Strongly-typed defaults via @dataclass (safe in IDEs/linters).
#   2) Optional YAML files.
#   3) Optional CLI overrides (OmegaConf if installed, graceful fallback).
#
# Usage (common):
#   from config import load_config
#   cfg = load_config("configs/base.yaml")   # or None for defaults
#   print(cfg.model.d, cfg.train.batch_size)
#
# CLI usage (when run as a script):
#   python config.py --config configs/base.yaml model.d=512 windows.W=1024
#
# Design notes:
# - Field names and defaults follow the architecture plan (v3):
#   windows (W/O), operators (deg, rank), integrator (steps, dt),
#   stitching (d_low), ledger (rate), duals (lambda/alpha via SDA), etc.
# - Validation enforces basic invariants (e.g., O < W, groups | d).
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Optional, Sequence
import os
import sys
import json
from pathlib import Path

# Optional deps (all graceful):
try:
    from omegaconf import OmegaConf
    _HAVE_OMEGA = True
except Exception:
    _HAVE_OMEGA = False

try:
    import yaml  # PyYAML (fallback if OmegaConf not installed)
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


# ----------------------------- Dataclass blocks ------------------------------

@dataclass
class WindowsConfig:
    W: int = 1024         # window length (recommended 512–1024)
    O: int = 64           # overlap ("collar"), ~5–10% of W
    rescale_to_unit: bool = True  # normalize local time domain to [0,1]


@dataclass
class ModelConfig:
    d: int = 768
    k: int = 128
    vocab_size: int = 50304     # GPT-2 compatible default; replace in data/tokenize
    groups: int = 16            # groupwise traceless norm
    d_low: int = 48             # low-D for stitching
    # Operator bank
    cheb_deg: int = 8           # Chebyshev degree (<=10)
    lowrank_r: int = 64         # Nyström rank (<=64)
    adv_kernel: int = 5
    osc_modes: int = 16
    # Quantization hints (capacity-aware; see modules/quant.py later)
    qat_enable: bool = False
    int4_spectral: bool = True
    int8_mix_ledger: bool = True


@dataclass
class IntegratorConfig:
    max_steps: int = 4      # usually 2–4
    dt: float = 1.0         # logical step size (scaled inside the block)
    stop_epsilon: float = 1e-4  # minimal ΔBKM-like improvement to continue
    cost_per_step: float = 1.0  # constant proxy; can be replaced by tiny MLP later


@dataclass
class StitchingConfig:
    align_lowD: bool = True
    reortho_every: int = 0         # 0 = lazy re-ortho (on demand)
    skl_weight: float = 1.0
    procrustes_weight: float = 1.0


@dataclass
class LedgerConfig:
    enable_virtual_rate: bool = True
    mixtures: int = 2
    target_bits_per_token: float = 0.0  # low rate target (encourages coarse)
    forget_when_no_gain: bool = True


@dataclass
class GateConfig:
    use_sparse_gate: bool = True
    n_ops: int = 4
    hidden: int = 64
    target_active_ops: int = 2        # enforce 1–2 active ops
    sparsity_weight: float = 1e-3


@dataclass
class KrylovConfig:
    enable: bool = True
    steps: int = 2   # 2–3
    trigger_delta_bkm: float = 1e-3


@dataclass
class DualsConfig:
    # Stochastic Dual Averaging over log-space (λ: capacity, α: complexity)
    lambda_init: float = 0.1
    alpha_init: float = 0.1
    sda_lr: float = 0.05
    sda_momentum: float = 0.9
    var_aware: bool = True


@dataclass
class TrainConfig:
    seed: int = 1337
    deterministic: bool = True
    batch_size: int = 8
    micro_batches: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_tokens: int = 1_000_000
    compile: bool = False       # torch.compile if desired
    grad_clip: float = 1.0
    fp16: bool = False
    bf16: bool = True


@dataclass
class DataConfig:
    dataset: str = "tiny-shakespeare"
    text_column: str = "text"
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class LoggingConfig:
    name: str = "lazyclo"
    level: str = "INFO"
    log_dir: str = "logs"
    file_name: str = "run.log"
    rotate_when: str = "D"      # "D" daily; or "" to disable time rotation
    rotate_bytes: int = 0       # 0 = disabled; else rotate by size
    backup_count: int = 7
    json: bool = False
    color: bool = True
    rank_zero_only: bool = True
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None


@dataclass
class ExperimentConfig:
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    krylov: KrylovConfig = field(default_factory=KrylovConfig)
    duals: DualsConfig = field(default_factory=DualsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # --------------------------- Validation rules ----------------------------
    def validate(self) -> None:
        W, O = self.windows.W, self.windows.O
        if not (1 <= O < W):
            raise ValueError(f"Invalid windows: O({O}) must be in [1, W-1] with W={W}.")
        if self.model.d % self.model.groups != 0:
            raise ValueError(f"Model groups({self.model.groups}) must divide d({self.model.d}).")
        if self.model.cheb_deg < 0 or self.model.cheb_deg > 16:
            raise ValueError("cheb_deg should be 0..16 (<=10 recommended).")
        if self.integrator.max_steps < 1 or self.integrator.max_steps > 16:
            raise ValueError("max_steps should be in 1..16.")
        if self.model.d_low <= 0 or self.model.d_low > self.model.d:
            raise ValueError("d_low must be in 1..d.")
        if self.train.micro_batches < 1 or self.train.batch_size < 1:
            raise ValueError("batch_size and micro_batches must be >= 1.")

    # Pretty serialize
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        if _HAVE_OMEGA:
            return OmegaConf.to_yaml(OmegaConf.create(self.to_dict()))
        if _HAVE_YAML:
            return yaml.safe_dump(self.to_dict(), sort_keys=False)
        return json.dumps(self.to_dict(), indent=2)


# ------------------------------ Load / Merge --------------------------------

def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def _merge_dicts(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (right-biased)."""
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    # Manual reconstruction keeps strong typing even w/o OmegaConf
    return ExperimentConfig(
        windows=WindowsConfig(**d.get("windows", {})),
        model=ModelConfig(**d.get("model", {})),
        integrator=IntegratorConfig(**d.get("integrator", {})),
        stitching=StitchingConfig(**d.get("stitching", {})),
        ledger=LedgerConfig(**d.get("ledger", {})),
        gate=GateConfig(**d.get("gate", {})),
        krylov=KrylovConfig(**d.get("krylov", {})),
        duals=DualsConfig(**d.get("duals", {})),
        train=TrainConfig(**d.get("train", {})),
        data=DataConfig(**d.get("data", {})),
        logging=LoggingConfig(**d.get("logging", {})),
    )


def load_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Sequence[str]] = None,
) -> ExperimentConfig:
    """
    Load configuration in three steps:
      1) Start from dataclass defaults.
      2) Merge YAML file if provided.
      3) Merge CLI overrides (requires OmegaConf).
    """
    base = default_config().to_dict()

    # YAML (file path or env LAZYCLO_CONFIG)
    path = yaml_path or os.environ.get("LAZYCLO_CONFIG", "")
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        if _HAVE_OMEGA:
            y = OmegaConf.to_container(OmegaConf.load(str(p)), resolve=True)
        elif _HAVE_YAML:
            with open(p, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
        else:
            raise RuntimeError("Neither OmegaConf nor PyYAML available to read YAML.")
        base = _merge_dicts(base, y)

    # CLI overrides (OmegaConf only)
    if cli_overrides and _HAVE_OMEGA:
        cli = OmegaConf.to_container(OmegaConf.from_cli(list(cli_overrides)), resolve=True)
        base = _merge_dicts(base, cli)

    cfg = _from_dict(base)
    cfg.validate()
    return cfg


def save_config(cfg: ExperimentConfig, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cfg.to_yaml())


# --------------------------------- __main__ ---------------------------------

def _parse_cli_args() -> Dict[str, Any]:
    # Minimal parser that forwards unknown args to OmegaConf.from_cli if present.
    import argparse
    parser = argparse.ArgumentParser(description="LazyCLO config loader")
    parser.add_argument("--config", type=str, default=None, help="YAML path")
    parser.add_argument("--print", action="store_true", help="Print final YAML")
    parser.add_argument("--save", type=str, default=None, help="Save merged YAML here")
    args, unknown = parser.parse_known_args()
    return {"args": args, "unknown": unknown}

if __name__ == "__main__":
    io = _parse_cli_args()
    args, unknown = io["args"], io["unknown"]
    cfg = load_config(args.config, unknown if _HAVE_OMEGA else None)

    # Optional: basic sanity echo
    print("[LazyCLO] Loaded configuration ✓")
    print(f"- windows: W={cfg.windows.W}, O={cfg.windows.O}")
    print(f"- model: d={cfg.model.d}, k={cfg.model.k}, groups={cfg.model.groups}, d_low={cfg.model.d_low}")
    print(f"- integrator: max_steps={cfg.integrator.max_steps}, dt={cfg.integrator.dt}, stop_eps={cfg.integrator.stop_epsilon}")

    if args.print:
        print("\n# --- Final merged config ---")
        print(cfg.to_yaml())

    if args.save:
        save_config(cfg, args.save)
        print(f"[LazyCLO] Saved merged config -> {args.save}")

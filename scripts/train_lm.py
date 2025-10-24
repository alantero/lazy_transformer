# scripts/train_lm.py
# -----------------------------------------------------------------------------
# CLI training entrypoint for Lazy Transformer (Phase 3+).
# Wraps train/loop.py components with presets, schedulers, checkpoints,
# and optional profiling. Minimal deps; falls back to toy data if no file.
# Usage example:
#   python scripts/train_lm.py --preset base_tiny --steps 200 --checkpoint-dir ./ckpts
#   python scripts/train_lm.py --preset bank4_r2_rate_on_sda --text-file data.txt
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import asdict
from typing import Optional, List, Dict, Any, Tuple
import argparse
import logging
import os
import sys
import math
import random

# Make repo root importable when running as a script
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
# ---- Performance knobs (safe defaults) ----
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # If PyTorch & GPU support it, prefer high matmul precision (enables TF32 on Ampere+)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass

import json
from contextlib import nullcontext


# tqdm for progress bars
from tqdm.auto import tqdm

# --- Project imports (robust fallbacks where possible) ------------------------
try:
    from configs.presets import get_preset, list_presets  # type: ignore
except Exception:
    def get_preset(name: str) -> Dict[str, Any]:
        raise RuntimeError("configs.presets.get_preset not found. Create configs/presets.py.")
    def list_presets() -> List[str]:
        return []

try:
    from data.tok_embed import TokenEmbedder, SimpleVocab, pack_batch  # type: ignore
except Exception:
    from data.tokenize import TokenEmbedder, SimpleVocab, pack_batch  # type: ignore

# Optional dataloader helpers
try:
    from data.dataloaders import collate_batch as _collate_batch, build_collar_mask as _build_collar_mask  # type: ignore
except Exception:
    _collate_batch = None  # type: ignore
    _build_collar_mask = None  # type: ignore

# Core loop pieces
from train.loop import ContinuousLM, LoopConfig, build_optimizer, train_step  # type: ignore
from train.checkpoints import CheckpointManager  # type: ignore
from optim.schedulers import make_scheduler  # type: ignore

# Profiling helpers (optional)
try:
    from utils.profile import Timer, ThroughputMeter, nvtx_range, record_function as prof_record_function, gpu_mem  # type: ignore
except Exception:
    Timer = None  # type: ignore
    ThroughputMeter = None  # type: ignore
    def nvtx_range(_name):  # type: ignore
        from contextlib import nullcontext
        return nullcontext()
    def prof_record_function(_name):  # type: ignore
        from contextlib import nullcontext
        return nullcontext()
    def gpu_mem():  # type: ignore
        return {"allocated": 0.0, "reserved": 0.0}

# HF-style config adapter (optional)
try:
    from utils.hf_config import load_hf_config, save_hf_config  # type: ignore
except Exception:
    load_hf_config = None  # type: ignore
    save_hf_config = None  # type: ignore


# ------------------------------ Helpers ---------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

def load_texts(path: Optional[str]) -> List[str]:
    if path is None:
        # Toy default (small char-level)
        return ["hello there", "general kenobi", "hello hello"]
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # keep non-empty
    return [ln for ln in lines if len(ln) > 0]

def build_batch_from_texts(
    texts: List[str],
    vocab: SimpleVocab,
    W: int,
    O: int,
) -> Tuple[Dict[str, torch.Tensor], LoopConfig]:
    seqs = [vocab.encode(t, mode="char", add_bos=True, add_eos=True) for t in texts]
    # Prefer real collate if available (pads to max len and builds mask)
    if _collate_batch is not None:
        bos_id = getattr(vocab, "bos_id", None)
        eos_id = getattr(vocab, "eos_id", None)
        batch = _collate_batch(seqs, pad_id=vocab.pad_id, bos_id=bos_id, eos_id=eos_id, max_len=None)
        if _build_collar_mask is not None:
            lens = batch["mask"].sum(dim=1).tolist()
            batch["collar_mask"] = _build_collar_mask(lens, W=W, O=O)
    else:
        tokens, mask = pack_batch(seqs, pad_id=vocab.pad_id)
        batch = {"tokens": tokens, "mask": mask, "targets": tokens.clone()}
    return batch, None  # cfg will be built by caller

def merge_cfg(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out

# --- Helper: filter/normalize config for LoopConfig ---
from dataclasses import fields as _dc_fields
def _sanitize_for_loopconfig(d: Dict[str, Any]) -> Dict[str, Any]:
    """Filter arbitrary config dict `d` down to the fields accepted by LoopConfig.
    - If `pad_id` is missing but `eos_id` is present (HF-style), set pad_id = eos_id.
    - Log and drop keys not present in LoopConfig to avoid `__init__` errors
      (e.g., bos_id, eos_id, unk_token, etc.).
    """
    allowed = {f.name for f in _dc_fields(LoopConfig)}
    d = dict(d)  # shallow copy
    # Consolidate pad/eos convention: prefer explicit pad_id; else fall back to eos_id
    if ("pad_id" not in d or d["pad_id"] is None) and ("eos_id" in d and d["eos_id"] is not None):
        try:
            d["pad_id"] = int(d["eos_id"])  # Strategy B: pad == eos
        except Exception:
            pass
    # Build filtered dict and collect ignored keys for logging
    out: Dict[str, Any] = {}
    ignored: List[str] = []
    for k, v in d.items():
        if k in allowed:
            out[k] = v
        else:
            ignored.append(k)
    if ignored:
        logging.info(f"[cfg] ignoring non-LoopConfig keys: {sorted(ignored)}")
    return out


# ------------------------------ NPY Dataset ---------------------------------
class NpyPackedDataset(Dataset):
    """
    Loads a .npy memmap with shape [N, seq_len] containing GPT-2 token IDs (int32).
    Returns torch.long tensors. Collate will shift to build targets.
    """
    def __init__(self, path: str):
        assert os.path.exists(path), f"File not found: {path}"
        self.path = path
        self.arr = np.load(path, mmap_mode="r")  # [N, S] int32

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __getitem__(self, idx: int):
        x = np.asarray(self.arr[idx], dtype=np.int32).copy()  # make writable to avoid PyTorch warning
        return torch.as_tensor(x, dtype=torch.long)

def _infer_or_check_seq_len_from_npy(cfg, train_path: str) -> int:
    """Ensure cfg.W matches dataset seq_len (or set it if missing)."""
    have = int(np.load(train_path, mmap_mode="r").shape[1])
    want = getattr(cfg, "W", None)
    # The model uses window size W; our packed windows are exactly seq_len.
    if want is None or want <= 0:
        setattr(cfg, "W", have)
        logging.info(f"[data] W not set in cfg; inferred W={have} from {train_path}")
    elif want != have:
        logging.warning(f"[data] dataset seq_len ({have}) != cfg.W ({want}); using cfg.W but verify your packing.")
    return have

def _npy_collate(seq_len: int):
    """Stack to [B,S] then shift to x/y (next-token) and add a full-ones mask."""
    def _fn(batch):
        x_full = torch.stack(batch, dim=0)          # [B, S]
        x = x_full[:, :seq_len-1].contiguous()      # [B, S-1]
        y = x_full[:, 1:seq_len].contiguous()       # [B, S-1]
        mask = torch.ones_like(x, dtype=torch.bool) # fixed windows, no pad
        return {"x": x, "y": y, "mask": mask}
    return _fn

def build_npy_dataloaders(
    cfg,
    train_path: str,
    val_path: str,
    micro_batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
):
    """Create DataLoaders from pre-packed NPY windows."""
    seq_len = _infer_or_check_seq_len_from_npy(cfg, train_path)
    train_ds = NpyPackedDataset(train_path)
    val_ds   = NpyPackedDataset(val_path)
    collate  = _npy_collate(seq_len)

    # (Optional) DDP samplers if user launches with torchrun
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    pw = bool(num_workers > 0)
    pf = 4 if num_workers > 0 else None

    train_kwargs = dict(
        dataset=train_ds, batch_size=micro_batch_size, shuffle=shuffle_train, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate,
        persistent_workers=pw,
    )
    val_kwargs = dict(
        dataset=val_ds, batch_size=micro_batch_size, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=collate,
        persistent_workers=pw,
    )
    if num_workers > 0:
        train_kwargs["prefetch_factor"] = pf
        val_kwargs["prefetch_factor"] = pf

    train_loader = DataLoader(**train_kwargs)
    val_loader   = DataLoader(**val_kwargs)
    logging.info(f"[data] train {tuple(train_ds.arr.shape)}  val {tuple(val_ds.arr.shape)}  seq_len={seq_len} | workers={num_workers} persistent={pw} prefetch={pf or 'default'}")
    return train_loader, val_loader, seq_len


# --- Helper: Fix vocab/pad from meta.json or NPY probe ---
def _maybe_fix_vocab_from_meta(cfg, train_path: str) -> None:
    """
    Ensure cfg.vocab_size and cfg.pad_id are consistent with the dataset/tokenizer.
    Preference order:
      1) meta.json next to the NPY files (written by fwe_tokenize_pack.py)
      2) quick probe of the NPY memmap to estimate max token id
    Always enforce: vocab_size >= pad_id+1 and > max_id seen in a small probe.
    """
    # Defaults from cfg (may be None)
    cur_vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    cur_pad   = getattr(cfg, "pad_id", None)

    # --- Try meta.json ---
    meta_vocab_candidates: List[int] = []
    meta_pad: Optional[int] = None
    try:
        base = os.path.dirname(os.path.abspath(train_path))
        meta_path = os.path.join(base, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            # candidates for vocab_size
            for key in ("vocab_size_hf", "vocab_size"):
                if key in meta and isinstance(meta[key], (int, float, str)):
                    try:
                        meta_vocab_candidates.append(int(meta[key]))
                    except Exception:
                        pass
            # try to infer from eos/pad too
            for key in ("eos_id", "pad_id"):
                if key in meta and isinstance(meta[key], (int, float, str)):
                    try:
                        meta_vocab_candidates.append(int(meta[key]) + 1)
                    except Exception:
                        pass
            # pad_id preference: meta.pad_id else meta.eos_id
            if "pad_id" in meta:
                meta_pad = int(meta["pad_id"])
            elif "eos_id" in meta:
                meta_pad = int(meta["eos_id"])

            if meta_vocab_candidates:
                v_meta = max(meta_vocab_candidates)
                if cur_vocab != v_meta:
                    logging.info(f"[data] meta.json suggests vocab_size={v_meta} (candidates={meta_vocab_candidates}); cfg.vocab_size was {cur_vocab}")
                cur_vocab = max(cur_vocab, v_meta)
            if meta_pad is not None:
                cur_pad = meta_pad
                logging.info(f"[data] meta.json sets pad_id={cur_pad}")
    except Exception as e:
        logging.warning(f"[data] failed to read meta.json for vocab/pad fix: {e}")

    # --- Probe small slice of NPY anyway (robustness) ---
    try:
        arr = np.load(train_path, mmap_mode="r")
        vmax = int(np.max(arr[: min(4096, arr.shape[0])]))  # small probe
        need = vmax + 1
        if cur_vocab < need:
            logging.info(f"[data] probe raises vocab_size to {need} (max token id={vmax})")
        cur_vocab = max(cur_vocab, need)
    except Exception as e:
        logging.warning(f"[data] vocab probe failed: {e}")

    # --- Finalize: ensure vocab_size > pad_id and pad defined ---
    if cur_pad is None:
        # If pad is still unknown, default to eos-style (last id)
        cur_pad = max(0, cur_vocab - 1)
        logging.info(f"[data] pad_id not provided; defaulting to {cur_pad}")
    if cur_vocab <= int(cur_pad):
        fixed = int(cur_pad) + 1
        logging.warning(f"[data] vocab_size {cur_vocab} <= pad_id {cur_pad}; bumping vocab_size -> {fixed}")
        cur_vocab = fixed

    # Write back to cfg
    cfg.vocab_size = int(cur_vocab)
    cfg.pad_id = int(cur_pad)
    logging.info(f"[data] resolved vocab/pad: vocab_size={cfg.vocab_size}, pad_id={cfg.pad_id}")

# ------------------------------ Sanity checks --------------------------------
@torch.no_grad()
def run_sanity_checks(model: torch.nn.Module, batch: Dict[str, torch.Tensor], cfg) -> None:
    """Log basic alignment and loss sanity metrics on a single batch.
    Enhanced: Checks vocab head size, target id ranges, and computes CE with/without pad ignore.
    Also compares model training-style loss to raw CE and off-by-one diagnosis.
    """
    import math
    device = next(model.parameters()).device
    tokens  = batch["tokens"].to(device)
    targets = batch["targets"].to(device)
    mask    = batch.get("mask", None)
    if mask is not None:
        mask = mask.to(device)

    eq_ratio = (tokens == targets).float().mean().item()
    pad_id = getattr(cfg, "pad_id", None)
    if pad_id is not None:
        pad_ratio = (targets == int(pad_id)).float().mean().item()
    else:
        pad_ratio = 0.0

    uniq = int(torch.unique(targets).numel())

    # --- Forward pass (no grad) on tokens/mask to get logits ---
    out = model.forward_tokens(tokens, mask)
    # Unpack logits only; ignore any extra outputs
    if isinstance(out, (tuple, list)) and len(out) >= 1:
        logits = out[0]
    else:
        logits = out

    B, T, V = logits.shape
    flat_logits = logits.reshape(B*T, V)
    flat_targets = targets.reshape(B*T)

    # Aligned labels: use provided targets (run_sanity_checks uses next-token y)
    aligned_y = targets

    # Aligned CE using the chosen labels
    ce_aligned = torch.nn.CrossEntropyLoss()(flat_logits, aligned_y.reshape(B*T)).item()

    # Compute CE without ignore_index (raw) and with ignore_index if pad_id is defined (for diagnostics)
    ce_raw = torch.nn.CrossEntropyLoss()(flat_logits, flat_targets).item()
    if pad_id is not None:
        ce_ignore = torch.nn.CrossEntropyLoss(ignore_index=int(pad_id))(flat_logits, flat_targets).item()
    else:
        ce_ignore = ce_raw

    # Also try CE if targets were mistakenly unshifted (i.e., equal to tokens) to detect off-by-one
    ce_vs_tokens = torch.nn.CrossEntropyLoss()(flat_logits, tokens.reshape(B*T)).item()

    # Range checks for targets vs head dimension
    tmin = int(targets.min().item())
    tmax = int(targets.max().item())
    vocab_size = int(getattr(cfg, "vocab_size", V))

    # Try computing the model's own training-style loss on this batch
    # (ContinuousLM.__call__ returns (loss, stats) with its internal convention)
    batch_eval = {"tokens": tokens, "targets": targets}
    if mask is not None:
        batch_eval["mask"] = mask
    try:
        model_loss, model_stats = model(batch_eval)  # type: ignore
        model_loss = float(model_loss.item()) if hasattr(model_loss, "item") else float(model_loss)
    except Exception as e:
        model_loss = float("nan")
        logging.warning(f"[sanity] could not compute model(batch) loss: {e}")
        model_stats = {}

    logging.info(
        "[sanity] eq_ratio(tokens==targets)=%0.4f | pad_ratio(targets==pad_id[%s])=%0.4f | "
        "unique_targets=%d | head_dim(V)=%d | cfg.vocab_size=%d | targets_range=[%d,%d] | "
        "CE_aligned=%0.3f | CE_raw=%0.3f | CE_ignore_pad=%0.3f | CE_vs_tokens=%0.3f | model_loss=%s"
        % (eq_ratio, str(pad_id), pad_ratio, uniq, V, vocab_size, tmin, tmax, ce_aligned, ce_raw, ce_ignore, ce_vs_tokens, f"{model_loss:.3f}" if math.isfinite(model_loss) else "nan")
    )

    # Heuristics to warn loudly if something is off
    if eq_ratio > 0.05:
        logging.warning("[sanity] High eq_ratio indicates shift may be broken (tokens likely equal to targets).")
    if pad_id is not None and pad_ratio > 0.01:
        logging.warning("[sanity] Non-negligible pad_id in targets; consider ignoring pad in loss and fixing packing.")
    if V != vocab_size:
        logging.warning(f"[sanity] Vocab head dim (V={V}) != cfg.vocab_size ({vocab_size}). Check final projection / tie_softmax.")
    if tmin < 0 or tmax >= V:
        logging.warning(f"[sanity] Target IDs out of range for head: min={tmin} max={tmax} vs V={V}.")

    # Alignment checks
    #if math.isfinite(model_loss) and abs(ce_aligned - model_loss) < 0.5:
    #    pass  # aligned labels match training loss (good)
    #else:
    #    logging.warning("[sanity] CE_aligned diverges from model_loss. Check that forward_tokens returns (logits, y_win) and that collate shift matches training.")
    # Alignment note: model_loss is authoritative (may include internal masking/normalization).
    # We log CE_* just for diagnostics; we don't expect it to match model_loss numerically.

    # Detect off-by-one: if CE_vs_tokens << CE_raw but CE_aligned ≈ model_loss
    if math.isfinite(model_loss) and (ce_vs_tokens + 0.5 < ce_raw) and (abs(ce_aligned - model_loss) < 0.5):
        logging.warning("[sanity] CE_vs_tokens << CE_raw while CE_aligned ≈ model_loss. Likely your sanity 'targets' are pre-shifted, while the model computes its own shift.")




# ------------------------------ CLI / Main ------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Lazy Transformer LM (windowed).")
    # Data
    p.add_argument("--text-file", type=str, default=None, help="Path to a text file (one sample per line). If omitted, toy data is used.")
    # Preset
    p.add_argument("--preset", type=str, default=None, help=f"Preset name from configs.presets. Known: {', '.join(list_presets()) or 'none'}")
    # Steps / logging
    p.add_argument("--steps", type=int, default=None, help="Training steps (overwrites cfg if set).")
    p.add_argument("--log-interval", type=int, default=10, help="Steps between logs.")
    # Model/loop overrides (common)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--O", type=int, default=None)
    p.add_argument("--d-model", dest="d_model", type=int, default=None)
    p.add_argument("--groups", type=int, default=None)
    p.add_argument("--cheb-deg", dest="cheb_deg", type=int, default=None)
    p.add_argument("--dt", type=float, default=None, help="Integrator time step (dt) for the dynamics/integrator.")
    p.add_argument("--integ-method", dest="method", type=str, default=None, choices=["euler", "heun"], help="Integrator method to use (euler or heun).")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    # Stitching / dual rate (shortlist)
    p.add_argument("--stitch-w", dest="stitch_w", type=float, default=None)
    p.add_argument("--use-dual-rate", dest="use_dual_rate", action="store_true")
    p.add_argument("--no-use-dual-rate", dest="use_dual_rate", action="store_false")
    p.set_defaults(use_dual_rate=None)
    p.add_argument("--target-bpp", dest="target_bpp", type=float, default=None,
                   help="Target bits-per-token for the dual-rate penalty. If not set, the model defaults to ~0.6*log2(vocab_size). Use together with --use-dual-rate.")
    # Scheduler
    p.add_argument("--scheduler", dest="scheduler_name", type=str, default=None,
                   choices=["warmup_cosine", "warmup_linear", "noam", "plateau"])
    p.add_argument("--warmup", dest="scheduler_warmup_steps", type=int, default=None)
    p.add_argument("--total-steps", dest="scheduler_total_steps", type=int, default=None)
    p.add_argument("--min-lr", dest="scheduler_min_lr", type=float, default=None)
    # Checkpoints
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--keep-last-k", type=int, default=None)
    p.add_argument("--resume", dest="resume_path", type=str, default=None)
    # Profiling
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-nvtx", action="store_true")
    p.add_argument("--profile-log-mem", action="store_true")

    # HF-style config
    p.add_argument("--hf-config", type=str, default=None, help="Path to a HF-style config.json or directory containing it.")
    p.add_argument("--hf-save", type=str, default=None, help="If set, save resolved LoopConfig as HF-style config.json into this directory.")

    # Logging level
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # NPY dataset (pre-packed windows)
    p.add_argument("--train-npy", dest="train_npy", type=str, default=None,
                   help="Path to train.npy (packed [N, seq_len] token IDs).")
    p.add_argument("--val-npy", dest="val_npy", type=str, default=None,
                   help="Path to val.npy (packed [N, seq_len] token IDs).")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=8,
                   help="Micro batch size per step when using NPY dataset.")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=4,
                   help="DataLoader workers for NPY dataset.")
    p.add_argument("--best-every", dest="best_every", type=int, default=1,
                   help="Check/save 'best' checkpoint every N steps (default: 1 = every step).")
    p.add_argument("--val-every", dest="val_every", type=int, default=0,
                   help="Run validation every N steps (0 disables periodic validation). Best checkpoint is selected by val_loss when validation runs.")
    p.add_argument("--debug-sanity", action="store_true",
               help="Run one-time sanity checks on targets/tokens alignment and pad usage before training.")

    # --- AR/diagnostic controls (propagated to LoopConfig) ---
    p.add_argument("--grads-border-only", dest="grads_border_only", action="store_true",
                  help="Backprop only on the collar/border region (windowed training).")
    p.add_argument("--no-grads-border-only", dest="grads_border_only", action="store_false",
                  help="Disable border-only grads; compute CE on all positions.")
    p.set_defaults(grads_border_only=None)

    p.add_argument("--debug-align", action="store_true",
                  help="Run strict autoregressive alignment diagnostics during training.")
    p.add_argument("--debug-align-every", type=int, default=None,
                  help="Frequency (in steps) for alignment debug logs.")
    p.add_argument("--debug-state-norm", action="store_true",
                  help="Log state norms (e.g., ||h|| and ||h_last||).")
    p.add_argument("--debug-topk", type=int, default=None,
                  help="If set, log top-k token ids/probs at last timestep during training.")

    p.add_argument("--state-w", type=float, default=0.0,
                   help="Peso del regularizador de no-expansión de energía de estado (0 = desactivado).")
    p.add_argument("--state-mask", type=str, choices=["all", "collar"], default="all",
                   help="Dónde aplicar el regularizador de estado: en todo ('all') o sólo en el collar ('collar').")
    p.add_argument("--state-kind", type=str, default="energy_contract",
                   help="Tipo de regularizador de estado. Por ahora: 'energy_contract'.")


    # Optional override for Chebyshev laplacian kind
    p.add_argument("--cheb-laplacian", dest="cheb_laplacian", type=str, default=None,
                  choices=["path_causal", "path", "cycle", "toeplitz"],
                  help="Laplacian kind for Chebyshev operator. Use 'path_causal' for strict causal LM; 'path' (Dirichlet) is non-causal and only for ablations.")

    return p
@torch.no_grad()
def evaluate_val_ce(model: torch.nn.Module, val_loader: DataLoader, pad_id: int) -> Dict[str, float]:
    """Evaluate cross-entropy/accuracy on the validation loader.

    Notes:
      - We DO NOT assume `model.device` exists (ContinuousLM doesn't expose it).
        Instead we get the device from the first parameter.
      - Uses ignore_index=pad_id to be robust to occasional EOS-as-pad in packed windows.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tok, total_acc = 0.0, 0, 0.0
    for batch in val_loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        out = model.forward_tokens(x, mask)      # may be Tensor or (logits, ...)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        B, T, V = logits.shape
        # Per-token CE (no reduction), then mask-out invalid positions before summing.
        per_tok = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1),
            ignore_index=int(pad_id),
            reduction="none",
        ).view(B, T)
        # valid positions: non-pad and (if provided) within mask
        valid = (y != int(pad_id))
        if mask is not None:
            valid = valid & mask
        per_tok = per_tok * valid.to(per_tok.dtype)
        batch_loss_sum = per_tok.sum()
        ntok = int(valid.sum().item())
        total_loss += float(batch_loss_sum.item())
        total_tok  += ntok
        pred = logits.argmax(-1)
        batch_acc = float((pred[valid] == y[valid]).float().mean().item()) if ntok > 0 else 0.0
        total_acc  += batch_acc
    mean_loss = total_loss / max(total_tok, 1)
    ppl = math.exp(mean_loss)
    acc = total_acc / max(1, len(val_loader))
    return {"val_loss": mean_loss, "val_ppl": ppl, "val_acc": acc}

# --- tqdm-enabled validation function ---
@torch.no_grad()
def evaluate_val_ce_tqdm(model: torch.nn.Module, val_loader: DataLoader, pad_id: int, desc: str) -> Dict[str, float]:
    """Same as evaluate_val_ce, but displays a tqdm over the validation loader."""
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tok, total_acc = 0.0, 0, 0.0
    with tqdm(val_loader, desc=desc, leave=False, dynamic_ncols=True) as pbar:
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            out = model.forward_tokens(x, mask)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            B, T, V = logits.shape
            # Per-token CE (no reduction), then apply mask before summing.
            per_tok = torch.nn.functional.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1),
                ignore_index=int(pad_id),
                reduction="none",
            ).view(B, T)
            valid = (y != int(pad_id))
            if mask is not None:
                valid = valid & mask
            per_tok = per_tok * valid.to(per_tok.dtype)
            batch_loss_sum = per_tok.sum()
            ntok = int(valid.sum().item())
            total_loss += float(batch_loss_sum.item())
            total_tok  += ntok
            pred = logits.argmax(-1)
            batch_acc = float((pred[valid] == y[valid]).float().mean().item()) if ntok > 0 else 0.0
            total_acc  += batch_acc
            # live metrics
            mean_loss = total_loss / max(total_tok, 1)
            ppl = math.exp(mean_loss)
            mean_acc = total_acc / max(1, pbar.n + 1)
            pbar.set_postfix({"loss": f"{mean_loss:.4f}", "ppl": f"{ppl:.1f}", "acc": f"{mean_acc:.3f}"})
    mean_loss = total_loss / max(total_tok, 1)
    ppl = math.exp(mean_loss)
    acc = total_acc / max(1, len(val_loader))
    return {"val_loss": mean_loss, "val_ppl": ppl, "val_acc": acc}

def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="[train] %(message)s")

    set_seed(int(args.seed))

    # --- Build cfg from preset or defaults -----------------------------------
    if args.preset is not None:
        preset_cfg = get_preset(args.preset)  # dict or LoopConfig-like dict
        if isinstance(preset_cfg, LoopConfig):
            preset_dict = asdict(preset_cfg)
        else:
            preset_dict = dict(preset_cfg)
    else:
        # Reasonable small default (close to train/loop demo)
        preset_dict = asdict(LoopConfig(
            W=16, O=4,
            vocab_size=128, d_model=64, groups=8,
            cheb_deg=6, cheb_laplacian="path_causal",
            skew_rank=8, R_rank=4,
            steps=2, dt=0.5, method="heun",
            tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
            pad_id=0,
            lr=3e-3, weight_decay=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_gauge=True,
            stitch_w=0.1, stitch_use_lowd=True, stitch_use_skl=False,
            scheduler_name="warmup_cosine",
            scheduler_total_steps=200,
            scheduler_warmup_steps=20,
            scheduler_min_lr=3e-4,
            profile=False, profile_nvtx=False, profile_log_mem=False,
            checkpoint_dir="./ckpts", save_every=0, keep_last_k=5, best_metric_name="loss",
        ))

    # Helper to load HF-like config (utils.hf_config or plain JSON)
    def _load_hf_like_config(path: str) -> Dict[str, Any]:
        d = {}
        # Try utils.hf_config first
        if load_hf_config is not None:
            try:
                cfg_hf = load_hf_config(path)
                d = asdict(cfg_hf)
                logging.info(f"[hf] loaded HF config via utils.hf_config from {path}")
                return d
            except Exception as e:
                logging.warning(f"[hf] hf_config loader failed for {path}: {e}")
        # Fallback: plain JSON; allow directory containing config.json
        cfg_path = path
        if os.path.isdir(cfg_path):
            cfg_path = os.path.join(cfg_path, "config.json")
        try:
            with open(cfg_path, "r") as f:
                d = json.load(f)
            logging.info(f"[hf] loaded plain JSON config from {cfg_path}")
        except Exception as e:
            logging.warning(f"[hf] failed to load plain JSON config from {cfg_path}: {e}")
        return d

    # CLI overrides
    overrides = {
        "W": args.W, "O": args.O, "d_model": args.d_model, "groups": args.groups,
        "cheb_deg": args.cheb_deg, "lr": args.lr, "weight_decay": args.weight_decay,
        "device": args.device, "stitch_w": args.stitch_w,
        "scheduler_name": args.scheduler_name, "scheduler_warmup_steps": args.scheduler_warmup_steps,
        "scheduler_total_steps": args.scheduler_total_steps, "scheduler_min_lr": args.scheduler_min_lr,
        "checkpoint_dir": args.checkpoint_dir, "save_every": args.save_every,
        "keep_last_k": args.keep_last_k, "resume_path": args.resume_path,
        "profile": args.profile, "profile_nvtx": args.profile_nvtx, "profile_log_mem": args.profile_log_mem,
        "grads_border_only": args.grads_border_only,
        "debug_align": getattr(args, "debug_align", None),
        "debug_align_every": getattr(args, "debug_align_every", None),
        "debug_state_norm": getattr(args, "debug_state_norm", None),
        "debug_topk": getattr(args, "debug_topk", None),
        "cheb_laplacian": getattr(args, "cheb_laplacian", None),
        "dt": args.dt,
        "method": args.method,
        "state_w": getattr(args, "state_w", None),
        "state_mask": getattr(args, "state_mask", None),
        "state_kind": getattr(args, "state_kind", None),
        "target_bpp": getattr(args, "target_bpp", None),
    }

    # --- Determine base config source and merge order ---
    # - If user passes --hf-config, use that.
    # - Else, if configs/config.json exists, use it as default.
    # - Else, fall back to preset_dict computed above.
    hf_dict = {}
    user_hf_path = args.hf_config
    default_hf_path = os.path.join("configs", "config.json")
    if user_hf_path is not None:
        hf_dict = _load_hf_like_config(user_hf_path)
    elif os.path.exists(default_hf_path):
        logging.info(f"[hf] using default config at {default_hf_path}")
        hf_dict = _load_hf_like_config(default_hf_path)
    else:
        logging.info("[hf] no HF-style config found; using preset base.")

    # Merge order: base (hf if present, else preset) -> CLI overrides
    base = hf_dict if hf_dict else preset_dict
    merged = merge_cfg(base, overrides)

    # Warn if user or base config selected a non-causal Laplacian for AR LM
    if merged.get("cheb_laplacian") == "path":
        logging.warning("[cfg] cheb_laplacian='path' is NON-causal (Dirichlet). Prefer 'path_causal' for autoregressive LM to avoid future leakage.")

    # Ensure compatibility: LoopConfig has no 'max_len' field.
    if merged.get("pos_kind") == "learned":
        logging.warning("[cfg] pos_kind='learned' requested but LoopConfig has no 'max_len'; falling back to 'sinusoidal'.")
        merged["pos_kind"] = "sinusoidal"

    merged = _sanitize_for_loopconfig(merged)
    cfg = LoopConfig(**merged)  # type: ignore[arg-type]

    # Steps: CLI --steps wins; else take scheduler_total_steps (if given); else 400
    steps = args.steps if args.steps is not None else (cfg.scheduler_total_steps or 400)

    # --- Data -----------------------------------------------------------------
    use_npy = (args.train_npy is not None) and (args.val_npy is not None)
    last_batch = None

    if use_npy:
        # Build DataLoaders from pre-packed NPY windows
        train_loader, val_loader, seq_len = build_npy_dataloaders(
            cfg,
            train_path=args.train_npy,
            val_path=args.val_npy,
            micro_batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pin_memory=True,
            drop_last=True,
        )
        # Fix vocab_size/pad_id from meta.json or probe
        _maybe_fix_vocab_from_meta(cfg, args.train_npy)
        # Sensible defaults unless user explicitly opted in via CLI:
        if args.use_dual_rate is None:
            cfg.use_dual_rate = False
            logging.info("[cfg] use_dual_rate not specified; disabling by default for LM training to avoid bpp clamping.")
        if args.grads_border_only is None:
            cfg.grads_border_only = False
            logging.info("[cfg] grads_border_only not specified; disabling by default (CE on all positions).")
        # Defensive assertion
        assert 0 <= int(cfg.pad_id) < int(cfg.vocab_size), f"pad_id {cfg.pad_id} must be within [0, vocab_size={cfg.vocab_size})"
        device = torch.device(cfg.device)
        model = ContinuousLM(cfg).to(device)
        opt = build_optimizer(model, cfg)
        sch = None
        if cfg.scheduler_name:
            name = cfg.scheduler_name.lower()
            total_steps = cfg.scheduler_total_steps if cfg.scheduler_total_steps > 0 else steps
            if name in ("warmup_cosine", "cosine"):
                sch = make_scheduler(
                    name, opt,
                    total_steps=total_steps,
                    warmup_steps=max(0, cfg.scheduler_warmup_steps),
                    base_lr=cfg.lr,
                    min_lr=cfg.scheduler_min_lr,
                    cycles=getattr(cfg, "scheduler_cycles", 0.5),
                )
            elif name in ("warmup_linear", "linear"):
                sch = make_scheduler(
                    name, opt,
                    total_steps=total_steps,
                    warmup_steps=max(0, cfg.scheduler_warmup_steps),
                    base_lr=cfg.lr,
                    min_lr=cfg.scheduler_min_lr,
                )
            elif name == "noam":
                sch = make_scheduler(
                    name, opt,
                    d_model=cfg.d_model,
                    warmup_steps=max(1, cfg.scheduler_warmup_steps or 4000),
                    scale=1.0,
                )
            elif name in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
                sch = make_scheduler(
                    "plateau", opt,
                    factor=getattr(cfg, "plateau_factor", 0.5),
                    patience=getattr(cfg, "plateau_patience", 200),
                    ema_alpha=getattr(cfg, "plateau_ema_alpha", 0.9),
                    threshold=getattr(cfg, "plateau_threshold", 1e-3),
                    minimize=getattr(cfg, "plateau_minimize", True),
                    base_lr=cfg.lr,
                    min_lr=cfg.scheduler_min_lr,
                )
        ckpt_mgr = CheckpointManager(
            cfg.checkpoint_dir,
            keep_last_k=cfg.keep_last_k,
            best_tag=cfg.best_metric_name,
            is_better=(lambda new, best: (best is None) or (new < best)),
        )
        # --- Step tracking for resume ---
        start_step = 0
        if cfg.resume_path:
            try:
                from train.checkpoints import load_checkpoint  # type: ignore
                ckpt = load_checkpoint(cfg.resume_path, model=model, optimizer=opt, scheduler=None, strict_model=False)
                logging.info(f"[ckpt] resumed from {cfg.resume_path} (step={ckpt.get('step')}, epoch={ckpt.get('epoch')})")
                start_step = int(ckpt.get("step", 0) or 0)
                logging.info(f"[ckpt] resume start_step={start_step}")
            except Exception as e:
                logging.warning(f"[ckpt] resume failed: {e}")
        # Advance scheduler to match resume step (kind-aware)
        sched_kind = (cfg.scheduler_name or "").lower() if cfg.scheduler_name else ""
        try:
            if sch is not None and start_step > 0:
                if sched_kind in ("warmup_cosine", "cosine", "warmup_linear", "linear", "noam"):
                    for _ in range(start_step):
                        sch.step()
                elif sched_kind in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
                    logging.info("[sched] ReduceLROnPlateau cannot be fast-forwarded reliably without metrics; leaving scheduler state as-is.")
        except Exception as e:
            logging.warning(f"[sched] failed to advance scheduler to step {start_step}: {e}")

        tm = ThroughputMeter() if ThroughputMeter is not None else None
        device_ctx = torch.cuda.amp.autocast if (cfg.device == "cuda" and hasattr(torch.cuda, "amp")) else nullcontext  # type: ignore

        # ---- AMP setup (autocast, GradScaler) ----
        use_amp = (cfg.device == "cuda" and torch.cuda.is_available())
        try:
            from torch import amp as _amp
        except Exception:
            class _Noop:
                def __init__(self, *a, **k): pass
                def __call__(self, *a, **k):
                    from contextlib import nullcontext
                    return nullcontext()
                def scale(self, x): return x
                def step(self, opt): opt.step()
                def update(self): pass
            autocast = _Noop()
            class GradScaler(_Noop): pass
        else:
            autocast = _amp.autocast
            GradScaler = _amp.GradScaler
        scaler = GradScaler(enabled=use_amp)



        # Optional one-time sanity checks on validation batch
        if args.debug_sanity:
            try:
                sample_batch = next(iter(val_loader))
            except StopIteration:
                sample_batch = next(iter(train_loader))
            # Build a legacy-style view for sanity (tokens/targets) from x/y
            legacy = {
                "tokens": sample_batch["x"].to(device, non_blocking=True),
                "targets": sample_batch["y"].to(device, non_blocking=True),
                "mask": sample_batch["mask"].to(device, non_blocking=True),
            }
            model.eval()
            run_sanity_checks(model, legacy, cfg)
            model.train()



        # --- Train loop over DataLoader ---------------------------------------
        def _train_step_ce(model, opt, batch):
            model.train()
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss, stats = model(batch)  # ContinuousLM returns (loss, stats)

            # Backward with gradient scaling (if AMP enabled)
            scaler.scale(loss).backward()

            # IMPORTANT: unscale before clipping so the clip threshold is applied in true scale
            if use_amp:
                try:
                    scaler.unscale_(opt)
                except Exception:
                    # If unscale is not available (e.g., Noop GradScaler), just continue
                    pass

            # Clip after unscale; 1.0 is conservative (adjust if needed)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step via scaler
            scaler.step(opt)
            scaler.update()

            stats = dict(stats)
            stats["loss"] = float(loss.item())
            return stats

        train_iter = iter(train_loader)
        pbar = tqdm(range(start_step, steps), initial=start_step, total=steps, desc="training", dynamic_ncols=True)
        for s in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Move to device
            for k in ("x", "y", "mask"):
                batch[k] = batch[k].to(device, non_blocking=True)

            if cfg.profile and Timer is not None:
                timer = Timer(sync_cuda=True)
                with (nvtx_range(f"step_{s}") if args.profile_nvtx else torch.autograd.profiler.record_function("train_step")):
                    timer.start()
                    stats = _train_step_ce(model, opt, batch)
                    dt = timer.stop()
                if tm is not None:
                    tok = int(stats.get("tokens", 0))
                    if tok > 0:
                        tm.update(tok, dt)
            else:
                stats = _train_step_ce(model, opt, batch)

            last_batch = batch  # keep last for sanity forward

            # tqdm live metrics (updates every step)
            lr = opt.param_groups[0]["lr"]
            postfix = {
                "loss": f"{stats['loss']:.4f}",
                "acc": f"{stats.get('acc', 0.0):.3f}",
                "bpp": f"{stats.get('bpp', 0.0):.3f}",
                "lr": f"{lr:.2e}",
            }
            # Optional finer-grained losses if the model reports them
            ce_val = stats.get("ce", stats.get("lm_ce"))
            st_val = stats.get("stitch", stats.get("stitch_loss"))
            if ce_val is not None:
                postfix["ce"] = f"{float(ce_val):.4f}"
            if st_val is not None:
                postfix["stitch"] = f"{float(st_val):.4f}"
            ls_val = stats.get("loss_state")
            if ls_val is not None:
                postfix["l_state"] = f"{float(ls_val):.4f}"
            if cfg.profile and tm is not None and tm.time_s > 0:
                postfix["ips"] = f"{tm.ips:.1f}"
            # Show dual-rate target if active
            try:
                if getattr(cfg, "use_dual_rate", False) and hasattr(model, "_sda_target"):
                    if getattr(model, "_dual_use_log", True):
                        bpp_tgt = math.exp(float(model._sda_target))
                    else:
                        bpp_tgt = float(model._sda_target)
                    postfix["bpp_tgt"] = f"{bpp_tgt:.2f}"
            except Exception:
                pass
            pbar.set_postfix(postfix)

            # Checkpoints
            ckpt_mgr.periodic_save(
                model=model, optimizer=opt, scheduler=None, step=s, epoch=0, cfg=asdict(cfg),
                every=cfg.save_every, extra={"tokens_seen": int(stats.get("tokens", 0))}
            )
            ran_val = False
            if int(args.val_every) > 0 and (s > start_step) and ((s % int(args.val_every)) == 0):
                # nested tqdm for validation; resumes outer pbar after it finishes
                val_metrics = evaluate_val_ce_tqdm(model, val_loader, pad_id=int(cfg.pad_id), desc=f"val @ step {s}")
                logging.info(f"[val] step {s} | val_loss={val_metrics['val_loss']:.4f} | val_ppl={val_metrics['val_ppl']:.3f} | val_acc={val_metrics['val_acc']:.3f}")
                metric_name = "val_loss"
                metric_val = float(val_metrics["val_loss"])
                ckpt_mgr.update_best(
                    metric_value=metric_val, model=model, optimizer=opt, scheduler=None,
                    step=s, epoch=0, cfg=asdict(cfg), extra={metric_name: metric_val}
                )
                ran_val = True
            if not ran_val:
                metric_name = cfg.best_metric_name
                metric_val = float(stats.get(metric_name, stats.get("loss", 0.0)))
                if int(args.best_every) <= 1 or (s % int(args.best_every) == 0):
                    ckpt_mgr.update_best(
                        metric_value=metric_val, model=model, optimizer=opt, scheduler=None,
                        step=s, epoch=0, cfg=asdict(cfg), extra={metric_name: metric_val}
                    )

            # Scheduler step: non-plateau every step; plateau only on validation
            if sch is not None:
                if sched_kind in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
                    if ran_val:
                        sch.step(metric_val)
                else:
                    sch.step()

        # End-of-run sanity forward
        model.eval()
        with torch.no_grad():
            ref = last_batch if last_batch is not None else next(iter(val_loader))
            x = ref["x"].to(device, non_blocking=True)
            m = ref.get("mask", None)
            if m is not None:
                m = m.to(device, non_blocking=True)
            out = model.forward_tokens(x, m)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            B, T, V = logits.shape
            print(f"[train] forward sanity: logits {B}x{T}x{V} ✓")

        # Optionally save resolved config in HF style
        if args.hf_save is not None and save_hf_config is not None:
            try:
                out_path = save_hf_config(cfg, args.hf_save)
                logging.info(f"[hf] saved resolved config to: {out_path}")
            except Exception as e:
                logging.warning(f"[hf] failed to save HF config: {e}")
        return
    else:
        # Fallback: toy/text mode (previous behavior)
        texts = load_texts(args.text_file)
        vocab = SimpleVocab.build_from_texts(texts, mode="char", add_unk=False)
        batch, _ = build_batch_from_texts(texts, vocab, cfg.W, cfg.O)
        # Defensive: ensure pad_id and vocab_size are set before model creation
        if getattr(cfg, "pad_id", None) is None:
            cfg.pad_id = 0
        if getattr(cfg, "vocab_size", None) is None:
            cfg.vocab_size = int(vocab.size if 'vocab' in locals() else 128)
        assert 0 <= int(cfg.pad_id) < int(cfg.vocab_size), f"pad_id {cfg.pad_id} must be within [0, vocab_size={cfg.vocab_size})"
        # Sensible defaults for text mode too
        if args.use_dual_rate is None:
            cfg.use_dual_rate = False
            logging.info("[cfg] use_dual_rate not specified; disabling by default (text mode).")
        if args.grads_border_only is None:
            cfg.grads_border_only = False
            logging.info("[cfg] grads_border_only not specified; disabling by default (text mode, CE on all positions).")

    # --- Model / Optim / Scheduler / Ckpt (text fallback) --------------------
    device = torch.device(cfg.device)
    model = ContinuousLM(cfg).to(device)
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)
    opt = build_optimizer(model, cfg)

    if args.debug_sanity:
        model.eval()
        run_sanity_checks(model, batch, cfg)
        model.train()

    sch = None
    if cfg.scheduler_name:
        name = cfg.scheduler_name.lower()
        total_steps = cfg.scheduler_total_steps if cfg.scheduler_total_steps > 0 else steps
        if name in ("warmup_cosine", "cosine"):
            sch = make_scheduler(
                name, opt,
                total_steps=total_steps,
                warmup_steps=max(0, cfg.scheduler_warmup_steps),
                base_lr=cfg.lr,
                min_lr=cfg.scheduler_min_lr,
                cycles=getattr(cfg, "scheduler_cycles", 0.5),
            )
        elif name in ("warmup_linear", "linear"):
            sch = make_scheduler(
                name, opt,
                total_steps=total_steps,
                warmup_steps=max(0, cfg.scheduler_warmup_steps),
                base_lr=cfg.lr,
                min_lr=cfg.scheduler_min_lr,
            )
        elif name == "noam":
            sch = make_scheduler(
                name, opt,
                d_model=cfg.d_model,
                warmup_steps=max(1, cfg.scheduler_warmup_steps or 4000),
                scale=1.0,
            )
        elif name in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
            sch = make_scheduler(
                "plateau", opt,
                factor=getattr(cfg, "plateau_factor", 0.5),
                patience=getattr(cfg, "plateau_patience", 200),
                ema_alpha=getattr(cfg, "plateau_ema_alpha", 0.9),
                threshold=getattr(cfg, "plateau_threshold", 1e-3),
                minimize=getattr(cfg, "plateau_minimize", True),
                base_lr=cfg.lr,
                min_lr=cfg.scheduler_min_lr,
            )

    # Define scheduler kind for later use
    sched_kind = (cfg.scheduler_name or "").lower() if cfg.scheduler_name else ""

    ckpt_mgr = CheckpointManager(
        cfg.checkpoint_dir,
        keep_last_k=cfg.keep_last_k,
        best_tag=cfg.best_metric_name,
        is_better=(lambda new, best: (best is None) or (new < best)),
    )
    # --- Step tracking for resume ---
    start_step = 0
    if cfg.resume_path:
        try:
            from train.checkpoints import load_checkpoint  # type: ignore
            ckpt = load_checkpoint(cfg.resume_path, model=model, optimizer=opt, scheduler=None, strict_model=False)
            logging.info(f"[ckpt] resumed from {cfg.resume_path} (step={ckpt.get('step')}, epoch={ckpt.get('epoch')})")
            start_step = int(ckpt.get("step", 0) or 0)
            logging.info(f"[ckpt] resume start_step={start_step}")
        except Exception as e:
            logging.warning(f"[ckpt] resume failed: {e}")
    # Advance scheduler to match resume step (kind-aware)
    try:
        if sch is not None and start_step > 0:
            if sched_kind in ("warmup_cosine", "cosine", "warmup_linear", "linear", "noam"):
                for _ in range(start_step):
                    sch.step()
            elif sched_kind in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
                logging.info("[sched] ReduceLROnPlateau cannot be fast-forwarded reliably without metrics; leaving scheduler state as-is.")
    except Exception as e:
        logging.warning(f"[sched] failed to advance scheduler to step {start_step}: {e}")

    tm = ThroughputMeter() if ThroughputMeter is not None else None

    pbar_txt = tqdm(range(start_step, steps), initial=start_step, total=steps, desc="training", dynamic_ncols=True)
    for s in pbar_txt:
        if cfg.profile and Timer is not None:
            timer = Timer(sync_cuda=True)
            with (nvtx_range(f"step_{s}") if args.profile_nvtx else torch.autograd.profiler.record_function("train_step")):
                timer.start()
                stats = train_step(model, batch, opt)
                dt = timer.stop()
            if tm is not None:
                tok = int(stats.get("tokens", 0))
                if tok > 0:
                    tm.update(tok, dt)
        else:
            stats = train_step(model, batch, opt)

        lr = opt.param_groups[0]["lr"]
        postfix = {
            "loss": f"{stats['loss']:.4f}",
            "acc": f"{stats.get('acc', 0.0):.3f}",
            "bpp": f"{stats.get('bpp', 0.0):.3f}",
            "lr": f"{lr:.2e}",
        }
        # Optional finer-grained losses if the model reports them
        ce_val = stats.get("ce", stats.get("lm_ce"))
        st_val = stats.get("stitch", stats.get("stitch_loss"))
        if ce_val is not None:
            postfix["ce"] = f"{float(ce_val):.4f}"
        if st_val is not None:
            postfix["stitch"] = f"{float(st_val):.4f}"
        ls_val = stats.get("loss_state")
        if ls_val is not None:
            postfix["l_state"] = f"{float(ls_val):.4f}"
        if cfg.profile and tm is not None and tm.time_s > 0:
            postfix["ips"] = f"{tm.ips:.1f}"
        # Show dual-rate target if active
        try:
            if getattr(cfg, "use_dual_rate", False) and hasattr(model, "_sda_target"):
                if getattr(model, "_dual_use_log", True):
                    bpp_tgt = math.exp(float(model._sda_target))
                else:
                    bpp_tgt = float(model._sda_target)
                postfix["bpp_tgt"] = f"{bpp_tgt:.2f}"
        except Exception:
            pass
        pbar_txt.set_postfix(postfix)

        ckpt_mgr.periodic_save(
            model=model, optimizer=opt, scheduler=None, step=s, epoch=0, cfg=asdict(cfg),
            every=cfg.save_every, extra={"tokens_seen": int(stats.get("tokens", 0))}
        )
        metric_name = cfg.best_metric_name
        metric_val = float(stats.get(metric_name, stats.get("loss", 0.0)))
        if int(args.best_every) <= 1 or (s % int(args.best_every) == 0):
            ckpt_mgr.update_best(
                metric_value=metric_val, model=model, optimizer=opt, scheduler=None,
                step=s, epoch=0, cfg=asdict(cfg), extra={metric_name: metric_val}
            )

        # Scheduler step: non-plateau every step; plateau only on validation
        # For text fallback, always use stats for metric_val
        metric_name = cfg.best_metric_name
        metric_val = float(stats.get(metric_name, stats.get("loss", 0.0)))
        if sch is not None:
            if sched_kind in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
                sch.step(metric_val)
            else:
                sch.step()

    model.eval()
    with torch.no_grad():
        out = model.forward_tokens(batch.get("tokens", batch.get("x")), batch.get("mask", None))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        B, T, V = logits.shape
        print(f"[train] forward sanity: logits {B}x{T}x{V} ✓")

if __name__ == "__main__":
    main()

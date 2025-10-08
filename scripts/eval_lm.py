# scripts/eval_lm.py
# -----------------------------------------------------------------------------
# CLI evaluation entrypoint for Lazy Transformer.
# Loads a checkpoint + preset (or defaults), builds a batch from text lines,
# and reports loss, bits-per-token (bpp), and perplexity. Optional profiling.
#
# Examples:
#   python scripts/eval_lm.py --ckpt ./ckpts/best_val_loss.pt --text-file data.txt
#   python scripts/eval_lm.py --preset base_tiny --num-iters 5
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import asdict
from typing import Optional, List, Dict, Any, Tuple
import argparse
import logging
import os
import sys
import random
import math

# Make repo root importable when running as a script
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

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
    from data.dataloaders import collate_batch as _collate_batch  # type: ignore
except Exception:
    _collate_batch = None  # type: ignore

from train.loop import ContinuousLM, LoopConfig  # type: ignore
from train.checkpoints import load_checkpoint  # type: ignore

# Optional profiling helpers
try:
    from utils.profile import Timer, nvtx_range, record_function as prof_record_function, gpu_mem  # type: ignore
except Exception:
    Timer = None  # type: ignore
    def nvtx_range(_name):  # type: ignore
        from contextlib import nullcontext
        return nullcontext()
    def prof_record_function(_name):  # type: ignore
        from contextlib import nullcontext
        return nullcontext()
    def gpu_mem():  # type: ignore
        return {"allocated": 0.0, "reserved": 0.0}


# ------------------------------ helpers ---------------------------------------

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
        # Tiny default toy set
        return ["hello there", "general kenobi", "hello hello"]
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    return [ln for ln in lines if len(ln) > 0]

def build_batch_from_texts(
    texts: List[str],
    vocab: SimpleVocab,
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    seqs = [vocab.encode(t, mode="char", add_bos=True, add_eos=True) for t in texts]
    if _collate_batch is not None:
        bos_id = getattr(vocab, "bos_id", None)
        eos_id = getattr(vocab, "eos_id", None)
        batch = _collate_batch(seqs, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, max_len=None)
        # Targets default: next-token of tokens (model handles shift)
        batch["targets"] = batch["tokens"].clone()
        return batch
    tokens, mask = pack_batch(seqs, pad_id=pad_id)
    return {"tokens": tokens, "mask": mask, "targets": tokens.clone()}

def merge_cfg(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


# ------------------------------ CLI / main ------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Lazy Transformer LM.")
    # Data
    p.add_argument("--text-file", type=str, default=None, help="Path to a text file (one sample per line). If omitted, uses toy data.")
    # Preset
    p.add_argument("--preset", type=str, default=None, help=f"Preset name from configs.presets. Known: {', '.join(list_presets()) or 'none'}")
    # Checkpoint
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to load. If omitted, evaluates random init.")
    # Device
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Override device.")
    # Overrides (common small set)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--O", type=int, default=None)
    p.add_argument("--d-model", dest="d_model", type=int, default=None)
    p.add_argument("--groups", type=int, default=None)
    # Iterations (repeat forward for timing avg)
    p.add_argument("--num-iters", type=int, default=1, help="Repeat evaluation passes to average timing.")
    # Profiling
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-nvtx", action="store_true")
    p.add_argument("--profile-log-mem", action="store_true")
    # Logging
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Seed
    p.add_argument("--seed", type=int, default=0)
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="[eval] %(message)s")
    set_seed(int(args.seed))

    # --- Build cfg from preset or defaults -----------------------------------
    if args.preset is not None:
        preset_cfg = get_preset(args.preset)
        if isinstance(preset_cfg, LoopConfig):
            preset_dict = asdict(preset_cfg)
        else:
            preset_dict = dict(preset_cfg)
    else:
        preset_dict = asdict(LoopConfig(
            W=16, O=4,
            vocab_size=128, d_model=64, groups=8,
            cheb_deg=6, cheb_laplacian="cycle",
            skew_rank=8, R_rank=4,
            steps=2, dt=0.5, method="heun",
            tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
            pad_id=0,
            lr=3e-3, weight_decay=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_gauge=True,
            stitch_w=0.0, stitch_use_lowd=False, stitch_use_skl=False,
            # profiling flags fed from CLI
        ))

    overrides = {
        "W": args.W, "O": args.O, "d_model": args.d_model, "groups": args.groups,
        "device": args.device,
        "profile": args.profile, "profile_nvtx": args.profile_nvtx, "profile_log_mem": args.profile_log_mem,
    }
    merged = merge_cfg(preset_dict, overrides)
    cfg = LoopConfig(**merged)  # type: ignore[arg-type]

    # --- Data -----------------------------------------------------------------
    texts = load_texts(args.text_file)
    vocab = SimpleVocab.build_from_texts(texts, mode="char", add_unk=False)
    batch = build_batch_from_texts(texts, vocab, pad_id=vocab.pad_id)

    # --- Model & checkpoint ---------------------------------------------------
    device = torch.device(cfg.device)
    model = ContinuousLM(cfg).to(device)
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)

    if args.ckpt:
        try:
            ckpt = load_checkpoint(args.ckpt, model=model, optimizer=None, scheduler=None, map_location=device, strict_model=False)
            logging.info(f"[ckpt] loaded: step={ckpt.get('step')}, epoch={ckpt.get('epoch')}")
        except Exception as e:
            logging.warning(f"[ckpt] load failed: {e}")

    # --- Eval passes ----------------------------------------------------------
    model.eval()
    total_loss = 0.0
    total_bpp = 0.0

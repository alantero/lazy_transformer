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
        x = np.asarray(self.arr[idx], dtype=np.int32)  # 1D [S]
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
    """Stack to [B,S] then shift to tokens/targets; add trivial mask of ones."""
    def _fn(batch):
        x = torch.stack(batch, dim=0)               # [B, S]
        tokens  = x[:, :seq_len-1]                   # [B, S-1]
        targets = x[:, 1:seq_len]                    # [B, S-1]
        mask    = torch.ones_like(tokens, dtype=torch.bool)
        return {"tokens": tokens, "targets": targets, "mask": mask}
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

    train_loader = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=shuffle_train, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch_size, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=collate
    )
    logging.info(f"[data] train {tuple(train_ds.arr.shape)}  val {tuple(val_ds.arr.shape)}  seq_len={seq_len}")
    return train_loader, val_loader, seq_len


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
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    # Stitching / dual rate (shortlist)
    p.add_argument("--stitch-w", dest="stitch_w", type=float, default=None)
    p.add_argument("--use-dual-rate", dest="use_dual_rate", action="store_true")
    p.add_argument("--no-use-dual-rate", dest="use_dual_rate", action="store_false")
    p.set_defaults(use_dual_rate=None)
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
    return p

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
            cheb_deg=6, cheb_laplacian="cycle",
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
    }
    # HF config merge (preset -> HF -> CLI). YAML is optional and not used here.
    hf_dict = {}
    if args.hf_config is not None and load_hf_config is not None:
        try:
            cfg_hf = load_hf_config(args.hf_config)
            hf_dict = asdict(cfg_hf)
        except Exception as e:
            logging.warning(f"[hf] failed to load HF config: {e}")

    merged = merge_cfg(preset_dict, hf_dict)
    merged = merge_cfg(merged, overrides)

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
        if cfg.resume_path:
            try:
                from train.checkpoints import load_checkpoint  # type: ignore
                ckpt = load_checkpoint(cfg.resume_path, model=model, optimizer=opt, scheduler=None, strict_model=False)
                logging.info(f"[ckpt] resumed from {cfg.resume_path} (step={ckpt.get('step')}, epoch={ckpt.get('epoch')})")
            except Exception as e:
                logging.warning(f"[ckpt] resume failed: {e}")

        tm = ThroughputMeter() if ThroughputMeter is not None else None
        device_ctx = torch.cuda.amp.autocast if (cfg.device == "cuda" and hasattr(torch.cuda, "amp")) else nullcontext  # type: ignore

        # --- Train loop over DataLoader ---------------------------------------
        train_iter = iter(train_loader)
        for s in range(steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Move to device
            for k in ("tokens", "targets", "mask"):
                if k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

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

            last_batch = batch  # keep last for sanity forward

            # Logs
            if (s % max(1, int(args.log_interval))) == 0 or s == steps - 1:
                lr = opt.param_groups[0]["lr"]
                msg = f"step {s}/{steps} | loss={stats['loss']:.4f} | acc={stats.get('acc', 0.0):.3f} | bpp={stats.get('bpp', 0.0):.3f} | lr={lr:.2e}"
                if cfg.profile and tm is not None and tm.time_s > 0:
                    msg += f" | ips={tm.ips:.1f}"
                if cfg.profile_log_mem and torch.cuda.is_available():
                    m = gpu_mem()
                    msg += f" | gpuMB={m['allocated']:.1f}/{m['reserved']:.1f}"
                logging.info(msg)

            # Checkpoints
            ckpt_mgr.periodic_save(
                model=model, optimizer=opt, scheduler=None, step=s, epoch=0, cfg=cfg,
                every=cfg.save_every, extra={"tokens_seen": int(stats.get("tokens", 0))}
            )
            metric_name = cfg.best_metric_name
            metric_val = float(stats.get(metric_name, stats.get("loss", 0.0)))
            ckpt_mgr.update_best(
                metric_value=metric_val, model=model, optimizer=opt, scheduler=None,
                step=s, epoch=0, cfg=cfg, extra={metric_name: metric_val}
            )

            if sch is not None:
                sch.step(s, metrics=float(stats.get("loss", 0.0)))

        # End-of-run sanity forward
        model.eval()
        with torch.no_grad():
            ref = last_batch if last_batch is not None else next(iter(val_loader))
            logits, *_ = model.forward_tokens(ref["tokens"], ref.get("mask", None))
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

    # --- Model / Optim / Scheduler / Ckpt (text fallback) --------------------
    device = torch.device(cfg.device)
    model = ContinuousLM(cfg).to(device)
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)
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

    # Resume (optional) — mirror NPY branch behavior
    if cfg.resume_path:
        try:
            from train.checkpoints import load_checkpoint  # type: ignore
            ckpt = load_checkpoint(cfg.resume_path, model=model, optimizer=opt, scheduler=None, strict_model=False)
            logging.info(f"[ckpt] resumed from {cfg.resume_path} (step={ckpt.get('step')}, epoch={ckpt.get('epoch')})")
        except Exception as e:
            logging.warning(f"[ckpt] resume failed: {e}")

    tm = ThroughputMeter() if ThroughputMeter is not None else None

    for s in range(steps):
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

        if (s % max(1, int(args.log_interval))) == 0 or s == steps - 1:
            lr = opt.param_groups[0]["lr"]
            msg = f"step {s}/{steps} | loss={stats['loss']:.4f} | acc={stats.get('acc', 0.0):.3f} | bpp={stats.get('bpp', 0.0):.3f} | lr={lr:.2e}"
            if cfg.profile and tm is not None and tm.time_s > 0:
                msg += f" | ips={tm.ips:.1f}"
            if cfg.profile_log_mem and torch.cuda.is_available():
                m = gpu_mem()
                msg += f" | gpuMB={m['allocated']:.1f}/{m['reserved']:.1f}"
            logging.info(msg)

        ckpt_mgr.periodic_save(
            model=model, optimizer=opt, scheduler=None, step=s, epoch=0, cfg=cfg,
            every=cfg.save_every, extra={"tokens_seen": int(stats.get("tokens", 0))}
        )
        metric_name = cfg.best_metric_name
        metric_val = float(stats.get(metric_name, stats.get("loss", 0.0)))
        ckpt_mgr.update_best(
            metric_value=metric_val, model=model, optimizer=opt, scheduler=None,
            step=s, epoch=0, cfg=cfg, extra={metric_name: metric_val}
        )

        if sch is not None:
            sch.step(s, metrics=float(stats.get("loss", 0.0)))

    model.eval()
    with torch.no_grad():
        logits, *_ = model.forward_tokens(batch["tokens"], batch.get("mask", None))
        B, T, V = logits.shape
        print(f"[train] forward sanity: logits {B}x{T}x{V} ✓")

if __name__ == "__main__":
    main()

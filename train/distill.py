# train/distill.py
# KD fine-tuning loop (Phase 10.1): border-only distillation (collars) with optional low-D logits projection.
# - Student = ContinuousLM (from train/loop.py)
# - Teacher = frozen model (same arch) loaded from ckpt or cloned
# - KD only on window collars (first/last O tokens of each window), CE on all (masked by pad)
# - Optional low-D projection of logits (V→dKD) shared by student & teacher for cheaper KD
#
# This script is self-contained for training logic; it reuses modules from the repo.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ safe imports ----------------------------------
# Make repo root importable when running as a script: python train/distill.py
if __package__ in (None, ""):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Model & data
from train.loop import ContinuousLM, LoopConfig  # student/teacher model
from data.tok_embed import SimpleVocab, pack_batch  

# Windows for collar mask construction
from utils.windows import slice_windows  # we only use window geometry
# Loss (task CE base)
from losses.task_loss import sequence_cross_entropy, shift_for_next_token


Tensor = torch.Tensor


# ------------------------------ configuration ---------------------------------

@dataclass
class DistillConfig:
    # Windows (must match the model’s)
    W: int = 256
    O: int = 32

    # KD
    alpha: float = 0.5            # weight of KD term
    temperature: float = 2.0      # temperature for KD
    proj_dim: Optional[int] = 128 # if set, use V→proj_dim projection before KD (low-D)
    proj_seed: int = 0
    collar_only: bool = True      # KD only on collars (first/last O of each window)
    # Teacher
    teacher_ckpt: Optional[str] = None   # path to a .pt/.pth with state_dict (optional)
    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.0
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pad_id: int = 0


# ------------------------------ KD utilities ----------------------------------

def build_collar_mask(T: int, W: int, O: int, device: torch.device) -> Tensor:
    """
    Return [T] bool mask that is True on collar positions (within O of any window boundary)
    for windows of size W with hop S=W-O (OLA). Handles tail-clipping.
    """
    if O <= 0 or W <= 2 * O:
        return torch.zeros(T, dtype=torch.bool, device=device)
    S = W - O
    m = torch.zeros(T, dtype=torch.bool, device=device)
    s = 0
    while s < T:
        e = s + W
        # left collar [s, s+O)
        m[s:min(s + O, T)] = True
        # right collar [e-O, e)
        if s + W - O < T:
            m[max(s + W - O, 0):min(e, T)] = True
        s += S
    return m


class LogitsProjector(nn.Module):
    """
    Fixed (non-trainable) projection for low-D KD: V → dKD using a random Gaussian matrix.
    Shared by student & teacher. Uses Xavier-like scaling.
    """
    def __init__(self, vocab_size: int, proj_dim: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        W = torch.empty(vocab_size, proj_dim, dtype=torch.float32)
        nn.init.xavier_normal_(W, gain=1.0)
        with torch.no_grad():
            W = W / math.sqrt(float(vocab_size))
        self.register_buffer("P", W, persistent=True)  # [V, d]

    def forward(self, logits: Tensor) -> Tensor:
        # logits: [B,T,V] → [B,T,d]
        return logits @ self.P  # [B,T,d]


def kd_kl_div(student_logits: Tensor,
              teacher_logits: Tensor,
              *,
              temperature: float = 2.0,
              mask: Optional[Tensor] = None) -> Tensor:
    """
    KL( teacher || student ) at temperature T, averaged over masked positions.
    Returns a scalar Tensor (mean over positions).
    """
    T = float(temperature)
    s_logp = F.log_softmax(student_logits / T, dim=-1)  # [B,T,C]
    t_logp = F.log_softmax(teacher_logits / T, dim=-1)
    t_p = t_logp.exp()
    # Elementwise KL over class dim
    kl_map = (t_p * (t_logp - s_logp)).sum(dim=-1)      # [B,T]
    # Standard T^2 factor for KD
    kl_map = (T * T) * kl_map
    if mask is not None:
        m = mask.to(dtype=kl_map.dtype, device=kl_map.device)
        denom = m.sum().clamp_min(1.0)
        return (kl_map * m).sum() / denom
    return kl_map.mean()


# ---------------------------- distillation runner -----------------------------

class Distiller(nn.Module):
    """
    Wraps student + (frozen) teacher, computes CE + KD (collar-only) loss.
    """
    def __init__(self, student: ContinuousLM, teacher: ContinuousLM, cfg: DistillConfig):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.cfg = cfg

        # Optional low-D projector
        self.projector: Optional[LogitsProjector] = None
        if cfg.proj_dim is not None:
            # Get vocab_size from student's head (or cfg)
            V = student.cfg.vocab_size
            self.projector = LogitsProjector(V, cfg.proj_dim, cfg.proj_seed)

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        tokens: Tensor = batch["tokens"]
        mask: Optional[Tensor] = batch.get("mask", None)
        targets: Optional[Tensor] = batch.get("targets", None)
        if targets is None:
            targets = tokens

        # --- Student forward (full) ---
        logits_s_full = self.student.forward_tokens(tokens, mask)  # [B,T,V]

        # Shift for next-token CE
        logits_s, targets_s, mask_s = shift_for_next_token(logits_s_full, targets, mask=mask)
        loss_ce, stats = sequence_cross_entropy(
            logits_s, targets_s, mask=mask_s, ignore_index=self.student.cfg.pad_id,
            label_smoothing=0.0, reduction="mean"
        )

        # --- Teacher forward (no-grad) ---
        with torch.no_grad():
            logits_t_full = self.teacher.forward_tokens(tokens, mask)  # [B,T,V]

        # --- Collar mask (KD region) ---
        B, T, V = logits_s_full.shape
        device = logits_s_full.device
        collar_1d = build_collar_mask(T, self.student.cfg.W, self.student.cfg.O, device)  # [T]
        mask_kd = collar_1d.unsqueeze(0).expand(B, T)  # [B,T]
        if mask is not None:
            mask_kd = mask_kd & mask.bool()

        # Match student/teacher to next-token alignment for KD as well
        logits_s_kd, _, mask_s_kd = shift_for_next_token(logits_s_full, targets, mask=mask_kd)
        logits_t_kd, _, _ = shift_for_next_token(logits_t_full, targets, mask=mask_kd)

        # Optional low-D projection
        if self.projector is not None:
            logits_s_kd = self.projector(logits_s_kd)   # [B,T,d]
            logits_t_kd = self.projector(logits_t_kd)   # [B,T,d]

        # --- KD loss on collars only ---
        loss_kd = kd_kl_div(
            logits_s_kd, logits_t_kd,
            temperature=self.cfg.temperature,
            mask=mask_s_kd
        )

        total = loss_ce + float(self.cfg.alpha) * loss_kd
        stats.update({
            "loss_ce": float(loss_ce.detach().item()),
            "loss_kd": float(loss_kd.detach().item()),
            "loss": float(total.detach().item()),
            "kd_alpha": float(self.cfg.alpha),
            "kd_T": float(self.cfg.temperature),
            "kd_mask_frac": float(mask_s_kd.float().mean().item()),
        })
        return total, stats


# --------------------------------- training API --------------------------------

def build_optimizer(model: nn.Module, cfg: DistillConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_step(distiller: Distiller, batch: Dict[str, Tensor], optim: torch.optim.Optimizer) -> Dict[str, float]:
    distiller.train()
    optim.zero_grad(set_to_none=True)
    loss, stats = distiller(batch)
    loss.backward()
    optim.step()
    return stats


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="[distill] %(message)s")
    torch.manual_seed(0)

    # Tiny toy data (char-level) to sanity check the pipeline
    texts = ["hello there", "general kenobi", "hello hello"]
    vocab = SimpleVocab.build_from_texts(texts, mode="char", add_unk=False)
    seqs = [vocab.encode(t, mode="char", add_bos=True, add_eos=True) for t in texts]
    tokens, mask = pack_batch(seqs, pad_id=vocab.pad_id)
    batch = {"tokens": tokens, "mask": mask, "targets": tokens.clone()}

    # Shared window/model config
    loop_cfg = LoopConfig(
        W=16, O=4,
        vocab_size=vocab.size, d_model=64, groups=8,
        cheb_deg=6, cheb_laplacian="cycle",
        skew_rank=8, R_rank=4,
        steps=2, dt=0.5, method="heun",
        tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
        pad_id=vocab.pad_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_gauge=True,
    )
    device = torch.device(loop_cfg.device)

    # Student & teacher
    student = ContinuousLM(loop_cfg).to(device)
    teacher = ContinuousLM(loop_cfg).to(device)

    # (Optional) load teacher checkpoint
    # If cfg.teacher_ckpt is provided, you would do:
    #   state = torch.load(cfg.teacher_ckpt, map_location=device)
    #   teacher.load_state_dict(state)
    # For this sanity test, we clone student weights into teacher and freeze:
    teacher.load_state_dict(student.state_dict())

    # Move batch to device
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)

    # Distillation config (low-D KD enabled for speed)
    cfg = DistillConfig(
        W=loop_cfg.W, O=loop_cfg.O,
        alpha=0.7, temperature=2.0, proj_dim=64, proj_seed=0,
        collar_only=True, teacher_ckpt=None,
        lr=3e-3, weight_decay=0.0,
        device=loop_cfg.device, pad_id=vocab.pad_id,
    )

    distiller = Distiller(student, teacher, cfg).to(device)
    opt = build_optimizer(distiller.student, cfg)

    # Train a few steps on a single batch (sanity)
    steps = 30
    for s in range(steps):
        stats = train_step(distiller, batch, opt)
        if (s % 5) == 0 or s == steps - 1:
            logging.info(f"step {s:02d} | loss={stats['loss']:.4f} | ce={stats['loss_ce']:.4f} | kd={stats['loss_kd']:.4f} | kdmask={stats['kd_mask_frac']:.3f}")

    # Forward-only check
    distiller.eval()
    with torch.no_grad():
        out = distiller(batch)
        print("[distill] forward OK ✓")

# losses/task_loss.py
# Sequence losses for language/token tasks (PyTorch-only).
# - Masked cross-entropy with ignore_index + label smoothing
# - Optional knowledge distillation (teacher logits, learnable α / T supported)
# - Stats: tokens, nll, ce, acc@1, acc@5, optional ΔBKM (if states provided)
# - Utilities: next-token shift, accuracy helpers, perplexity

from __future__ import annotations
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

import math

__all__ = [
    "shift_for_next_token",
    "sequence_cross_entropy",
    "sequence_ce_with_distillation",
    "perplexity",
    "token_accuracy",
    "sequence_cross_entropy_autoregressive",
]


# ------------------------------ small helpers --------------------------------

def shift_for_next_token(
    logits: Tensor, targets: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Align (logits, targets, mask) for next-token prediction:
      logits[:, :-1] vs targets[:, 1:].
    Shapes:
      logits:  [B, T, V]
      targets: [B, T] (int64)
      mask:    [B, T] (bool) or None
    Returns:
      logits_s, targets_s, mask_s with T' = T-1
    """
    if logits.dim() != 3 or targets.dim() != 2:
        raise ValueError("logits must be [B,T,V] and targets [B,T].")
    B, T, V = logits.shape
    if targets.shape[:2] != (B, T):
        raise ValueError("targets must match logits on (B,T).")
    logits_s = logits[:, :-1, :]          # [B, T-1, V]
    targets_s = targets[:, 1:]            # [B, T-1]
    mask_s = None if mask is None else mask[:, 1:].to(torch.bool)
    return logits_s, targets_s, mask_s


def _flatten_for_loss(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Flatten time & batch: [B,T,V] → [B*T,V], [B*T].
    """
    if logits.dim() != 3 or targets.dim() != 2:
        raise ValueError("logits must be [B,T,V] and targets [B,T].")
    B, T, V = logits.shape
    return logits.reshape(B * T, V), targets.reshape(B * T)



def _valid_mask_like(targets: Tensor, mask: Optional[Tensor], ignore_index: Optional[int]) -> Tensor:
    """Build a boolean mask over [B,T] for valid tokens."""
    if mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = mask.to(torch.bool)
    if ignore_index is not None:
        valid = valid & (targets != ignore_index)
    return valid


# Collar mask helper
def _apply_collar_time_mask(mask: Optional[Tensor], T: int, left: int = 0, right: int = 0, device=None) -> Optional[Tensor]:
    """
    Optionally zero-out (ignore) the first `left` and last `right` time positions for CE.
    If `mask` is given, it is AND-ed with the collar mask.
    Shapes:
      mask: [B, T] or None
    Returns:
      [B, T] or None
    """
    if (left <= 0) and (right <= 0):
        return mask
    if device is None and mask is not None:
        device = mask.device
    B = None if mask is None else mask.shape[0]
    collar = torch.ones((B if B is not None else 1, T), dtype=torch.bool, device=device)
    if left > 0:
        collar[:, :min(left, T)] = False
    if right > 0:
        collar[:, max(T - right, 0):] = False
    if mask is None:
        return collar
    return mask.to(torch.bool) & collar
# ------------------------------- convenience ---------------------------------
# ----------------------------------- __main__ ---------------------------------


# ------------------------------- main losses ---------------------------------

def sequence_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",          # 'mean' (token-average), 'sum', or 'none'
    # extras (optional):
    h_prev: Optional[Tensor] = None,
    h_next: Optional[Tensor] = None,
    topk: int = 5,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Masked sequence cross-entropy with optional label smoothing.

    Args:
      logits: [B,T,V] (unnormalized)
      targets:[B,T]   (int64)
      mask:   [B,T] bool (1=valid, 0=ignore). Combined with ignore_index if set.
      ignore_index: label to ignore in loss/accuracy (e.g., pad_id)
      label_smoothing: ε in [0,1). CE = (1-ε)*CE_hard + ε*CE_uniform
      reduction: 'mean' averages over *valid* tokens only.
      h_prev, h_next: optional states to report ΔBKM = mean((h_next - h_prev)^2)
      topk: report acc@k in stats (acc is acc@1)

    Returns:
      loss, stats dict (tokens, nll, ce, acc, acc@5, [bkm])
    """
    if label_smoothing < 0 or label_smoothing >= 1:
        raise ValueError("label_smoothing must be in [0,1).")

    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)           # [B,T,V]

    # Valid mask
    valid = _valid_mask_like(targets, mask, ignore_index)

    if valid.any():
        lp_flat, tgt_flat = _flatten_for_loss(log_probs, targets)
        m_flat = valid.reshape(-1)

        lp_flat = lp_flat[m_flat]            # [N,V]
        tgt_flat = tgt_flat[m_flat]          # [N]
        N = lp_flat.size(0)
    else:
        zero = logits.sum() * 0.0
        stats: Dict[str, float] = {"tokens": 0.0, "nll": 0.0, "ce": 0.0, "acc": 0.0, "acc@5": 0.0}
        # Optional ΔBKM stat (0.0 if not computable)
        if (h_prev is not None) and (h_next is not None):
            with torch.no_grad():
                stats["bkm"] = float((h_next - h_prev).pow(2).mean().item())
        return zero, stats

    # Hard CE term (negative log-likelihood)
    nll = F.nll_loss(lp_flat, tgt_flat, reduction="none")   # [N]

    if label_smoothing > 0.0:
        # Uniform CE term (same for all classes): - mean(log_probs)
        u = -lp_flat.mean(dim=-1)                           # [N]
        ce_flat = (1.0 - label_smoothing) * nll + label_smoothing * u
    else:
        ce_flat = nll

    # Reduction on valid tokens
    if reduction == "mean":
        loss = ce_flat.mean()
    elif reduction == "sum":
        loss = ce_flat.sum()
    elif reduction == "none":
        ce_full = torch.zeros(B * T, device=logits.device, dtype=ce_flat.dtype)
        ce_full[m_flat] = ce_flat
        loss = ce_full.view(B, T)
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'.")

    # Accuracies on valid tokens (top-1 + top-k)
    with torch.no_grad():
        pred1 = log_probs.argmax(dim=-1)            # [B,T]
        correct1 = (pred1[valid] == targets[valid]).sum().item()
        total = int(valid.sum().item())
        acc1 = (correct1 / max(total, 1)) if total > 0 else 0.0

        k = max(1, min(int(topk), V))
        if k > 1:
            topk_idx = log_probs.topk(k, dim=-1).indices                    # [B,T,k]
            matchk = (topk_idx == targets.unsqueeze(-1)).any(dim=-1)        # [B,T]
            acck = (matchk[valid].sum().item() / max(total, 1)) if total > 0 else 0.0
        else:
            acck = acc1

        stats = {
            "tokens": float(total),
            "nll": float(nll.mean().item()),
            "ce": float(ce_flat.mean().item()),
            "acc": float(acc1),
            f"acc@{k}": float(acck),
        }

        # Optional ΔBKM stat
        if (h_prev is not None) and (h_next is not None):
            stats["bkm"] = float((h_next - h_prev).pow(2).mean().item())

    return loss, stats


def sequence_ce_with_distillation(
    logits_student: Tensor,
    targets: Tensor,
    *,
    teacher_logits: Optional[Tensor] = None,
    alpha: Union[float, Tensor] = 0.0,      # blend: (1-α)*CE_hard + α*T^2*KL(student||teacher)
    temperature: Union[float, Tensor] = 1.0,
    mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    # extras:
    h_prev: Optional[Tensor] = None,
    h_next: Optional[Tensor] = None,
    topk: int = 5,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Cross-entropy with optional knowledge distillation (teacher logits).
    - `alpha` and `temperature` can be floats or 0-D Tensors/nn.Parameters (learnable).
    - Distillation term is T^2 * KL(softmax(S/T) || softmax(Teacher/T)).
    """
    # Hard CE part (+ stats incl. acc@k and optional bkm)
    ce_hard, stats = sequence_cross_entropy(
        logits_student, targets,
        mask=mask, ignore_index=ignore_index,
        label_smoothing=label_smoothing, reduction=reduction,
        h_prev=h_prev, h_next=h_next, topk=topk,
    )

    if teacher_logits is None:
        return ce_hard, stats

    # Treat α and T as tensors (allow gradients if provided as nn.Parameter)
    def _to_scalar_tensor(x: Union[float, Tensor]) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(device=logits_student.device, dtype=logits_student.dtype)
        return torch.tensor(float(x), device=logits_student.device, dtype=logits_student.dtype)

    alphaT = _to_scalar_tensor(alpha)
    TT = torch.clamp(_to_scalar_tensor(temperature), min=1e-8)

    # Build valid mask
    B, Tlen, V = logits_student.shape
    valid = _valid_mask_like(targets, mask, ignore_index)
    if not valid.any():
        # No valid tokens → just return CE
        return ce_hard, stats

    # Distillation KL (computed unconditionally to preserve gradients wrt α/T)
    s = logits_student / TT
    t = teacher_logits / TT
    log_p = F.log_softmax(s, dim=-1)     # [B,T,V]
    q = F.softmax(t, dim=-1)

    log_p_f, _ = _flatten_for_loss(log_p, targets)
    q_f, _ = _flatten_for_loss(q, targets)
    m_f = valid.reshape(-1)
    log_p_f = log_p_f[m_f]                # [N,V]
    q_f = q_f[m_f]                        # [N,V]

    kl_flat = F.kl_div(log_p_f, q_f, log_target=False, reduction="none").sum(dim=-1)  # [N]
    if reduction not in {"mean", "sum"}:
        raise ValueError("Distillation supports reduction 'mean' or 'sum' only.")
    kl = kl_flat.mean() if reduction == "mean" else kl_flat.sum()

    loss = (1.0 - alphaT) * ce_hard + alphaT * (TT * TT) * kl

    # Update stats
    stats = dict(stats)
    stats["kl"] = float(kl.detach().item())
    stats["alpha"] = float(alphaT.detach().item())
    stats["T"] = float(TT.detach().item())
    return loss, stats


# ------------------------------- convenience ---------------------------------

def perplexity(loss_mean: float) -> float:
    """Convert mean CE (nat) to perplexity (exp)."""
    return float(torch.exp(torch.tensor(loss_mean)).item())


def token_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    k: int = 1,
) -> float:
    """Top-k accuracy over valid tokens."""
    if logits.dim() != 3:
        raise ValueError("logits must be [B,T,V].")
    B, T, V = logits.shape
    k = max(1, min(int(k), V))

    if mask is not None:
        valid = mask.to(torch.bool)
    else:
        valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid = valid & (targets != ignore_index)

    total = int(valid.sum().item())
    if total == 0:
        return 0.0

    if k == 1:
        pred = logits.argmax(dim=-1)
        correct = (pred[valid] == targets[valid]).sum().item()
        return float(correct / total)
    else:
        topk_idx = logits.topk(k, dim=-1).indices
        matchk = (topk_idx == targets.unsqueeze(-1)).any(dim=-1)
        correct = matchk[valid].sum().item()
        return float(correct / total)


# ------------------------------- autoregressive CE wrapper --------------------

def sequence_cross_entropy_autoregressive(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    topk: int = 5,
    collar: Optional[Tuple[int, int]] = None,
    bpp: bool = True,
    h_prev: Optional[Tensor] = None,
    h_next: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Strict autoregressive CE:
      - Shifts to (logits[:, :-1], targets[:, 1:]) internally.
      - Applies optional time-collar on the CE mask (not on the forward).
      - Reports CE, acc, acc@k, tokens, and (optionally) bits-per-token.
    """
    if logits.dim() != 3 or targets.dim() != 2:
        raise ValueError("logits must be [B,T,V] and targets [B,T].")
    B, T, V = logits.shape
    if targets.shape[:2] != (B, T):
        raise ValueError("targets must match logits on (B,T).")
    # shift
    logits_s, targets_s, mask_s = shift_for_next_token(logits, targets, mask=mask)
    # optional collar on CE only
    if collar is not None:
        left, right = int(collar[0]), int(collar[1])
        mask_s = _apply_collar_time_mask(mask_s, T - 1, left=left, right=right, device=(mask_s.device if mask_s is not None else logits.device))
    # delegate to base CE
    loss, stats = sequence_cross_entropy(
        logits_s, targets_s,
        mask=mask_s, ignore_index=ignore_index,
        label_smoothing=label_smoothing, reduction=reduction,
        h_prev=h_prev, h_next=h_next, topk=topk,
    )
    if bpp:
        # CE is in nats; convert to bits/token
        try:
            ce_mean = float(stats["ce"])
            stats["bpp"] = ce_mean / math.log(2)
        except Exception:
            pass
    return loss, stats


# ----------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V = 3, 7, 19
    pad_id = 0

    # Random logits/targets with some padding
    logits = torch.randn(B, T, V)
    targets = torch.randint(1, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    # introduce padding on last positions
    mask[:, -1] = 0
    # set targets at masked positions to pad_id (⚠️ index with a single boolean mask)
    targets[~mask] = pad_id

    # 1) Plain CE vs F.cross_entropy baseline (no smoothing)
    loss, st = sequence_cross_entropy(logits, targets, mask=mask, ignore_index=pad_id, label_smoothing=0.0)
    # Baseline:
    base = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )
    denom = mask.sum().item()  # valid tokens
    base_mean = base / max(denom, 1)
    print(f"[task_loss] CE mean={loss.item():.6f} vs baseline={base_mean.item():.6f}")
    assert abs(loss.item() - base_mean.item()) < 1e-6
    print(f"[task_loss] acc={st['acc']:.3f}, acc@5={st.get('acc@5', 0.0):.3f}")

    # 2) With label smoothing
    ls = 0.1
    loss_ls, st_ls = sequence_cross_entropy(logits, targets, mask=mask, ignore_index=pad_id, label_smoothing=ls, topk=5)
    print(f"[task_loss] CE (ls={ls}) mean={loss_ls.item():.6f}, acc@1={st_ls['acc']:.3f}, acc@5={st_ls['acc@5']:.3f}")

    # 3) Distillation term (α,T as floats)
    teacher = logits + 0.1 * torch.randn_like(logits)
    loss_kd, st_kd = sequence_ce_with_distillation(
        logits, targets, teacher_logits=teacher, alpha=0.5, temperature=2.0,
        mask=mask, ignore_index=pad_id, label_smoothing=0.0
    )
    print(f"[task_loss] CE+KD mean={loss_kd.item():.6f}, KL={st_kd['kl']:.6f}, α={st_kd['alpha']:.2f}, T={st_kd['T']:.2f}")

    # 4) Distillation with learnable α and T (demonstration; grads exist)
    alpha_param = nn.Parameter(torch.tensor(0.3))
    T_param = nn.Parameter(torch.tensor(1.5))
    loss_kd2, st_kd2 = sequence_ce_with_distillation(
        logits, targets, teacher_logits=teacher, alpha=alpha_param, temperature=T_param,
        mask=mask, ignore_index=pad_id
    )
    # Backward to ensure graph is valid
    loss_kd2.backward(retain_graph=True)
    print(f"[task_loss] CE+KD (learnable α,T) = {loss_kd2.item():.6f}")

    # 5) Next-token shift utility sanity
    logits_s, targets_s, mask_s = shift_for_next_token(logits, targets, mask=mask)
    assert logits_s.shape[1] == T - 1 and targets_s.shape[1] == T - 1
    print("[task_loss] next-token shift shapes ok")

    # 6) token_accuracy helper (top-1 / top-5)
    acc1 = token_accuracy(logits, targets, mask=mask, ignore_index=pad_id, k=1)
    acc5 = token_accuracy(logits, targets, mask=mask, ignore_index=pad_id, k=5)
    print(f"[task_loss] token_accuracy: top1={acc1:.3f}, top5={acc5:.3f}")

    print("[task_loss] All good ✓")

    # 7) Autoregressive wrapper with collar and bpp
    loss_ar, st_ar = sequence_cross_entropy_autoregressive(
        logits, targets, mask=mask, ignore_index=pad_id, label_smoothing=0.0, topk=5, collar=(1, 1)
    )
    print(f"[task_loss] AR CE mean={loss_ar.item():.6f}, bpp={st_ar.get('bpp', 0.0):.3f}, acc@1={st_ar['acc']:.3f}")


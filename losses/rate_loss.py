# losses/rate_loss.py
# Rate-style auxiliary losses (PyTorch-only, no repo-global deps).
# - Bits-per-token (from NLL or from entropy)
# - Budget penalties (hinge / L2 / Lagrange-form)
# - Optional gate sparsity via Bernoulli KL to a prior
#
# Typical use:
#   loss, stats = rate_loss(
#       logits, targets=targets, mask=mask,
#       target_bpp=1.5, budget_mode="hinge", budget_coeff=0.1,
#       gates=gate_tensor, gate_prior=0.1, gate_coeff=1e-4, ignore_index=pad_id
#   )

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
_Budget = Literal["hinge", "l2", "lagrange"]


# ------------------------------- small utilities -------------------------------

def _masked_mean(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    """
    Mean over positions with optional [B,T] mask (1=keep, 0=ignore).
    x can be [B,T] or [B,T,*] (we average over all dims).
    """
    if mask is None:
        return x.mean()
    # Expand mask to x's shape (ones on trailing dims)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    m = mask.to(dtype=x.dtype, device=x.device)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def _nats_to_bits(nats: Tensor) -> Tensor:
    return nats / math.log(2.0)


def _entropy_from_logits(logits: Tensor, *, temperature: float = 1.0) -> Tensor:
    """
    Per-token entropy in NATs. logits [B,T,V] → H [B,T].
    """
    z = logits / float(temperature)
    logp = F.log_softmax(z, dim=-1)          # [B,T,V]
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)              # [B,T]
    return H


def _nll_per_token(logits: Tensor, targets: Tensor, *, ignore_index: Optional[int] = None) -> Tensor:
    """
    Per-token NLL in NATs. logits [B,T,V], targets [B,T] -> nll [B,T].
    `ignore_index` positions get nll=0 here (you should mask them out in the mean).
    """
    logp = F.log_softmax(logits, dim=-1)                       # [B,T,V]
    tgt = targets.long()
    if ignore_index is not None:
        mask = (tgt != int(ignore_index))
        # For ignored positions, set a dummy index (0) but zero out later
        tgt = torch.where(mask, tgt, torch.zeros_like(tgt))
    else:
        mask = torch.ones_like(tgt, dtype=torch.bool)

    # gather log-prob of the target class
    lp = logp.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)  # [B,T]
    nll = -lp
    # Zero out ignored entries (so masked mean can be used safely)
    nll = nll * mask.to(dtype=nll.dtype)
    return nll


def bernoulli_kl(p: Tensor, q: float, *, eps: float = 1e-8) -> Tensor:
    """
    Elementwise KL( Bernoulli(p) || Bernoulli(q) ) in NATs.
    p: arbitrary shape (values in [0,1]); q: scalar prior in (0,1).
    Returns tensor same shape as p.
    """
    p = p.clamp(eps, 1.0 - eps)
    q = float(max(min(q, 1.0 - eps), eps))
    return p * (torch.log(p) - math.log(q)) + (1.0 - p) * (torch.log(1.0 - p) - math.log(1.0 - q))


# ----------------------------------- public API --------------------------------

@dataclass
class RateConfig:
    # How to measure bits-per-token
    use_entropy: bool = False          # if True: use H(p) (uncertainty). If False: use NLL(targets).
    temperature: float = 1.0

    # Budget penalty on average bits-per-token (bpp)
    target_bpp: Optional[float] = None
    budget_mode: _Budget = "hinge"     # 'hinge' | 'l2' | 'lagrange'
    budget_coeff: float = 0.0          # multiplier for hinge/L2; ignored for 'lagrange'
    lambda_rate: float = 0.0           # Lagrange multiplier (only for 'lagrange')

    # Optional gate sparsity via Bernoulli KL to prior
    gate_prior: Optional[float] = None # e.g., 0.1 → encourage ~10% open
    gate_coeff: float = 0.0            # scaling for gate KL (in bits if want → multiply by 1/ln2 externally)

    # Misc
    ignore_index: Optional[int] = None # for NLL path
    name: str = "rate"


def bits_per_token(
    logits: Tensor,
    *,
    targets: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    use_entropy: bool = False,
    temperature: float = 1.0,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute average bits-per-token (scalar Tensor) and stats.
    If `use_entropy=False` and `targets` provided → use NLL(targets).
    Else → use entropy H(p). Mask excludes positions from the average.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,T,V], got {tuple(logits.shape)}")
    B, T, V = logits.shape
    stats: Dict[str, float] = {}

    if (not use_entropy) and (targets is not None):
        nll = _nll_per_token(logits, targets, ignore_index=ignore_index)  # [B,T] nats
        bpp_map = _nats_to_bits(nll)
        bpp = _masked_mean(bpp_map, mask)
        stats["bpp_from"] = 0.0  # 0 → NLL
    else:
        H = _entropy_from_logits(logits, temperature=temperature)         # [B,T] nats
        bpp_map = _nats_to_bits(H)
        bpp = _masked_mean(bpp_map, mask)
        stats["bpp_from"] = 1.0  # 1 → entropy

    stats["bpp"] = float(bpp.detach().item())
    stats["bpp_max"] = math.log2(float(V))
    return bpp, stats


def rate_budget_penalty(
    bpp: Tensor,
    *,
    target_bpp: Optional[float],
    mode: _Budget = "hinge",
    coeff: float = 0.0,
    lambda_rate: float = 0.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Penalty encouraging average bpp <= target.
      - 'hinge':   coeff * max(0, bpp - target)
      - 'l2':      coeff * (max(0, bpp - target))^2
      - 'lagrange':lambda_rate * (bpp - target)
    If target_bpp is None → zero penalty.
    Returns (loss, stats).
    """
    stats: Dict[str, float] = {}
    if target_bpp is None:
        return torch.zeros_like(bpp), stats

    gap = bpp - float(target_bpp)
    if mode == "hinge":
        pen = coeff * torch.clamp(gap, min=0.0)
    elif mode == "l2":
        pen = coeff * torch.clamp(gap, min=0.0).pow(2)
    elif mode == "lagrange":
        pen = lambda_rate * gap
        stats["lambda_rate"] = float(lambda_rate)
    else:
        raise ValueError("mode must be one of {'hinge','l2','lagrange'}.")

    stats["bpp_gap"] = float(gap.detach().item())
    stats["rate_pen"] = float(pen.detach().item())
    return pen, stats


def gate_kl_penalty(
    gates: Optional[Tensor],
    *,
    prior: Optional[float],
    coeff: float = 0.0,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    KL( Bernoulli(g) || Bernoulli(prior) ) averaged over tokens/channels, scaled by coeff.
    gates: [B,T,D] or [B,T,G] or [B,T,1]; values in [0,1].
    """
    stats: Dict[str, float] = {}
    if gates is None or prior is None or coeff == 0.0:
        return torch.zeros((), device=gates.device if isinstance(gates, torch.Tensor) else "cpu"), stats  # type: ignore[return-value]

    kl_map = bernoulli_kl(gates, prior)  # same shape as gates (nats)
    if mask is not None:
        while mask.dim() < kl_map.dim():
            mask = mask.unsqueeze(-1)
        kl_map = kl_map * mask.to(dtype=kl_map.dtype, device=kl_map.device)

    kl_mean = kl_map.mean()
    loss = coeff * _nats_to_bits(kl_mean)  # scale in bits if desired (human-friendly)
    stats["gate_kl_bits"] = float(loss.detach().item()) / max(coeff, 1e-12)
    stats["gate_mean"] = float(gates.mean().detach().item())
    return loss, stats


def rate_loss(
    logits: Tensor,
    *,
    targets: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    gates: Optional[Tensor] = None,
    cfg: Optional[RateConfig] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    High-level helper to compute:
      total = rate_budget_penalty(bpp) + gate_kl_penalty(gates)
    Returns (loss, stats).
    """
    if cfg is None:
        cfg = RateConfig()

    # 1) bits-per-token
    bpp, st_bpp = bits_per_token(
        logits,
        targets=targets,
        mask=mask,
        use_entropy=cfg.use_entropy,
        temperature=cfg.temperature,
        ignore_index=cfg.ignore_index,
    )

    # 2) budget penalty
    pen_rate, st_budget = rate_budget_penalty(
        bpp,
        target_bpp=cfg.target_bpp,
        mode=cfg.budget_mode,
        coeff=cfg.budget_coeff,
        lambda_rate=cfg.lambda_rate,
    )

    # 3) gate KL
    pen_gate, st_gate = gate_kl_penalty(
        gates,
        prior=cfg.gate_prior,
        coeff=cfg.gate_coeff,
        mask=mask,
    )

    total = pen_rate + pen_gate
    stats: Dict[str, float] = {"bpp": st_bpp["bpp"], "bpp_max": st_bpp["bpp_max"], **st_budget, **st_gate}
    return total, stats


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[rate_loss] Running sanity tests...")

    B, T, V = 3, 9, 17
    D = 12
    pad_id = 0

    logits = torch.randn(B, T, V)
    targets = torch.randint(1, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -1] = 0
    targets[:, -1] = pad_id  # mark last as pad

    # 1) bpp from NLL vs from entropy
    bpp_nll, st1 = bits_per_token(logits, targets=targets, mask=mask, use_entropy=False, ignore_index=pad_id)
    bpp_ent, st2 = bits_per_token(logits, mask=mask, use_entropy=True)
    print(f"  bpp (nll)={float(bpp_nll):.3f}, bpp (entropy)={float(bpp_ent):.3f}, bpp_max≈{st1['bpp_max']:.2f}")

    # 2) Budget penalties
    pen_hinge, st_h = rate_budget_penalty(bpp_nll, target_bpp=float(st1["bpp_max"]) * 0.6, mode="hinge", coeff=0.5)
    pen_l2, st_l2 = rate_budget_penalty(bpp_nll, target_bpp=float(st1["bpp_max"]) * 0.6, mode="l2", coeff=0.25)
    pen_lagr, st_lg = rate_budget_penalty(bpp_nll, target_bpp=float(st1["bpp_max"]) * 0.6, mode="lagrange", lambda_rate=0.1)
    print(f"  budget: hinge={float(pen_hinge):.3e}, l2={float(pen_l2):.3e}, lagr={float(pen_lagr):.3e}")
    assert pen_hinge.ndim == 0 and pen_l2.ndim == 0 and pen_lagr.ndim == 0

    # 3) Gate KL (sparsity)
    gates = torch.sigmoid(torch.randn(B, T, D) * 0.5)
    kl_loss, stg = gate_kl_penalty(gates, prior=0.1, coeff=1e-3, mask=mask)
    print(f"  gate KL bits={stg['gate_kl_bits']:.3e}, gate_mean={stg['gate_mean']:.3f}")
    assert kl_loss.item() >= 0.0

    # 4) Full helper
    cfg = RateConfig(
        use_entropy=False, ignore_index=pad_id,
        target_bpp=st1["bpp_max"] * 0.5, budget_mode="hinge", budget_coeff=0.1,
        gate_prior=0.1, gate_coeff=1e-3, name="rate",
    )
    loss, stats = rate_loss(logits, targets=targets, mask=mask, gates=gates, cfg=cfg)
    print("  total:", float(loss), "| stats:", {k: f"{v:.3e}" for k, v in stats.items()})
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0

    print("[rate_loss] All good ✓")

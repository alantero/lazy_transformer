# modules/ledger.py
# Lightweight “ledger” for online training stats and capacity signals.
# PyTorch-only + optional utils.ema for EMAs.
#
# What this provides
# ------------------
# • Running EMAs (scalar) for things like entropy / loss / tokens seen.
# • Capacity signals from logits (per-token): cap = 1 − H(p)/log(V)  ∈ [0,1].
#   - Shapes returned as [B,T,1] by default, with helpers to expand to [B,T,G]/[B,T,D].
# • Tiny utils to register scalars, update EMAs, and snapshot stats for logging.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# EMA helper (safe import whether run as module or script)
try:
    from utils.ema import EMA  # type: ignore
except Exception:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.ema import EMA  # type: ignore

Tensor = torch.Tensor


# ----------------------------- small EMA utilities ----------------------------

def _safe_mean(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if mask is not None:
        x = x * mask.to(dtype=x.dtype, device=x.device)
        denom = mask.to(dtype=x.dtype, device=x.device).sum().clamp_min(1.0)
    else:
        # mean over all elements when no mask is provided
        denom = torch.tensor(float(x.numel()), device=x.device, dtype=x.dtype)
    return x.sum() / denom


# ------------------------------- capacity helpers -----------------------------

@torch.no_grad()
def logits_entropy(logits: Tensor, *, temperature: float = 1.0, mask: Optional[Tensor] = None) -> Tensor:
    """
    Per-token entropy (natural log):
      logits: [B,T,V]  →  H: [B,T]
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,T,V], got {tuple(logits.shape)}")
    z = logits / float(temperature)
    p = F.softmax(z, dim=-1)
    # Avoid log(0): clamp probs
    p = p.clamp_min(1e-12)
    H = -(p * p.log()).sum(dim=-1)  # [B,T]
    if mask is not None:
        H = H * mask.to(dtype=H.dtype, device=H.device)
    return H


@torch.no_grad()
def capacity_from_entropy(H: Tensor, V: int, *, clamp: bool = True) -> Tensor:
    """
    Normalize entropy to capacity ∈ [0,1]:
       cap = 1 − H / log(V)
    Returns [B,T] if input H is [B,T].
    """
    Hmax = math.log(float(V)) if V > 0 else 1.0
    cap = 1.0 - (H / float(Hmax))
    if clamp:
        cap = cap.clamp(0.0, 1.0)
    return cap


def _broadcast_cap(cap_bt: Tensor, *, d_model: Optional[int] = None, groups: Optional[int] = None) -> Tensor:
    """
    Expand cap [B,T] to:
      • [B,T,1] (default if neither d_model nor groups provided)
      • [B,T,G] if groups provided
      • [B,T,D] if d_model provided
    Priority: if both provided, expand to [B,T,D].
    """
    if cap_bt.dim() != 2:
        raise ValueError("cap must be [B,T].")
    B, T = cap_bt.shape
    if d_model is not None:
        return cap_bt.view(B, T, 1).expand(B, T, d_model)
    if groups is not None:
        return cap_bt.view(B, T, 1).expand(B, T, groups)
    return cap_bt.view(B, T, 1)


# ---------------------------------- the ledger --------------------------------

@dataclass
class LedgerState:
    steps: int = 0
    tokens_seen: int = 0
    ema_alpha: float = 0.90


class CapacityLedger(nn.Module):
    """
    Online stats + capacity signals.
      • Call update(logits, mask, loss) each step to refresh EMAs.
      • Call capacity(logits, mask, ...) to get per-token cap tensors for gating.
    """
    def __init__(self, vocab_size: int, *, ema_alpha: float = 0.90, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = torch.float32):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.state = LedgerState(steps=0, tokens_seen=0, ema_alpha=float(ema_alpha))
        self.entropy_ema = EMA(alpha=float(ema_alpha), debias=False)
        self.loss_ema    = EMA(alpha=float(ema_alpha), debias=False)

    # ------------------------------- updates ----------------------------------

    @torch.no_grad()
    def update(
        self,
        *,
        logits: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        loss: Optional[Tensor | float] = None,
    ) -> Dict[str, float]:
        """
        Update ledger with current batch stats. Any subset of inputs is allowed.
          logits: [B,T,V] for entropy stats
          mask:   [B,T]   tokens to count (1) vs pad (0)
          loss:   scalar   task loss for EMA
        Returns a dict of float scalars for logging.
        """
        s = self.state
        out: Dict[str, float] = {}
        alpha = s.ema_alpha

        if mask is not None:
            s.tokens_seen += int(mask.to(dtype=torch.int64).sum().item())
            out["tokens_seen"] = float(s.tokens_seen)

        if logits is not None:
            H = logits_entropy(logits, mask=mask)  # [B,T]
            H_mean = float(_safe_mean(H, mask=mask).item())
            e_val = self.entropy_ema.update(torch.tensor(H_mean, device=logits.device))
            out["entropy"] = H_mean
            out["entropy_ema"] = float(e_val if not isinstance(e_val, torch.Tensor) else e_val.item())

        if loss is not None:
            loss_val = float(loss) if not isinstance(loss, torch.Tensor) else float(loss.detach().item())
            # choose a safe device even if logits is None
            if isinstance(self.entropy_ema.value(), torch.Tensor):
                dev = self.entropy_ema.value().device
            else:
                dev = logits.device if logits is not None else torch.device("cpu")
            l_val = self.loss_ema.update(torch.tensor(loss_val, device=dev))
            out["loss_ema"] = float(l_val if not isinstance(l_val, torch.Tensor) else l_val.item())

        s.steps += 1
        out["steps"] = float(s.steps)
        return out

    # ------------------------------ capacities --------------------------------

    @torch.no_grad()
    def capacity(
        self,
        logits: Tensor,
        *,
        mask: Optional[Tensor] = None,
        temperature: float = 1.0,
        normalize: bool = True,
        clamp: bool = True,
        d_model: Optional[int] = None,
        groups: Optional[int] = None,
        stop_grad: bool = True,
        fill_masked_with: float = 0.0,
    ) -> Tensor:
        """
        Compute capacity per token from logits.
          logits: [B,T,V] → cap in one of { [B,T,1], [B,T,G], [B,T,D] }
        Args:
          temperature: softmax temperature for entropy
          normalize:   if True, cap = 1 − H/log(V); else returns raw H (negated)
          clamp:       clamp cap to [0,1] if normalize
          d_model/groups: broadcast shape helper (priority: D > G > 1)
          stop_grad:   detach returned capacity
          fill_masked_with: capacity value for masked (pad) positions
        """
        H = logits_entropy(logits, temperature=temperature, mask=None)  # [B,T], do not zero yet
        if normalize:
            cap = capacity_from_entropy(H, self.vocab_size, clamp=clamp)  # [B,T]
        else:
            cap = -H  # higher “capacity” when entropy low → more negative H; keep as a signal

        if mask is not None and mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        if mask is not None:
            cap = torch.where(mask.to(dtype=torch.bool, device=cap.device), cap, torch.as_tensor(fill_masked_with, device=cap.device, dtype=cap.dtype))

        cap_btX = _broadcast_cap(cap, d_model=d_model, groups=groups)  # [B,T,1/G/D]
        return cap_btX.detach() if stop_grad else cap_btX

    # ------------------------------- bookkeeping -------------------------------

    def snapshot(self) -> Dict[str, float]:
        """Return a plain dict of current scalar EMAs and counters (for logging)."""
        s = self.state
        e = self.entropy_ema.value()
        l = self.loss_ema.value()
        out = dict(
            steps=float(s.steps),
            tokens_seen=float(s.tokens_seen),
            entropy_ema=float(e if not isinstance(e, torch.Tensor) else e.item()) if e is not None else 0.0,
            loss_ema=float(l if not isinstance(l, torch.Tensor) else l.item()) if l is not None else 0.0,
        )
        return out


    def get_extra_state(self):
        """
        Provide extra state to be saved alongside the module's state_dict
        without overriding nn.Module.state_dict(). This keeps compatibility
        with PyTorch's recursion and avoids signature mismatches.
        """
        return {
            "vocab_size": int(self.vocab_size),
            "state": asdict(self.state),
            "ema_entropy": self.entropy_ema.state_dict(),
            "ema_loss": self.loss_ema.state_dict(),
        }

    def set_extra_state(self, state):
        """
        Restore payload saved by get_extra_state(). Also supports older flat
        payloads for backward compatibility.
        """
        if not isinstance(state, dict):
            return

        # vocab size
        try:
            if "vocab_size" in state:
                self.vocab_size = int(state["vocab_size"])
        except Exception:
            pass

        # dataclass LedgerState
        st = state.get("state", None)
        if isinstance(st, dict):
            try:
                self.state.steps = int(st.get("steps", self.state.steps))
                self.state.tokens_seen = int(st.get("tokens_seen", self.state.tokens_seen))
                self.state.ema_alpha = float(st.get("ema_alpha", self.state.ema_alpha))
            except Exception:
                pass

        # EMA payloads (new nested form)
        ema_e = state.get("ema_entropy", None)
        if isinstance(ema_e, dict):
            try:
                self.entropy_ema.load_state_dict(ema_e)
            except Exception:
                pass
        else:
            # backward-compat: accept old flat key 'entropy_mean'
            ent = state.get("entropy_mean", None)
            if ent is not None:
                try:
                    self.entropy_ema.reset(torch.as_tensor(ent))
                except Exception:
                    pass

        ema_l = state.get("ema_loss", None)
        if isinstance(ema_l, dict):
            try:
                self.loss_ema.load_state_dict(ema_l)
            except Exception:
                pass
        else:
            los = state.get("loss_mean", None)
            if los is not None:
                try:
                    self.loss_ema.reset(torch.as_tensor(los))
                except Exception:
                    pass


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("[ledger] Running sanity tests...")

    B, T, V = 2, 7, 11
    D, G = 12, 3

    # Fake logits / mask
    logits = torch.randn(B, T, V)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -1] = 0  # last token padded

    # 1) Entropy + capacity
    H = logits_entropy(logits)
    print(f"  entropy: shape={tuple(H.shape)}, mean={float(H.mean()):.3f}")
    assert H.shape == (B, T)

    cap_bt1 = capacity_from_entropy(H, V)
    assert cap_bt1.min().item() >= 0.0 and cap_bt1.max().item() <= 1.0
    print(f"  cap[BT]: min={float(cap_bt1.min()):.3f}, max={float(cap_bt1.max()):.3f}")

    # 2) Ledger update + snapshot
    led = CapacityLedger(vocab_size=V, ema_alpha=0.9)
    stats = led.update(logits=logits, mask=mask, loss=1.23)
    snap = led.snapshot()
    print("  update stats:", {k: f"{v:.3f}" for k, v in stats.items()})
    print("  snapshot:    ", {k: f"{v:.3f}" for k, v in snap.items()})
    assert "entropy_ema" in stats and "steps" in stats

    # 3) Capacity tensors for gating
    cap_tok = led.capacity(logits, mask=mask)  # [B,T,1]
    cap_grp = led.capacity(logits, mask=mask, groups=G)  # [B,T,G]
    cap_chn = led.capacity(logits, mask=mask, d_model=D)  # [B,T,D]
    print(f"  cap shapes:  {tuple(cap_tok.shape)}, {tuple(cap_grp.shape)}, {tuple(cap_chn.shape)}")
    assert cap_tok.shape == (B, T, 1)
    assert cap_grp.shape == (B, T, G)
    assert cap_chn.shape == (B, T, D)
    assert torch.all(cap_tok[mask == 0] == 0.0), "Masked tokens should have 0 capacity by default."

    # 4) Save / load round-trip
    payload = led.state_dict()
    led2 = CapacityLedger(vocab_size=V, ema_alpha=0.9)
    led2.load_state_dict(payload)
    snap2 = led2.snapshot()
    assert abs(snap2["entropy_ema"] - snap["entropy_ema"]) < 1e-6

    print("[ledger] All good ✓")

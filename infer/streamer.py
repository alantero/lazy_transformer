# infer/streamer.py
# Streaming inference with sliding windows:
#   • Memory scales with (window W + a few “bits” stats), not full sequence length.
#   • Local-optimum early stop (based on entropy/plateau in recent steps).
#   • Optional “teleports”: periodically truncate context (keep only collar) or on low capacity.
#
# Assumes a model with `forward_tokens(tokens, mask)->[B,T,V]` (e.g., ContinuousLM from train/loop.py).
# Self-contained; only depends on torch and (optionally) the repo for the demo in __main__.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ------------------------------- helpers --------------------------------------

def _last_indices_from_mask(mask: Tensor) -> Tensor:
    """mask [B,T] bool → [B] index of last valid token per sequence."""
    lengths = mask.to(dtype=torch.int64).sum(dim=1)  # [B]
    idx = (lengths - 1).clamp_min(0)
    return idx


@torch.no_grad()
def _last_logits(model: nn.Module, tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Return logits on the last valid position of each sequence: [B,V]."""
    model.eval()
    logits = model.forward_tokens(tokens, mask)  # [B,T,V]
    B, T, V = logits.shape
    if mask is None:
        last_idx = torch.full((B,), T - 1, dtype=torch.long, device=logits.device)
    else:
        last_idx = _last_indices_from_mask(mask)
    return logits[torch.arange(B, device=logits.device), last_idx]  # [B,V]


def _top_k_top_p_filtering(logits: Tensor, top_k: int = 0, top_p: float = 1.0) -> Tensor:
    """Apply top-k and/or nucleus (top-p) filtering on [B,V] logits (in-place safe via clone)."""
    B, V = logits.shape
    z = logits.clone()

    # top-k
    if top_k > 0 and top_k < V:
        kth = torch.topk(z, top_k, dim=-1).values[:, -1].unsqueeze(-1)  # [B,1]
        z[z < kth] = -float("inf")

    # top-p
    if top_p < 1.0:
        probs = F.softmax(z, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask_p = cum > top_p
        mask_p[:, 0] = False
        scatter_mask = torch.zeros_like(mask_p, dtype=torch.bool)
        scatter_mask.scatter_(dim=-1, index=sorted_idx, src=mask_p)
        z[scatter_mask] = -float("inf")

    return z


def _sample_from_logits(logits: Tensor, *, temperature: float, top_k: int, top_p: float) -> Tuple[Tensor, Tensor]:
    """Sample id from [B,V] logits with T/top-k/top-p. Returns (ids[B], probs[B])."""
    z = logits / float(max(temperature, 1e-8))
    zf = _top_k_top_p_filtering(z, top_k=top_k, top_p=top_p)
    p = F.softmax(zf, dim=-1)
    ids = torch.multinomial(p, num_samples=1).squeeze(-1)
    probs = p.gather(dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
    return ids, probs


def _entropy_last(logits_last: Tensor) -> Tensor:
    """Shannon entropy (nats) for [B,V] distributions at the last step."""
    logp = F.log_softmax(logits_last, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)  # [B]


def _append_tokens(tokens: Tensor, mask: Tensor, next_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """Append next_ids[B] to tokens[B,T], update mask, return new tensors."""
    B = tokens.shape[0]
    next_ids = next_ids.view(B, 1)
    new_tokens = torch.cat([tokens, next_ids], dim=1)
    next_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
    new_mask = torch.cat([mask, next_mask], dim=1)
    return new_tokens, new_mask


def _trim_to_window(tokens: Tensor, mask: Tensor, W: int) -> Tuple[Tensor, Tensor]:
    """Keep only the last W positions of tokens/mask."""
    if tokens.size(1) <= W:
        return tokens, mask
    return tokens[:, -W:], mask[:, -W:]


# ----------------------------- streamer config --------------------------------

@dataclass
class StreamerConfig:
    # windowing
    W: int = 256
    O: int = 32

    # sampling
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    # stopping by local optimum / plateau
    stop_on_optimum: bool = True
    opt_window: int = 8               # recent history of entropies
    opt_patience: int = 4             # require >= this many “non-improving” steps to stop
    opt_eps: float = 1e-3             # minimum improvement in normalized entropy to count as “better”
    min_new_tokens: int = 1           # don't stop before producing at least this many tokens

    # teleports (context truncation)
    teleport_every: Optional[int] = None      # every N accepted tokens, force a teleport
    teleport_on_low_capacity: bool = False    # if True, teleports when capacity < cap_threshold
    cap_threshold: float = 0.15               # capacity ~ 1 - H/log(V); lower → less confident
    keep_collar_on_teleport: bool = True      # keep last O tokens as collar when teleporting

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pad_id: int = 0
    eos_id: Optional[int] = None


# --------------------------------- Streamer -----------------------------------

class Streamer:
    """
    Sliding-window autoregressive generator:
      - maintains only the last W tokens + tiny stats (entropy EMA), not full history
      - yields newly generated tokens and final stats
      - can early-stop on local optimum and perform periodic/conditional teleports
    """
    def __init__(self, model: nn.Module, cfg: StreamerConfig):
        self.model = model.eval()
        self.cfg = cfg
        self._ent_hist: List[float] = []
        self._accepted = 0

    @torch.no_grad()
    def _should_stop(self, H_norm: float, produced: int) -> bool:
        if not self.cfg.stop_on_optimum:
            return False
        if produced < max(1, self.cfg.min_new_tokens):
            return False

        # track history and check plateau
        self._ent_hist.append(H_norm)
        K = min(len(self._ent_hist), self.cfg.opt_window)
        if K < self.cfg.opt_window:
            return False
        recent = self._ent_hist[-K:]  # most recent entropies (normalized)
        # check if best improvement over this window is < eps
        best = min(recent)
        improvements = [recent[i] - best for i in range(K)]
        non_improving = sum(1 for dv in improvements if dv < self.cfg.opt_eps)
        return non_improving >= self.cfg.opt_patience

    @torch.no_grad()
    def _maybe_teleport(self, tokens: Tensor, mask: Tensor, cap: float) -> Tuple[Tensor, Tensor, bool]:
        did = False
        # time-based teleport
        if self.cfg.teleport_every is not None and self.cfg.teleport_every > 0:
            if (self._accepted > 0) and (self._accepted % self.cfg.teleport_every == 0):
                tokens, mask = self._do_teleport(tokens, mask)
                did = True
        # capacity-based teleport
        if (not did) and self.cfg.teleport_on_low_capacity and (cap < self.cfg.cap_threshold):
            tokens, mask = self._do_teleport(tokens, mask)
            did = True
        return tokens, mask, did

    @torch.no_grad()
    def _do_teleport(self, tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Keep only the collar (last O tokens) or empty context
        if self.cfg.keep_collar_on_teleport and self.cfg.O > 0:
            keep = min(tokens.size(1), self.cfg.O)
            return tokens[:, -keep:], mask[:, -keep:]
        else:
            # keep just last token as seed (or BOS if pad_id serves as BOS)
            keep = 1
            return tokens[:, -keep:], mask[:, -keep:]

    @torch.no_grad()
    def generate(
        self,
        tokens: Tensor,
        mask: Optional[Tensor],
        *,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        Stream-generate up to `max_new_tokens`, returning:
          { 'tokens': [B, T+N_stream],
            'mask': [B, T+N_stream],
            'accept': int, 'propose': int,
            'teleports': int, 'stopped_optimum': bool }
        Only the last W context is kept internally at each step.
        """
        device = tokens.device
        B, T0 = tokens.shape
        m = mask if mask is not None else torch.ones_like(tokens, dtype=torch.bool, device=device)
        out_new: List[Tensor] = []
        teleports = 0
        self._ent_hist.clear()
        self._accepted = 0

        # trim start (if prompt longer than W)
        tokens, m = _trim_to_window(tokens, m, self.cfg.W)

        stopped_opt = False
        for _ in range(max_new_tokens):
            # logits at last position, sample a token
            last = _last_logits(self.model, tokens, m)     # [B,V]
            V = last.size(-1)
            H = _entropy_last(last)                        # [B]
            Hn = (H / math.log(float(V))).mean().item()    # normalized entropy in [0,1] approx
            cap = max(0.0, 1.0 - Hn)                       # capacity proxy

            next_ids, _ = _sample_from_logits(
                last,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
            )
            tokens, m = _append_tokens(tokens, m, next_ids)
            tokens, m = _trim_to_window(tokens, m, self.cfg.W)
            out_new.append(next_ids)
            self._accepted += B

            # Teleport policy (optional)
            tokens, m, did_tp = self._maybe_teleport(tokens, m, cap)
            if did_tp:
                teleports += 1

            # Early stop on local optimum/plateau (normalized entropy)
            if self._should_stop(Hn, produced=len(out_new)):
                stopped_opt = True
                break

            # EOS stop (optional)
            if self.cfg.eos_id is not None:
                if bool((next_ids == int(self.cfg.eos_id)).all().item()):
                    break

        # concat outputs
        if out_new:
            add = torch.stack(out_new, dim=1)  # [B,N]
            out_tokens = torch.cat([tokens[:, :tokens.size(1) - add.size(1)], add], dim=1) if tokens.size(1) > add.size(1) else torch.cat([tokens[:, :0], add], dim=1)
            # Reconstruct full stream by reusing the already-trimmed tokens as final state:
            # For users: return the original prompt concatenated with generated tokens only
            gen_only = add
            full_tokens = torch.cat([torch.zeros_like(tokens[:, :0]), gen_only], dim=1)  # placeholder to keep API consistent
            # But users usually want original prompt + gen. We don't carry full prompt to save memory,
            # so we return only the final window tokens and the generated tail for clarity:
            final_tokens = torch.cat([tokens, torch.zeros_like(tokens[:, :0])], dim=1)  # equals tokens
        else:
            gen_only = torch.empty(B, 0, dtype=tokens.dtype, device=device)
            final_tokens = tokens

        # For convenience, return the final window state AND the generated chunk separately.
        out = {
            "final_window_tokens": tokens,           # [B, ≤W] final window state
            "final_window_mask": m,                  # [B, ≤W]
            "generated": gen_only,                   # [B, N_gen] (only the newly sampled tokens)
            "accept": int(self._accepted),
            "propose": int(self._accepted),          # equal here (no speculative rejection)
            "teleports": teleports,
            "stopped_optimum": stopped_opt,
        }
        return out


# ---------------------------------- __main__ ----------------------------------

if __name__ == "__main__":
    # Minimal sanity: run with an untrained ContinuousLM (distribution is random but pipeline works).
    torch.manual_seed(0)
    print("[streamer] Running sanity test...")

    # Safe import of the model + tiny tokenizer for a demo
    # Ensure repo root is on sys.path when running as a script
    if __package__ in (None, ""):
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    try:
        from train.loop import ContinuousLM, LoopConfig
    except Exception as e:
        raise SystemExit(f"Cannot import model (train/loop.py). Error: {e}")

    try:
        from data.tok_embed import SimpleVocab, pack_batch  # preferred
    except Exception:
        from data.tokenize import SimpleVocab, pack_batch   # fallback

    # Tiny vocab / prompt
    texts = ["hello there", "general kenobi", "hello hello"]
    vocab = SimpleVocab.build_from_texts(texts, mode="char", add_unk=False)
    enc = lambda s: vocab.encode(s, mode="char", add_bos=True, add_eos=False)

    prompts = ["hello "]
    seqs = [enc(s) for s in prompts]
    tokens, mask = pack_batch(seqs, pad_id=vocab.pad_id)  # [B,T]
    B, T = tokens.shape

    # Model config (toy)
    cfg_m = LoopConfig(
        W=32, O=8,
        vocab_size=vocab.size, d_model=64, groups=8,
        cheb_deg=4, cheb_laplacian="cycle",
        skew_rank=8, R_rank=4,
        steps=2, dt=0.5, method="heun",
        tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
        pad_id=vocab.pad_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_gauge=True,
    )
    device = torch.device(cfg_m.device)
    model = ContinuousLM(cfg_m).to(device).eval()

    tokens = tokens.to(device)
    mask = mask.to(device)

    # Streamer config
    cfg_s = StreamerConfig(
        W=cfg_m.W, O=cfg_m.O,
        temperature=1.0, top_k=0, top_p=1.0,
        stop_on_optimum=True, opt_window=6, opt_patience=3, opt_eps=5e-4, min_new_tokens=2,
        teleport_every=10, teleport_on_low_capacity=True, cap_threshold=0.10, keep_collar_on_teleport=True,
        device=str(device), pad_id=vocab.pad_id, eos_id=vocab.eos_id,
    )

    streamer = Streamer(model, cfg_s)

    out = streamer.generate(tokens, mask, max_new_tokens=24)
    gen = out["generated"].cpu()
    for i in range(gen.shape[0]):
        decoded = vocab.decode(gen[i].tolist(), mode="char")
        print(f"  sample[{i}] → '{decoded}' | teleports={out['teleports']} | stop_opt={out['stopped_optimum']}")

    print("[streamer] Done ✓")

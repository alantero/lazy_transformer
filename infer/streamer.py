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


def _unwrap_logits(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def _last_logits(model: nn.Module, tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Return logits on the last valid position of each sequence: [B,V]."""
    model.eval()
    #logits = model.forward_tokens(tokens, mask)  # [B,T,V]
    logits = _unwrap_logits(model.forward_tokens(tokens, mask))  # [B,T,V]
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

# -------------------- Decoding anti-collapse helpers -------------------------

def apply_repetition_penalty(logits: Tensor, generated: List[int], penalty: float, window: int = 256) -> Tensor:
    """Scale logits of recently generated token ids by 1/penalty (CTRL-style)."""
    if penalty is None or penalty <= 1.0 or len(generated) == 0:
        return logits
    recent = generated[-int(window):]
    if len(recent) == 0:
        return logits
    ids = torch.tensor(recent, device=logits.device, dtype=torch.long).unique()
    logits[:, ids] = logits[:, ids] / penalty
    return logits

def block_repeated_ngrams(logits: Tensor, generated: List[int], n: int, vocab_size: int, window: int = 256) -> Tensor:
    """Mask logits for tokens that would complete a repeated n-gram."""
    if n is None or n <= 1 or len(generated) < n - 1:
        return logits
    ctx = generated[-(n-1):]
    hist = generated[-int(window):]
    blocked = []
    for i in range(len(hist) - (n - 1)):
        if hist[i:i+n-1] == ctx:
            nxt = hist[i + (n - 1)]
            blocked.append(nxt)
    if not blocked:
        return logits
    blocked = torch.tensor(list(set(blocked)), device=logits.device, dtype=torch.long)
    logits[:, blocked] = -float("inf")
    return logits

def typical_filtering(logits: Tensor, p: float = 0.9) -> Tensor:
    """Typical decoding filtering: keep minimal mass around expected self-information."""
    if p >= 1.0:
        return logits
    logp = torch.log_softmax(logits, dim=-1)
    pvals = logp.exp()
    self_info = -logp
    dev = torch.abs(self_info - (pvals * self_info).sum(dim=-1, keepdim=True))  # deviation from typical SI
    # sort by deviation (ascending), keep the minimal-mass set whose cumulative prob >= p
    order = torch.argsort(dev, dim=-1)  # [B, V]
    sorted_p = torch.gather(pvals, -1, order)
    csum = torch.cumsum(sorted_p, dim=-1)
    # Build a mask of tokens to drop (True=drop). Start all dropped, then un-drop the prefix.
    drop = torch.ones_like(logits, dtype=torch.bool)
    B, V = logits.shape
    for b in range(B):
        k = int((csum[b] >= p).nonzero(as_tuple=False)[0].item())  # index of cutoff (inclusive)
        keep_idx = order[b, :k+1]
        drop[b, keep_idx] = False
    out = logits.clone()
    out[drop] = -float("inf")
    return out

def min_p_filtering(logits: Tensor, min_p: float = 0.0) -> Tensor:
    """Eta/min-p sampling: mask tokens whose prob < min_p * max_prob."""
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    maxp = probs.max(dim=-1, keepdim=True).values
    mask = probs < (min_p * maxp)
    out = logits.clone()
    out[mask] = -float("inf")
    return out


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

    # decoding controls
    method: str = "sample"           # "sample" | "typical" | "greedy"
    typical_p: float = 0.9
    repetition_penalty: float = 1.0  # 1.0 = off
    no_repeat_ngram_size: int = 0    # 0 = off
    min_p: float = 0.0               # 0.0 = off (eta/min-p)

    # teleports (context truncation)
    teleport_every: Optional[int] = None      # every N accepted tokens, force a teleport
    teleport_on_low_capacity: bool = False    # if True, teleports when capacity < cap_threshold
    cap_threshold: float = 0.15               # capacity ~ 1 - H/log(V); lower → less confident
    keep_collar_on_teleport: bool = True      # keep last O tokens as collar when teleporting

    # debugging
    debug_topk: bool = False           # if True, print top-10 tokens at the last position
    debug_topk_all: bool = False       # if True, print at every step; otherwise only first step

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
        # Keep a copy of the full, untrimmed prompt for decoding and repetition history
        prompt_tokens_full = tokens.clone()
        # Per-batch repetition histories (tail of prompt) and alive mask (for EOS stop)
        hist_ids: List[List[int]] = [
            prompt_tokens_full[b].tolist()[-int(self.cfg.W + self.cfg.O):] for b in range(B)
        ]
        alive = torch.ones(B, dtype=torch.bool, device=device)
        out_new: List[Tensor] = []
        teleports = 0
        self._ent_hist.clear()
        self._accepted = 0

        # trim start (if prompt longer than W)
        tokens, m = _trim_to_window(tokens, m, self.cfg.W)

        stopped_opt = False
        for _ in range(max_new_tokens):
            # logits at last position, then apply anti-collapse filters
            last = _last_logits(self.model, tokens, m)     # [B,V]
            V = last.size(-1)
            H = _entropy_last(last)                        # [B]
            Hn = (H / math.log(float(V))).mean().item()    # normalized entropy in [0,1] approx
            cap = max(0.0, 1.0 - Hn)                       # capacity proxy

            # Optional debug: show top-10 tokens/probs at the last position (sample 0)
            if getattr(self.cfg, "debug_topk", False) and ((self._accepted == 0) or getattr(self.cfg, "debug_topk_all", False)):
                 with torch.no_grad():
                      probs_dbg = torch.softmax(last, dim=-1)
                      k = int(min(10, V))
                      topk_dbg = torch.topk(probs_dbg, k=k, dim=-1)
                      ids_dbg = topk_dbg.indices[0].tolist()
                      vals_dbg = [float(v) for v in topk_dbg.values[0]]
                      argmax_id = int(torch.argmax(last[0]).item())
                      step_idx = int(self._accepted // max(1, tokens.size(0)))
                      logit_range = float((last[0].max() - last[0].min()).item())
                      logit_l2 = float(last[0].norm(p=2).item())
                      print(f"[debug] step={step_idx} top{k}_ids(sample0)=", ids_dbg)
                      print(f"[debug] step={step_idx} top{k}_probs(sample0)=", vals_dbg)
                      print(f"[debug] step={step_idx} argmax_id(sample0)=", argmax_id, "| logit_range=", logit_range, "| logit_l2=", logit_l2)


            # Repetition penalty & n-gram blocking — apply per batch using each sample's history
            for b in range(B):
                lb = last[b:b+1]
                hb = hist_ids[b]
                lb = apply_repetition_penalty(lb, hb, self.cfg.repetition_penalty, window=self.cfg.W + self.cfg.O)
                lb = block_repeated_ngrams(lb, hb, self.cfg.no_repeat_ngram_size, V, window=self.cfg.W + self.cfg.O)
                last[b:b+1] = lb

            # Method-specific filtering + sampling pipeline
            # Order (HF-style): temperature -> typical (optional) -> min-p -> top-k/top-p -> sample/greedy
            if self.cfg.method == "greedy":
                # Greedy does not use temperature or stochastic filters
                z = last
                next_ids = torch.argmax(z, dim=-1)
                # Update per-batch histories with the newly chosen ids (only for alive samples)
                for b in range(B):
                    if bool(alive[b]):
                        hist_ids[b].append(int(next_ids[b].item()))
                        # Keep only a bounded tail to avoid unbounded growth
                        if len(hist_ids[b]) > int(self.cfg.W + self.cfg.O) * 4:
                            hist_ids[b] = hist_ids[b][-int(self.cfg.W + self.cfg.O):]
            else:
                # 1) Temperature
                z = last / float(max(self.cfg.temperature, 1e-8))
                # 2) Typical (optional)
                if self.cfg.method == "typical":
                    z = typical_filtering(z, p=self.cfg.typical_p)
                # 3) Min-p (eta)
                z = min_p_filtering(z, min_p=self.cfg.min_p)
                # 4) Top-k / Top-p
                z = _top_k_top_p_filtering(z, top_k=self.cfg.top_k, top_p=self.cfg.top_p)
                # 5) Sample
                p = torch.softmax(z, dim=-1)
                next_ids = torch.multinomial(p, num_samples=1).squeeze(-1)
                # Update per-batch histories with the newly chosen ids (only for alive samples)
                for b in range(B):
                    if bool(alive[b]):
                        hist_ids[b].append(int(next_ids[b].item()))
                        # Keep only a bounded tail to avoid unbounded growth
                        if len(hist_ids[b]) > int(self.cfg.W + self.cfg.O) * 4:
                            hist_ids[b] = hist_ids[b][-int(self.cfg.W + self.cfg.O):]
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

            # EOS stop per-sample (for B>1 keep generating for unfinished samples)
            if self.cfg.eos_id is not None:
                eos_tensor = torch.tensor(int(self.cfg.eos_id), device=device, dtype=next_ids.dtype)
                alive = alive & (next_ids != eos_tensor)
                if not bool(alive.any().item()):
                    break

        # concat outputs
        if out_new:
            gen_only = torch.stack(out_new, dim=1)  # [B, N_gen]
        else:
            gen_only = torch.empty(B, 0, dtype=tokens.dtype, device=device)

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
    import argparse, json
    from torch.serialization import add_safe_globals
    try:
        from torch.torch_version import TorchVersion
        add_safe_globals([TorchVersion])
    except Exception:
        pass
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Sliding-window streaming for CLO")
    parser.add_argument("--hf-config", type=str, default="configs/config.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="In a quiet village at the edge of the forest,")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method", type=str, default="sample", choices=["sample", "typical", "greedy"])
    parser.add_argument("--typical-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument("--no-stop-on-optimum", action="store_true", help="Disable entropy plateau early-stop.")
    parser.add_argument("--eos-stop", action="store_true", help="Stop when EOS is generated.")
    parser.add_argument("--debug-topk", action="store_true", help="Print top-10 token ids/probs at first decoding step.")
    parser.add_argument("--debug-topk-all", action="store_true", help="Print top-10 token ids/probs at every decoding step.")
    parser.add_argument("--zero-head-bias", action="store_true", help="Set output head bias to zero at load time (debugging collapse).")
    parser.add_argument("--print-head-biais-topk", type=int, default=0, help="If >0, print top-K tokens by output head bias (debug).")
    args = parser.parse_args()

    # Ensure repo root is on sys.path when running as a script
    if __package__ in (None, ""):
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from train.loop import ContinuousLM, LoopConfig
    with open(args.hf_config, "r") as f:
        raw = json.load(f)
    if str(raw.get("pos_kind", "sinusoidal")).lower() == "learned":
        raw["pos_kind"] = "sinusoidal"

    # Load tokenizer early to resolve vocab/pad consistently with training (GPT-2: vocab=50257, eos/pad=50256)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = 10_000_000
    vocab_from_tok = int(tok.vocab_size)
    pad_from_tok = int(tok.eos_token_id)

    # --- Load checkpoint (for weights and possibly cfg) ---
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[load] weights_only=True failed ({e}); retrying with weights_only=False (trusted local checkpoint)")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # If checkpoint carries a config, prefer it for core dims (protects against mismatches)
    ckcfg = None
    if isinstance(state, dict):
        ckcfg = state.get("cfg") or state.get("config") or None
    if isinstance(ckcfg, dict):
        for k in ["W","O","vocab_size","pad_id","tie_softmax","d_model","groups",
                  "cheb_deg","cheb_laplacian","skew_rank","R_rank","factor_rank",
                  "pos_kind","use_gauge"]:
            if k in ckcfg:
                raw[k] = ckcfg[k]

    # Build model config from (raw ⟵ maybe-ckpt ⟵ tokenizer guards)
    vocab_cfg = int(raw.get("vocab_size", vocab_from_tok) or vocab_from_tok)
    pad_cfg = int(raw.get("pad_id", pad_from_tok))
    vocab_final = max(vocab_cfg, vocab_from_tok, pad_cfg + 1)
    pad_final = pad_cfg if pad_cfg < vocab_final else (vocab_final - 1)

    cfg_m = LoopConfig(
        W=raw.get("W", 1024), O=raw.get("O", 64),
        vocab_size=vocab_final, d_model=raw.get("d_model", 384), groups=raw.get("groups", 16),
        cheb_deg=raw.get("cheb_deg", 8), cheb_laplacian=raw.get("cheb_laplacian", "cycle"),
        skew_rank=raw.get("skew_rank", 8), R_rank=raw.get("R_rank", 4),
        steps=2, dt=0.5, method="heun",
        tie_softmax=raw.get("tie_softmax", True), factor_rank=raw.get("factor_rank", 64),
        pos_kind=raw.get("pos_kind", "sinusoidal"), pad_id=pad_final,
        lr=raw.get("lr", 5e-4), weight_decay=raw.get("weight_decay", 0.1),
        device=args.device, use_gauge=raw.get("use_gauge", True),
        stitch_w=raw.get("stitch_w", 0.10), stitch_use_lowd=raw.get("stitch_use_lowd", True), stitch_use_skl=raw.get("stitch_use_skl", False),
        scheduler_name="warmup_cosine", scheduler_total_steps=200, scheduler_warmup_steps=20, scheduler_min_lr=3e-4,
        profile=False, profile_nvtx=False, profile_log_mem=False,
        checkpoint_dir="./ckpts", save_every=0, keep_last_k=5, best_metric_name="loss",
    )
    device = torch.device(args.device)
    model = ContinuousLM(cfg_m).to(device).eval()

    # Load weights (accept both {'model': sd} and raw sd)
    sd = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} e.g. {missing[:5]}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} e.g. {unexpected[:5]}")

    # --- Inspect/zero output head bias (robust) ---
    try:
        # Heuristic: find a Linear whose out_features == vocab_size (the LM head)
        def _find_vocab_linear(module: nn.Module, vocab_size: int) -> Optional[nn.Linear]:
            for m in module.modules():
                if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == int(vocab_size):
                    return m  # first match
            return None

        head_module: Optional[nn.Linear] = None

        # 1) Try common attribute names
        for name in ("output_head", "head", "lm_head"):
            if hasattr(model, name):
                cand = getattr(model, name)
                if isinstance(cand, nn.Linear) and getattr(cand, "out_features", None) == int(cfg_m.vocab_size):
                    head_module = cand
                    break

        # 2) Fallback: scan the whole model
        if head_module is None:
            head_module = _find_vocab_linear(model, int(cfg_m.vocab_size))

        if head_module is None:
            print("[warn] could not locate output head Linear (to vocab).")
        else:
            print(f"[debug] found output head: {head_module.__class__.__name__}(in={getattr(head_module,'in_features',None)}, out={getattr(head_module,'out_features',None)})")

            # Print top-K bias entries if requested
            if args.print_head_biais_topk and int(args.print_head_biais_topk) > 0:
                if getattr(head_module, "bias", None) is not None:
                    with torch.no_grad():
                        b = head_module.bias.detach().cpu()
                        k = int(min(int(args.print_head_biais_topk), b.numel()))
                        topv, topi = torch.topk(b, k=k, largest=True)
                        print(f"[debug] head.bias top{k} ids:", topi.tolist())
                        print(f"[debug] head.bias top{k} values:", [float(v) for v in topv])
                else:
                    print("[debug] head has no bias (nothing to print).")

            # Zero the bias if requested
            if args.zero_head_bias:
                if getattr(head_module, "bias", None) is not None:
                    with torch.no_grad():
                        head_module.bias.zero_()
                    print("[debug] zeroed output head bias.")
                else:
                    print("[debug] head has no bias to zero.")

        # --- Fallback: scan for a vocab-sized bias Parameter (tied-head case) ---
        if head_module is None:
            bias_param = None
            bias_name = None
            vocab_N = int(cfg_m.vocab_size)
            for name, p in model.named_parameters():
                # Look for 1D bias-like parameters exactly of size vocab_size
                if p.ndim == 1 and p.numel() == vocab_N:
                    # Heuristics: prefer names that contain 'bias' or 'out'
                    if ('bias' in name.lower()) or ('out' in name.lower()) or ('head' in name.lower()):
                        bias_param = p
                        bias_name = name
                        break
            # If none matched the heuristic, accept the first 1D param of vocab size
            if bias_param is None:
                for name, p in model.named_parameters():
                    if p.ndim == 1 and p.numel() == vocab_N:
                        bias_param = p
                        bias_name = name
                        break

            if bias_param is not None:
                print(f"[debug] found vocab-sized bias param: {bias_name} shape={tuple(bias_param.shape)}")
                if args.print_head_biais_topk and int(args.print_head_biais_topk) > 0:
                    with torch.no_grad():
                        b = bias_param.detach().cpu()
                        k = int(min(int(args.print_head_biais_topk), b.numel()))
                        topv, topi = torch.topk(b, k=k, largest=True)
                        print(f"[debug] bias(top){k} ids:", topi.tolist())
                        print(f"[debug] bias(top){k} values:", [float(v) for v in topv])
                if args.zero_head_bias:
                    with torch.no_grad():
                        bias_param.zero_()
                    print(f"[debug] zeroed vocab-sized bias param: {bias_name}")
            else:
                print("[warn] could not locate output head Linear nor a vocab-sized bias Parameter.")
    except Exception as e:
        print(f"[warn] head-bias debug failed: {e}")








    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    tokens = enc["input_ids"].to(device)
    mask = enc.get("attention_mask", None)
    if mask is not None: mask = mask.to(device)

    print(f"[debug] prompt_len_tokens={tokens.size(1)} | model_W={int(raw.get('W', cfg_m.W))} | model_O={int(raw.get('O', cfg_m.O))}")

    # Sanity: head dimension must match vocab_size
    with torch.no_grad():
        t_short = tokens[:, -min(tokens.size(1), cfg_m.W):]
        m_short = None if mask is None else mask[:, -t_short.size(1):]
        head_test = _unwrap_logits(model.forward_tokens(t_short, m_short))
    V = int(head_test.size(-1))
    if V != int(cfg_m.vocab_size):
        raise SystemExit(f"[error] head dim V={V} != cfg.vocab_size={int(cfg_m.vocab_size)}. "
                         f"Check tokenizer/ckpt/config alignment.")

    # Debug: show what common suspect IDs decode to (e.g., 11 is often ',' in GPT-2)
    try:
        for suspect in [11, tok.eos_token_id, 0]:
            if 0 <= int(suspect) < V:
                print(f"[debug] suspect id {int(suspect)} decodes to: '{tok.decode([int(suspect)], skip_special_tokens=False)}'")
    except Exception as _e:
        print(f"[warn] token decode debug failed: {_e}")

    cfg_s = StreamerConfig(
        W=cfg_m.W, O=cfg_m.O,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        method=args.method, typical_p=args.typical_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_p=args.min_p,
        stop_on_optimum=not args.no_stop_on_optimum, opt_window=8, opt_patience=4, opt_eps=1e-3, min_new_tokens=max(2, args.min_new_tokens),
        teleport_every=None, teleport_on_low_capacity=False, cap_threshold=0.10, keep_collar_on_teleport=True,
        debug_topk=bool(args.debug_topk),
        debug_topk_all=bool(args.debug_topk_all),
        device=str(device), pad_id=cfg_m.pad_id, eos_id=int(tok.eos_token_id) if args.eos_stop else None,
    )
    streamer = Streamer(model, cfg_s)
    out = streamer.generate(tokens, mask, max_new_tokens=args.max_new_tokens)

    # Reconstruct full text: original prompt + generated continuation
    full_out = torch.cat([enc["input_ids"].to(device), out["generated"].to(device)], dim=1) if out["generated"].numel() > 0 else enc["input_ids"].to(device)
    text = tok.decode(full_out[0].tolist(), skip_special_tokens=True)
    print("\n=== Streamed Output ===\n" + text)
    print(f"[stats] teleports={out['teleports']} stopped_optimum={out['stopped_optimum']}")

"""
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
"""

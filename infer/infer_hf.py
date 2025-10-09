# infer_hf.py
# ------------------------------------------------------------
# Inference for your CLO model using a Hugging Face-compatible wrapper.
# Supported via `model.generate(...)`:
#   - Greedy
#   - Sampling (temperature/top_k/top_p)
#   - Beam search
#   - Typical decoding (typical_p)
#   - Contrastive search (penalty_alpha + top_k)
#   - Repetition penalty, no_repeat_ngram_size
#
# Requirements:
#   pip install transformers
#   (and your repo deps to import ContinuousLM / LoopConfig)
# ------------------------------------------------------------

import os, json, math
import torch
import torch.nn as nn

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    GPT2TokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# Ensure repository root is importable regardless of CWD
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_this_dir)  # parent of infer/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import your implementation (try both top-level and package-style paths)
try:
    from train.loop import ContinuousLM, LoopConfig  # noqa: E402
except ModuleNotFoundError:
    from lazy_transformer.train.loop import ContinuousLM, LoopConfig  # noqa: E402


# ----------------------------- HF Config Wrapper -----------------------------

class CLOHFConfig(PretrainedConfig):
    model_type = "clo"

    def __init__(
        self,
        vocab_size=50257,
        W=1024,
        O=64,
        d_model=384,
        groups=16,
        cheb_deg=8,
        cheb_laplacian="cycle",
        skew_rank=8,
        R_rank=4,
        tie_softmax=True,
        factor_rank=64,
        pos_kind="sinusoidal",   # we use sinusoidal to avoid 'max_len'
        pad_id=50256,
        lr=5e-4,
        weight_decay=0.1,
        device="cuda",
        use_gauge=True,
        stitch_w=0.10,
        stitch_use_lowd=True,
        stitch_use_skl=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.W = W
        self.O = O
        self.d_model = d_model
        self.groups = groups
        self.cheb_deg = cheb_deg
        self.cheb_laplacian = cheb_laplacian
        self.skew_rank = skew_rank
        self.R_rank = R_rank
        self.tie_softmax = tie_softmax
        self.factor_rank = factor_rank
        self.pos_kind = pos_kind
        self.pad_id = pad_id
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.use_gauge = use_gauge
        self.stitch_w = stitch_w
        self.stitch_use_lowd = stitch_use_lowd
        self.stitch_use_skl = stitch_use_skl

    # For generate() compatibility
    @property
    def eos_token_id(self):
        # Use GPT-2 EOS (50256) by default
        return 50256

    @property
    def bos_token_id(self):
        # GPT-2 does not use BOS by default; keep None
        return None


# ------------------------------ HF Model Wrapper -----------------------------

class CLOForCausalLM(PreTrainedModel):
    config_class = CLOHFConfig

    def __init__(self, config: CLOHFConfig):
        super().__init__(config)
        # Build your internal model from LoopConfig
        loop_cfg = LoopConfig(
            W=config.W, O=config.O,
            vocab_size=config.vocab_size,
            d_model=config.d_model, groups=config.groups,
            cheb_deg=config.cheb_deg, cheb_laplacian=config.cheb_laplacian,
            skew_rank=config.skew_rank, R_rank=config.R_rank,
            steps=2, dt=0.5, method="heun",
            tie_softmax=config.tie_softmax, factor_rank=config.factor_rank,
            pos_kind=config.pos_kind, pad_id=config.pad_id,
            lr=config.lr, weight_decay=config.weight_decay,
            device=config.device, use_gauge=config.use_gauge,
            stitch_w=config.stitch_w, stitch_use_lowd=config.stitch_use_lowd, stitch_use_skl=config.stitch_use_skl,
            # scheduler params are not needed for inference
            scheduler_name="warmup_cosine", scheduler_total_steps=200, scheduler_warmup_steps=20, scheduler_min_lr=3e-4,
            profile=False, profile_nvtx=False, profile_log_mem=False,
            checkpoint_dir="./ckpts", save_every=0, keep_last_k=5, best_metric_name="loss",
        )
        self.inner = ContinuousLM(loop_cfg)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = False,
        past_key_values=None,
        **kwargs
    ):
        # The inner model accepts (tokens, mask) and returns logits [B, T, V]
        logits, *_ = self.inner.forward_tokens(input_ids, attention_mask)
        loss = None
        if labels is not None:
            # standard next-token cross-entropy (shifted)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size),
                            shift_labels.view(-1))
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # we do not implement cache in this wrapper
            hidden_states=None,
            attentions=None
        )

    # Methods to let generate() run without kv-cache
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _reorder_cache(self, past_key_values, beam_idx):
        # No past_key_values; nothing to reorder
        return past_key_values


# ------------------------------ Load utility ------------------------------

def load_clo_hf(
    hf_config_path: str = "configs/config.json",
    checkpoint_path: str = "ckpts/clo30m_owt/best_loss_step00003938.pt",
    device: str = "cuda"
):
    # Load your training JSON config (adapted to CLOHFConfig)
    with open(hf_config_path, "r") as f:
        raw = json.load(f)

    # Force pos_kind="sinusoidal" to match training fallback
    raw.setdefault("pos_kind", "sinusoidal")

    cfg = CLOHFConfig(**raw)
    model = CLOForCausalLM(cfg)
    # Load weights from your training checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)  # supports {'model': state_dict} or raw state_dict
    missing, unexpected = model.inner.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # useful for batch generation
    return model, tok


# ------------------------------ Usage examples -------------------------------

@torch.inference_mode()
def demo_generation(model: CLOForCausalLM, tok: GPT2TokenizerFast, prompt: str, device="cuda"):
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None: attn = attn.to(device)

    print("\n=== Greedy ===")
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))

    print("\n=== Sampling (top-k/top-p/temperature) ===")
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))

    print("\n=== Beam search ===")
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=120,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))

    print("\n=== Typical decoding ===")
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=120,
        do_sample=True,
        typical_p=0.8,     # typical p
        temperature=0.9,
        repetition_penalty=1.1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))

    print("\n=== Contrastive search ===")
    # Requires penalty_alpha > 0 and top_k >= 1; no sampling
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=120,
        penalty_alpha=0.6,   # strength/entropy
        top_k=6,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLO HF-compatible inference")
    parser.add_argument("--hf-config", type=str, default="configs/config.json", help="Path to training JSON config")
    parser.add_argument("--checkpoint", type=str, default="ckpts/clo30m_owt/best_loss_step00003938.pt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--prompt", type=str, default="In a quiet village at the edge of the forest,", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Number of new tokens to generate")
    parser.add_argument("--method", type=str, default="sample", choices=["greedy","sample","beam","typical","contrastive"], help="Decoding method")

    # Common controls
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty >= 1.0")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Block repeated n-grams of this size (0 disables)")

    # Sampling params
    parser.add_argument("--temperature", type=float, default=0.8, help="Softmax temperature for sampling/typical")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k for sampling/contrastive")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--typical-p", type=float, default=0.8, help="Typical decoding p value")

    # Beam params
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping for beam search")

    # Contrastive params
    parser.add_argument("--penalty-alpha", type=float, default=0.6, help="Contrastive search penalty alpha")

    # Export
    parser.add_argument("--export-dir", type=str, default=None, help="If set, save_pretrained() to this directory")

    args = parser.parse_args()

    model, tok = load_clo_hf(args["hf_config"] if isinstance(args, dict) else args.hf_config,
                             args["checkpoint"] if isinstance(args, dict) else args.checkpoint,
                             device=args["device"] if isinstance(args, dict) else args.device)

    enc = tok(args["prompt"] if isinstance(args, dict) else args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(args.device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(args.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        no_repeat_ngram_size=max(0, args.no_repeat_ngram_size),
        repetition_penalty=max(1.0, args.repetition_penalty),
    )

    if args.method == "greedy":
        gen_kwargs.update(dict(do_sample=False))

    elif args.method == "sample":
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ))

    elif args.method == "beam":
        gen_kwargs.update(dict(
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
        ))

        # generate defaults to greedy within each beam

    elif args.method == "typical":
        gen_kwargs.update(dict(
            do_sample=True,
            typical_p=args.typical_p,
            temperature=args.temperature,
        ))

    elif args.method == "contrastive":
        gen_kwargs.update(dict(
            penalty_alpha=args.penalty_alpha,
            top_k=max(1, args.top_k),
        ))

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            **gen_kwargs
        )
    print("\n=== Output ===\n")
    print(tok.decode(out[0], skip_special_tokens=True))

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        model.save_pretrained(args.export_dir)
        tok.save_pretrained(args.export_dir)
        print(f"[export] Saved HF-compatible model+tokenizer to: {args.export_dir}")

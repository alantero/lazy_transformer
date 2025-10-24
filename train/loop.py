# train/loop.py
# Minimal end-to-end training loop (Phase 3).
# Pipeline per window: tokens → embed → (groupwise norm) → (gauge) → cheb → portham → integrate → head → CE.
# Windows are overlap-added back to the full sequence before computing the next-token CE.
# PyTorch-only. Uses the modules we built in this project.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

#
# ------------------------------ safe imports ----------------------------------
# When executed as a script (python train/loop.py), make repo root importable.
if __package__ in (None, ""):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from data.tok_embed import TokenEmbedder, SimpleVocab, pack_batch  # preferred
except Exception:
    from data.tokenize import TokenEmbedder, SimpleVocab, pack_batch   # fallback

# Optionally import dataloaders utilities (collate_batch, build_collar_mask)
try:
    from data.dataloaders import collate_batch as _collate_batch, build_collar_mask as _build_collar_mask  # type: ignore
except Exception:
    _collate_batch = None  # type: ignore
    _build_collar_mask = None  # type: ignore

from utils.windows import slice_windows, reconstruct_from_windows
from modules.normalize import groupwise_norm
from operators.cheb import ChebFilter1D
# Optional advective operator (used only if use_op_bank=True and available)
try:
    from operators.advective import Advective1D as _AdvectiveClass  # type: ignore
except Exception:
    _AdvectiveClass = None  # type: ignore
from modules.portham import PortHamiltonianStep
from modules.integrator import Integrator
from modules.output_head import OutputHead
from losses.task_loss import sequence_cross_entropy, shift_for_next_token
from losses.rate_loss import bits_per_token
from losses.stitching_loss import StitchingOverlapLoss
from losses.regularizers import gate_l0_proxy, opbank_cosine_decorrelation, energy_oscillation_penalty
from optim.sda import DualSDA
from modules.quant import calibrate_model, toggle_fakequant_collect
from train.checkpoints import CheckpointManager  # checkpoint management
from optim.schedulers import make_scheduler  # learning-rate schedulers
from utils.profile import Timer, record_function as prof_record_function, nvtx_range, ThroughputMeter, gpu_mem
from modules.ledger import CapacityLedger
from contextlib import nullcontext


# ------------------------------ optional gauge --------------------------------
# Gauge is optional in the minimal loop; if not available we fallback to identity.
try:
    # whichever name you used in modules/gauge.py
    from modules.gauge import CapacityGauge as _GaugeClass  # type: ignore
except Exception:
    try:
        from modules.gauge import Gauge as _GaugeClass  # type: ignore
    except Exception:
        _GaugeClass = None  # type: ignore

# ------------------------------ optional gate ---------------------------------
try:
    from modules.gate import GroupGate as _GateClass  # type: ignore
except Exception:
    _GateClass = None  # type: ignore


# ------------------------------- config dataclass ------------------------------

@dataclass
class LoopConfig:
    # windows
    W: int = 256
    O: int = 32

    # model dims
    vocab_size: int = 128
    d_model: int = 128
    groups: int = 8

    # chebyshev filter
    cheb_deg: int = 8
    cheb_laplacian: str = "path_causal"  # 'path_causal' (strictly causal AR), 'path' (Dirichlet, non-causal), 'cycle' (periodic wrap)

    # port-hamiltonian
    skew_rank: int = 16
    R_rank: int = 4

    # integrator
    steps: int = 2
    dt: float = 0.5
    method: str = "heun"  # RK2 default (v3-style)

    # embedder
    tie_softmax: bool = True
    factor_rank: Optional[int] = None
    pos_kind: str = "sinusoidal"
    pad_id: int = 0

    # integrator nonlinearity blocks
    nl_diffusion_enabled: bool = False
    nl_p: float = 1.5
    nl_huber_delta: float = 1e-3
    nl_a_min: float = 0.2
    nl_a_max: float = 5.0
    nl_recompute_iters: int = 1
    nl_reduce: str = "group"  # or "channel"

    # port-hamiltonian nonlinear options (field-side)
    ph_nl_enabled: bool = False
    ph_use_nl_R: bool = False
    ph_use_nl_H: bool = False
    ph_nl_rank: int = 8

    # gate options
    use_gate: bool = False
    gate_topk: int = 2
    gate_per_channel: bool = False
    gate_gamma: float = 1.2

    # operator bank / top-k real
    use_op_bank: bool = False
    gate_topk_real: int = 2

    # output head multiplicative branch
    head_use_mul: bool = False
    head_mul_rank: int = 16
    head_mul_gate_init: float = 0.0

    # extra debug toggles
    debug_log_A: bool = False    # print p-Lap params/clamps once
    debug_log_gate: bool = False # print gate config once
    debug_energy: bool = False   # log E_pre/E_post for a sample

    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.0

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_gauge: bool = True  # can disable if module is not present

    # grad policy
    grads_border_only: bool = True     # compute grads only on collar (overlap) regions
    refresh_on_drift: bool = True      # if drift between bulk and collar is large, recompute bulk with grads
    drift_thresh: float = 1e-2         # threshold for drift refresh

    # stitching (overlap consistency)
    stitch_w: float = 0.0           # weight; if 0, disabled
    stitch_use_skl: bool = False    # symmetric KL on logits overlaps
    stitch_use_lowd: bool = True    # low-D Procrustes MSE on hidden overlaps (here d_low = D)
    stitch_rank: Optional[int] = None
    stitch_temperature: float = 1.0

    # dual/rate (optional)
    use_dual_rate: bool = True
    target_bpp: Optional[float] = None   # if None, defaults to 0.6 * log2(V)
    dual_lr: float = 1e-2
    dual_ema: float = 0.9
    dual_use_log: bool = True            # update in log-space (on bpp)
    dual_var_aware: bool = True          # variance-aware normalization of the dual gap

    # quantization: brief post-pruning calibration
    quant_calibrate_after_prune: bool = True
    quant_calib_batches: int = 64
    freeze_qparams_after_calib: bool = True

    # checkpoints
    checkpoint_dir: str = "_tmp_ckpts"
    save_every: int = 0            # 0 = disable periodic saves
    keep_last_k: int = 5
    best_metric_name: str = "loss" # which stat to track as "best" (e.g., 'loss' or 'bpp')
    resume_path: Optional[str] = None

    # scheduler (optional)
    scheduler_name: Optional[str] = None  # e.g., 'warmup_cosine', 'warmup_linear', 'noam', 'plateau'
    scheduler_total_steps: int = 0        # if 0, demo will default to 40
    scheduler_warmup_steps: int = 0
    scheduler_min_lr: float = 0.0
    scheduler_cycles: float = 0.5         # only for cosine

    # plateau-specific
    plateau_factor: float = 0.5
    plateau_patience: int = 200
    plateau_ema_alpha: float = 0.9
    plateau_threshold: float = 1e-3
    plateau_minimize: bool = True

    # profiling (optional)
    profile: bool = False           # measure wall time per step and compute tokens/sec
    profile_nvtx: bool = False      # wrap steps in NVTX ranges (Nsight)
    profile_log_mem: bool = False   # log CUDA memory (allocated/reserved)

    # debugging / diagnostics
    debug_align: bool = False           # compute CE alignment diagnostics inside the model
    debug_align_every: int = 200        # steps between debug logs (approx; model-side counter)
    debug_state_norm: bool = False      # log hidden-state norm stats before head projection
    debug_topk: int = 0                 # if >0, log top-k token ids/probs at the first debug step

    # state regularizer (energy non-expansion)
    state_w: float = 0.0                 # weight of L_state (0 => disabled)
    state_mask: str = "all"             # 'all' or 'collar'
    state_kind: str = "energy_contract" # future-proof: one kind for now

    # regularizers (optional)
    reg_gate_l0: float = 0.0            # weight for L0-like gate sparsity
    reg_gate_from_logits: bool = False  # if True, interpret gate input as logits
    reg_op_decor: float = 0.0           # weight for operator output decorrelation
    reg_energy_osc: float = 0.0         # tiny penalty for energy increases across steps
    reg_energy_hinge: float = 0.0       # margin before penalizing energy increases


# ----------------------------------- model ------------------------------------

class ContinuousBlock(nn.Module):
    """
    A single “physics + integration” block acting on one window.
    Expects input [B*T_w, W, D] and returns the same shape.
    """
    def __init__(
        self,
        d: int,
        groups: int,
        cheb_deg: int,
        laplacian: str,
        skew_rank: int,
        R_rank: int,
        integ_steps: int,
        dt: float,
        method: str,
        use_gauge: bool,
        use_op_bank: bool = False,
    ):
        super().__init__()
        self.d = d
        self.groups = groups
        self._use_op_bank = bool(use_op_bank)

        # Optional gauge (capacity scaling). If not present, identity.
        if use_gauge and _GaugeClass is not None:
            try:
                self.gauge = _GaugeClass(d=d, groups=groups)  # type: ignore
            except TypeError:
                self.gauge = _GaugeClass(d, groups)  # type: ignore
        else:
            self.gauge = nn.Identity()

        # ---------------- gate (optional, attached by LM if requested) ----------------
        if use_gauge and _GateClass is not None and True:  # gate is independent from gauge flag
            self.gate = None  # placeholder; created in LM using cfg
        else:
            self.gate = None

        # Chebyshev spectral filter over time axis.
        # IMPORTANT FIX: ChebFilter1D signature does NOT take `d`; use K/groups/laplacian.
        lap_mode = laplacian  # expected in {'path_causal','path','cycle'}
        self.cheb = ChebFilter1D(K=cheb_deg, groups=groups, laplacian=lap_mode)

        # Port-Hamiltonian field + integrator
        self.field = PortHamiltonianStep(
            d=d, groups=groups, skew_rank=skew_rank, R_rank=R_rank, traceless=True,
            use_nl_R=getattr(self, "_ph_use_nl_R", False),
            use_nl_H=getattr(self, "_ph_use_nl_H", False),
            nl_rank=getattr(self, "_ph_nl_rank", 8),
        )
        self.integ = Integrator(self.field, method=method, dt=dt, steps=integ_steps)

        # Optional operator bank inside PortHamiltonianStep (for ctx/top-k real)
        if self._use_op_bank and hasattr(self.field, 'set_op_bank'):
            # Causal-safe by default: only include Chebyshev (path_causal Laplacian)
            ops = [self.cheb]
            try:
                self.field.set_op_bank(ops)
            except Exception:
                pass

    def forward(self, x: torch.Tensor, step_hook=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B', W, D]
        Returns:
            post: [B', W, D]  (after integration)
            pre:  [B', W, D]  (before integration, Chebyshev output)
        """
        # Groupwise norm on features (per-group normalization)
        x = groupwise_norm(x, groups=self.groups, axis=-1, eps=1e-6)

        # Gauge — some gauges take only x; others (e.g., dual) may need a capacity signal.
        if isinstance(self.gauge, nn.Identity):
            g = x
        else:
            try:
                g = self.gauge(x)  # type: ignore
            except TypeError:
                # Minimal fallback capacity: ones (TODO: replace by real capacity signal)
                cap = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
                g = self.gauge(x, cap)  # type: ignore

        # Chebyshev filter (or defer to op-bank if enabled)
        if getattr(self, '_use_op_bank', False) and getattr(self.field, 'op_bank', None):
            y = g  # defer ops to field via op_bank
        else:
            y = self.cheb(g)

        # Integrate Port-Hamiltonian dynamics
        integ_kwargs = {}
        # gate (if the LM attached one)
        if getattr(self, "gate", None) is not None:
            integ_kwargs.update({
                "gate": self.gate,
                "gate_kwargs": {"topk_real": int(getattr(self, "_gate_topk_real", 2))},
            })
        # nl diffusion
        if getattr(self, "_nl_enabled", False):
            integ_kwargs["nl_diffusion"] = {
                "enabled": True,
                "p": float(getattr(self, "_nl_p", 1.5)),
                "huber_delta": float(getattr(self, "_nl_huber_delta", 1e-3)),
                "a_min": float(getattr(self, "_nl_a_min", 0.2)),
                "a_max": float(getattr(self, "_nl_a_max", 5.0)),
                "recompute_iters": int(getattr(self, "_nl_recompute_iters", 1)),
                "reduce": str(getattr(self, "_nl_reduce", "group")),
                "causal": True,
            }
        # ph-nl use
        if getattr(self, "_ph_enabled", False):
            integ_kwargs["ph_nl"] = {"enabled": True}
        # groups for operators along channel dim
        integ_kwargs["groups"] = int(self.groups)

        z = self.integ(y, step_hook=step_hook, **integ_kwargs)  # [B', W, D]
        return z, y


class ContinuousLM(nn.Module):
    """
    End-to-end LM:
      - TokenEmbedder (E_V + E_d) with optional tying
      - Windowed processing via ContinuousBlock (Cheb + PortHam + Integrator)
      - OutputHead tied to the embedder (default)
    """
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = TokenEmbedder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            pos_kind=cfg.pos_kind,
            max_len=None,                 # only used if learned
            tie_softmax=cfg.tie_softmax,
            factor_rank=cfg.factor_rank,
            pad_id=cfg.pad_id,
        )

        self.block = ContinuousBlock(
            d=cfg.d_model,
            groups=cfg.groups,
            cheb_deg=cfg.cheb_deg,
            laplacian=cfg.cheb_laplacian,
            skew_rank=cfg.skew_rank,
            R_rank=cfg.R_rank,
            integ_steps=cfg.steps,
            dt=cfg.dt,
            method=cfg.method,
            use_gauge=cfg.use_gauge,
            use_op_bank=cfg.use_op_bank,
        )
        self.block._gate_topk_real = int(cfg.gate_topk_real)
        # Propagate PH-NL flags
        self.block._ph_enabled   = bool(cfg.ph_nl_enabled)
        self.block._ph_use_nl_R  = bool(cfg.ph_use_nl_R)
        self.block._ph_use_nl_H  = bool(cfg.ph_use_nl_H)
        self.block._ph_nl_rank   = int(cfg.ph_nl_rank)
        # Propagate NL diffusion flags
        self.block._nl_enabled         = bool(cfg.nl_diffusion_enabled)
        self.block._nl_p               = float(cfg.nl_p)
        self.block._nl_huber_delta     = float(cfg.nl_huber_delta)
        self.block._nl_a_min           = float(cfg.nl_a_min)
        self.block._nl_a_max           = float(cfg.nl_a_max)
        self.block._nl_recompute_iters = int(cfg.nl_recompute_iters)
        self.block._nl_reduce          = str(cfg.nl_reduce)
        # Optional gate
        if cfg.use_gate and (_GateClass is not None):
            self.gate = _GateClass(d_model=cfg.d_model, groups=cfg.groups, mode="mul",
                                   per_channel=bool(cfg.gate_per_channel),
                                   sparse_topk=int(max(1, cfg.gate_topk)),
                                   gamma=float(cfg.gate_gamma))
            self.block.gate = self.gate
        else:
            self.gate = None

        # Canonical naming: the logits projection is `self.head` only.
        # We do not export `lm_head.*` aliases in state_dict; inference must use `head`.
        self.head = OutputHead(
            vocab_size=None if cfg.tie_softmax else cfg.vocab_size,
            d_model=cfg.d_model,
            tie_weight=cfg.tie_softmax,
            get_tied=(lambda: (self.embed._effective_token_weight(), self.embed.out_bias)) if cfg.tie_softmax else None,
            learn_bias=cfg.tie_softmax and (self.embed.out_bias is None),
            temperature=1.0,
            use_mul=bool(cfg.head_use_mul),
            mul_rank=int(cfg.head_mul_rank),
            mul_gate_init=float(cfg.head_mul_gate_init),
        )
        # Ensure compatibility with checkpoints that include a scalar output scale
        # Register inside the head module so the state_dict key is exactly 'head.weight_mul'
        if not hasattr(self.head, 'weight_mul'):
            self.head.register_parameter('weight_mul', nn.Parameter(torch.ones(())))

        # --- debug flags (cfg-only; set via arguments/config) ---
        self._aux_gate_pen = None
        self._aux_decor_pen = None
        self._aux_energy_pen = None
        self._debug_align = bool(cfg.debug_align)
        self._debug_align_every = int(max(1, cfg.debug_align_every))
        self._debug_state_norm = bool(cfg.debug_state_norm)
        self._debug_topk = int(max(0, cfg.debug_topk))
        self._debug_step = 0

        # Stitching loss (optional)
        self.stitch = None
        if cfg.stitch_w > 0.0 and (cfg.stitch_use_skl or cfg.stitch_use_lowd):
            self.stitch = StitchingOverlapLoss(
                W=cfg.W, O=cfg.O, layout="bnwd",
                use_skl=cfg.stitch_use_skl, skl_weight=1.0, temperature=cfg.stitch_temperature,
                use_lowd=cfg.stitch_use_lowd, lowd_weight=1.0, rank=cfg.stitch_rank,
            )

        # Dual for rate/bpp (log-space + variance-aware normalization)
        self.sda = None
        self._dual_use_log = bool(cfg.dual_use_log)
        self._dual_var_aware = bool(cfg.dual_var_aware)
        if cfg.use_dual_rate:
            tgt_bpp = cfg.target_bpp if (cfg.target_bpp is not None) else 0.6 * math.log2(cfg.vocab_size)
            target_for_sda = math.log(max(tgt_bpp, 1e-6)) if self._dual_use_log else float(tgt_bpp)
            # Keep a copy of the target in the working domain to normalize gaps
            self._sda_target = float(target_for_sda)
            # EMA stats for variance-aware normalization (on the working domain)
            self.register_buffer("_logbpp_mean", torch.tensor(self._sda_target, dtype=torch.float32))
            self.register_buffer("_logbpp_var", torch.tensor(1.0, dtype=torch.float32))
            # Dual object
            self.sda = DualSDA({"bpp": target_for_sda}, lrs={"bpp": cfg.dual_lr}, ema_alpha=cfg.dual_ema, use_sda=True)

        # Capacity/telemetry ledger (entropy-based capacity signal)
        self.ledger = CapacityLedger(vocab_size=cfg.vocab_size, ema_alpha=0.9)

        # --- Debug/sanity logs for new features ---
        if cfg.debug_log_gate and (self.gate is not None):
            logging.info(f"[cfg] gate enabled: topk={cfg.gate_topk}, per_channel={cfg.gate_per_channel}, gamma={cfg.gate_gamma}")
        if cfg.debug_log_A and bool(cfg.nl_diffusion_enabled):
            logging.info(f"[cfg] p-Lap enabled: p={cfg.nl_p}, δ={cfg.nl_huber_delta}, a∈[{cfg.nl_a_min},{cfg.nl_a_max}], recompute={cfg.nl_recompute_iters}, reduce={cfg.nl_reduce}")
        if cfg.ph_nl_enabled:
            logging.info(f"[cfg] PH-NL enabled: use_nl_R={cfg.ph_use_nl_R}, use_nl_H={cfg.ph_use_nl_H}, nl_rank={cfg.ph_nl_rank}")

    @torch.no_grad()
    def _shape_windows(self, h: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Slice h [B,T,D] along T into windows with overlap.
        Returns:
          xw: [B*nw, W, D]
          B, T, D, nw: original sizes to unshape later
        Note: utils.windows.slice_windows(axis=1) returns [B, D, n_windows, W] for torch.
        """
        B, T, D = h.shape
        W, O = self.cfg.W, self.cfg.O

        win = slice_windows(h, W, O, axis=1, pad=True)  # [B, D, n_win, W]
        x = win.permute(0, 2, 3, 1).contiguous()        # [B, n_win, W, D]
        B_, nwin, W_, D_ = x.shape
        assert B_ == B and W_ == W and D_ == D
        xw = x.view(B * nwin, W, D)                     # merge batch & windows
        return xw, B, T, D, nwin

    def _unshape_and_reconstruct(self, yw: torch.Tensor, B: int, T: int, D: int, nwin: int) -> torch.Tensor:
        """
        yw: [B*nw, W, D] → reconstruct full sequence [B, T, D] with OLA normalization.
        utils.windows.reconstruct_from_windows expects last dim=W and windows axis=-2.
        Our layout for reconstruct: [B, D, n_win, W].
        """
        W, O = self.cfg.W, self.cfg.O
        y = yw.view(B, nwin, W, D)                  # [B, n_win, W, D]
        y_for_recon = y.permute(0, 3, 1, 2).contiguous()  # [B, D, n_win, W]
        yr = reconstruct_from_windows(y_for_recon, n=T, W=W, O=O, axis=-2, window_fn="ones")
        return yr.permute(0, 2, 1).contiguous()     # [B, T, D]

    @torch.no_grad()
    def _maybe_debug_state(self, h_full: torch.Tensor, logits: torch.Tensor, tokens: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> None:
        if not self._debug_state_norm and not self._debug_align:
            return
        # step gating
        do_log = False
        if self.training:
            do_log = (self._debug_step % self._debug_align_every) == 0
            self._debug_step += 1
        else:
            # always log once in eval if enabled
            do_log = True
        if not do_log:
            return

        B, T, D = h_full.shape
        _, _, V = logits.shape
        device = h_full.device

        if self._debug_state_norm:
            # Norm of last hidden, and of all positions
            h_last = h_full[:, -1, :]                  # [B, D]
            n_last = torch.linalg.vector_norm(h_last, dim=-1)  # [B]
            n_all = torch.linalg.vector_norm(h_full.reshape(B * T, D), dim=-1)  # [B*T]
            mean_last = float(n_last.mean().item())
            std_last = float(n_last.std(unbiased=False).item())
            mean_all = float(n_all.mean().item())
            std_all = float(n_all.std(unbiased=False).item())
            logging.info(f"[debug] ||h_last|| mean={mean_last:.3f} std={std_last:.3f} | ||h|| mean={mean_all:.3f} std={std_all:.3f}")
            # Log entropy and logit range at last time-step
            last_logits = logits[:, -1, :]  # [B, V]
            logit_range = (last_logits.max() - last_logits.min()).item()
            p = F.softmax(last_logits, dim=-1)
            H = -(p * (p.clamp_min(1e-12).log())).sum(-1).mean().item()
            logging.info(f"[debug] last-step: logit_range={logit_range:.3f} | entropy={H:.3f}")

        if self._debug_align:
            # If we have tokens, compute canonical next-token shift diagnostics
            if tokens is not None:
                logits_s, targets_s, mask_s = shift_for_next_token(logits, tokens, mask=mask)
                Bs, Ts, V = logits_s.shape
                ce_full = F.cross_entropy(logits_s.reshape(-1, V), targets_s.reshape(-1), reduction="mean")
                if Ts > 1:
                    ce_left  = F.cross_entropy(logits_s[:, :-1, :].reshape(-1, V), targets_s[:, :-1].reshape(-1), reduction="mean")
                    ce_right = F.cross_entropy(logits_s[:, 1:,  :].reshape(-1, V), targets_s[:, 1: ].reshape(-1), reduction="mean")
                else:
                    ce_left = ce_full
                    ce_right = ce_full
                acc_full = float((logits_s.argmax(-1) == targets_s).float().mean().item())
                logging.info(f"[debug] CE(full)={float(ce_full.item()):.4f} | CE(<)={float(ce_left.item()):.4f} | CE(>)={float(ce_right.item()):.4f} | acc_full={acc_full:.3f}")

                if self._debug_topk > 0:
                    probs = F.softmax(logits_s[0, -1], dim=-1)
                    topk = min(self._debug_topk, V)
                    val, idx = torch.topk(probs, k=topk, dim=-1)
                    logging.info(f"[debug] top{topk}_ids(last@0)={idx.tolist()} | top{topk}_probs={val.tolist()}")
            else:
                # Fallback: diagnostics vs greedy teacher (not strictly next-token)
                y = logits.argmax(dim=-1)
                Bf, Tf, V = logits.shape
                ce_full = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="mean")
                if Tf > 1:
                    ce_left  = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), y[:, :-1].reshape(-1), reduction="mean")
                    ce_right = F.cross_entropy(logits[:, 1:,  :].reshape(-1, V), y[:, 1: ].reshape(-1), reduction="mean")
                else:
                    ce_left = ce_full
                    ce_right = ce_full
                acc_full = float((logits.argmax(-1) == y).float().mean().item())
                logging.info(f"[debug] CE(full)={float(ce_full.item()):.4f} | CE(<)={float(ce_left.item()):.4f} | CE(>)={float(ce_right.item()):.4f} | acc_full={acc_full:.3f}")
                if self._debug_topk > 0:
                    probs = F.softmax(logits[0, -1], dim=-1)
                    topk = min(self._debug_topk, V)
                    val, idx = torch.topk(probs, k=topk, dim=-1)
                    logging.info(f"[debug] top{topk}_ids(last@0)={idx.tolist()} | top{topk}_probs={val.tolist()}")

        return

    def forward_tokens(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Canonical forward for LM given batch of token ids and optional mask.

        Args:
            tokens: LongTensor [B, T]
            mask: Optional Bool/Byte [B, T]

        Returns:
            logits: [B, T, V]
            y_win_bnwd: [B, n_win, W, D]   # post-integration hidden by windows
            logits_win: Optional [B, n_win, W, V]  # logits per window if stitching KL is enabled
            h_pre: [B, T, D]  # pre-integration (Chebyshev) hidden, reconstructed
            h_post: [B, T, D] # post-integration hidden, reconstructed
        """
        # clear aux penalties for this call
        self._aux_gate_pen = None
        self._aux_decor_pen = None
        self._aux_energy_pen = None

        # 1) Embed tokens -> [B, T, D]
        h0 = self.embed(tokens)

        # 2) Slice into windows
        xw, B, T, D, nwin = self._shape_windows(h0)
        W, O = self.cfg.W, self.cfg.O
        use_border_only = self.training and self.cfg.grads_border_only and (O > 0) and (W > 2 * O)

        # Decide whether to log integrator steps
        want_integ_log = (self._debug_state_norm or self._debug_align or 
                          self.cfg.debug_log_A or self.cfg.debug_log_gate or self.cfg.debug_energy)
        # step gating: log at step 0 and every _debug_align_every steps
        step_gate = (self._debug_step % self._debug_align_every == 0)
        want_integ_log = want_integ_log and step_gate

        def _pretty_integ(info: Dict[str, float]) -> str:
            parts = []
            # time/step
            if 's_idx' in info and 'dt' in info:
                parts.append(f"s={int(info['s_idx'])}, dt={info['dt']:.3f}")
            # gate stats
            if 'gate_ent' in info:
                parts.append(f"gate_ent={info['gate_ent']:.3f}")
            if 'gate_active' in info:
                parts.append(f"gate_active={info['gate_active']:.2f}")
            # diffusion A(h) stats
            if 'A_mean' in info:
                parts.append(f"Aμ={info['A_mean']:.3f}")
            if 'A_min' in info and 'A_max' in info:
                parts.append(f"A∈[{info['A_min']:.3f},{info['A_max']:.3f}]")
            # simple energies
            if 'E_pre' in info and 'E_post' in info:
                parts.append(f"E: {info['E_pre']:.3f}→{info['E_post']:.3f}")
            # fallback: any other interesting keys
            for k in sorted(info.keys()):
                if k in ('s_idx','dt','gate_ent','gate_active','A_mean','A_min','A_max','E_pre','E_post'):
                    continue
                v = info[k]
                if isinstance(v, (int, float)):
                    parts.append(f"{k}={float(v):.3f}")
            return " | ".join(parts) if parts else str(info)

        if want_integ_log:
            def integ_hook(info: Dict[str, float]):
                logging.info(f"[integ] {_pretty_integ(info)}")
        else:
            integ_hook = None

        if use_border_only:
            # no-grad bulk
            with torch.no_grad():
                y_post_ng, y_pre_ng = self.block(xw, step_hook=integ_hook)  # [B*nw, W, D], [B*nw, W, D]
            # grad pass
            y_post_g, y_pre_g = self.block(xw, step_hook=integ_hook)      # [B*nw, W, D], [B*nw, W, D]
            # compose: bulk from no-grad, collars from grad
            y_post = y_post_ng.detach().clone()
            y_pre = y_pre_ng.detach().clone()
            if O > 0:
                y_post[:, :O, :] = y_post_g[:, :O, :]
                y_post[:, W - O:, :] = y_post_g[:, W - O:, :]
                y_pre[:, :O, :] = y_pre_g[:, :O, :]
                y_pre[:, W - O:, :] = y_pre_g[:, W - O:, :]

            # optional refresh on drift
            if self.cfg.refresh_on_drift:
                bulk = y_post_ng[:, O:W - O, :]
                coll = torch.cat([y_post_g[:, :O, :], y_post_g[:, W - O:, :]], dim=1)
                drift = (bulk.mean() - coll.mean()).abs().item()
                if drift > float(self.cfg.drift_thresh):
                    y_post = y_post_g
                    y_pre = y_pre_g
        else:
            # full-grad path
            y_post, y_pre = self.block(xw, step_hook=integ_hook)

        # --- optional regularizers on windowed tensors ---
        try:
            if (self.gate is not None) and (float(self.cfg.reg_gate_l0) > 0.0):
                # Try to obtain gate weights by reusing gate on pre-integrated features
                y_for_gate = y_pre  # [B*nw, W, D]
                out_gate = None
                try:
                    out_gate = self.block.gate(y_for_gate, return_gate=True)
                except TypeError:
                    # some gate impls may not accept return_gate; attempt without and skip
                    out_gate = None
                if isinstance(out_gate, (tuple, list)) and len(out_gate) >= 2:
                    w = out_gate[1]
                    # interpret as probs by default
                    pen = gate_l0_proxy(w, coeff=float(self.cfg.reg_gate_l0), from_logits=bool(self.cfg.reg_gate_from_logits))
                    self._aux_gate_pen = pen.to(y_for_gate)
        except Exception:
            pass

        try:
            if bool(self.cfg.use_op_bank) and (float(self.cfg.reg_op_decor) > 0.0) and getattr(self.block.field, 'op_bank', None):
                # Evaluate each operator once on the same pre state and decorrelate their pooled outputs
                Ys = []
                for op in list(self.block.field.op_bank):
                    try:
                        Ys.append(op.op(y_pre, ctx=None))  # [B*nw, W, D]
                    except Exception:
                        pass
                if len(Ys) >= 2:
                    pen = opbank_cosine_decorrelation(Ys, coeff=float(self.cfg.reg_op_decor))
                    self._aux_decor_pen = pen.to(y_pre)
        except Exception:
            pass

        # 3) Save hidden by window for stitching (post-integration)
        y_win_bnwd = y_post.view(B, nwin, W, D)  # [B, n_win, W, D]

        # 4) Reconstruct sequences (both post and pre states)
        h_post = self._unshape_and_reconstruct(y_post, B, T, D, nwin)        # [B, T, D]
        h_pre  = self._unshape_and_reconstruct(y_pre,  B, T, D, nwin)        # [B, T, D]

        # Energy oscillation penalty (post/pre reconstructed)
        if float(self.cfg.reg_energy_osc) > 0.0:
            with torch.no_grad():
                E_pair = torch.stack([
                    (h_pre*h_pre).mean(),
                    (h_post*h_post).mean()
                ], dim=0)  # [2]
            self._aux_energy_pen = energy_oscillation_penalty(E_pair, coeff=float(self.cfg.reg_energy_osc), hinge=float(self.cfg.reg_energy_hinge)).to(h_pre)

        logits = self.head(h_post, mask=None, mask_behavior="none", traceless=True)  # [B, T, V]
        # Apply legacy output scaling only if shape is safe (scalar or per-vocab [V]).
        # Newer checkpoints use multiplicative branches inside OutputHead and should NOT be re-scaled here.
        wm = getattr(self.head, 'weight_mul', None)
        if isinstance(wm, nn.Parameter):
            try:
                if wm.ndim == 0:
                    # scalar: ok
                    logits = logits * wm
                elif wm.ndim == 1 and wm.shape[0] == logits.shape[-1]:
                    # per-vocab vector [V]: ok
                    logits = logits * wm
                else:
                    # Any other shape (e.g., [D], [V,D], [D,V]) is handled inside OutputHead; skip here.
                    logging.info(
                        f"[loop] skip external weight_mul scale (shape={tuple(wm.shape)} not in {{(), (V,)}}); handled by OutputHead."
                    )
            except Exception as e:
                logging.warning(f"[loop] failed to apply legacy weight_mul scaling: {e}")

        # 5) Optional debug (state norms / CE alignment)
        self._maybe_debug_state(h_post, logits, tokens, mask)

        # --- debug energy log ---
        if self.cfg.debug_energy:
            with torch.no_grad():
                e_pre = float((h_pre*h_pre).mean().item())
                e_post = float((h_post*h_post).mean().item())
                logging.info(f"[debug] energy: E_pre={e_pre:.4f} → E_post={e_post:.4f}")

        # 6) Prepare logits per window if SKL is enabled
        logits_win = None
        if (self.stitch is not None) and self.cfg.stitch_use_skl:
            log_w = slice_windows(logits, self.cfg.W, self.cfg.O, axis=1, pad=True)  # [B, V, n_win, W]
            logits_win = log_w.permute(0, 2, 3, 1).contiguous()  # [B, n_win, W, V]

        return logits, y_win_bnwd, logits_win, h_pre, h_post
    def _state_loss(self, h_pre: torch.Tensor, h_post: torch.Tensor, mask_bt: torch.Tensor, kind: str = "energy_contract") -> Tuple[torch.Tensor, Dict[str, float]]:
        # h_pre, h_post: [B, T, D]; mask_bt: [B, T] bool
        # energy per token = ||h||^2 along channel dim
        e_pre = (h_pre * h_pre).sum(dim=-1)              # [B, T]
        e_post = (h_post * h_post).sum(dim=-1)           # [B, T]
        # relative positive increase (contractive w.r.t. energy)
        rel_increase = (e_post - e_pre) / (e_pre + 1e-8)
        pen = torch.relu(rel_increase)
        if mask_bt is not None:
            pen = pen[mask_bt]
            e_pre_m = e_pre[mask_bt]
            e_post_m = e_post[mask_bt]
        else:
            e_pre_m = e_pre.reshape(-1)
            e_post_m = e_post.reshape(-1)
        if pen.numel() > 0:
            loss_state = pen.mean()
            H_pre = float(e_pre_m.mean().detach().item())
            H_post = float(e_post_m.mean().detach().item())
            dH_pos = float(torch.relu(e_post_m - e_pre_m).mean().detach().item())
        else:
            loss_state = pen.sum() * 0.0
            H_pre = H_post = dH_pos = 0.0
        return loss_state, {"H_pre": H_pre, "H_post": H_post, "dH_pos": dH_pos}

    def _build_collar_seq_mask(self, B: int, T: int, device, dtype=torch.bool):
        """
        Build a [B,T] boolean mask marking positions in the collar (first O and last O of each window).
        """
        W, O = self.cfg.W, self.cfg.O
        base = torch.zeros(B, T, 1, device=device)
        win = slice_windows(base, W, O, axis=1, pad=True)  # [B, 1, nwin, W]
        mw = torch.zeros_like(win, dtype=torch.bool)
        if O > 0:
            mw[:, :, :, :O] = True
            mw[:, :, :, W - O:] = True
        # reconstruct to [B, 1, T]
        mr = reconstruct_from_windows(mw, n=T, W=W, O=O, axis=-2, window_fn="ones")
        # squeeze the singleton dim (1) to get [B, T]
        return (mr > 0).squeeze(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training/eval step.

        Preferred (canonical) path:
          batch = {'x': Long[B,T], 'y': Long[B,T], 'mask': Bool[B,T]}
          where x is the input sequence, y is next-token targets (shifted by +1),
          and mask marks valid positions (all ones for fixed windows).

        Backward-compatibility path:
          batch = {'tokens': Long[B,T], optional 'mask': Bool[B,T], optional 'targets': Long[B,T]}
          In this case we perform the global next-token shift inside this method.
        """
        pad_id = int(self.cfg.pad_id)

        # Detect canonical vs legacy batch format
        if ("x" in batch) and ("y" in batch):
            x: torch.Tensor = batch["x"]
            y: torch.Tensor = batch["y"]
            mask: Optional[torch.Tensor] = batch.get("mask", None)

            logits, y_win, logits_win, h_pre, h_post = self.forward_tokens(x, mask)  # [B,T,V], ...
            B, T, V = logits.shape
            assert V == self.cfg.vocab_size, f"head_dim(V)={V} != cfg.vocab_size={self.cfg.vocab_size}"

            # Ledger update on valid tokens (pads excluded)
            valid_mask = (y != pad_id)
            if mask is not None:
                valid_mask = valid_mask & mask.bool()
            led_stats = self.ledger.update(logits=logits, mask=valid_mask, loss=None)

            # Collar-only loss logic
            use_collar_ce = self.training and self.cfg.grads_border_only and (self.cfg.O > 0) and (self.cfg.W > 2 * self.cfg.O)
            # Build supervision mask: start from mask or all ones, AND with (y != pad_id)
            if mask is not None:
                sup_mask = mask.bool() & (y != pad_id)
            else:
                sup_mask = (y != pad_id)
            if use_collar_ce:
                collar_mask = self._build_collar_seq_mask(B, T, logits.device)
                sup_mask = sup_mask & collar_mask
            # Build mask for state regularizer
            if self.cfg.state_mask.lower() == "collar":
                collar_mask = self._build_collar_seq_mask(B, T, logits.device)
                state_mask_bt = collar_mask
            else:
                state_mask_bt = torch.ones(B, T, dtype=torch.bool, device=logits.device)
            # Also exclude PADs/right-padding using the available supervision mask logic
            if 'y' in locals():
                state_mask_bt = state_mask_bt & (y != pad_id)
                if mask is not None:
                    state_mask_bt = state_mask_bt & mask.bool()
            elif 'targets_s' in locals():
                # legacy path must use the shifted mask to match T after shift
                if 'mask_s' in locals() and (mask_s is not None):
                    state_mask_bt = state_mask_bt & mask_s.bool()
            # Extra alignment check (masked fraction only as info)
            if self._debug_align and ((self._debug_step % max(1, self._debug_align_every)) == 1):
                with torch.no_grad():
                    used_frac = float(sup_mask.float().mean().item())
                    logging.info(f"[debug] sup_mask used fraction={used_frac:.3f} (collar+valid)")
            ce = F.cross_entropy(
                logits.reshape(-1, V),
                y.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
            )
            # Mask and normalize by count
            ce_masked = ce.view(B, T)[sup_mask]
            if ce_masked.numel() > 0:
                loss = ce_masked.mean()
            else:
                loss = ce_masked.sum() * 0.0  # zero, keep grad

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = ((pred == y)[sup_mask].float().mean().item()) if sup_mask.any() else 0.0

            # Bits-per-token via helper (NLL path); aligned with y/mask
            bpp_tensor, _ = bits_per_token(logits, targets=y, mask=sup_mask, use_entropy=False, ignore_index=pad_id)
            bpp_val = float(bpp_tensor.detach().item())

            stats: Dict[str, float] = {"acc": acc, "bpp": bpp_val, "tokens": float(sup_mask.sum().item())}

            if led_stats:
                # keep only plain floats
                for k, v in led_stats.items():
                    if isinstance(v, (int, float)):
                        stats[k] = float(v)

        else:
            # Legacy path: accept 'tokens' and perform shift inside
            tokens: torch.Tensor = batch["tokens"]
            mask: Optional[torch.Tensor] = batch.get("mask", None)
            targets: Optional[torch.Tensor] = batch.get("targets", None)

            logits, y_win, logits_win, h_pre, h_post = self.forward_tokens(tokens, mask)
            if targets is None:
                targets = tokens  # teacher forcing baseline

            # Global next-token shift
            logits_s, targets_s, mask_s = shift_for_next_token(logits, targets, mask=mask)
            B, T, V = logits_s.shape
            assert V == self.cfg.vocab_size, f"head_dim(V)={V} != cfg.vocab_size={self.cfg.vocab_size}"

            # Ledger update on valid tokens (pads excluded)
            valid_mask = (targets_s != pad_id)
            if mask_s is not None:
                valid_mask = valid_mask & mask_s.bool()
            led_stats = self.ledger.update(logits=logits_s, mask=valid_mask, loss=None)

            use_collar_ce = self.training and self.cfg.grads_border_only and (self.cfg.O > 0) and (self.cfg.W > 2 * self.cfg.O)
            # Build supervision mask: start from mask_s or all ones, AND with (targets_s != pad_id)
            if mask_s is not None:
                sup_mask = mask_s.bool() & (targets_s != pad_id)
            else:
                sup_mask = (targets_s != pad_id)
            if use_collar_ce:
                collar_mask = self._build_collar_seq_mask(B, T, logits_s.device)
                sup_mask = sup_mask & collar_mask
            # Build mask for state regularizer
            if self.cfg.state_mask.lower() == "collar":
                collar_mask = self._build_collar_seq_mask(B, T, logits_s.device)
                state_mask_bt = collar_mask
            else:
                state_mask_bt = torch.ones(B, T, dtype=torch.bool, device=logits_s.device)
            # Also exclude PADs/right-padding using the available supervision mask logic
            if 'y' in locals():
                state_mask_bt = state_mask_bt & (y != pad_id)
                if mask is not None:
                    state_mask_bt = state_mask_bt & mask.bool()
            elif 'targets_s' in locals():
                # legacy path must use the shifted mask to match T after shift
                if 'mask_s' in locals() and (mask_s is not None):
                    state_mask_bt = state_mask_bt & mask_s.bool()
            # Extra alignment check (masked fraction only as info)
            if self._debug_align and ((self._debug_step % max(1, self._debug_align_every)) == 1):
                with torch.no_grad():
                    used_frac = float(sup_mask.float().mean().item())
                    logging.info(f"[debug] sup_mask used fraction={used_frac:.3f} (collar+valid)")
            ce = F.cross_entropy(
                logits_s.reshape(-1, V),
                targets_s.reshape(-1),
                ignore_index=pad_id,
                reduction="none",
            )
            ce_masked = ce.view(B, T)[sup_mask]
            if ce_masked.numel() > 0:
                loss = ce_masked.mean()
            else:
                loss = ce_masked.sum() * 0.0

            with torch.no_grad():
                pred = logits_s.argmax(dim=-1)
                acc = ((pred == targets_s)[sup_mask].float().mean().item()) if sup_mask.any() else 0.0

            bpp_tensor, _ = bits_per_token(logits_s, targets=targets_s, mask=sup_mask, use_entropy=False, ignore_index=pad_id)
            bpp_val = float(bpp_tensor.detach().item())

            stats = {"acc": acc, "bpp": bpp_val, "tokens": float(sup_mask.sum().item())}

            if led_stats:
                # keep only plain floats
                for k, v in led_stats.items():
                    if isinstance(v, (int, float)):
                        stats[k] = float(v)

        # Optional stitching loss on overlaps (mask pads/right-padding)
        if self.stitch is not None and (self.cfg.O > 0):
            st_loss_terms = []
            st_stats: Dict[str, float] = {}

            # Build window-level validity mask from token mask (exclude pads/right-padding)
            Bm, Tm = logits.shape[:2]
            if 'mask' in locals() and (mask is not None):
                base_mask = mask.bool()
            else:
                base_mask = torch.ones(Bm, Tm, dtype=torch.bool, device=logits.device)

            mw = slice_windows(base_mask.unsqueeze(1).float(), self.cfg.W, self.cfg.O, axis=1, pad=True)  # [B,1,nw,W]
            mask_win = (mw > 0.5).squeeze(1)  # [B,nw,W]

            if self.cfg.stitch_use_skl and logits_win is not None:
                skl, st = self.stitch(
                    logits=logits_win,
                    h_lowd=None,
                    mask=mask_win,
                )
                st_loss_terms.append(skl)
                st_stats.update({"st_skl": st.get("skl", 0.0)})
            if self.cfg.stitch_use_lowd and y_win is not None:
                lmse, st = self.stitch(
                    logits=None,
                    h_lowd=y_win,
                    mask=mask_win,
                )
                st_loss_terms.append(lmse)
                st_stats.update({"st_lowd": st.get("mse_lowd", 0.0)})
            if st_loss_terms:
                st_total = sum(st_loss_terms)
                loss = loss + self.cfg.stitch_w * st_total
                stats.update({"loss_stitch": float(st_total.detach().item()), "stitch_w": float(self.cfg.stitch_w)})
                stats.update(st_stats)

        # State regularizer (energy non-expansion, contractive penalty)
        if float(self.cfg.state_w) > 0.0:
            l_state, st_e = self._state_loss(h_pre, h_post, state_mask_bt, kind=self.cfg.state_kind)
            loss = loss + float(self.cfg.state_w) * l_state
            stats.update({
                "loss_state": float(l_state.detach().item()),
                "state_w": float(self.cfg.state_w),
                **st_e,
            })

        # Add optional regularizers collected in forward_tokens
        if isinstance(self._aux_gate_pen, torch.Tensor):
            loss = loss + self._aux_gate_pen
            stats["loss_gate_l0"] = float(self._aux_gate_pen.detach().item())
        if isinstance(self._aux_decor_pen, torch.Tensor):
            loss = loss + self._aux_decor_pen
            stats["loss_op_decor"] = float(self._aux_decor_pen.detach().item())
        if isinstance(self._aux_energy_pen, torch.Tensor):
            loss = loss + self._aux_energy_pen
            stats["loss_energy_osc"] = float(self._aux_energy_pen.detach().item())

        # Dual penalty in log-space with variance-aware normalization (unchanged)
        if self.sda is not None and ("bpp" in stats):
            with torch.no_grad():
                metric = math.log(max(stats["bpp"], 1e-8)) if self._dual_use_log else float(stats["bpp"])
                alpha = float(self.cfg.dual_ema)
                m = float(self._logbpp_mean.item())
                m_new = alpha * m + (1.0 - alpha) * metric
                v = float(self._logbpp_var.item())
                v_new = alpha * v + (1.0 - alpha) * (metric - m_new) ** 2
                self._logbpp_mean.fill_(m_new)
                self._logbpp_var.fill_(max(v_new, 1e-12))

                if self._dual_var_aware:
                    gap_norm = (metric - self._sda_target) / math.sqrt(float(self._logbpp_var.item()))
                    metric_for_sda = self._sda_target + gap_norm
                else:
                    metric_for_sda = metric

            pen, st_dual = self.sda.penalty({"bpp": torch.tensor(metric_for_sda, device=logits.device)})
            loss = loss + pen
            stats.update(st_dual)

        return loss, stats


# -------------------------------- training utils ------------------------------

def build_optimizer(model: nn.Module, cfg: LoopConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_step(model: ContinuousLM, batch: Dict[str, torch.Tensor], optim: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    optim.zero_grad(set_to_none=True)
    loss, stats = model(batch)
    loss.backward()
    optim.step()

    # Update duals (log-space, metric domain consistent with target)
    if hasattr(model, "sda") and (model.sda is not None) and ("bpp" in stats):
      try:
        metric = math.log(max(stats["bpp"], 1e-8)) if getattr(model, "_dual_use_log", True) else float(stats["bpp"])
        model.sda.update({"bpp": metric})
      except Exception:
        pass

    out = {"loss": float(loss.detach().item()), **stats}
    return out


# ----------------------------------- __main__ --------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[loop] %(message)s")
    torch.manual_seed(0)

    # Tiny toy data (char-level) to overfit a single batch
    texts = ["hello there", "general kenobi", "hello hello"]
    vocab = SimpleVocab.build_from_texts(texts, mode="char", add_unk=False)
    seqs = [vocab.encode(t, mode="char", add_bos=True, add_eos=True) for t in texts]
    # Prefer the dataloader collate if available; fallback to pack_batch otherwise
    if _collate_batch is not None:
        bos_id = getattr(vocab, "bos_id", None)
        eos_id = getattr(vocab, "eos_id", None)
        batch = _collate_batch(seqs, pad_id=vocab.pad_id, bos_id=bos_id, eos_id=eos_id, max_len=None)
        # Optional: collar mask for overlaps (not strictly needed by the model yet)
        if _build_collar_mask is not None:
            lens = batch["mask"].sum(dim=1).tolist()
            cfg = LoopConfig(
                W=16, O=4,
                vocab_size=vocab.size, d_model=64, groups=8,
                cheb_deg=6, cheb_laplacian="path_causal",
                skew_rank=8, R_rank=4,
                steps=2, dt=0.5, method="ph-strang-nl",
                tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
                pad_id=vocab.pad_id,
                lr=3e-3, weight_decay=0.0,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_gauge=True,
                stitch_w=0.1, stitch_use_lowd=True, stitch_use_skl=False,
                scheduler_name="warmup_cosine",
                scheduler_total_steps=40,
                scheduler_warmup_steps=5,
                scheduler_min_lr=3e-4,
                nl_diffusion_enabled=True,
                head_use_mul=True,
                use_op_bank=True,
            )
            batch["collar_mask"] = _build_collar_mask(lens, W=cfg.W, O=cfg.O)
        else:
            cfg = LoopConfig(
                W=16, O=4,
                vocab_size=vocab.size, d_model=64, groups=8,
                cheb_deg=6, cheb_laplacian="path_causal",
                skew_rank=8, R_rank=4,
                steps=2, dt=0.5, method="ph-strang-nl",
                tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
                pad_id=vocab.pad_id,
                lr=3e-3, weight_decay=0.0,
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_gauge=True,
                stitch_w=0.1, stitch_use_lowd=True, stitch_use_skl=False,
                scheduler_name="warmup_cosine",
                scheduler_total_steps=40,
                scheduler_warmup_steps=5,
                scheduler_min_lr=3e-4,
                nl_diffusion_enabled=True,
                head_use_mul=True,
                use_op_bank=True,
            )
    else:
        tokens, mask = pack_batch(seqs, pad_id=vocab.pad_id)
        batch = {"tokens": tokens, "mask": mask, "targets": tokens.clone()}
        cfg = LoopConfig(
            W=16, O=4,
            vocab_size=vocab.size, d_model=64, groups=8,
            cheb_deg=6, cheb_laplacian="path_causal",
            skew_rank=8, R_rank=4,
            steps=2, dt=0.5, method="ph-strang-nl",
            tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
            pad_id=vocab.pad_id,
            lr=3e-3, weight_decay=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_gauge=True,
            stitch_w=0.1, stitch_use_lowd=True, stitch_use_skl=False,
            scheduler_name="warmup_cosine",
            scheduler_total_steps=40,
            scheduler_warmup_steps=5,
            scheduler_min_lr=3e-4,
            profile=True,
            profile_nvtx=False,
            profile_log_mem=True,
            nl_diffusion_enabled=True,
            head_use_mul=True,
            use_op_bank=True,
        )

    device = torch.device(cfg.device)
    model = ContinuousLM(cfg).to(device)
    cfg_dict = asdict(cfg)

    # Move batch to device
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)

    opt = build_optimizer(model, cfg)

    # Throughput meter for profiling
    tm = ThroughputMeter()


    # Optional LR scheduler
    sch = None
    if cfg.scheduler_name:
        name = cfg.scheduler_name.lower()
        # total steps default for the demo if unspecified
        total_steps = cfg.scheduler_total_steps if cfg.scheduler_total_steps > 0 else 40
        if name in ("warmup_cosine", "cosine"):
            sch = make_scheduler(
                name, opt,
                total_steps=total_steps,
                warmup_steps=max(0, cfg.scheduler_warmup_steps),
                base_lr=cfg.lr,
                min_lr=cfg.scheduler_min_lr,
                cycles=cfg.scheduler_cycles,
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
                factor=cfg.plateau_factor,
                patience=cfg.plateau_patience,
                ema_alpha=cfg.plateau_ema_alpha,
                threshold=cfg.plateau_threshold,
                minimize=cfg.plateau_minimize,
                base_lr=cfg.lr,
                min_lr=cfg.scheduler_min_lr,
            )



    # Checkpoint manager (periodic + best)
    ckpt_mgr = CheckpointManager(
        cfg.checkpoint_dir,
        keep_last_k=cfg.keep_last_k,
        best_tag=cfg.best_metric_name,
        is_better=(lambda new, best: (best is None) or (new < best)),
    )

    # Optional: brief post-pruning calibration (Phase 3, 9.2)
    if cfg.quant_calibrate_after_prune:
        def _repeat_batch(b, n):
            for _ in range(n):
                # yield as single positional arg so model(batch) is called by calibrate_model
                yield (b,)
        with torch.no_grad():
            calibrate_model(model, data_iter=_repeat_batch(batch, cfg.quant_calib_batches), num_batches=cfg.quant_calib_batches)
        if cfg.freeze_qparams_after_calib:
            toggle_fakequant_collect(model, False)

    # Overfit a single batch a few steps (sanity)
    steps = 40
    for s in range(steps):
        # Profiling: measure step wall time (CUDA-synced) and wrap with optional NVTX
        if cfg.profile or cfg.profile_nvtx:
            timer = Timer(sync_cuda=True)
            ctx = nvtx_range(f"step_{s}") if cfg.profile_nvtx else nullcontext()
            with ctx, prof_record_function("train_step"):
                timer.start()
                stats = train_step(model, batch, opt)
                dt = timer.stop()  # seconds
            # Update throughput (items = tokens processed this step)
            tok = int(stats.get("tokens", 0))
            if tok > 0:
                tm.update(tok, dt)
        else:
            stats = train_step(model, batch, opt)
            dt = None

        cur_lr = opt.param_groups[0]["lr"]
        extra_prof = ""
        if cfg.profile and tm.time_s > 0:
            extra_prof += f", ips={tm.ips:.1f}"  # tokens per second
        if cfg.profile_log_mem and torch.cuda.is_available():
            m = gpu_mem()
            extra_prof += f", gpuMB={m['allocated']:.1f}/{m['reserved']:.1f}"
        if (s % 5) == 0 or s == steps - 1:
            logging.info(
                f"step {s:02d} | loss={stats['loss']:.4f} | acc={stats.get('acc', 0.0):.3f} "
                f"| tokens={int(stats.get('tokens', 0))} | lr={cur_lr:.2e}{extra_prof}"
            )

        # Periodic checkpoint save
        ckpt_mgr.periodic_save(
            model=model, optimizer=opt, scheduler=None,
            step=s, epoch=0, cfg=cfg_dict, every=cfg.save_every,
            extra={"tokens_seen": int(stats.get("tokens", 0))},
        )
        # Update best using chosen metric (fallback to loss)
        metric_name = cfg.best_metric_name
        metric_val = float(stats.get(metric_name, stats.get("loss", 0.0)))
        ckpt_mgr.update_best(
            metric_value=metric_val,
            model=model, optimizer=opt, scheduler=None,
            step=s, epoch=0, cfg=cfg_dict,
            extra={metric_name: metric_val},
        )

        # Step scheduler (supports metric-driven and step-driven)
        if sch is not None:
            sch.step(s, metrics=float(stats.get("loss", 0.0)))

    # Forward-only throughput sanity
    model.eval()
    with torch.no_grad():
        logits, *_ = model.forward_tokens(batch["tokens"], batch["mask"])
        B, T, V = logits.shape
        print(f"[loop] forward logits: {B}×{T}×{V} ✓")

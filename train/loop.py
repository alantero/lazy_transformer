# train/loop.py
# Minimal end-to-end training loop (Phase 3).
# Pipeline per window: tokens → embed → (groupwise norm) → (gauge) → cheb → portham → integrate → head → CE.
# Windows are overlap-added back to the full sequence before computing the next-token CE.
# PyTorch-only. Uses the modules we built in this project.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import logging
import math
import torch
import torch.nn as nn

# ------------------------------ safe imports ----------------------------------
# When executed as a script (python train/loop.py), make repo root importable.
if __package__ in (None, ""):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from data.tok_embed import TokenEmbedder, SimpleVocab, pack_batch  # preferred
except Exception:
    from data.tokenize import TokenEmbedder, SimpleVocab, pack_batch   # fallback

from utils.windows import slice_windows, reconstruct_from_windows
from modules.normalize import groupwise_norm
from operators.cheb import ChebFilter1D
from modules.portham import PortHamiltonianStep
from modules.integrator import Integrator
from modules.output_head import OutputHead
from losses.task_loss import sequence_cross_entropy, shift_for_next_token
from losses.rate_loss import bits_per_token
from losses.stitching_loss import StitchingOverlapLoss
from optim.sda import DualSDA
from modules.quant import calibrate_model, toggle_fakequant_collect


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
    cheb_laplacian: str = "cycle"  # 'cycle' (periodic) | 'path' (Dirichlet)

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
    ):
        super().__init__()
        self.d = d
        self.groups = groups

        # Optional gauge (capacity scaling). If not present, identity.
        if use_gauge and _GaugeClass is not None:
            try:
                self.gauge = _GaugeClass(d=d, groups=groups)  # type: ignore
            except TypeError:
                self.gauge = _GaugeClass(d, groups)  # type: ignore
        else:
            self.gauge = nn.Identity()

        # Chebyshev spectral filter over time axis.
        # IMPORTANT FIX: ChebFilter1D signature does NOT take `d`; use K/groups/laplacian.
        self.cheb = ChebFilter1D(K=cheb_deg, groups=groups, laplacian=laplacian)

        # Port-Hamiltonian field + integrator
        self.field = PortHamiltonianStep(
            d=d, groups=groups, skew_rank=skew_rank, R_rank=R_rank, traceless=True
        )
        self.integ = Integrator(self.field, method=method, dt=dt, steps=integ_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B', W, D]
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

        # Chebyshev filter (temporal operator)
        y = self.cheb(g)  # API computes its own 1D Laplacian per window length and mode

        # Integrate Port-Hamiltonian dynamics
        z = self.integ(y)  # [B', W, D]
        return z


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
        )

        # Output head: tie to embedder weights (lazy-friendly)
        self.head = OutputHead(
            vocab_size=None if cfg.tie_softmax else cfg.vocab_size,
            d_model=cfg.d_model,
            tie_weight=cfg.tie_softmax,
            get_tied=(lambda: (self.embed._effective_token_weight(), self.embed.out_bias)) if cfg.tie_softmax else None,
            learn_bias=cfg.tie_softmax and (self.embed.out_bias is None),
            temperature=1.0,
        )

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
            tgt_bpp = cfg.target_bpp if cfg.target_bpp is not None else 0.6 * math.log2(cfg.vocab_size)
            target_for_sda = math.log(max(tgt_bpp, 1e-6)) if self._dual_use_log else float(tgt_bpp)
            # Keep a copy of the target in the working domain to normalize gaps
            self._sda_target = float(target_for_sda)
            # EMA stats for variance-aware normalization (on the working domain)
            self.register_buffer("_logbpp_mean", torch.tensor(self._sda_target, dtype=torch.float32))
            self.register_buffer("_logbpp_var", torch.tensor(1.0, dtype=torch.float32))
            # Dual object
            self.sda = DualSDA({"bpp": target_for_sda}, lrs={"bpp": cfg.dual_lr}, ema_alpha=cfg.dual_ema, use_sda=True)

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

    def forward_tokens(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        tokens: [B,T] → logits: [B,T,V], y_win_bnwd: [B, n_win, W, D] (optional), logits_win: [B, n_win, W, V] (optional)
        """
        h0 = self.embed(tokens)                             # [B,T,D]
        # Per-window processing
        xw, B, T, D, nwin = self._shape_windows(h0)

        W, O = self.cfg.W, self.cfg.O
        use_border_only = self.training and self.cfg.grads_border_only and (O > 0) and (W > 2 * O)

        if use_border_only:
            # 1) no-grad full-window pass (bulk cheap)
            with torch.no_grad():
                y_ng = self.block(xw)                       # [B*nw, W, D] (detached)
            # 2) grad pass (recompute) – we will keep only collars from this result
            y_g = self.block(xw)                            # [B*nw, W, D] (with grads)

            # compose final window outputs: bulk from no-grad, collars from grad
            y = y_ng.detach().clone()
            if O > 0:
                y[:, :O, :] = y_g[:, :O, :]
                y[:, W - O:, :] = y_g[:, W - O:, :]

            # optional refresh by drift: compare mean of bulk vs collars
            if self.cfg.refresh_on_drift:
                bulk = y_ng[:, O:W - O, :]
                coll = torch.cat([y_g[:, :O, :], y_g[:, W - O:, :]], dim=1)
                drift = (bulk.mean() - coll.mean()).abs().item()
                if drift > float(self.cfg.drift_thresh):
                    # recompute (or simply keep) the full grad result
                    y = y_g
        else:
            # regular full-grad path
            y = self.block(xw)

        # Keep windowed hidden for stitching loss if needed
        y_win_bnwd = y.view(B, nwin, W, D)  # [B, n_win, W, D]

        h_full = self._unshape_and_reconstruct(y, B, T, D, nwin)  # [B,T,D]
        # Head (tied or learned). Keep logits unmasked here; CE handles ignore_index.
        logits = self.head(h_full, mask=None, mask_behavior="none", traceless=True)

        # Prepare logits windows [B, n_win, W, V] for stitching if needed
        logits_win = None
        if self.stitch is not None and self.cfg.stitch_use_skl:
            log_w = slice_windows(logits, self.cfg.W, self.cfg.O, axis=1, pad=True)  # [B, V, n_win, W]
            logits_win = log_w.permute(0, 2, 3, 1).contiguous()  # [B, n_win, W, V]

        return logits, y_win_bnwd if (self.stitch is not None and self.cfg.stitch_use_lowd) else None, logits_win

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        batch = { 'tokens': Long[B,T], optional 'mask': Bool[B,T], optional 'targets': Long[B,T] }
        Computes next-token CE on the FULL sequence (global shift), not per-window.
        """
        tokens: torch.Tensor = batch["tokens"]
        mask: Optional[torch.Tensor] = batch.get("mask", None)
        targets: Optional[torch.Tensor] = batch.get("targets", None)

        logits, y_win, logits_win = self.forward_tokens(tokens, mask)

        if targets is None:
            targets = tokens  # teacher forcing: predict next of the input sequence

        # Next-token shift (global)
        logits_s, targets_s, mask_s = shift_for_next_token(logits, targets, mask=mask)

        loss, stats = sequence_cross_entropy(
            logits_s, targets_s, mask=mask_s, ignore_index=self.cfg.pad_id,
            label_smoothing=0.0, reduction="mean"
        )

        # Bits-per-token (for ledger/rate): use NLL path
        bpp_tensor, _st_bpp = bits_per_token(
            logits_s, targets=targets_s, mask=mask_s, use_entropy=False, ignore_index=self.cfg.pad_id
        )
        bpp_val = float(bpp_tensor.detach().item())
        stats["bpp"] = bpp_val

        # Optional stitching loss on overlaps
        if self.stitch is not None and (self.cfg.O > 0):
            st_loss_terms = []
            st_stats: Dict[str, float] = {}
            if self.cfg.stitch_use_skl and logits_win is not None:
                skl, st = self.stitch(
                    logits=logits_win,
                    h_lowd=None,
                    mask=None,
                )
                st_loss_terms.append(skl)
                st_stats.update({"st_skl": st.get("skl", 0.0)})
            if self.cfg.stitch_use_lowd and y_win is not None:
                lmse, st = self.stitch(
                    logits=None,
                    h_lowd=y_win,
                    mask=None,
                )
                st_loss_terms.append(lmse)
                st_stats.update({"st_lowd": st.get("mse_lowd", 0.0)})
            if st_loss_terms:
                st_total = sum(st_loss_terms)
                loss = loss + self.cfg.stitch_w * st_total
                stats.update({"loss_stitch": float(st_total.detach().item()), "stitch_w": float(self.cfg.stitch_w)})
                stats.update(st_stats)

        # Dual penalty in log-space with variance-aware normalization
        if self.sda is not None:
            with torch.no_grad():
                metric = math.log(max(bpp_val, 1e-8)) if self._dual_use_log else bpp_val
                # EMA mean/var on the working domain
                alpha = float(self.cfg.dual_ema)
                m = float(self._logbpp_mean.item())
                # Update mean first
                m_new = alpha * m + (1.0 - alpha) * metric
                # Update variance as EMA of squared deviation around the updated mean
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
    tokens, mask = pack_batch(seqs, pad_id=vocab.pad_id)
    batch = {"tokens": tokens, "mask": mask, "targets": tokens.clone()}

    cfg = LoopConfig(
        W=16, O=4,
        vocab_size=vocab.size, d_model=64, groups=8,
        cheb_deg=6, cheb_laplacian="cycle",
        skew_rank=8, R_rank=4,
        steps=2, dt=0.5, method="heun",
        tie_softmax=True, factor_rank=16, pos_kind="sinusoidal",
        pad_id=vocab.pad_id,
        lr=3e-3, weight_decay=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_gauge=True,
        stitch_w=0.1, stitch_use_lowd=True, stitch_use_skl=False,
    )

    device = torch.device(cfg.device)
    model = ContinuousLM(cfg).to(device)

    # Move batch to device
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)

    opt = build_optimizer(model, cfg)

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
        stats = train_step(model, batch, opt)
        if (s % 5) == 0 or s == steps - 1:
            logging.info(f"step {s:02d} | loss={stats['loss']:.4f} | acc={stats.get('acc', 0.0):.3f} | tokens={int(stats.get('tokens', 0))}")

    # Forward-only throughput sanity
    model.eval()
    with torch.no_grad():
        logits, *_ = model.forward_tokens(batch["tokens"], batch["mask"])
        B, T, V = logits.shape
        print(f"[loop] forward logits: {B}×{T}×{V} ✓")

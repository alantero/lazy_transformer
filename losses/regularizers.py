# losses/regularizers.py
# Regularizers and auxiliary penalties (PyTorch-only, no repo-global deps).
# Includes: parameter L1/L2, group-lasso, orthogonality, temporal TV/smoothness,
# gate sparsity (Bernoulli-KL), and a small aggregator to sum enabled terms.

from __future__ import annotations
from typing import Iterable, Callable, Dict, Tuple, Optional, Any, Union, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Array = Union[Tensor, float]


# ------------------------------ similarity helpers ---------------------------

def _pairwise_cosine_matrix(Ys: Sequence[Tensor], eps: float = 1e-8) -> Optional[Tensor]:
    """
    Build a pairwise cosine-similarity matrix over a list of feature tensors.
    Each tensor Y has shape [*, D] or [B, T, D]; we flatten all leading dims to N×D, L2-normalize, and compute C = U U^T.
    Returns [M, M] (M=len(Ys)), or None if len(Ys) < 2.
    """
    M = len(Ys)
    if M < 2:
        return None
    U_list = []
    for Y in Ys:
        if Y is None:
            continue
        Y2 = Y
        if Y2.dim() >= 2:
            Y2 = Y2.reshape(-1, Y2.shape[-1])  # [N, D]
        else:
            Y2 = Y2.unsqueeze(0)
        U = F.normalize(Y2.float(), p=2, dim=-1, eps=eps)  # [N, D]
        # pooled mean direction for stability (avoid O(N^2) memory)
        u_bar = U.mean(dim=0, keepdim=True)                 # [1, D]
        U_list.append(u_bar)
    if not U_list:
        return None
    Ustack = torch.cat(U_list, dim=0)   # [M, D]
    C = (Ustack @ Ustack.t()).clamp(min=-1.0, max=1.0)  # [M, M]
    return C


# ------------------------------ param utilities ------------------------------

def _iter_params(
    params_or_modules: Union[Iterable[nn.Parameter], nn.Module, Iterable[nn.Module]],
    *,
    requires_grad_only: bool = True,
    include_bias: bool = True,
    include_norm: bool = True,
) -> Iterable[nn.Parameter]:
    """
    Yield parameters from either:
      - an iterable of nn.Parameter
      - a single nn.Module
      - an iterable of nn.Module
    with simple filters (requires_grad, bias/norm include).
    """
    def _is_norm(m: nn.Module) -> bool:
        return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d))

    if isinstance(params_or_modules, nn.Module):
        for m in params_or_modules.modules():
            for name, p in m.named_parameters(recurse=False):
                if requires_grad_only and (not p.requires_grad):
                    continue
                if (not include_bias) and name.endswith("bias"):
                    continue
                if (not include_norm) and _is_norm(m):
                    continue
                yield p
        return

    # Iterable of modules?
    it = iter(params_or_modules)
    try:
        first = next(it)
    except StopIteration:
        return
    # Put back the first
    params_list: Iterable[nn.Parameter]
    if isinstance(first, nn.Module):
        modules = [first, *it]
        for mod in modules:
            yield from _iter_params(mod, requires_grad_only=requires_grad_only, include_bias=include_bias, include_norm=include_norm)
        return
    else:
        # We assume iterable of parameters
        params = [first, *it]  # type: ignore[assignment]
        for p in params:  # type: ignore[assignment]
            if requires_grad_only and (not p.requires_grad):
                continue
            yield p  # type: ignore[misc]


# ------------------------------ param penalties ------------------------------

def l2_weight_decay(
    params_or_modules: Union[Iterable[nn.Parameter], nn.Module, Iterable[nn.Module]],
    coeff: float,
    *,
    include_bias: bool = False,
    include_norm: bool = False,
) -> Tensor:
    """∑ ||θ||² with optional exclusion of bias/normalization parameters."""
    if coeff == 0.0:
        return torch.zeros((), device="cpu")
    total = None
    for p in _iter_params(params_or_modules, include_bias=include_bias, include_norm=include_norm):
        term = torch.sum(p.float() ** 2)
        total = term if total is None else (total + term)
    if total is None:
        total = torch.zeros((), device="cpu")
    return coeff * total


def l1_sparsity(
    params_or_modules: Union[Iterable[nn.Parameter], nn.Module, Iterable[nn.Module]],
    coeff: float,
    *,
    include_bias: bool = True,
    include_norm: bool = True,
) -> Tensor:
    """∑ ||θ||₁."""
    if coeff == 0.0:
        return torch.zeros((), device="cpu")
    total = None
    for p in _iter_params(params_or_modules, include_bias=include_bias, include_norm=include_norm):
        term = torch.sum(torch.abs(p.float()))
        total = term if total is None else (total + term)
    if total is None:
        total = torch.zeros((), device="cpu")
    return coeff * total


def group_lasso(
    x: Tensor,
    *,
    groups: int,
    axis: int = -1,
    coeff: float = 1e-4,
    eps: float = 1e-12,
) -> Tensor:
    """
    Group-Lasso penalty over last (or chosen) dimension.
      - x is split into 'groups' equal-sized chunks along `axis`
      - penalty = sum over groups of L2-norm of each group vector
    """
    if coeff == 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if axis < 0:
        axis += x.ndim
    C = x.shape[axis]
    if C % groups != 0:
        raise ValueError(f"'groups' must divide the chosen axis size. Got size={C}, groups={groups}.")
    cg = C // groups
    # Move target axis to last; reshape to (..., G, cg)
    x_last = x.movedim(axis, -1)
    new_shape = x_last.shape[:-1] + (groups, cg)
    xl = x_last.reshape(new_shape)
    # L2 across intra-group channels, then sum across groups and batches
    norms = torch.sqrt(torch.clamp((xl ** 2).sum(dim=-1), min=eps))  # (..., G)
    return coeff * norms.sum()


def orthogonality_penalty(W: Tensor, *, coeff: float = 1e-4, identity: Optional[Tensor] = None) -> Tensor:
    """
    Frobenius ||WᵀW - I||². If I not provided, uses identity of size W.shape[1].
    """
    if coeff == 0.0:
        return torch.zeros((), device=W.device, dtype=W.dtype)
    k = W.shape[1]
    I = identity if identity is not None else torch.eye(k, device=W.device, dtype=W.dtype)
    gram = W.transpose(-2, -1) @ W
    return coeff * torch.sum((gram - I) ** 2)


def lowrank_factor_orthogonality(A: Tensor, B: Optional[Tensor] = None, *, coeff: float = 1e-4) -> Tensor:
    """
    Encourage A (and optionally B) to have orthonormal columns: ||AᵀA - I||² + ||BᵀB - I||².
    Works batch/group wise if A is [..., Cg, r].
    """
    if coeff == 0.0:
        return torch.zeros((), device=A.device, dtype=A.dtype)
    def _pen(F: Tensor) -> Tensor:
        r = F.shape[-1]
        I = torch.eye(r, device=F.device, dtype=F.dtype)
        G = F.transpose(-2, -1) @ F
        return torch.sum((G - I) ** 2)
    total = _pen(A)
    if B is not None:
        total = total + _pen(B)
    return coeff * total


# ------------------------------ sequence penalties ---------------------------

def total_variation(
    x: Tensor,
    *,
    axis: int = 1,         # time/sequence dim in [B,T,D]
    p: Literal[1, 2] = 1,
    coeff: float = 1e-4,
) -> Tensor:
    """
    TV penalty across the given axis:
      p=1: ∑ |Δ x|      (robust edges)
      p=2: ∑ ||Δ x||²   (smooth)
    """
    if coeff == 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if axis < 0:
        axis += x.ndim
    dx = torch.diff(x, dim=axis)
    if p == 1:
        pen = torch.sum(torch.abs(dx))
    elif p == 2:
        pen = torch.sum(dx ** 2)
    else:
        raise ValueError("p must be 1 or 2.")
    return coeff * pen


def laplacian_smoothness(
    x: Tensor,
    L: Tensor,
    *,
    coeff: float = 1e-4,
    time_axis: int = 1,
) -> Tensor:
    """
    Graph/temporal smoothness: ∑_{b,d} x_{b,: ,d}^T L x_{b,: ,d}
    Inputs:
      x: [B, T, D] (or arbitrary as long as L applies along `time_axis`)
      L: [T, T] symmetric PSD laplacian-like matrix
    """
    if coeff == 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if time_axis < 0:
        time_axis += x.ndim
    # Move time axis to dim=1 and features to last for convenience
    x_std = x.movedim(time_axis, 1)  # [B, T, *]
    # Collapse trailing dims (treat everything except B and T as feature-index)
    B, T = x_std.shape[:2]
    tail = int(torch.numel(x_std) // (B * T))
    x_btF = x_std.reshape(B, T, tail)  # [B, T, F]
    # y = L @ x  along T
    y = torch.einsum("tu, b u f -> b t f", L.to(x_btF), x_btF)
    # sum over b,t,f of x*y
    pen = torch.einsum("btf, btf -> ", x_btF, y)
    return coeff * pen


# ------------------------------ gate penalties -------------------------------

def gate_l1(g: Tensor, *, coeff: float = 1e-4) -> Tensor:
    """
    L1 on gates g ∈ [0,1] to encourage sparsity (closed gates).
    """
    if coeff == 0.0:
        return torch.zeros((), device=g.device, dtype=g.dtype)
    return coeff * torch.sum(torch.abs(g))


def gate_bernoulli_kl(
    g: Tensor,
    *,
    p_prior: float = 0.1,
    coeff: float = 1e-4,
    eps: float = 1e-8,
) -> Tensor:
    """
    KL( Bernoulli(g) || Bernoulli(p_prior) ) averaged over positions, scaled by coeff.
    Encourages g toward sparse prior if p_prior < 0.5.
    """
    if coeff == 0.0:
        return torch.zeros((), device=g.device, dtype=g.dtype)
    p = torch.clamp(g, eps, 1.0 - eps)
    q = torch.tensor(float(p_prior), device=g.device, dtype=g.dtype).clamp(eps, 1.0 - eps)
    kl = p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (torch.log(1.0 - p) - torch.log(1.0 - q))
    return coeff * kl.mean()


def gate_l0_proxy(
    g: Tensor,
    *,
    coeff: float = 1e-4,
    from_logits: bool = False,
    beta: float = 2.0,
) -> Tensor:
    """
    L0-like surrogate for gates in [0,1] or for their logits.
    If from_logits=True, we map logits `a` to expected activation via a smooth hard-concrete CDF proxy: p = sigmoid(a / beta).
    Else, we assume `g` \in [0,1] and take E[L0] ≈ sum(g), which coincides with L1 on [0,1].
    """
    if coeff == 0.0:
        return torch.zeros((), device=g.device, dtype=g.dtype)
    if from_logits:
        p = torch.sigmoid(g.float() / beta)
        return coeff * p.sum()
    else:
        return coeff * g.float().clamp(0.0, 1.0).sum()


# ------------------------------ operator-bank penalties ----------------------

def opbank_cosine_decorrelation(
    Ys: Sequence[Tensor],
    *,
    coeff: float = 1e-4,
    eps: float = 1e-8,
    exclude_diag: bool = True,
) -> Tensor:
    """
    Penalize cosine similarity between a list of operator outputs.
    Args:
      Ys: list/seq of tensors (each [..., D]); e.g., outputs of different ops on the same input.
      coeff: scale; set 0 to disable.
      exclude_diag: if True, ignore self-similarities on the diagonal.
    Returns: scalar penalty.
    """
    if coeff == 0.0 or len(Ys) <= 1:
        return torch.zeros((), dtype=torch.float32)
    C = _pairwise_cosine_matrix(Ys, eps=eps)
    if C is None:
        return torch.zeros((), dtype=torch.float32)
    if exclude_diag:
        M = C.shape[0]
        mask = torch.ones_like(C, dtype=torch.bool)
        mask.fill_(True)
        mask.fill_diagonal_(False)
        val = torch.abs(C[mask]).mean() if mask.any() else torch.zeros((), dtype=C.dtype, device=C.device)
    else:
        val = torch.abs(C).mean()
    return coeff * val


# ------------------------------ energy smoothing -----------------------------

def energy_oscillation_penalty(
    E: Tensor,
    *,
    coeff: float = 1e-5,
    hinge: float = 0.0,
) -> Tensor:
    """
    Tiny penalty to discourage upward energy jumps between consecutive steps.
    Args:
      E: tensor of energies per-step [S] or [B,S].
      hinge: allowed increase margin; only increases above `hinge` are penalized.
    Returns: coeff * sum( relu( (E[t+1]-E[t]) - hinge ) ).
    """
    if coeff == 0.0:
        return torch.zeros((), dtype=torch.float32)
    if E.dim() == 1:
        dE = E[1:] - E[:-1]
    else:
        dE = E[..., 1:] - E[..., :-1]
    inc = F.relu(dE - float(hinge))
    return coeff * inc.sum()


# ------------------------------ aggregator helper ----------------------------

def sum_regularizers(terms: Dict[str, Optional[Tensor]]) -> Tuple[Tensor, Dict[str, float]]:
    """
    Sum a dict of {name: tensor or None} into a single scalar penalty.
    Returns (total, scalars) where scalars has float() values for logging.
    """
    total = None
    scalars: Dict[str, float] = {}
    for k, v in terms.items():
        if v is None:
            continue
        # Preserve dtype/device; ignore 0-dim contradictions by summing lazily
        scalars[k] = float(v.detach().cpu())
        total = v if total is None else (total + v)
    if total is None:
        total = torch.zeros((), dtype=torch.float32)
    return total, scalars


# ---------------------------------- __main__ ---------------------------------

if __name__ == "__main__":
    print("[regularizers] Running sanity tests...")
    torch.manual_seed(0)

    # Dummy module with some params
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=True)
            self.norm = nn.LayerNorm(8)

        def forward(self, x: Tensor) -> Tensor:
            return self.norm(self.lin(x))

    m = Tiny()
    x = torch.randn(4, 16, 8)

    # L2 / L1 on params
    reg_l2 = l2_weight_decay(m, 1e-3, include_bias=False, include_norm=False)
    reg_l1 = l1_sparsity(m.parameters(), 1e-4)

    # Group lasso on features (groups=4 over last dim)
    gl = group_lasso(x, groups=4, axis=-1, coeff=1e-3)

    # Orthogonality on a weight
    W = m.lin.weight  # [8,8]
    ortho = orthogonality_penalty(W, coeff=1e-3)

    # Low-rank factor orthogonality (fake factors)
    A = torch.randn(3, 6, 4)  # [G,Cg,r]
    B = torch.randn(3, 6, 4)
    lr_ortho = lowrank_factor_orthogonality(A, B, coeff=1e-3)

    # Temporal penalties
    tv1 = total_variation(x, axis=1, p=1, coeff=1e-4)
    tv2 = total_variation(x, axis=1, p=2, coeff=1e-4)

    # Laplacian smoothness (cycle Laplacian over T)
    Bsz, T, D = x.shape
    L = torch.zeros(T, T)
    idx = torch.arange(T)
    L[idx, idx] = 2.0
    L[idx, (idx - 1) % T] = L[idx, (idx + 1) % T] = -1.0
    smooth = laplacian_smoothness(x, L, coeff=1e-4, time_axis=1)

    # Gate penalties
    g = torch.sigmoid(torch.randn_like(x))  # fake gates in (0,1)
    g_l1 = gate_l1(g, coeff=1e-4)
    g_kl = gate_bernoulli_kl(g, p_prior=0.1, coeff=1e-4)

    total, scalars = sum_regularizers({
        "l2": reg_l2,
        "l1": reg_l1,
        "gl": gl,
        "ortho": ortho,
        "lr_ortho": lr_ortho,
        "tv1": tv1,
        "tv2": tv2,
        "smooth": smooth,
        "g_l1": g_l1,
        "g_kl": g_kl,
    })

    # New regs
    g_logits = torch.randn_like(x[..., :2])
    l0_from_logits = gate_l0_proxy(g_logits, coeff=1e-4, from_logits=True)
    l0_from_probs = gate_l0_proxy(torch.sigmoid(g_logits), coeff=1e-4, from_logits=False)

    Y1, Y2, Y3 = torch.randn(8, 16), torch.randn(8, 16), torch.randn(8, 16)
    decor = opbank_cosine_decorrelation([Y1, Y2, Y3], coeff=1e-4)

    E = torch.tensor([1.0, 0.9, 0.93, 0.91])
    eosc = energy_oscillation_penalty(E, coeff=1e-5, hinge=0.0)

    total2, scal2 = sum_regularizers({
        "l0_logits": l0_from_logits,
        "l0_probs": l0_from_probs,
        "decor": decor,
        "eosc": eosc,
    })
    assert total2.item() >= 0.0

    print("  scalars:", {k: f"{v:.3e}" for k, v in scalars.items()})
    assert total.item() >= 0.0
    print("[regularizers] All good ✓")

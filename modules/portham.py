# modules/portham.py
# Port-Hamiltonian step A = (J - R) G over features (last dim), per time step.
# J is skew-symmetric, R is PSD, G is groupwise block-diagonal mixing. PyTorch only.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- small utils ---------------------------------

def _traceless_last(x: torch.Tensor) -> torch.Tensor:
    """Subtract mean on last dim (keeps shape)."""
    return x - x.mean(dim=-1, keepdim=True)

def _traceless_groupwise(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Subtract per-group mean on the last dim. x: [..., D], D % groups == 0.
    """
    *prefix, D = x.shape
    if groups < 1 or D % groups != 0:
        raise ValueError(f"'groups' must divide D. Got D={D}, groups={groups}.")
    cg = D // groups
    xg = x.view(*prefix, groups, cg)                      # [..., G, cg]
    mu = xg.mean(dim=-1, keepdim=True)                   # [..., G, 1]
    yg = xg - mu
    return yg.view(*prefix, D)

def _apply_blockdiag(W: torch.Tensor, h: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Block-diagonal matmul with G blocks.
    W: [G, cg, cg], h: [..., D], D=G*cg. Returns [..., D].
    """
    *prefix, D = h.shape
    G, cg, cg2 = W.shape
    if cg != cg2 or D != G * cg or G != groups:
        raise ValueError(f"Incompatible shapes: W={tuple(W.shape)}, h last dim={D}, groups={groups}.")
    xg = h.view(*prefix, G, cg)                              # [..., G, cg]
    yg = torch.einsum('gij,...gj->...gi', W, xg)             # [..., G, cg]
    return yg.reshape(*prefix, D)


# ------------------------------ main module ----------------------------------

class PortHamiltonianStep(nn.Module):
    """
    Linear field f(h) = (J - R)·G·h on features (last dim).
      - J: skew-symmetric (full or low-rank).
      - R: PSD (diag softplus + optional low-rank B B^T).
      - G: groupwise block-diagonal (identity init).
    I/O: [B, T, D] → [B, T, D].
    """
    def __init__(
        self,
        d: int,
        groups: int = 1,
        skew_rank: int = 16,     # default low-rank J (v3-style)
        R_rank: int = 0,         # extra PSD rank via B B^T
        use_diag_R: bool = True,
        traceless: bool = True,  # if True, apply groupwise traceless on input
        eps: float = 1e-6,
    ):
        super().__init__()
        if d <= 0:
            raise ValueError("d must be > 0")
        if groups < 1 or d % groups != 0:
            raise ValueError(f"'groups' must divide d. Got d={d}, groups={groups}.")
        self.d = int(d)
        self.groups = int(groups)
        self.cg = d // groups
        self.skew_rank = int(skew_rank)
        self.R_rank = int(R_rank)
        self.use_diag_R = bool(use_diag_R)
        self.traceless = bool(traceless)
        self.eps = float(eps)

        # --- G: block-diagonal (groupwise) mixing, identity init ---
        W = torch.eye(self.cg).repeat(self.groups, 1, 1)     # [G, cg, cg]
        self.G = nn.Parameter(W)                              # learnable block weights

        # --- J: skew-symmetric ---
        if self.skew_rank > 0:
            r = self.skew_rank
            self.U = nn.Parameter(torch.zeros(d, r))
            self.V = nn.Parameter(torch.zeros(d, r))
            nn.init.normal_(self.U, std=0.02)
            nn.init.normal_(self.V, std=0.02)
            self.J_scale = nn.Parameter(torch.tensor(0.0))   # lazy-minimal init
            self._mode_full_J = False
        else:
            # Full skew via antisymmetrization
            self.M = nn.Parameter(torch.zeros(d, d))
            nn.init.normal_(self.M, std=0.02)
            self.J_scale = nn.Parameter(torch.tensor(0.0))   # lazy-minimal init
            self._mode_full_J = True

        # --- R: PSD = diag(softplus) + B B^T ---
        if self.use_diag_R:
            # Start near zero dissipation; softplus(rho)+eps ≥ eps.
            self.rho = nn.Parameter(torch.full((d,), -4.0))
        else:
            self.register_parameter("rho", None)
        if self.R_rank > 0:
            self.B = nn.Parameter(torch.zeros(d, self.R_rank))
            nn.init.normal_(self.B, std=0.02)
        else:
            self.register_parameter("B", None)
        self.R_scale = nn.Parameter(torch.tensor(0.0))       # lazy-minimal init

    # ------------------------------- internals --------------------------------

    def _apply_G(self, h: torch.Tensor) -> torch.Tensor:
        return _apply_blockdiag(self.G, h, self.groups)

    def _apply_J(self, y: torch.Tensor) -> torch.Tensor:
        if self._mode_full_J:
            J = self.M - self.M.t()                           # skew
            return self.J_scale * (y @ J.t())
        else:
            # (U V^T − V U^T) y = U(V^T y) − V(U^T y)
            U, V = self.U, self.V
            tU = y @ U                                        # [N, r]
            tV = y @ V                                        # [N, r]
            out = tU @ V.t() - tV @ U.t()                     # [N, D]
            return self.J_scale * out

    def _apply_R(self, y: torch.Tensor) -> torch.Tensor:
        out = 0.0
        if self.use_diag_R:
            diag = F.softplus(self.rho) + self.eps            # [D] ≥ eps
            out = y * diag                                    # elementwise on last dim
        if self.R_rank > 0:
            t = y @ self.B                                    # [N, r]
            out = out + t @ self.B.t()
        return self.R_scale * out

    # -------------------------------- forward ---------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, T, D] → y: [B, T, D] with A = (J - R) G.
        Applies groupwise traceless if `self.traceless` (for consistency with normalize.py).
        """
        if h.dim() != 3:
            raise ValueError(f"h must be [B,T,D], got {tuple(h.shape)}")
        B, T, D = h.shape
        if D != self.d:
            raise ValueError(f"Expected last dim D={self.d}, got {D}")

        # Traceless preprocessing
        if self.traceless:
            x = _traceless_groupwise(h, self.groups)
        else:
            x = h

        y = x.reshape(B * T, D)       # flatten time/batch for linear ops on last dim
        y = self._apply_G(y)          # block-diagonal mixing
        Jy = self._apply_J(y)         # skew part
        Ry = self._apply_R(y)         # dissipative part (PSD)
        out = Jy - Ry
        return out.view(B, T, D)


# ----------------------------------- __main__ --------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, G = 2, 32, 16, 4
    step = PortHamiltonianStep(d=D, groups=G, skew_rank=16, R_rank=4, use_diag_R=True, traceless=True)
    h = torch.randn(B, T, D)

    # Forward: with lazy-minimal init (J_scale=R_scale=0), output should be ~0
    y = step(h)
    max_abs = y.abs().max().item()
    print("[portham] out shape:", tuple(y.shape), "max|y|:", f"{max_abs:.2e}")
    assert max_abs < 1e-6

    # Check J skew-symmetry numerically
    with torch.no_grad():
        if step._mode_full_J:
            J = (step.M - step.M.t()) * step.J_scale
        else:
            J = step.J_scale * (step.U @ step.V.t() - step.V @ step.U.t())
        sym_inf = (J + J.t()).abs().max().item()
        print(f"[portham] ||J+J^T||_∞ ≈ {sym_inf:.2e}")
        assert sym_inf < 1e-6

        # Check R PSD: v^T R v >= 0 for random v (should be ≈0 with scale=0)
        v = torch.randn(64, D)
        q = (v * step._apply_R(v)).sum(dim=1)
        print(f"[portham] R PSD min(q) ≈ {float(q.min().item()):.2e}")
        assert q.min().item() > -1e-8

        # Traceless groupwise effect: per-group means are ~0
        x = _traceless_groupwise(h, G)
        cg = D // G
        xg = x.view(B, T, G, cg)
        mu_g = xg.mean(dim=-1)
        mu_max = mu_g.abs().max().item()
        print(f"[portham] groupwise traceless max|μ| ≈ {mu_max:.2e}")
        assert mu_max < 1e-6

    print("[portham] All good ✓")

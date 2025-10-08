# tests/test_krylov.py
# Residual reduction on a toy SPD system w/ optional krylov module.

import os
import sys
import torch
import pytest

# Safe-import: ensure repo root on sys.path when running as a script
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import krylov lazily; skip test if not present
try:
    import modules.krylov as kry
except Exception:
    kry = None


@pytest.mark.skipif(kry is None, reason="modules.krylov not available")
def test_krylov_residual_reduction():
    # Build SPD matrix A via 1D Laplacian on path (Dirichlet)
    n = 32
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i-1] = -1.0
        if i < n-1:
            A[i, i+1] = -1.0

    b = torch.randn(n)
    x0 = torch.zeros(n)

    # Prefer a generic API; support multiple signatures
    if hasattr(kry, "KrylovRefiner"):
        KR = kry.KrylovRefiner
        ref = None
        # Try different constructor signatures
        for kw in ({"max_steps": 3}, {"steps": 3}, {}):
            try:
                ref = KR(**kw)
                break
            except TypeError:
                ref = None
        if ref is None:
            pytest.skip("KrylovRefiner constructor signature unsupported.")

        # Try different call signatures / method names
        x1 = None
        matvec = lambda v: A @ v

        # 1) Try calling the module itself
        for call_kw in ({}, {"steps": 3}):
            try:
                x1 = ref(matvec, b, x0, **call_kw)
                break
            except (TypeError, NotImplementedError):
                x1 = None

        # 2) Try common method names on the instance
        if x1 is None:
            for meth in ("refine", "run", "solve"):
                if hasattr(ref, meth):
                    fn = getattr(ref, meth)
                    for call_kw in ({}, {"steps": 3}):
                        try:
                            x1 = fn(matvec, b, x0, **call_kw)
                            break
                        except TypeError:
                            x1 = None
                    if x1 is not None:
                        break

        if x1 is None:
            # fall back to functional API if present, else skip
            if hasattr(kry, "krylov_refine"):
                try:
                    x1 = kry.krylov_refine(matvec, b, x0, steps=3)
                except TypeError:
                    x1 = kry.krylov_refine(matvec, b, x0)
            else:
                pytest.skip("KrylovRefiner call signature/methods unsupported and no functional API available.")

    elif hasattr(kry, "krylov_refine"):
        # Functional API
        try:
            x1 = kry.krylov_refine(lambda v: A @ v, b, x0, steps=3)
        except TypeError:
            x1 = kry.krylov_refine(lambda v: A @ v, b, x0)
    else:
        pytest.skip("No known Krylov refiner API in modules.krylov.")

    r0 = torch.norm(A @ x0 - b).item()
    r1 = torch.norm(A @ x1 - b).item()
    assert r1 < r0, f"residual did not decrease: r0={r0:.3e} r1={r1:.3e}"


if __name__ == "__main__":
    # Run via pytest even when invoked with python
    raise SystemExit(pytest.main([__file__]))

"""
tests/test_nfw.py — Verify NFW analytical functions
Run: python tests/test_nfw.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from src.nfw import NFWProfile

def test_nfw():
    print("=" * 50)
    print("TEST: NFW Analytical Functions")
    print("=" * 50)
    nfw    = NFWProfile(rho_c=0.1, Rs=20.0)
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name} {detail}")
            passed += 1
        else:
            print(f"  [FAIL] {name} {detail}")
            failed += 1

    # T1: rho at r=Rs
    r_test    = torch.tensor([[20.0, 0.0, 0.0]])
    rho_at_Rs = nfw.density(r_test).item()
    expected  = nfw.rho_c / 4.0
    err       = abs(rho_at_Rs - expected) / expected
    check("rho(Rs) = rho_c/4", err < 1e-4, f"  (err={err:.2e})")

    # T2: Potential negative everywhere
    coords  = nfw.sample_coords(1000)
    phi_all = nfw.potential(coords)
    check("Phi < 0 everywhere", (phi_all < 0).all().item())

    # T3: Density positive everywhere
    rho_all = nfw.density(coords)
    check("rho > 0 everywhere", (rho_all > 0).all().item())

    # T4: Poisson equation: laplacian == 4piG*rho
    lap  = nfw.laplacian(coords)
    ref  = 4.0 * np.pi * 1.0 * rho_all
    merr = (lap - ref).abs().max().item()
    check("Poisson: nabla^2 Phi = 4piG*rho", merr < 1e-8, f"  (max_err={merr:.2e})")

    # T5: Density decreases with radius
    r_near = torch.tensor([[1.0, 0.0, 0.0]])
    r_far  = torch.tensor([[80.0, 0.0, 0.0]])
    check("rho decreasing with r",
          nfw.density(r_near).item() > nfw.density(r_far).item())

    # T6: Sampling stays inside sphere
    r_max   = 100.0
    samp    = nfw.sample_coords(5000, r_max=r_max)
    radii   = (samp**2).sum(dim=1).sqrt()
    check("All samples inside sphere", (radii <= r_max + 1e-3).all().item())

    print(f"\nResult: {passed} passed / {passed+failed} total")
    return failed == 0

if __name__ == '__main__':
    ok = test_nfw()
    sys.exit(0 if ok else 1)

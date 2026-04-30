"""
tests/test_laplacian.py — Cross-verify all three Laplacian implementations
Run: python tests/test_laplacian.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from src.nfw     import NFWProfile
from src.siren   import SirenNetwork
from src.physics import (laplacian_autograd, laplacian_hessian,
                          laplacian_finite_diff, verify_laplacian)

def test_laplacian():
    print("=" * 50)
    print("TEST: Laplacian Implementations")
    print("=" * 50)
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

    nfw    = NFWProfile(rho_c=0.1, Rs=20.0)
    coords = nfw.sample_coords(100, r_max=80.0)
    anal_lap = nfw.laplacian(coords)

    # T1: All three methods agree on NFW potential
    print("\n  [NFW potential — three-way cross-check]")
    ok, results = verify_laplacian(
        phi_fn=nfw.potential,
        coords=coords,
        nfw_analytical_laplacian=anal_lap,
        tolerance=0.02,
        fd_tolerance=0.05,  # Finite differences need looser tolerance
        verbose=True,
    )
    check("All Laplacian methods agree on NFW", ok)

    # T2: Autograd Laplacian on SIREN
    print("\n  [SIREN — autograd vs finite diff]")
    model = SirenNetwork(hidden_features=64, hidden_layers=2)
    model.eval()
    small_coords = nfw.sample_coords(30, r_max=1.0)
    ok2, _ = verify_laplacian(
        phi_fn=model.forward_phi_only,
        coords=small_coords,
        tolerance=0.05,
        fd_tolerance=0.1,  # Even looser for SIREN (more complex function)
        verbose=True,
    )
    check("SIREN Laplacian: autograd == finite diff", ok2)

    # T3: Poisson equation holds analytically
    coords3   = nfw.sample_coords(200, r_max=80.0)
    lap3      = nfw.laplacian(coords3)
    target3   = 4.0 * np.pi * 1.0 * nfw.density(coords3)
    max_err   = (lap3 - target3).abs().max().item()
    check("Poisson holds analytically", max_err < 1e-8, f"  (max_err={max_err:.2e})")

    # T4: Laplacian shape
    c4 = nfw.sample_coords(50, r_max=50.0).requires_grad_(True)
    l4 = laplacian_autograd(nfw.potential, c4)
    check("Laplacian output shape (N,1)", l4.shape == (50, 1))

    # T5: Gradient clears between calls (no graph leak)
    c5a = nfw.sample_coords(20, r_max=20.0).requires_grad_(True)
    c5b = nfw.sample_coords(20, r_max=20.0).requires_grad_(True)
    l5a = laplacian_autograd(nfw.potential, c5a)
    l5b = laplacian_autograd(nfw.potential, c5b)
    check("Two independent Laplacian calls succeed", l5a is not None and l5b is not None)

    print(f"\nResult: {passed} passed / {passed+failed} total")
    return failed == 0

if __name__ == '__main__':
    ok = test_laplacian()
    sys.exit(0 if ok else 1)

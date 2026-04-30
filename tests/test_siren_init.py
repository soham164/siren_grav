"""
tests/test_siren_init.py — Verify SIREN weight initialization
Run: python tests/test_siren_init.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.siren import SirenNetwork, verify_siren_initialization

def test_siren_init():
    print("=" * 50)
    print("TEST: SIREN Initialization")
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

    # T1: Activation distribution at each layer
    model  = SirenNetwork(hidden_features=256, hidden_layers=4, omega_0=30.0)
    ok, _  = verify_siren_initialization(model, n_test=10_000, verbose=True)
    check("Layer activations uniform in [-1,1]", ok)

    # T2: Forward pass shape
    x          = torch.randn(32, 3)
    phi, rho   = model(x)
    check("Output phi shape (32,1)", phi.shape == (32, 1))
    check("Output rho shape (32,1)", rho.shape  == (32, 1))

    # T3: Rho strictly positive
    check("Rho > 0 (Softplus)", (rho > 0).all().item())

    # T4: Gradients flow
    x2       = torch.randn(8, 3, requires_grad=True)
    phi2, _  = model(x2)
    phi2.sum().backward()
    check("Gradients flow (no NaN)", x2.grad is not None and not x2.grad.isnan().any())

    # T5: Different omega_0 values
    for omega in [10.0, 30.0, 50.0]:
        m    = SirenNetwork(omega_0=omega)
        ok_o, _ = verify_siren_initialization(m, n_test=5_000, verbose=False)
        check(f"omega_0={omega} initializes correctly", ok_o)

    print(f"\nResult: {passed} passed / {passed+failed} total")
    return failed == 0

if __name__ == '__main__':
    ok = test_siren_init()
    sys.exit(0 if ok else 1)

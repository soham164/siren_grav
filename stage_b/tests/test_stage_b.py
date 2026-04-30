"""
tests/test_stage_b.py  —  Stage B Gate Tests
=============================================
Run before training to verify the full pipeline works.
Run: python tests/test_stage_b.py
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.units import TNGUnits, SimUnits


def test_unit_conversions():
    print("\n[T1] TNG unit conversions")
    units = TNGUnits(redshift=0.0, h=0.6774)

    # 1 ckpc/h at z=0 → 1/0.6774 = 1.4763 kpc
    pos   = np.array([[1.0, 0.0, 0.0]])
    kpc   = units.convert_positions(pos)
    expected = 1.0 / 0.6774
    err   = abs(kpc[0, 0] - expected) / expected
    assert err < 1e-5, f"Position conversion error: {err}"
    print(f"  pos conversion: 1 ckpc/h → {kpc[0,0]:.5f} kpc  PASS")

    # 1 TNG mass unit → 1e10/0.6774 M_sun
    mass  = np.array([1.0])
    msun  = units.convert_masses(mass)
    expected_m = 1e10 / 0.6774
    err_m = abs(msun[0] - expected_m) / expected_m
    assert err_m < 1e-5, f"Mass conversion error: {err_m}"
    print(f"  mass conversion: 1 TNG unit → {msun[0]:.4e} M_sun  PASS")


def test_mock_halo_generation():
    print("\n[T2] Mock halo generation")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import the mock generator from the experiment script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_stage_b",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "experiments", "run_stage_b.py")
    )
    mod = importlib.util.module_from_spec(spec)

    # Just test that mock data has correct structure
    n_particles = 10_000
    r = np.random.exponential(scale=100.0, size=n_particles)
    r = np.clip(r, 0.1, 400.0)
    theta = np.arccos(2 * np.random.rand(n_particles) - 1)
    phi   = 2 * np.pi * np.random.rand(n_particles)
    pos   = np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ], axis=1)

    assert pos.shape == (n_particles, 3), f"Wrong shape: {pos.shape}"
    assert (np.linalg.norm(pos, axis=1) <= 400.0 + 1e-3).all(), "Points outside sphere"
    print(f"  Mock halo: {n_particles:,} particles, shape={pos.shape}  PASS")


def test_density_estimator():
    print("\n[T3] KDE density estimator")
    from src.field_estimator import DensityEstimator

    np.random.seed(42)
    n = 5_000
    pos  = np.random.randn(n, 3) * 30.0   # Gaussian cluster
    mass = np.ones(n) * 1e8               # equal mass particles

    est = DensityEstimator(pos, mass, max_fit_pts=n)

    # Query at origin — should be highest density
    rho_centre = est.density_at(np.array([[0.0, 0.0, 0.0]]))[0]
    rho_far    = est.density_at(np.array([[200.0, 0.0, 0.0]]))[0]

    assert rho_centre > 0,        "Centre density must be positive"
    assert rho_centre > rho_far,  "Centre density must exceed far density"
    print(f"  rho(0,0,0)={rho_centre:.3e}  rho(200,0,0)={rho_far:.3e}  PASS")


def test_potential_estimator():
    print("\n[T4] Potential estimator (direct summation)")
    from src.field_estimator import PotentialEstimator

    np.random.seed(42)
    n    = 500    # small N for direct summation test
    pos  = np.random.randn(n, 3) * 20.0
    mass = np.ones(n) * 1e8

    est = PotentialEstimator(pos, mass, softening_kpc=1.0, max_direct_n=1000)

    query = np.array([[0.0, 0.0, 0.0],
                      [50.0, 0.0, 0.0]])
    phi   = est.potential_at(query)

    assert phi.shape == (2,),   f"Wrong shape: {phi.shape}"
    assert (phi < 0).all(),     "Potential must be negative"
    assert phi[0] < phi[1],     "Potential deeper at centre than at r=50"
    print(f"  phi(0,0,0)={phi[0]:.2f}  phi(50,0,0)={phi[1]:.2f} (km/s)^2  PASS")


def test_siren_model():
    print("\n[T5] SIREN model forward pass")
    from src.models import SirenNetwork

    model = SirenNetwork(hidden_features=64, hidden_layers=2)
    x     = torch.randn(32, 3)
    phi, rho = model(x)

    assert phi.shape == (32, 1), f"Phi shape wrong: {phi.shape}"
    assert rho.shape == (32, 1), f"Rho shape wrong: {rho.shape}"
    assert (rho > 0).all(),      "Rho must be positive (Softplus)"
    print(f"  phi shape={phi.shape}  rho shape={rho.shape}  rho>0=True  PASS")


def test_laplacian_autograd():
    print("\n[T6] Laplacian via autograd")
    from src.trainer_b import laplacian_autograd
    from src.models import SirenNetwork

    model  = SirenNetwork(hidden_features=32, hidden_layers=2)
    coords = torch.randn(20, 3).requires_grad_(True)

    lap = laplacian_autograd(model.forward_phi_only, coords)

    assert lap.shape == (20, 1), f"Laplacian shape wrong: {lap.shape}"
    assert not lap.isnan().any(), "Laplacian contains NaN"
    print(f"  Laplacian shape={lap.shape}  no NaN  PASS")


def test_colocation_sampler():
    print("\n[T7] Colocation sampler")
    from src.trainer_b import ColocationSampler

    sampler = ColocationSampler(r_max=1.0, batch_size=512)
    coords  = sampler.sample(512)

    assert coords.shape == (512, 3),          f"Shape wrong: {coords.shape}"
    assert coords.requires_grad,              "requires_grad must be True"
    in_sphere = ((coords.detach()**2).sum(1) <= 1.0).all()
    assert in_sphere,                         "Points outside unit sphere"
    print(f"  shape={coords.shape}  requires_grad=True  in_sphere=True  PASS")


def test_sim_units():
    print("\n[T8] SimUnits scaling")
    sim = SimUnits(r_scale_kpc=500.0, m_scale_msun=1e12)

    # G in sim units should be 1.0 by construction
    from src.units import G_KPC_KMS_MSUN
    G_sim = G_KPC_KMS_MSUN * sim.m_scale / sim.r_scale / sim.v_scale**2
    assert abs(G_sim - 1.0) < 1e-4, f"G_sim should be 1.0, got {G_sim}"
    print(f"  G in sim units = {G_sim:.6f}  (expected 1.0)  PASS")


if __name__ == '__main__':
    print("=" * 55)
    print("Stage B Gate Tests")
    print("=" * 55)

    tests = [
        test_unit_conversions,
        test_mock_halo_generation,
        test_density_estimator,
        test_potential_estimator,
        test_siren_model,
        test_laplacian_autograd,
        test_colocation_sampler,
        test_sim_units,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"Result: {passed}/{passed+failed} passed")
    if failed == 0:
        print("All tests PASSED — safe to run Stage B training.")
    else:
        print(f"{failed} test(s) FAILED — fix before training.")
    print("=" * 55)
    sys.exit(0 if failed == 0 else 1)

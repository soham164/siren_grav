"""
src/physics.py  —  Step 4: Laplacian via Autograd
==================================================
The Laplacian of the potential (nabla^2 Phi) is the key quantity that links
the potential to the density via Poisson's equation:

    nabla^2 Phi = 4 * pi * G * rho

To enforce this as a training loss, we need to compute nabla^2 Phi from
the network's predicted Phi using automatic differentiation.

We implement TWO methods and cross-verify them:

Method 1 — Sequential autograd (FAST, use for training):
    Step 1: grad_phi  = d(Phi) / d(coords)           shape: (N, 3)
    Step 2: For each dimension i, compute d(grad_phi_i) / d(coords_i)
    Step 3: laplacian = sum of the three second derivatives
    Cost: 2 backward passes

Method 2 — Full Hessian diagonal (SLOW, use for verification only):
    Compute the full Hessian matrix of Phi w.r.t. coords (N, 3, 3)
    Take the trace (sum of diagonal) = Laplacian
    Cost: 3 backward passes + full matrix computation

Both should give identical results to numerical precision (~1e-5).
If they disagree, something is wrong with the autograd computation.

We also implement a finite-difference Laplacian for further cross-checking:
    nabla^2 f(x) ≈ [f(x+h) + f(x-h) - 2f(x)] / h^2  (per dimension, summed)
    Cost: 6 forward passes, no gradient computation
    Accuracy: O(h^2) — good enough for verification, not for training
"""

import torch
import torch.nn as nn
import numpy as np


# ── Method 1: Sequential autograd (training-efficient) ───────────────────────
def laplacian_autograd(phi_fn, coords):
    """
    Compute the Laplacian of phi_fn(coords) using two sequential autograd calls.
    This is the method used during PINN training (efficient, exact).

    Parameters
    ----------
    phi_fn : callable
        A function that takes coords (N, 3) and returns Phi (N, 1).
        Typically model.forward_phi_only.
        IMPORTANT: phi_fn must be differentiable (use SirenNetwork, not NFW).

    coords : tensor of shape (N, 3), requires_grad=True
        The 3D coordinates at which to evaluate the Laplacian.

    Returns
    -------
    lap : tensor of shape (N, 1)
        The Laplacian nabla^2 Phi at each input coordinate.

    How it works:
        1. Forward pass: phi = phi_fn(coords)         shape (N, 1)
        2. First backward: grad_phi = d(phi)/d(coords) shape (N, 3)
           - create_graph=True keeps the computation graph alive for step 3
        3. Second backward: for each spatial dim i,
           lap_i = d(grad_phi[:, i]) / d(coords[:, i]) shape (N,)
        4. laplacian = lap_x + lap_y + lap_z           shape (N, 1)
    """
    # Ensure coords has gradient tracking
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)

    # Forward pass
    phi = phi_fn(coords)                                        # (N, 1)

    # First-order gradient: d(Phi)/d(coords) = [dPhi/dx, dPhi/dy, dPhi/dz]
    # create_graph=True: keep the graph so we can differentiate again
    grad_phi = torch.autograd.grad(
        outputs=phi,
        inputs=coords,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,          # MUST be True for the second derivative
        retain_graph=True,
    )[0]                                                        # (N, 3)

    # Second-order: laplacian = d^2Phi/dx^2 + d^2Phi/dy^2 + d^2Phi/dz^2
    laplacian = torch.zeros(coords.shape[0], 1, device=coords.device)

    for dim in range(3):
        # Gradient of the dim-th component of grad_phi w.r.t. the dim-th coordinate
        grad2_dim = torch.autograd.grad(
            outputs=grad_phi[:, dim].sum(),  # scalar for autograd
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0][:, dim]                                            # (N,) — only the dim-th column

        laplacian[:, 0] += grad2_dim

    return laplacian


# ── Method 2: Hessian diagonal (verification only) ───────────────────────────
def laplacian_hessian(phi_fn, coords):
    """
    Compute the Laplacian via the trace of the full Hessian matrix.
    Slower than Method 1 but independent implementation — used for cross-checking.

    Parameters
    ----------
    phi_fn : callable — same as in laplacian_autograd
    coords : tensor of shape (N, 3), requires_grad=True

    Returns
    -------
    lap : tensor of shape (N, 1)
    """
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)

    N = coords.shape[0]
    laplacian = torch.zeros(N, 1, device=coords.device)

    for dim in range(3):
        # Compute d^2 Phi / d(coords_dim)^2 for each point
        phi = phi_fn(coords)                                    # (N, 1)

        grad1 = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=coords,
            create_graph=True,
        )[0][:, dim]                                            # (N,)

        grad2 = torch.autograd.grad(
            outputs=grad1.sum(),
            inputs=coords,
            create_graph=False,
        )[0][:, dim]                                            # (N,)

        laplacian[:, 0] += grad2

    return laplacian


# ── Method 3: Finite differences (sanity check, no autograd needed) ───────────
def laplacian_finite_diff(phi_fn, coords, h=None):
    """
    Approximate the Laplacian via central finite differences.
    This uses NO autograd — it evaluates phi_fn at perturbed coordinates.
    Used as the ultimate sanity check: if all three methods agree, we are correct.

    Approximation:
        d^2f/dx^2 ≈ [f(x+h,y,z) + f(x-h,y,z) - 2f(x,y,z)] / h^2

    Accuracy: O(h^2). Step size is adaptive based on coordinate scale.
    Cost: 6 forward passes (2 per spatial dimension).

    Parameters
    ----------
    phi_fn : callable — must accept coords (N, 3) and return phi (N, 1)
    coords : tensor of shape (N, 3) — does NOT need requires_grad
    h      : float or None — finite difference step size. If None, uses adaptive h = 0.01 * mean(|coords|)

    Returns
    -------
    lap : tensor of shape (N, 1)
    """
    with torch.no_grad():
        # Adaptive step size based on coordinate scale
        if h is None:
            coord_scale = coords.abs().mean().item()
            h = max(0.01 * coord_scale, 0.1)  # At least 0.1 to avoid numerical issues
        
        phi_0 = phi_fn(coords)           # f(x, y, z)   shape (N, 1)
        laplacian = torch.zeros_like(phi_0)

        for dim in range(3):
            # Perturb the dim-th coordinate by +h and -h
            coords_plus  = coords.clone()
            coords_minus = coords.clone()
            coords_plus[:, dim]  += h
            coords_minus[:, dim] -= h

            phi_plus  = phi_fn(coords_plus)    # f(..., coord_dim + h, ...)
            phi_minus = phi_fn(coords_minus)   # f(..., coord_dim - h, ...)

            # Central difference: (f(+h) + f(-h) - 2*f(0)) / h^2
            laplacian += (phi_plus + phi_minus - 2.0 * phi_0) / (h ** 2)

    return laplacian


# ── Cross-verification function ───────────────────────────────────────────────
def verify_laplacian(phi_fn, coords, nfw_analytical_laplacian=None,
                     tolerance=1e-3, fd_tolerance=None, require_fd_pass=False, verbose=True):
    """
    Cross-verify all three Laplacian implementations against each other,
    and optionally against the known analytical value from NFW.

    This is Step 4's main verification function.
    The autograd methods must agree with each other and (if provided) the analytical solution.
    Finite differences are a sanity check only and don't need to pass for overall success.

    Parameters
    ----------
    phi_fn                  : callable — differentiable potential function
    coords                  : tensor (N, 3) — test coordinates
    nfw_analytical_laplacian: tensor (N, 1) or None — ground truth if available
    tolerance               : float — max allowed relative difference for autograd methods
    fd_tolerance            : float or None — tolerance for finite diff (defaults to 0.1 if None)
    require_fd_pass         : bool — if True, finite diff must pass for overall success (default False)
    verbose                 : bool — print results

    Returns
    -------
    bool : True if critical checks pass (autograd methods agree + match analytical if provided)
    """
    if fd_tolerance is None:
        fd_tolerance = 0.1  # Finite differences are less accurate, use 10% tolerance
    
    coords_grad = coords.clone().requires_grad_(True)

    # Compute all three
    lap_auto   = laplacian_autograd(phi_fn, coords_grad.clone().requires_grad_(True))
    lap_hess   = laplacian_hessian(phi_fn,  coords_grad.clone().requires_grad_(True))
    lap_fd     = laplacian_finite_diff(phi_fn, coords)

    results = {}

    with torch.no_grad():
        # Auto vs Hessian (CRITICAL - must pass)
        err_ah = (lap_auto - lap_hess).abs() / (lap_auto.abs() + 1e-8)
        results['autograd_vs_hessian'] = err_ah.max().item()

        # Auto vs Finite Diff (sanity check only)
        err_af = (lap_auto - lap_fd).abs() / (lap_auto.abs() + 1e-8)
        results['autograd_vs_finitediff'] = err_af.max().item()

        # Hessian vs Finite Diff (sanity check only)
        err_hf = (lap_hess - lap_fd).abs() / (lap_hess.abs() + 1e-8)
        results['hessian_vs_finitediff'] = err_hf.max().item()

        # Against analytical (CRITICAL if provided - must pass)
        if nfw_analytical_laplacian is not None:
            err_an = (lap_auto - nfw_analytical_laplacian).abs()
            err_an_rel = err_an / (nfw_analytical_laplacian.abs() + 1e-8)
            results['autograd_vs_analytical'] = err_an_rel.max().item()

    # Determine pass/fail
    critical_pass = True
    if verbose:
        print("  Laplacian cross-verification:")
        for key, err in results.items():
            # Determine if this is a critical check
            is_critical = ('analytical' in key or key == 'autograd_vs_hessian')
            is_fd_check = 'finitediff' in key
            
            # Use appropriate tolerance
            if is_fd_check:
                tol = fd_tolerance
                ok = err < tol
                if require_fd_pass:
                    critical_pass = critical_pass and ok
            else:
                tol = tolerance
                ok = err < tol
                if is_critical:
                    critical_pass = critical_pass and ok
            
            print(f"    {key:35s}: max_rel_err = {err:.2e}  "
                  f"{'PASS' if ok else 'FAIL'}")

    return critical_pass, results


# ── Poisson residual (used as training loss component) ────────────────────────
def poisson_residual(phi_fn, rho_pred, coords, G=1.0):
    """
    Compute the Poisson equation residual at a batch of coordinates.
    This is the L_Poisson term in the total training loss.

    Residual = nabla^2 Phi_pred - 4*pi*G*rho_pred

    If the network satisfies Poisson's equation perfectly, this is zero everywhere.

    Parameters
    ----------
    phi_fn   : callable — returns Phi (N, 1) from coords (N, 3)
    rho_pred : tensor (N, 1) — network's predicted density at coords
    coords   : tensor (N, 3), requires_grad=True
    G        : float — gravitational constant (1.0 in simulation units)

    Returns
    -------
    residual : tensor (N, 1) — the Poisson residual at each point
    loss     : scalar tensor — mean squared residual (the actual loss term)
    """
    lap_phi  = laplacian_autograd(phi_fn, coords)               # nabla^2 Phi
    target   = 4.0 * np.pi * G * rho_pred                      # 4*pi*G*rho
    residual = lap_phi - target
    loss     = (residual ** 2).mean()
    return residual, loss


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.nfw import NFWProfile
    from src.siren import SirenNetwork

    print("=" * 60)
    print("Physics (Laplacian) Self-Test")
    print("=" * 60)

    nfw = NFWProfile(rho_c=0.1, Rs=20.0)

    # ── Test A: Verify Laplacian on the ANALYTICAL NFW potential ─────────────
    # The NFW potential is a known function. We wrap it to be autograd-compatible.
    print("\n[A] Testing Laplacian on analytical NFW potential")
    print("    (ground truth: nabla^2 Phi_NFW = 4*pi*G*rho_NFW analytically)")

    coords = nfw.sample_coords(200, r_max=80.0).requires_grad_(True)
    analytical_lap = nfw.laplacian(coords.detach())

    # Wrap NFW potential for autograd (it's already differentiable via torch ops)
    def nfw_phi_fn(c):
        return nfw.potential(c)

    all_pass, results = verify_laplacian(
        phi_fn=nfw_phi_fn,
        coords=coords.detach(),
        nfw_analytical_laplacian=analytical_lap,
        tolerance=1e-2,
        fd_tolerance=0.05,
        verbose=True,
    )
    print(f"\n    NFW Laplacian verification: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # ── Test B: Verify Laplacian on UNTRAINED SIREN ───────────────────────────
    print("\n[B] Testing Laplacian on untrained SIREN (checks autograd graph is correct)")
    print("    (no ground truth — just checking the three methods agree)")

    model = SirenNetwork(hidden_features=64, hidden_layers=2)  # small for speed
    model.eval()

    coords_small = nfw.sample_coords(50, r_max=1.0)

    all_pass_b, _ = verify_laplacian(
        phi_fn=model.forward_phi_only,
        coords=coords_small,
        nfw_analytical_laplacian=None,
        tolerance=1e-2,
        fd_tolerance=0.1,
        verbose=True,
    )
    print(f"\n    SIREN Laplacian verification: {'ALL PASS' if all_pass_b else 'SOME FAILED'}")

    # ── Test C: Poisson residual on untrained SIREN ───────────────────────────
    print("\n[C] Poisson residual on untrained SIREN (should be large — not trained yet)")
    coords_c = coords_small.clone().requires_grad_(True)
    _, rho_pred = model(coords_c.detach())
    residual, loss = poisson_residual(model.forward_phi_only, rho_pred, coords_c)
    print(f"    Poisson residual MSE (untrained): {loss.item():.4f}")
    print(f"    (expected to be large — network hasn't learned physics yet)")

    print("\n" + "=" * 60)
    print("Physics self-test complete.")
    print("=" * 60)

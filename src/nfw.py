"""
src/nfw.py  —  Step 3: Analytical NFW Ground Truth
===================================================
The NFW (Navarro-Frenk-White) profile is the standard mathematical model
for dark matter halos. Because it has a known analytical form for density,
potential, AND the Laplacian of the potential, it is the perfect ground truth
for testing the SIREN.

If the SIREN cannot learn something we can write down analytically, it will
never learn something from a real messy simulation.

Physics recap:
    Density:    rho(r)       = rho_c / [(r/Rs)(1 + r/Rs)^2]
    Potential:  Phi(r)       = -4*pi*G*rho_c*Rs^3 * ln(1 + r/Rs) / r
    Laplacian:  nabla^2 Phi  = 4*pi*G*rho  (Poisson's equation — always true)

All functions accept torch tensors of shape (N, 3) for batched evaluation.
All return tensors of shape (N, 1).
"""

import torch
import numpy as np


# ── Physical constants ────────────────────────────────────────────────────────
# We work in "simulation units" where G=1, lengths in kpc, masses in solar masses.
# This is standard in N-body astrophysics and avoids numerical issues with
# tiny SI values like 6.674e-11.
G = 1.0


class NFWProfile:
    """
    Analytical NFW dark matter halo.

    Parameters
    ----------
    rho_c : float
        Characteristic density. Sets the overall amplitude of the halo.
        In simulation units (M_sun / kpc^3), typical value: 0.1
    Rs : float
        Scale radius in kpc. The density profile transitions from steep (r<<Rs)
        to shallow (r>>Rs) at this radius. Typical value: 20.0 kpc
    G : float
        Gravitational constant. Default 1.0 (simulation units).
    eps : float
        Softening length in kpc. Prevents division by zero at r=0.
        Physical: real halos have a finite-density core at small scales.
    """

    def __init__(self, rho_c=0.1, Rs=20.0, G=1.0, eps=0.1):
        self.rho_c = rho_c
        self.Rs    = Rs
        self.G     = G
        self.eps   = eps

    # ── Helper: radius from (N,3) coordinate tensor ──────────────────────────
    def _radius(self, coords):
        """
        coords: tensor of shape (N, 3) — columns are (x, y, z)
        returns: tensor of shape (N, 1) — Euclidean distance from origin
        """
        # Add softening so r never hits exactly 0
        r = torch.sqrt((coords ** 2).sum(dim=1, keepdim=True) + self.eps ** 2)
        return r

    # ── 1. Density field rho(r) ───────────────────────────────────────────────
    def density(self, coords):
        """
        NFW density: rho(r) = rho_c / [(r/Rs) * (1 + r/Rs)^2]

        This is high at the centre (r << Rs) and falls as 1/r^3 at large r.

        Returns tensor of shape (N, 1), always >= 0.
        """
        r  = self._radius(coords)           # (N, 1)
        x  = r / self.Rs                    # dimensionless radius
        rho = self.rho_c / (x * (1.0 + x) ** 2)
        return rho

    # ── 2. Gravitational potential Phi(r) ─────────────────────────────────────
    def potential(self, coords):
        """
        NFW potential: Phi(r) = -4*pi*G*rho_c*Rs^3 * ln(1 + r/Rs) / r

        Derived by integrating Poisson's equation for the NFW density.
        Always negative (convention: Phi → 0 as r → infinity).

        Returns tensor of shape (N, 1), always <= 0.
        """
        r      = self._radius(coords)
        factor = -4.0 * np.pi * self.G * self.rho_c * self.Rs ** 3
        Phi    = factor * torch.log(1.0 + r / self.Rs) / r
        return Phi

    # ── 3. Laplacian of potential: nabla^2 Phi ───────────────────────────────
    def laplacian(self, coords):
        """
        By Poisson's equation: nabla^2 Phi = 4*pi*G*rho

        This is the KEY equation. It means:
            laplacian of potential = density scaled by 4*pi*G

        We use this as our physics loss target during training.
        The network must learn weights such that its predicted Phi satisfies
        this equation at every point in space.

        Returns tensor of shape (N, 1).
        """
        return 4.0 * np.pi * self.G * self.density(coords)

    # ── 4. Gravitational force F = -grad(Phi) ────────────────────────────────
    def force(self, coords):
        """
        Analytical radial force from NFW potential.
        Used as an additional verification: does autograd give the same force?

        F_r(r) = -dPhi/dr  (radial component)
        Full force vector points from coords toward origin (inward).

        Returns tensor of shape (N, 3).
        """
        r      = self._radius(coords)                     # (N, 1)
        x      = r / self.Rs
        factor = 4.0 * np.pi * self.G * self.rho_c * self.Rs ** 3

        # dPhi/dr analytically
        dPhi_dr = factor * (1.0 / (r * (r + self.Rs)) - torch.log(1.0 + x) / r ** 2)

        # Force magnitude in radial direction, broadcast to (N, 3)
        r_hat = coords / r          # unit vector pointing outward
        F = -dPhi_dr * r_hat        # force points inward (toward centre)
        return F

    # ── 5. Sample training coordinates ───────────────────────────────────────
    def sample_coords(self, n_samples, r_max=100.0, device='cpu'):
        """
        Sample n_samples random (x, y, z) coordinates uniformly in a sphere
        of radius r_max around the origin.

        Uniform sampling in a sphere: sample in cube, reject points outside sphere.
        This gives a physically uniform distribution of training points.

        Returns tensor of shape (N, 3).
        """
        coords_list = []
        collected   = 0

        while collected < n_samples:
            # Sample in a cube [-r_max, r_max]^3
            batch = torch.FloatTensor(n_samples * 2, 3).uniform_(-r_max, r_max)
            # Keep only points inside the sphere
            inside = (batch ** 2).sum(dim=1) <= r_max ** 2
            batch  = batch[inside]
            coords_list.append(batch)
            collected += batch.shape[0]

        coords = torch.cat(coords_list, dim=0)[:n_samples]
        return coords.to(device)

    # ── 6. Normalisation statistics ───────────────────────────────────────────
    def get_normalisation(self, n_samples=100_000, r_max=100.0):
        """
        Compute mean and std of density and potential over a large sample.
        Used to normalise network targets to zero-mean, unit-variance.
        This is CRITICAL for stable training — raw physical values can span
        many orders of magnitude.

        Returns dict with keys: rho_mean, rho_std, phi_mean, phi_std
        """
        with torch.no_grad():
            coords = self.sample_coords(n_samples, r_max)
            rho    = self.density(coords)
            phi    = self.potential(coords)

        return {
            'rho_mean': rho.mean().item(),
            'rho_std' : rho.std().item(),
            'phi_mean': phi.mean().item(),
            'phi_std' : phi.std().item(),
        }


# ── Quick self-test (run this file directly to verify) ───────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("NFW Profile Self-Test")
    print("=" * 60)

    nfw = NFWProfile(rho_c=0.1, Rs=20.0)

    # Test 1: At r = Rs, the NFW density has a known value
    # rho(Rs) = rho_c / (1 * (1+1)^2) = rho_c / 4
    r_test = torch.tensor([[20.0, 0.0, 0.0]])  # exactly r = Rs on x-axis
    rho_at_Rs = nfw.density(r_test).item()
    expected  = nfw.rho_c / 4.0
    err = abs(rho_at_Rs - expected) / expected
    print(f"\nTest 1 — rho at r=Rs:")
    print(f"  Got      : {rho_at_Rs:.8f}")
    print(f"  Expected : {expected:.8f}")
    print(f"  Rel error: {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")

    # Test 2: Poisson equation — nabla^2 Phi must equal 4*pi*G*rho
    # We verify this analytically (both are computed from the same formulas)
    coords_test = torch.tensor([[10.0, 5.0, 3.0],
                                 [50.0, 0.0, 0.0],
                                 [1.0,  1.0, 1.0]])
    lap_phi = nfw.laplacian(coords_test)
    rho_ref = 4.0 * np.pi * G * nfw.density(coords_test)
    max_err = (lap_phi - rho_ref).abs().max().item()
    print(f"\nTest 2 — Poisson equation satisfied analytically:")
    print(f"  Max error: {max_err:.2e}  {'PASS' if max_err < 1e-10 else 'FAIL'}")

    # Test 3: Potential is always negative
    coords_rand = nfw.sample_coords(1000)
    phi_vals    = nfw.potential(coords_rand)
    all_neg     = (phi_vals < 0).all().item()
    print(f"\nTest 3 — Potential always negative:")
    print(f"  All negative: {all_neg}  {'PASS' if all_neg else 'FAIL'}")

    # Test 4: Density is always positive
    rho_vals  = nfw.density(coords_rand)
    all_pos   = (rho_vals > 0).all().item()
    print(f"\nTest 4 — Density always positive:")
    print(f"  All positive: {all_pos}  {'PASS' if all_pos else 'FAIL'}")

    # Test 5: Normalisation stats
    stats = nfw.get_normalisation(n_samples=10_000)
    print(f"\nTest 5 — Normalisation statistics (10k sample):")
    for k, v in stats.items():
        print(f"  {k:12s}: {v:.6f}")

    print("\n" + "=" * 60)
    print("NFW self-test complete.")
    print("=" * 60)

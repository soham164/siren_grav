"""
src/units.py  —  IllustrisTNG Unit Conversions
===============================================
IllustrisTNG does NOT store data in physical kpc / M_sun / km/s.
It uses a set of comoving simulation units that MUST be converted before
any physics calculation. Getting this wrong silently produces nonsense.

TNG internal units (Snapshot 99, z=0, a=1):
    Positions  : ckpc/h  (comoving kpc divided by h=0.6774)
    Masses     : 1e10 M_sun / h
    Velocities : km/s * sqrt(a)   [peculiar velocity, a=scale factor]
    Densities  : 1e10 M_sun/h / (ckpc/h)^3

At z=0: scale factor a=1, so comoving = physical.

Cosmological parameters for TNG100-1:
    h     = 0.6774   (Hubble parameter, dimensionless)
    Omega_m = 0.3089
    Omega_L = 0.6911

We convert everything to:
    Positions  : physical kpc
    Masses     : M_sun
    Velocities : km/s
    Densities  : M_sun / kpc^3

Then for the neural network we further normalise to simulation units
where G=1, which is standard in N-body astrophysics.
"""

import numpy as np

# ── TNG100-1 cosmological parameters ─────────────────────────────────────────
H0    = 67.74          # km/s/Mpc — Hubble constant
h     = 0.6774         # dimensionless Hubble parameter
OMEGA_M = 0.3089
OMEGA_L = 0.6911

# ── Snapshot 99 is z=0, scale factor a=1 ─────────────────────────────────────
SCALE_FACTOR = 1.0     # a = 1/(1+z) = 1 at z=0

# ── Gravitational constant in useful units ────────────────────────────────────
# G in units of kpc * (km/s)^2 / M_sun
G_KPC_KMS_MSUN = 4.3009e-6   # kpc (km/s)^2 M_sun^-1

# G in simulation units where [length]=kpc, [mass]=1e10 M_sun, [vel]=km/s
G_SIM = G_KPC_KMS_MSUN * 1e10   # = 4.3009e-4  kpc (km/s)^2 (1e10 M_sun)^-1


class TNGUnits:
    """
    Handles all unit conversions for a TNG snapshot.

    Parameters
    ----------
    redshift : float — snapshot redshift (0.0 for Snapshot 99)
    h        : float — dimensionless Hubble parameter (0.6774 for TNG)
    """

    def __init__(self, redshift=0.0, h=0.6774):
        self.z = redshift
        self.a = 1.0 / (1.0 + redshift)   # scale factor
        self.h = h

        # Conversion factors (multiply TNG value by these)
        self.pos_to_kpc    = self.a / h          # ckpc/h → physical kpc
        self.mass_to_msun  = 1e10 / h            # 1e10 M_sun/h → M_sun
        self.vel_to_kms    = np.sqrt(self.a)     # km/s*sqrt(a) → km/s
        self.dens_to_msun_kpc3 = (1e10 / h) / (self.a / h)**3  # density

        print(f"TNG Units initialised for z={redshift} (a={self.a:.4f}):")
        print(f"  pos  × {self.pos_to_kpc:.6f}  → physical kpc")
        print(f"  mass × {self.mass_to_msun:.4e} → M_sun")
        print(f"  vel  × {self.vel_to_kms:.6f}  → km/s")

    def convert_positions(self, pos_raw):
        """
        Convert raw TNG positions (ckpc/h) to physical kpc.
        pos_raw: numpy array of shape (N, 3)
        """
        return pos_raw * self.pos_to_kpc

    def convert_masses(self, mass_raw):
        """
        Convert raw TNG masses (1e10 M_sun/h) to M_sun.
        mass_raw: numpy array of shape (N,)
        """
        return mass_raw * self.mass_to_msun

    def convert_velocities(self, vel_raw):
        """
        Convert raw TNG peculiar velocities (km/s * sqrt(a)) to km/s.
        vel_raw: numpy array of shape (N, 3)
        """
        return vel_raw * self.vel_to_kms

    def convert_density(self, dens_raw):
        """
        Convert raw TNG gas density to M_sun/kpc^3.
        """
        return dens_raw * self.dens_to_msun_kpc3

    def centre_positions(self, pos_kpc, centre_kpc):
        """
        Translate positions so the halo centre is at the origin.
        pos_kpc    : (N, 3) array in physical kpc
        centre_kpc : (3,)   array — halo centre in physical kpc
        Returns    : (N, 3) centred positions in kpc
        """
        return pos_kpc - centre_kpc

    def virial_radius_kpc(self, group_r200, to_kpc=True):
        """
        Convert group R200 (virial radius) from TNG units to kpc.
        TNG stores R200 in ckpc/h.
        """
        if to_kpc:
            return group_r200 * self.pos_to_kpc
        return group_r200


# ── Simulation unit normalisation ────────────────────────────────────────────
class SimUnits:
    """
    After converting to physical kpc / M_sun / km/s, further rescale
    to simulation units for the neural network.

    We choose:
        [length] = r_scale  kpc   (e.g. virial radius of the halo)
        [mass]   = m_scale  M_sun (e.g. total halo mass)
        G_sim    = 1.0            (sets the velocity scale implicitly)

    This makes all network inputs O(1) which is critical for stable training.

    The velocity scale follows from G=1:
        [vel]^2 = G * [mass] / [length]
        [vel]   = sqrt(G_phys * m_scale / r_scale)  km/s
    """

    def __init__(self, r_scale_kpc, m_scale_msun):
        self.r_scale  = r_scale_kpc    # kpc
        self.m_scale  = m_scale_msun   # M_sun

        # Velocity scale: sqrt(G_phys * M / R)
        self.v_scale  = np.sqrt(G_KPC_KMS_MSUN * m_scale_msun / r_scale_kpc)  # km/s

        # Density scale: M / R^3
        self.rho_scale = m_scale_msun / r_scale_kpc**3   # M_sun/kpc^3

        # Potential scale: G*M/R = v_scale^2
        self.phi_scale = self.v_scale**2                 # (km/s)^2

        print(f"\nSimulation unit normalisation:")
        print(f"  r_scale   = {r_scale_kpc:.2f} kpc")
        print(f"  m_scale   = {m_scale_msun:.3e} M_sun")
        print(f"  v_scale   = {self.v_scale:.2f} km/s")
        print(f"  rho_scale = {self.rho_scale:.3e} M_sun/kpc^3")
        print(f"  phi_scale = {self.phi_scale:.3e} (km/s)^2")

    def pos_to_sim(self, pos_kpc):
        """Physical kpc → dimensionless [-r_max/r_scale, r_max/r_scale]"""
        return pos_kpc / self.r_scale

    def rho_to_sim(self, rho_msun_kpc3):
        """M_sun/kpc^3 → dimensionless"""
        return rho_msun_kpc3 / self.rho_scale

    def phi_to_sim(self, phi_kms2):
        """(km/s)^2 → dimensionless"""
        return phi_kms2 / self.phi_scale

    def sim_to_rho(self, rho_sim):
        """Dimensionless → M_sun/kpc^3"""
        return rho_sim * self.rho_scale

    def sim_to_phi(self, phi_sim):
        """Dimensionless → (km/s)^2"""
        return phi_sim * self.phi_scale


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("Units Self-Test")
    print("=" * 55)

    units = TNGUnits(redshift=0.0, h=0.6774)

    # At z=0, 1 ckpc/h should convert to 1/0.6774 = 1.4763 kpc
    pos_test = np.array([[1.0, 0.0, 0.0]])
    pos_kpc  = units.convert_positions(pos_test)
    expected = 1.0 / 0.6774
    err      = abs(pos_kpc[0, 0] - expected) / expected
    print(f"\nPos conversion (1 ckpc/h → {expected:.4f} kpc):")
    print(f"  Got {pos_kpc[0,0]:.4f}  err={err:.2e}  "
          f"{'PASS' if err < 1e-5 else 'FAIL'}")

    # Mass: 1 TNG unit = 1e10/0.6774 = 1.4764e10 M_sun
    mass_test = np.array([1.0])
    mass_msun = units.convert_masses(mass_test)
    expected_m = 1e10 / 0.6774
    err_m = abs(mass_msun[0] - expected_m) / expected_m
    print(f"\nMass conversion (1 TNG unit → {expected_m:.4e} M_sun):")
    print(f"  Got {mass_msun[0]:.4e}  err={err_m:.2e}  "
          f"{'PASS' if err_m < 1e-5 else 'FAIL'}")

    # SimUnits: check G=1 in sim units
    # For a halo with R200=500 kpc, M200=1e12 M_sun
    sim = SimUnits(r_scale_kpc=500.0, m_scale_msun=1e12)
    print(f"\nG in physical units : {G_KPC_KMS_MSUN:.4e} kpc (km/s)^2 M_sun^-1")
    print(f"G in sim units      : {G_KPC_KMS_MSUN * sim.m_scale / sim.r_scale / sim.v_scale**2:.4f}  (should be 1.0)")

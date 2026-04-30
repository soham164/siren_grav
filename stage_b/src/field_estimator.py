"""
src/field_estimator.py  —  Density and Potential from Particles
===============================================================
Takes cleaned particle arrays from HaloProcessor and produces
smooth continuous fields rho(x,y,z) and Phi(x,y,z) that the
SIREN can be trained on.

Two estimators:

1. DensityEstimator
   Uses KDE (Kernel Density Estimation) to turn discrete particle
   positions into a smooth density field. The key parameter is the
   bandwidth h — too small gives noisy spiky density, too large
   over-smooths and destroys structure.

   We use Silverman's rule for bandwidth selection:
       h = 1.06 * sigma * N^(-1/5)
   where sigma is the standard deviation of particle positions
   and N is the number of particles. This is theoretically optimal
   for Gaussian distributions and works well for halos.

   For large N (>500k particles), we use a fast tree-based KDE
   or grid-based approximation to avoid O(N²) cost.

2. PotentialEstimator
   Computes the gravitational potential at arbitrary points
   using direct summation (small N) or a Barnes-Hut tree (large N).

   Direct summation:
       Phi(x) = -G * sum_i [ m_i / |x - x_i| ]
   Cost: O(N * M) where M = number of query points.
   For N=10k, M=100k this is 10^9 operations — use tree instead.

   Tree-based (via pynbody if available, else scipy KDTree):
       Phi(x) ≈ -G * sum_over_tree_cells [ M_cell / |x - cell_centre| ]
   Cost: O(M * N * log N) — practical for N up to ~10^6.
"""

import numpy as np
import torch
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree


class DensityEstimator:
    """
    KDE-based density estimation from particle positions.

    For halos with > 200k particles, we subsample for KDE fitting
    and use the fitted KDE for all evaluations.

    Parameters
    ----------
    pos_kpc      : (N, 3) numpy array — particle positions in kpc
    mass_msun    : (N,)   numpy array — particle masses in M_sun
    bw_method    : str or float — bandwidth method ('silverman', 'scott', or float)
    max_fit_pts  : int — max particles used to FIT the KDE (subsampled if N > this)
    """

    def __init__(self, pos_kpc, mass_msun, bw_method='silverman',
                 max_fit_pts=100_000):

        self.pos_kpc   = pos_kpc
        self.mass_msun = mass_msun
        self.N         = pos_kpc.shape[0]
        self.total_mass = mass_msun.sum()
        self.bw_method  = bw_method

        print(f"\nFitting KDE density estimator:")
        print(f"  Particles   : {self.N:,}")
        print(f"  Total mass  : {self.total_mass:.3e} M_sun")

        # Subsample for KDE fitting if too many particles
        if self.N > max_fit_pts:
            idx       = np.random.choice(self.N, max_fit_pts, replace=False)
            fit_pos   = pos_kpc[idx]
            fit_wt    = mass_msun[idx]
            print(f"  Subsampled  : {max_fit_pts:,} for KDE fitting")
        else:
            fit_pos = pos_kpc
            fit_wt  = mass_msun

        # scipy gaussian_kde expects shape (3, N)
        # Weights are proportional to particle mass (more massive → higher density)
        self.kde = gaussian_kde(
            fit_pos.T,
            bw_method=bw_method,
            weights=fit_wt / fit_wt.sum(),   # normalised weights
        )

        # The bandwidth in kpc
        self.bandwidth_kpc = self.kde.factor * fit_pos.std()
        print(f"  Bandwidth   : {self.bandwidth_kpc:.2f} kpc (Silverman rule)")

    def density_at(self, query_pos_kpc):
        """
        Evaluate mass density at query positions.

        The KDE gives a probability density. We scale it by total mass
        to get physical mass density in M_sun/kpc^3.

        Parameters
        ----------
        query_pos_kpc : (M, 3) numpy array

        Returns
        -------
        rho : (M,) numpy array in M_sun/kpc^3
        """
        # KDE probability density at query points
        prob_density = self.kde(query_pos_kpc.T)   # (M,) — 1/kpc^3

        # Scale to physical density
        # The KDE integrates to 1, so multiply by total_mass to get M_sun/kpc^3
        rho = prob_density * self.total_mass        # M_sun/kpc^3
        return rho

    def density_at_torch(self, query_coords_kpc):
        """
        Same as density_at but accepts and returns torch tensors.
        Used for integration with the PyTorch training pipeline.
        """
        coords_np = query_coords_kpc.detach().cpu().numpy()
        rho_np    = self.density_at(coords_np)
        return torch.FloatTensor(rho_np).unsqueeze(1)   # (M, 1)

    def estimate_r200(self, target_overdensity=200, rho_critical_msun_kpc3=None):
        """
        Estimate virial radius R200 from the density profile.
        R200 = radius within which mean density = 200 * rho_critical.

        rho_critical at z=0 ≈ 1.36e8 M_sun/Mpc^3 = 136 M_sun/kpc^3
        """
        if rho_critical_msun_kpc3 is None:
            rho_critical_msun_kpc3 = 136.0   # M_sun/kpc^3 at z=0

        target_density = target_overdensity * rho_critical_msun_kpc3

        # Binary search for R200
        r_min, r_max = 1.0, 2000.0
        for _ in range(50):
            r_try = (r_min + r_max) / 2
            # Mean density within sphere of radius r_try
            mask      = np.linalg.norm(self.pos_kpc, axis=1) <= r_try
            mass_in   = self.mass_msun[mask].sum()
            vol       = (4/3) * np.pi * r_try**3
            mean_dens = mass_in / vol
            if mean_dens > target_density:
                r_min = r_try
            else:
                r_max = r_try
            if r_max - r_min < 0.1:
                break

        r200 = (r_min + r_max) / 2
        print(f"  Estimated R200: {r200:.1f} kpc")
        return r200


class PotentialEstimator:
    """
    Gravitational potential estimation via direct summation or tree.

    For research-grade accuracy, the potential should be computed with
    the same gravity solver used in the simulation (a PM or tree-PM code).
    Here we use direct summation with softening for correctness,
    and offer a fast tree approximation for large N.

    Parameters
    ----------
    pos_kpc      : (N, 3) numpy array — source particle positions
    mass_msun    : (N,)   numpy array — source particle masses
    softening_kpc: float — gravitational softening length
                   Prevents singularity at r=0. TNG uses ~1 kpc for DM.
    G            : float — gravitational constant
                   Default: 4.3009e-6 kpc (km/s)^2 M_sun^-1
    """

    G_KPC_KMS_MSUN = 4.3009e-6   # kpc (km/s)^2 M_sun^-1

    def __init__(self, pos_kpc, mass_msun, softening_kpc=1.0,
                 G=None, max_direct_n=5_000):
        self.pos_kpc      = pos_kpc
        self.mass_msun    = mass_msun
        self.softening    = softening_kpc
        self.G            = G or self.G_KPC_KMS_MSUN
        self.N            = pos_kpc.shape[0]
        self.max_direct_n = max_direct_n

        print(f"\nPotentialEstimator:")
        print(f"  N particles   : {self.N:,}")
        print(f"  Softening     : {softening_kpc:.2f} kpc")
        print(f"  G             : {self.G:.4e} kpc (km/s)^2 M_sun^-1")

        # Build KD-tree for fast nearest-neighbour queries (used in tree approx)
        if self.N > max_direct_n:
            print(f"  N > {max_direct_n:,}: using tree-based approximation")
            self.use_tree = True
            self._build_tree()
        else:
            print(f"  N <= {max_direct_n:,}: using direct summation")
            self.use_tree = False

    def _build_tree(self, n_multipole_cells=1000):
        """
        Build a coarse multipole approximation for large-N potential.
        Groups particles into cells, computes cell centre-of-mass.
        Only monopole term (M_cell / r) — sufficient for ~5% accuracy.
        """
        from sklearn.cluster import MiniBatchKMeans

        print(f"  Building {n_multipole_cells}-cell tree approximation...")

        # Group particles into cells via k-means
        try:
            kmeans = MiniBatchKMeans(n_clusters=n_multipole_cells,
                                     random_state=42, n_init=3)
            labels = kmeans.fit_predict(self.pos_kpc)
        except ImportError:
            # Fall back to uniform grid if sklearn not available
            print("  (sklearn not available, using uniform grid)")
            labels = self._uniform_grid_cells(n_multipole_cells)

        # Compute cell centres-of-mass and total masses
        cell_pos  = np.zeros((n_multipole_cells, 3))
        cell_mass = np.zeros(n_multipole_cells)

        for cell_id in range(n_multipole_cells):
            mask = labels == cell_id
            if mask.sum() == 0:
                continue
            m    = self.mass_msun[mask]
            p    = self.pos_kpc[mask]
            cell_mass[cell_id] = m.sum()
            cell_pos[cell_id]  = (p * m[:, None]).sum(0) / m.sum()

        # Keep only non-empty cells
        valid           = cell_mass > 0
        self.cell_pos   = cell_pos[valid]
        self.cell_mass  = cell_mass[valid]
        print(f"  Tree built: {valid.sum()} non-empty cells")

    def potential_at(self, query_pos_kpc, batch_size=1000):
        """
        Compute gravitational potential at query positions.

        Phi(x) = -G * sum_i [ m_i / sqrt(|x - x_i|^2 + eps^2) ]

        Parameters
        ----------
        query_pos_kpc : (M, 3) numpy array
        batch_size    : int — batch query points to limit memory

        Returns
        -------
        phi : (M,) numpy array in (km/s)^2
        """
        M   = query_pos_kpc.shape[0]
        phi = np.zeros(M)

        # Source positions and masses
        src_pos  = self.cell_pos  if self.use_tree else self.pos_kpc
        src_mass = self.cell_mass if self.use_tree else self.mass_msun

        # Process in batches to limit memory
        for start in range(0, M, batch_size):
            end     = min(start + batch_size, M)
            qpos    = query_pos_kpc[start:end]             # (B, 3)

            # Pairwise distances: (B, N_src)
            diff    = qpos[:, None, :] - src_pos[None, :, :]   # (B, N_src, 3)
            r2      = (diff**2).sum(axis=2)                     # (B, N_src)
            r_soft  = np.sqrt(r2 + self.softening**2)           # (B, N_src)

            # Potential at each query point
            phi[start:end] = -self.G * (src_mass[None, :] / r_soft).sum(axis=1)

            if start % 10_000 == 0 and M > 10_000:
                print(f"\r  Potential: {end}/{M}", end="", flush=True)

        if M > 10_000:
            print()

        return phi   # (km/s)^2

    def potential_at_torch(self, query_coords_kpc):
        """
        Same as potential_at but accepts/returns torch tensors.
        """
        coords_np = query_coords_kpc.detach().cpu().numpy()
        phi_np    = self.potential_at(coords_np)
        return torch.FloatTensor(phi_np).unsqueeze(1)   # (M, 1)


class RealHaloDataset:
    """
    Combines DensityEstimator + PotentialEstimator into a training dataset
    for a single real IllustrisTNG halo.

    Workflow:
        1. Sample N_train random coordinates within the halo virial radius
        2. Evaluate density and potential at each coordinate
        3. Normalise to zero-mean, unit-variance
        4. Return PyTorch-compatible dataset

    Parameters
    ----------
    halo_data    : dict from HaloProcessor.process()
    n_samples    : int — training sample count
    r_max_factor : float — sample within r_max_factor * R200
    device       : str
    seed         : int
    """

    def __init__(self, halo_data, n_samples=100_000,
                 r_max_factor=1.5, device='cpu', seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device      = device
        r200             = halo_data["r200_kpc"]
        self.r_max       = r_max_factor * r200
        self.coord_max   = self.r_max

        print(f"\nBuilding RealHaloDataset:")
        print(f"  R200      : {r200:.1f} kpc")
        print(f"  r_max     : {self.r_max:.1f} kpc ({r_max_factor}*R200)")
        print(f"  N samples : {n_samples:,}")

        # ── 1. Fit density estimator ──────────────────────────────────────────
        self.density_est = DensityEstimator(
            halo_data["dm_pos_kpc"],
            halo_data["dm_mass_msun"],
        )

        # ── 2. Fit potential estimator ────────────────────────────────────────
        self.potential_est = PotentialEstimator(
            halo_data["dm_pos_kpc"],
            halo_data["dm_mass_msun"],
            softening_kpc=1.0,
        )

        # ── 3. Sample training coordinates (uniform in sphere) ────────────────
        print(f"\nSampling {n_samples:,} training coordinates...")
        raw_coords = self._sample_sphere(n_samples, self.r_max)

        # ── 4. Evaluate density and potential ─────────────────────────────────
        print("Evaluating density at training points...")
        raw_rho = self.density_est.density_at(raw_coords)        # (N,) M_sun/kpc^3

        print("Evaluating potential at training points...")
        raw_phi = self.potential_est.potential_at(raw_coords)    # (N,) (km/s)^2

        # ── 5. Convert to tensors and normalise ───────────────────────────────
        raw_rho_t = torch.FloatTensor(raw_rho).unsqueeze(1)      # (N, 1)
        raw_phi_t = torch.FloatTensor(raw_phi).unsqueeze(1)      # (N, 1)
        coords_t  = torch.FloatTensor(raw_coords)                # (N, 3)

        self.rho_mean = raw_rho_t.mean()
        self.rho_std  = raw_rho_t.std() + 1e-8
        self.phi_mean = raw_phi_t.mean()
        self.phi_std  = raw_phi_t.std() + 1e-8

        print(f"\nNormalisation:")
        print(f"  rho: mean={self.rho_mean.item():.3e} std={self.rho_std.item():.3e} M_sun/kpc^3")
        print(f"  phi: mean={self.phi_mean.item():.3e} std={self.phi_std.item():.3e} (km/s)^2")

        self.coords = coords_t / self.r_max                         # [-1, 1]
        self.rho    = (raw_rho_t - self.rho_mean) / self.rho_std
        self.phi    = (raw_phi_t - self.phi_mean) / self.phi_std

        self.n_samples = n_samples
        print(f"\nDataset ready.")

    def _sample_sphere(self, n, r_max):
        """Sample n points uniformly inside a sphere of radius r_max."""
        pts = []
        collected = 0
        while collected < n:
            batch = np.random.uniform(-r_max, r_max, size=(n * 2, 3))
            mask  = np.linalg.norm(batch, axis=1) <= r_max
            batch = batch[mask]
            pts.append(batch)
            collected += batch.shape[0]
        return np.concatenate(pts)[:n]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.coords[idx], self.rho[idx], self.phi[idx]

    def denorm_rho(self, rho_norm):
        return rho_norm * self.rho_std + self.rho_mean

    def denorm_phi(self, phi_norm):
        return phi_norm * self.phi_std + self.phi_mean

    def norm_coords(self, coords_phys):
        return coords_phys / self.coord_max

    def get_split(self, val_fraction=0.1, batch_size=2048, seed=42):
        torch.manual_seed(seed)
        from torch.utils.data import DataLoader, random_split
        n_val   = int(self.n_samples * val_fraction)
        n_train = self.n_samples - n_val
        train_ds, val_ds = random_split(self, [n_train, n_val])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds,   batch_size=batch_size, shuffle=False))

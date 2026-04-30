"""
src/dataset.py  —  Step 5: Dataset and Data Pipeline
=====================================================
Two dataset classes:

1. NFWDataset
   Samples random (x,y,z) coordinates inside a sphere,
   computes the exact NFW density and potential at each point,
   normalises everything to zero-mean unit-variance for stable training.

2. ColocationSampler
   Generates random coordinate batches with NO labels.
   Used exclusively for the physics (Poisson) loss — we enforce
   nabla^2 Phi = 4*pi*G*rho at these points during training,
   no ground-truth label needed.

Why normalise?
    Raw physical values span many orders of magnitude:
        Phi  ranges from ~ -50   to ~ -0.01  (kpc^2/s^2 in sim units)
        Rho  ranges from ~  0.001 to ~  10   (M_sun/kpc^3)
    A network trained on unnormalised values will focus its capacity on
    the large values and ignore the small ones. Normalising to zero-mean,
    unit-variance makes every region of the halo equally important.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NFWDataset(Dataset):
    """
    PyTorch Dataset wrapping the analytical NFW profile.

    Every __getitem__ call returns a tuple:
        (coords_normalised, rho_normalised, phi_normalised)

    The normalisation statistics are computed once during __init__
    from a large sample, then stored and applied consistently.

    Parameters
    ----------
    nfw       : NFWProfile instance — the analytical ground truth
    n_samples : int   — number of training points
    r_max     : float — maximum radius of the sampling sphere (kpc)
    device    : str   — 'cpu' or 'cuda'
    seed      : int   — random seed for reproducibility
    """

    def __init__(self, nfw, n_samples=200_000, r_max=100.0,
                 device='cpu', seed=42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.nfw      = nfw
        self.r_max    = r_max
        self.device   = device
        self.n_samples = n_samples

        print(f"  Building NFWDataset: {n_samples:,} samples, r_max={r_max} kpc")

        # ── 1. Sample coordinates ─────────────────────────────────────────────
        raw_coords = nfw.sample_coords(n_samples, r_max=r_max, device=device)

        # ── 2. Compute ground truth values ────────────────────────────────────
        with torch.no_grad():
            raw_rho = nfw.density(raw_coords)    # (N, 1)
            raw_phi = nfw.potential(raw_coords)  # (N, 1)

        # ── 3. Compute normalisation statistics ───────────────────────────────
        # Use the training data itself for statistics (standard practice)
        self.coord_max = r_max  # coordinates normalised to [-1, 1] by dividing by r_max

        self.rho_mean = raw_rho.mean()
        self.rho_std  = raw_rho.std() + 1e-8    # +eps to avoid division by zero

        self.phi_mean = raw_phi.mean()
        self.phi_std  = raw_phi.std() + 1e-8

        print(f"  Normalisation stats:")
        print(f"    rho: mean={self.rho_mean.item():.4f}, std={self.rho_std.item():.4f}")
        print(f"    phi: mean={self.phi_mean.item():.4f}, std={self.phi_std.item():.4f}")

        # ── 4. Normalise and store ────────────────────────────────────────────
        self.coords = raw_coords / r_max               # coords in [-1, 1]
        self.rho    = (raw_rho - self.rho_mean) / self.rho_std
        self.phi    = (raw_phi - self.phi_mean) / self.phi_std

        # Sanity check: normalised values should be near zero-mean
        assert abs(self.rho.mean().item()) < 0.05, "rho normalisation failed"
        assert abs(self.phi.mean().item()) < 0.05, "phi normalisation failed"
        print(f"  Normalisation check: PASSED")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.coords[idx], self.rho[idx], self.phi[idx]

    # ── De-normalisation helpers (for evaluation and plotting) ────────────────
    def denorm_coords(self, coords_norm):
        """Convert normalised coords back to physical kpc."""
        return coords_norm * self.coord_max

    def denorm_rho(self, rho_norm):
        """Convert normalised density back to physical units."""
        return rho_norm * self.rho_std + self.rho_mean

    def denorm_phi(self, phi_norm):
        """Convert normalised potential back to physical units."""
        return phi_norm * self.phi_std + self.phi_mean

    def norm_coords(self, coords_phys):
        """Convert physical coordinates to normalised [-1, 1]."""
        return coords_phys / self.coord_max

    def get_dataloader(self, batch_size=4096, shuffle=True, num_workers=0):
        """Return a PyTorch DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(self.device == 'cuda'),
        )

    def get_split(self, val_fraction=0.1, seed=42):
        """
        Split into train and validation sets.

        Returns
        -------
        train_loader, val_loader : DataLoader, DataLoader
        """
        torch.manual_seed(seed)
        n_val   = int(self.n_samples * val_fraction)
        n_train = self.n_samples - n_val
        train_ds, val_ds = torch.utils.data.random_split(self, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=4096, shuffle=False)
        return train_loader, val_loader


class ColocationSampler:
    """
    Generates batches of random coordinates with NO labels.
    Used exclusively for the physics (Poisson) loss during training.

    At each training step, we sample a fresh batch of colocation points,
    compute the Laplacian of the predicted Phi at those points via autograd,
    and penalise its deviation from 4*pi*G*rho_pred.

    No ground truth is needed at colocation points — that is the whole point
    of a physics-informed approach. The physics equation IS the supervision.

    Parameters
    ----------
    r_max      : float — same as dataset r_max, normalised to [-1, 1]
    device     : str
    batch_size : int — number of colocation points per training step
    """

    def __init__(self, r_max=1.0, device='cpu', batch_size=4096):
        # r_max=1.0 because we work in normalised coordinates
        self.r_max      = r_max
        self.device     = device
        self.batch_size = batch_size

    def sample(self, n=None):
        """
        Sample a fresh random batch of coordinates inside the unit sphere.

        Returns
        -------
        coords : tensor (batch_size, 3), requires_grad=True
            Random coordinates in [-1, 1]^3, inside the unit sphere.
            requires_grad=True is essential — autograd must track these
            to compute the Laplacian.
        """
        n = n or self.batch_size

        # Sample in cube, reject outside sphere (uniform sphere sampling)
        coords_list = []
        collected   = 0
        while collected < n:
            batch   = torch.FloatTensor(n * 2, 3).uniform_(-self.r_max, self.r_max)
            inside  = (batch ** 2).sum(dim=1) <= self.r_max ** 2
            batch   = batch[inside]
            coords_list.append(batch)
            collected += batch.shape[0]

        coords = torch.cat(coords_list, dim=0)[:n].to(self.device)
        return coords.requires_grad_(True)   # CRITICAL: must be True for Laplacian


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.nfw import NFWProfile

    print("=" * 60)
    print("Dataset Self-Test")
    print("=" * 60)

    nfw = NFWProfile(rho_c=0.1, Rs=20.0)

    print("\nBuilding NFWDataset (50k samples)...")
    dataset = NFWDataset(nfw, n_samples=50_000, r_max=100.0)

    # Test 1: Dataset length
    print(f"\nTest 1 — Length: {len(dataset)} (expected: 50000)  "
          f"{'PASS' if len(dataset) == 50_000 else 'FAIL'}")

    # Test 2: Item shapes
    coords, rho, phi = dataset[0]
    print(f"\nTest 2 — Item shapes:")
    print(f"  coords: {coords.shape}  (expected: [3])    {'PASS' if coords.shape == (3,) else 'FAIL'}")
    print(f"  rho:    {rho.shape}    (expected: [1])    {'PASS' if rho.shape == (1,) else 'FAIL'}")
    print(f"  phi:    {phi.shape}    (expected: [1])    {'PASS' if phi.shape == (1,) else 'FAIL'}")

    # Test 3: Normalisation quality
    all_coords = dataset.coords
    all_rho    = dataset.rho
    all_phi    = dataset.phi

    coords_in_range = (all_coords.abs() <= 1.0).all()
    rho_mean_ok     = abs(all_rho.mean().item()) < 0.05
    rho_std_ok      = abs(all_rho.std().item()  - 1.0) < 0.05
    phi_mean_ok     = abs(all_phi.mean().item()) < 0.05
    phi_std_ok      = abs(all_phi.std().item()  - 1.0) < 0.05

    print(f"\nTest 3 — Normalisation quality:")
    print(f"  Coords in [-1,1]:  {'PASS' if coords_in_range else 'FAIL'}")
    print(f"  Rho mean ~ 0:      {all_rho.mean().item():.4f}  {'PASS' if rho_mean_ok else 'FAIL'}")
    print(f"  Rho std  ~ 1:      {all_rho.std().item():.4f}   {'PASS' if rho_std_ok else 'FAIL'}")
    print(f"  Phi mean ~ 0:      {all_phi.mean().item():.4f}  {'PASS' if phi_mean_ok else 'FAIL'}")
    print(f"  Phi std  ~ 1:      {all_phi.std().item():.4f}   {'PASS' if phi_std_ok else 'FAIL'}")

    # Test 4: De-normalisation round-trip
    coords_phys  = dataset.denorm_coords(all_coords[:10])
    coords_back  = dataset.norm_coords(coords_phys)
    roundtrip_err = (coords_back - all_coords[:10]).abs().max().item()
    print(f"\nTest 4 — Coord round-trip error: {roundtrip_err:.2e}  "
          f"{'PASS' if roundtrip_err < 1e-5 else 'FAIL'}")

    # Test 5: DataLoader
    loader = dataset.get_dataloader(batch_size=512)
    batch  = next(iter(loader))
    print(f"\nTest 5 — DataLoader batch shapes:")
    print(f"  coords: {batch[0].shape}  phi: {batch[2].shape}  rho: {batch[1].shape}")

    # Test 6: Colocation sampler
    print(f"\nTest 6 — ColocationSampler:")
    sampler = ColocationSampler(r_max=1.0, batch_size=1024)
    col_batch = sampler.sample()
    in_sphere = ((col_batch.detach() ** 2).sum(dim=1) <= 1.0).all()
    has_grad  = col_batch.requires_grad
    print(f"  All points in unit sphere: {'PASS' if in_sphere else 'FAIL'}")
    print(f"  requires_grad=True:        {'PASS' if has_grad else 'FAIL'}")
    print(f"  Shape: {col_batch.shape}  (expected: [1024, 3])")

    # Test 7: Train/val split
    train_loader, val_loader = dataset.get_split(val_fraction=0.1)
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"\nTest 7 — Train/val split: {n_train} train / {n_val} val  "
          f"(expected ~45000 / ~5000)")

    print("\n" + "=" * 60)
    print("Dataset self-test complete.")
    print("=" * 60)

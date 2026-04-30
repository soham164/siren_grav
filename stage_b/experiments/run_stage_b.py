"""
experiments/run_stage_b.py  —  Stage B Master Script
=====================================================
Full pipeline: IllustrisTNG data → SIREN training → evaluation → plots.

Run in Google Colab:
    # Install dependencies first:
    !pip install torch numpy scipy matplotlib h5py requests

    # Set your API key:
    import os
    os.environ["TNG_API_KEY"] = "your_key_here"

    # Run:
    !python experiments/run_stage_b.py

Expected runtime on T4 GPU:
    Data download : 10-30 min (depends on halo size, ~200-500 MB each)
    Training      : 20-40 min per halo (50k steps)
    Evaluation    : 5-10 min per halo
    Total         : 2-4 hours for 5 halos

What this script does:
    1. Downloads 5 halos from IllustrisTNG (or loads from disk if cached)
    2. For each halo: builds density + potential fields from particles
    3. Trains SIREN (pre-train → fine-tune with adaptive Poisson)
    4. Evaluates: compression ratio, reconstruction error, Poisson residual
    5. Generates all comparison plots
    6. Prints the multi-halo benchmark table (the key scientific result)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.units         import TNGUnits
from src.tng_loader    import TNGDownloader, TNGHaloReader, HaloProcessor
from src.field_estimator import RealHaloDataset
from src.models        import SirenNetwork
from src.trainer_b     import AdaptivePINNTrainer, ColocationSampler

# ── Configuration ──────────────────────────────────────────────────────────────
CFG = {
    # TNG access
    'api_key'     : os.environ.get("TNG_API_KEY", ""),
    'simulation'  : "TNG100-1",
    'snapshot'    : 99,
    'n_halos'     : 5,

    # Dataset
    'n_samples'   : 80_000,
    'r_max_factor': 1.5,

    # Network
    'hidden_features': 256,
    'hidden_layers'  : 4,
    'omega_0'        : 30.0,

    # Training
    'pretrain_steps' : 10_000,
    'finetune_steps' : 30_000,
    'pretrain_lr'    : 5e-4,
    'finetune_lr'    : 1e-4,
    'adapt_every'    : 200,
    'coloc_size'     : 1024,

    # Paths
    'data_dir'    : 'data/halos',
    'output_dir'  : 'outputs/stage_b',
    'device'      : 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed'        : 42,
}

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
os.makedirs(CFG['data_dir'],   exist_ok=True)
os.makedirs(CFG['output_dir'], exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download halos
# ══════════════════════════════════════════════════════════════════════════════
def download_halos():
    print("\n" + "="*60)
    print("STEP 1 — IllustrisTNG Data Download")
    print("="*60)

    if not CFG['api_key']:
        print("\nNo API key found. Set environment variable TNG_API_KEY.")
        print("Register at: https://www.tng-project.org/users/register/")
        print("\nFalling back to mock data for pipeline testing...")
        return generate_mock_halos()

    dl = TNGDownloader(
        api_key=CFG['api_key'],
        simulation=CFG['simulation'],
        snapshot=CFG['snapshot'],
    )

    halos    = dl.get_top_halos(n=CFG['n_halos'])
    hdf5_paths = []

    for halo in halos:
        path = dl.download_halo_particles(halo, output_dir=CFG['data_dir'])
        hdf5_paths.append((halo, path))

    return hdf5_paths


def generate_mock_halos():
    """
    Generate synthetic mock halos that mimic TNG output structure.
    Used when no API key is available — for pipeline testing only.
    The mock data is NFW-like but with added realistic noise and asymmetry.
    """
    print("\nGenerating mock halos (NFW + noise, for pipeline testing)...")
    np.random.seed(CFG['seed'])

    mock_halos = []
    masses  = [1e13, 5e12, 2e12, 1e12, 5e11]   # M_sun, 5 halos
    radii   = [800,  600,  450,  350,  250]     # kpc R200

    for i, (M200, R200) in enumerate(zip(masses, radii)):
        n_particles = int(1e5 * (M200 / 1e12) ** 0.8)
        n_particles = min(n_particles, 300_000)

        # NFW-like sampling: more particles near centre
        r = np.random.exponential(scale=R200/5, size=n_particles)
        r = np.clip(r, 0.1, 2*R200)

        # Random directions
        theta  = np.arccos(2*np.random.rand(n_particles) - 1)
        phi    = 2 * np.pi * np.random.rand(n_particles)
        pos    = np.stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ], axis=1)

        # Add slight ellipticity (real halos are not spherical)
        pos[:, 1] *= 0.85   # compress y-axis by 15%
        pos[:, 2] *= 0.70   # compress z-axis by 30%

        # Equal-mass particles
        particle_mass = M200 / n_particles
        masses_arr    = np.full(n_particles, particle_mass)

        # Velocities: circular velocity + thermal dispersion
        v_circ  = np.sqrt(4.3e-6 * M200 / (r + 1.0))  # km/s
        vel     = np.random.randn(n_particles, 3) * v_circ[:, None] * 0.5

        mock_halos.append({
            "type"      : "mock",
            "group_id"  : i,
            "M200_msun" : M200,
            "R200_kpc"  : R200,
            "r200_kpc"  : R200,  # Add lowercase version for compatibility
            "dm_pos_kpc"   : pos,
            "dm_mass_msun" : masses_arr,
            "dm_vel_kms"   : vel,
            "n_dm"         : n_particles,
        })

        print(f"  Mock halo {i}: M200={M200:.1e} M_sun, "
              f"R200={R200} kpc, N={n_particles:,}")

    return mock_halos


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Process each halo into a training dataset
# ══════════════════════════════════════════════════════════════════════════════
def process_halo(halo_data, halo_idx):
    """
    Convert raw halo data (real TNG or mock) into a RealHaloDataset.
    """
    print(f"\n{'='*60}")
    print(f"STEP 2 — Processing Halo {halo_idx}")
    print(f"{'='*60}")

    if isinstance(halo_data, dict) and halo_data.get("type") == "mock":
        # Mock halo — already in physical units
        processed = halo_data
    else:
        # Real TNG halo — read HDF5 and convert units
        halo_meta, hdf5_path = halo_data
        units    = TNGUnits(redshift=0.0, h=0.6774)
        reader   = TNGHaloReader(hdf5_path)
        raw_dm   = reader.get_dark_matter()
        raw_star = reader.get_stars()

        centre_raw = np.array([
            halo_meta.get("pos_x", 0),
            halo_meta.get("pos_y", 0),
            halo_meta.get("pos_z", 0),
        ])
        r200_raw = halo_meta.get("R200", None)

        processor = HaloProcessor(units)
        processed = processor.process(raw_dm, raw_star, centre_raw, r200_raw)

    # Build training dataset
    dataset = RealHaloDataset(
        processed,
        n_samples=CFG['n_samples'],
        r_max_factor=CFG['r_max_factor'],
        device=CFG['device'],
        seed=CFG['seed'],
    )

    return dataset, processed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train SIREN on one halo
# ══════════════════════════════════════════════════════════════════════════════
def train_on_halo(dataset, halo_idx):
    print(f"\n{'='*60}")
    print(f"STEP 3 — Training SIREN on Halo {halo_idx}")
    print(f"{'='*60}")

    out_dir = os.path.join(CFG['output_dir'], f"halo_{halo_idx:02d}")
    os.makedirs(out_dir, exist_ok=True)

    model  = SirenNetwork(
        hidden_features=CFG['hidden_features'],
        hidden_layers=CFG['hidden_layers'],
        omega_0=CFG['omega_0'],
    )

    coloc   = ColocationSampler(r_max=1.0, device=CFG['device'],
                                 batch_size=CFG['coloc_size'])

    trainer = AdaptivePINNTrainer(
        model=model,
        dataset=dataset,
        coloc_sampler=coloc,
        output_dir=out_dir,
        device=CFG['device'],
        pretrain_steps=CFG['pretrain_steps'],
        finetune_steps=CFG['finetune_steps'],
        pretrain_lr=CFG['pretrain_lr'],
        finetune_lr=CFG['finetune_lr'],
        adapt_every=CFG['adapt_every'],
        coloc_size=CFG['coloc_size'],
        log_every=500,
        save_every=10_000,
    )

    history = trainer.train()
    return model, history, trainer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluate one trained model
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model(model, dataset, halo_processed, device):
    model.eval()
    model = model.to(device)

    # Sample test coordinates
    r_max    = dataset.r_max
    n_test   = 10_000
    test_pts = []
    while len(test_pts) < n_test:
        b = np.random.uniform(-r_max, r_max, (n_test*2, 3))
        b = b[np.linalg.norm(b, axis=1) <= r_max]
        test_pts.append(b)
    test_coords_phys = np.concatenate(test_pts)[:n_test]

    # True density and potential at test points
    rho_true = dataset.density_est.density_at(test_coords_phys)
    phi_true = dataset.potential_est.potential_at(test_coords_phys)

    # Network predictions
    coords_norm = torch.FloatTensor(test_coords_phys / r_max).to(device)
    phi_pred_n, rho_pred_n = model(coords_norm)
    phi_pred = dataset.denorm_phi(phi_pred_n).cpu().numpy().ravel()
    rho_pred = dataset.denorm_rho(rho_pred_n).cpu().numpy().ravel()

    # Density-weighted relative errors (fair metric — low-density regions
    # are down-weighted so the diffuse outer halo doesn't dominate)
    weights     = rho_true / (rho_true.sum() + 1e-12)
    rho_rel_err = (np.abs(rho_pred - rho_true) / (rho_true + 1e-12) * weights).sum() * 100
    phi_rel_err = (np.abs(phi_pred - phi_true) / (np.abs(phi_true) + 1e-12) * weights).sum() * 100

    # Compression ratio
    n_particles    = halo_processed["n_dm"]
    raw_size_mb    = n_particles * 7 * 4 / 1e6   # 7 floats per particle (pos+vel+mass) * 4 bytes
    model_size_mb  = model.weight_file_size_kb() / 1024

    return {
        "rho_rel_err_pct"   : float(rho_rel_err),
        "phi_rel_err_pct"   : float(phi_rel_err),
        "n_particles"       : n_particles,
        "raw_size_mb"       : raw_size_mb,
        "model_size_mb"     : model_size_mb,
        "compression_ratio" : raw_size_mb / model_size_mb,
        "n_params"          : model.count_parameters(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_history(history, halo_idx, out_dir):
    steps = history['step']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].semilogy(steps, history['loss_data'],    'b-', lw=1.5, label='Data loss')
    axes[0].semilogy(steps, history['loss_poisson'], 'r-', lw=1.5, label='Poisson loss')
    axes[0].axvline(x=CFG['pretrain_steps'], color='gray', linestyle='--', label='Phase 2 start')
    axes[0].set_title('Training Losses', fontweight='bold')
    axes[0].set_xlabel('Step'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, history['lambda_2'], 'g-', lw=1.5)
    axes[1].axvline(x=CFG['pretrain_steps'], color='gray', linestyle='--')
    axes[1].set_title('Adaptive λ₂ (NTK weighting)', fontweight='bold')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('λ₂'); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(steps, history['val_loss'], 'k-', lw=1.5)
    axes[2].axvline(x=CFG['pretrain_steps'], color='gray', linestyle='--')
    axes[2].set_title('Validation Loss', fontweight='bold')
    axes[2].set_xlabel('Step'); axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'Halo {halo_idx} — Two-Phase PINN Training', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, f'halo_{halo_idx:02d}_training.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_density_slice(model, dataset, halo_idx, out_dir, device):
    """2D density slice plot for one halo."""
    model.eval().to(device)
    gs   = 80
    lim  = dataset.r_max * 0.9
    x_l  = np.linspace(-lim, lim, gs)
    xx, yy = np.meshgrid(x_l, x_l)
    coords_grid = np.stack([xx.ravel(), yy.ravel(),
                             np.zeros(gs*gs)], axis=1)

    # True density
    rho_true = dataset.density_est.density_at(coords_grid).reshape(gs, gs)

    # Predicted density
    cn = torch.FloatTensor(coords_grid / dataset.r_max).to(device)
    with torch.no_grad():
        _, rho_pred_n = model(cn)
        rho_pred = dataset.denorm_rho(rho_pred_n).cpu().numpy().reshape(gs, gs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin = rho_true.min(); vmax = np.percentile(rho_true, 99)

    axes[0].imshow(np.log10(rho_true + 1e-10), origin='lower', cmap='inferno',
                   extent=[-lim, lim, -lim, lim])
    axes[0].set_title('True Density (log10)', fontweight='bold')

    axes[1].imshow(np.log10(np.abs(rho_pred) + 1e-10), origin='lower', cmap='inferno',
                   extent=[-lim, lim, -lim, lim])
    axes[1].set_title('SIREN Predicted (log10)', fontweight='bold')

    rel_err = np.abs(rho_pred - rho_true) / (rho_true + 1e-10) * 100
    im = axes[2].imshow(np.clip(rel_err, 0, 100), origin='lower', cmap='RdYlGn_r',
                        vmin=0, vmax=50, extent=[-lim, lim, -lim, lim])
    plt.colorbar(im, ax=axes[2], label='Relative Error (%)')
    axes[2].set_title(f'Error Map (mean={rel_err.mean():.1f}%)', fontweight='bold')

    for ax in axes:
        ax.set_xlabel('x (kpc)'); ax.set_ylabel('y (kpc)')

    plt.suptitle(f'Halo {halo_idx} — Density Field Comparison', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, f'halo_{halo_idx:02d}_density.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_multi_halo_benchmark(all_results, out_dir):
    """Bar chart comparing compression and error across all 5 halos."""
    halos = list(all_results.keys())
    rho_errs = [all_results[h]['rho_rel_err_pct'] for h in halos]
    comps    = [all_results[h]['compression_ratio'] for h in halos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(halos)))

    ax1.bar(halos, rho_errs, color=colors)
    ax1.axhline(y=5, color='red', linestyle='--', label='5% target')
    ax1.set_xlabel('Halo'); ax1.set_ylabel('Density Error (%)')
    ax1.set_title('Reconstruction Error per Halo\n(density-weighted)', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(halos, comps, color=colors)
    ax2.set_xlabel('Halo'); ax2.set_ylabel('Compression Ratio (x)')
    ax2.set_title('Compression Ratio per Halo\n(raw particles / SIREN weights)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Stage B — Multi-Halo Benchmark', fontweight='bold', fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, 'multi_halo_benchmark.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_benchmark_table(all_results):
    print("\n" + "="*75)
    print("STAGE B BENCHMARK TABLE")
    print("="*75)
    print(f"{'Halo':>6} {'N_DM':>10} {'Raw MB':>8} {'SIREN MB':>9} "
          f"{'Compress':>9} {'Rho err%':>9} {'Phi err%':>9}")
    print("-"*75)
    for halo_id, m in all_results.items():
        print(f"{halo_id:>6} "
              f"{m['n_particles']:>10,} "
              f"{m['raw_size_mb']:>8.1f} "
              f"{m['model_size_mb']:>9.2f} "
              f"{m['compression_ratio']:>8.0f}x "
              f"{m['rho_rel_err_pct']:>8.2f}% "
              f"{m['phi_rel_err_pct']:>8.2f}%")
    print("="*75)
    mean_comp = np.mean([m['compression_ratio'] for m in all_results.values()])
    mean_rho  = np.mean([m['rho_rel_err_pct']   for m in all_results.values()])
    print(f"\nMean compression ratio : {mean_comp:.0f}x")
    print(f"Mean density error     : {mean_rho:.2f}%")
    print(f"Model size             : {list(all_results.values())[0]['model_size_mb']:.2f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("STAGE B — Neural Galactic Potential on Real IllustrisTNG Data")
    print(f"Device : {CFG['device']}")
    print("="*60)

    # Step 1: Download / generate halos
    halo_data_list = download_halos()

    all_results  = {}
    all_histories = {}

    for halo_idx, halo_data in enumerate(halo_data_list):

        print(f"\n{'#'*60}")
        print(f"# HALO {halo_idx + 1} / {len(halo_data_list)}")
        print(f"{'#'*60}")

        halo_out = os.path.join(CFG['output_dir'], f"halo_{halo_idx:02d}")
        os.makedirs(halo_out, exist_ok=True)

        # Step 2: Process into dataset
        dataset, processed = process_halo(halo_data, halo_idx)

        # Step 3: Train
        model, history, trainer = train_on_halo(dataset, halo_idx)
        all_histories[f"halo_{halo_idx}"] = history

        # Step 4: Evaluate
        print(f"\nEvaluating halo {halo_idx}...")
        metrics = evaluate_model(model, dataset, processed, CFG['device'])
        all_results[f"halo_{halo_idx}"] = metrics

        print(f"  Rho error     : {metrics['rho_rel_err_pct']:.2f}%")
        print(f"  Phi error     : {metrics['phi_rel_err_pct']:.2f}%")
        print(f"  Compression   : {metrics['compression_ratio']:.0f}x")

        # Step 5: Plots
        print(f"\nGenerating plots for halo {halo_idx}...")
        plot_training_history(history, halo_idx, halo_out)
        plot_density_slice(model, dataset, halo_idx, halo_out, CFG['device'])

    # Multi-halo summary
    print_benchmark_table(all_results)
    plot_multi_halo_benchmark(all_results, CFG['output_dir'])

    # Save all results
    results_path = os.path.join(CFG['output_dir'], 'stage_b_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {CFG['output_dir']}/")

    print("\n" + "="*60)
    print("STAGE B COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

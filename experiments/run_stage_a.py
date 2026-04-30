"""
experiments/run_stage_a.py  —  Step 7: The Full Stage A Experiment
===================================================================
This is the master script. It runs everything in sequence:

    1. Builds the NFW ground truth
    2. Runs all self-tests (gate: abort if anything fails)
    3. Trains three models for comparison:
        - ReLU baseline (standard network, no physics)
        - SIREN data-only (no physics loss)
        - SIREN + Poisson loss (full PINN)
    4. Evaluates all three on a held-out test set
    5. Produces the benchmark table (Table 1 in the paper)
    6. Saves all plots to outputs/

Run this in Google Colab:
    !python experiments/run_stage_a.py

Expected runtime:
    CPU : ~20-40 minutes
    GPU : ~3-5 minutes
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nfw     import NFWProfile
from src.siren   import SirenNetwork, verify_siren_initialization
from src.physics import verify_laplacian, laplacian_autograd
from src.dataset import NFWDataset, ColocationSampler
from src.trainer import Trainer

# ── Configuration ─────────────────────────────────────────────────────────────
CFG = {
    # NFW halo parameters
    'rho_c'       : 0.1,
    'Rs'          : 20.0,
    'r_max'       : 100.0,

    # Dataset
    'n_train'     : 200_000,
    'n_test'      : 20_000,

    # Network
    'hidden_features' : 256,
    'hidden_layers'   : 4,
    'omega_0'         : 30.0,

    # Training
    'total_steps'     : 30_000,   # Reduce to 10_000 for quick test
    'warmup_steps'    : 8_000,
    'learning_rate'   : 5e-4,
    'lambda_1'        : 1.0,
    'lambda_2_max'    : 0.005,
    'batch_size'      : 4096,
    'coloc_size'      : 4096,
    'log_every'       : 1000,
    'save_every'      : 10_000,

    # Output
    'output_dir'      : 'outputs/stage_a',
    'device'          : 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed'            : 42,
}

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
os.makedirs(CFG['output_dir'], exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Gate tests (abort if any fail)
# ═══════════════════════════════════════════════════════════════════════════════
def run_gate_tests(nfw):
    print("\n" + "=" * 60)
    print("PHASE 0 — Gate Tests (must all pass before training)")
    print("=" * 60)
    all_pass = True

    # Gate 1: NFW self-test
    print("\n[Gate 1] NFW analytical functions")
    r_test    = torch.tensor([[CFG['Rs'], 0.0, 0.0]])
    rho_at_Rs = nfw.density(r_test).item()
    expected  = nfw.rho_c / 4.0
    err       = abs(rho_at_Rs - expected) / expected
    ok        = err < 1e-4
    all_pass  = all_pass and ok
    print(f"  rho(Rs) = {rho_at_Rs:.6f}, expected {expected:.6f}, err={err:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # Gate 2: SIREN initialization
    print("\n[Gate 2] SIREN weight initialization")
    test_model = SirenNetwork(
        hidden_features=CFG['hidden_features'],
        hidden_layers=CFG['hidden_layers'],
        omega_0=CFG['omega_0']
    )
    ok, _ = verify_siren_initialization(test_model, n_test=10_000, verbose=True)
    all_pass = all_pass and ok
    print(f"  Overall: {'PASS' if ok else 'FAIL'}")

    # Gate 3: Laplacian cross-verification
    print("\n[Gate 3] Laplacian implementation (autograd vs finite diff)")
    coords_test   = nfw.sample_coords(100, r_max=CFG['r_max'])
    analytical_lap = nfw.laplacian(coords_test)
    ok, _ = verify_laplacian(
        phi_fn=nfw.potential,
        coords=coords_test,
        nfw_analytical_laplacian=analytical_lap,
        tolerance=0.02,
        fd_tolerance=0.05,  # Finite differences need looser tolerance
        verbose=True,
    )
    all_pass = all_pass and ok
    print(f"  Overall: {'PASS' if ok else 'FAIL'}")

    if not all_pass:
        print("\nSome gate tests FAILED. Aborting.")
        sys.exit(1)

    print("\nAll gate tests PASSED. Proceeding to training.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Build dataset
# ═══════════════════════════════════════════════════════════════════════════════
def build_dataset(nfw):
    print("\n" + "=" * 60)
    print("PHASE 1 — Building Dataset")
    print("=" * 60)
    dataset = NFWDataset(
        nfw,
        n_samples=CFG['n_train'],
        r_max=CFG['r_max'],
        device=CFG['device'],
        seed=CFG['seed'],
    )
    coloc = ColocationSampler(
        r_max=1.0,
        device=CFG['device'],
        batch_size=CFG['coloc_size'],
    )
    # Test dataset
    n_test_coords = nfw.sample_coords(CFG['n_test'], r_max=CFG['r_max'])
    with torch.no_grad():
        test_rho = nfw.density(n_test_coords)
        test_phi = nfw.potential(n_test_coords)

    print(f"\nDataset ready: {len(dataset):,} training samples")
    return dataset, coloc, n_test_coords, test_rho, test_phi


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Train all three models
# ═══════════════════════════════════════════════════════════════════════════════
def build_relu_baseline(hidden_features, hidden_layers):
    """
    Standard ReLU network — same architecture as SIREN but with ReLU activations.
    This is the comparison baseline to demonstrate spectral bias.
    """
    layers = [nn.Linear(3, hidden_features), nn.ReLU()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]

    backbone = nn.Sequential(*layers)

    class ReLUNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone         = backbone
            self.head_phi         = nn.Linear(hidden_features, 1)
            self.head_rho_linear  = nn.Linear(hidden_features, 1)
            self.softplus         = nn.Softplus(beta=10)

        def forward(self, coords):
            f   = self.backbone(coords)
            phi = self.head_phi(f)
            rho = self.softplus(self.head_rho_linear(f))
            return phi, rho

        def forward_phi_only(self, coords):
            return self.head_phi(self.backbone(coords))

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def weight_file_size_kb(self):
            return self.count_parameters() * 4 / 1024

    return ReLUNetwork()


def train_model(name, model, dataset, coloc, nfw, use_physics=True):
    """Train a single model and return its history."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    out_dir = os.path.join(CFG['output_dir'], name.lower().replace(' ', '_'))

    trainer = Trainer(
        model=model,
        dataset=dataset,
        coloc_sampler=coloc,
        nfw=nfw,
        output_dir=out_dir,
        device=CFG['device'],
        learning_rate=CFG['learning_rate'],
        total_steps=CFG['total_steps'],
        warmup_steps=CFG['warmup_steps'] if use_physics else CFG['total_steps'],
        lambda_1=CFG['lambda_1'],
        lambda_2_max=CFG['lambda_2_max'] if use_physics else 0.0,
        batch_size=CFG['batch_size'],
        coloc_size=CFG['coloc_size'],
        log_every=CFG['log_every'],
        save_every=CFG['save_every'],
        G=1.0,
    )

    history = trainer.train()
    return history, trainer


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_model(model, dataset, test_coords, test_rho, test_phi, nfw, device):
    """
    Compute all evaluation metrics for one trained model.

    Returns dict with:
        rho_rel_error   : mean relative error on density
        phi_rel_error   : mean relative error on potential
        rho_weighted_error : density-weighted relative error (fairer metric)
        phi_weighted_error : density-weighted relative error (fairer metric)
        poisson_residual: mean |nabla^2 Phi - 4piG*rho| / |4piG*rho|
        weight_size_kb  : model file size
    """
    model.eval()
    model = model.to(device)

    # Normalise test coords
    coords_norm = dataset.norm_coords(test_coords).to(device)

    # Predictions (normalised)
    phi_pred_norm, rho_pred_norm = model(coords_norm)

    # De-normalise
    phi_pred = dataset.denorm_phi(phi_pred_norm).cpu()
    rho_pred = dataset.denorm_rho(rho_pred_norm).cpu()

    # Standard relative errors (can be dominated by low-density regions)
    rho_rel_err = ((rho_pred - test_rho).abs() / (test_rho.abs() + 1e-8)).mean().item()
    phi_rel_err = ((phi_pred - test_phi).abs() / (test_phi.abs() + 1e-8)).mean().item()
    
    # Density-weighted errors (fairer metric - weights by local density)
    # This gives more importance to the dense inner regions where physics matters most
    density_weights = test_rho / (test_rho.sum() + 1e-8)  # Normalize to sum to 1
    rho_weighted_err = (density_weights * (rho_pred - test_rho).abs() / (test_rho.abs() + 1e-8)).sum().item()
    phi_weighted_err = (density_weights * (phi_pred - test_phi).abs() / (test_phi.abs() + 1e-8)).sum().item()

    # Poisson residual (on a subset — autograd needs grad)
    # Note: We need to enable gradients temporarily for this computation
    n_poisson = min(500, test_coords.shape[0])
    
    try:
        # Temporarily enable gradients for Poisson residual computation
        with torch.enable_grad():
            model.train()  # Enable gradient tracking in model
            
            pc = test_coords[:n_poisson].to(device).requires_grad_(True)
            pc_norm = dataset.norm_coords(pc)
            
            def phi_phys_fn(c_norm):
                p, _ = model(c_norm)
                return dataset.denorm_phi(p)

            lap_phi = laplacian_autograd(phi_phys_fn, pc_norm)
            rho_at_pc = nfw.density(pc.detach()).to(device)
            target    = 4.0 * np.pi * rho_at_pc
            poisson_rel = ((lap_phi - target).abs() / (target.abs() + 1e-8)).mean().item()
            
            model.eval()  # Set back to eval mode
    except RuntimeError as e:
        # If gradient computation fails (e.g., for ReLU baseline), set to NaN
        print(f"    Warning: Could not compute Poisson residual: {e}")
        poisson_rel = float('nan')

    return {
        'rho_rel_error'   : rho_rel_err * 100,    # percentage
        'phi_rel_error'   : phi_rel_err * 100,
        'rho_weighted_error': rho_weighted_err * 100,  # density-weighted (fairer)
        'phi_weighted_error': phi_weighted_err * 100,  # density-weighted (fairer)
        'poisson_residual': poisson_rel * 100,
        'weight_size_kb'  : model.weight_file_size_kb(),
        'n_params'        : model.count_parameters(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Plotting
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(histories, labels, output_dir):
    """Plot training loss curves for all models side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = ['#E8593C', '#2E6DA4', '#1D9E75']

    for (hist, label, color) in zip(histories, labels, colors):
        steps = hist['step']
        axes[0].plot(steps, hist['loss_data'],    color=color, label=label, linewidth=1.5)
        axes[1].plot(steps, hist['loss_poisson'], color=color, label=label, linewidth=1.5)
        axes[2].plot(steps, hist['val_loss_data'],color=color, label=label, linewidth=1.5)

    for ax, title in zip(axes, ['Data Loss (MSE)', 'Poisson Loss', 'Validation Loss']):
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Stage A Training Curves — SIREN vs ReLU vs PINN', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_density_comparison(models_dict, dataset, nfw, output_dir, device):
    """
    2D slice plot: true NFW density vs predictions from all three models.
    Slice through z=0 plane.
    """
    grid_size = 100
    x_lin     = np.linspace(-80, 80, grid_size)
    y_lin     = np.linspace(-80, 80, grid_size)
    xx, yy    = np.meshgrid(x_lin, y_lin)
    zz        = np.zeros_like(xx)

    coords_grid = torch.FloatTensor(
        np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    ).to(device)

    # True NFW density
    with torch.no_grad():
        true_rho = nfw.density(coords_grid).cpu().numpy().reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = true_rho.min(), true_rho.max()

    # True density
    im = axes[0].imshow(true_rho, origin='lower', cmap='inferno',
                        vmin=vmin, vmax=vmax,
                        extent=[-80, 80, -80, 80])
    axes[0].set_title('Ground Truth\n(NFW analytical)', fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Density')

    # Model predictions
    model_names = list(models_dict.keys())
    for i, (name, model) in enumerate(models_dict.items()):
        model.eval().to(device)
        coords_norm = dataset.norm_coords(coords_grid)
        with torch.no_grad():
            _, rho_pred_norm = model(coords_norm)
            rho_pred = dataset.denorm_rho(rho_pred_norm).cpu().numpy()
        rho_grid = rho_pred.reshape(grid_size, grid_size)

        ax = axes[i + 1]
        im = ax.imshow(rho_grid, origin='lower', cmap='inferno',
                       vmin=vmin, vmax=vmax,
                       extent=[-80, 80, -80, 80])
        err = abs(rho_grid - true_rho) / (true_rho + 1e-8) * 100
        ax.set_title(f'{name}\n(mean err: {err.mean():.1f}%)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Density')

    for ax in axes:
        ax.set_xlabel('x (kpc)')
        ax.set_ylabel('y (kpc)')

    plt.suptitle('Density Field — z=0 Slice Comparison', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'density_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_radial_profile(models_dict, dataset, nfw, output_dir, device):
    """
    Radial density profile: log-log plot of density vs radius.
    Shows how well each model captures the NFW 1/r^3 falloff.
    """
    r_vals = np.logspace(np.log10(1.0), np.log10(90.0), 200)
    coords_radial = torch.FloatTensor(
        np.stack([r_vals, np.zeros_like(r_vals), np.zeros_like(r_vals)], axis=1)
    ).to(device)

    fig, ax = plt.subplots(figsize=(8, 6))

    # True NFW
    with torch.no_grad():
        true_rho = nfw.density(coords_radial).cpu().numpy().ravel()
    ax.loglog(r_vals, true_rho, 'k-', linewidth=2.5, label='NFW Ground Truth', zorder=5)

    # Model predictions
    colors = ['#E8593C', '#2E6DA4', '#1D9E75']
    styles = ['--', '-.', ':']
    for (name, model), color, style in zip(models_dict.items(), colors, styles):
        model.eval().to(device)
        coords_norm = dataset.norm_coords(coords_radial)
        with torch.no_grad():
            _, rho_pred_norm = model(coords_norm)
            rho_pred = dataset.denorm_rho(rho_pred_norm).cpu().numpy().ravel()
        ax.loglog(r_vals, np.abs(rho_pred), color=color, linestyle=style,
                  linewidth=2, label=name)

    ax.axvline(x=nfw.Rs, color='gray', linestyle=':', alpha=0.7, label=f'Rs = {nfw.Rs} kpc')
    ax.set_xlabel('Radius r (kpc)', fontsize=12)
    ax.set_ylabel('Density ρ(r)', fontsize=12)
    ax.set_title('Radial Density Profile — Model Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'radial_profile.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_poisson_residual(model, dataset, nfw, output_dir, device, name="PINN_SIREN"):
    """
    2D map of the Poisson residual |nabla^2 Phi - 4piG*rho| / |4piG*rho|.
    Should be near-zero everywhere if the physics constraint is satisfied.
    """
    grid_size = 40    # smaller grid — autograd is slower
    x_lin     = np.linspace(-60, 60, grid_size)
    y_lin     = np.linspace(-60, 60, grid_size)
    xx, yy    = np.meshgrid(x_lin, y_lin)

    residuals = np.zeros((grid_size, grid_size))

    model.eval().to(device)
    def phi_phys_fn(c):
        p, _ = model(dataset.norm_coords(c))
        return dataset.denorm_phi(p)

    for i in range(grid_size):
        for j in range(grid_size):
            c = torch.FloatTensor([[xx[i,j], yy[i,j], 0.0]]).to(device)
            c = c.requires_grad_(True)
            lap   = laplacian_autograd(phi_phys_fn, c)
            rho_t = nfw.density(c.detach())
            target = 4.0 * np.pi * rho_t
            with torch.no_grad():
                res = (lap - target).abs() / (target.abs() + 1e-8)
            residuals[i, j] = res.item()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(residuals * 100, origin='lower', cmap='RdYlGn_r',
                   vmin=0, vmax=20, extent=[-60, 60, -60, 60])
    plt.colorbar(im, ax=ax, label='Poisson Residual (%)')
    ax.set_xlabel('x (kpc)', fontsize=12)
    ax.set_ylabel('y (kpc)', fontsize=12)
    ax.set_title(f'Poisson Residual Map — {name}\n'
                 f'(mean: {residuals.mean()*100:.1f}%  '
                 f'max: {residuals.max()*100:.1f}%)',
                 fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(output_dir, 'poisson_residual_map.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Print benchmark table
# ═══════════════════════════════════════════════════════════════════════════════
def print_benchmark_table(results):
    """Print the Stage A benchmark table — this becomes Table 1 in the paper."""
    print("\n" + "=" * 90)
    print("STAGE A BENCHMARK TABLE")
    print("=" * 90)
    print(f"{'Model':<25} {'Rho err%':>10} {'Rho wtd%':>10} {'Phi err%':>10} {'Phi wtd%':>10} "
          f"{'Poisson%':>10} {'Size KB':>10}")
    print("-" * 90)

    for name, metrics in results.items():
        print(f"{name:<25} "
              f"{metrics['rho_rel_error']:>9.2f}% "
              f"{metrics['rho_weighted_error']:>9.2f}% "
              f"{metrics['phi_rel_error']:>9.2f}% "
              f"{metrics['phi_weighted_error']:>9.2f}% "
              f"{metrics['poisson_residual']:>9.2f}% "
              f"{metrics['weight_size_kb']:>9.1f}")

    print("-" * 90)
    print("Rho err%     : mean relative error on density (all points)")
    print("Rho wtd%     : density-weighted error (fairer - emphasizes dense regions)")
    print("Phi err%     : mean relative error on potential (all points)")
    print("Phi wtd%     : density-weighted error (fairer - emphasizes dense regions)")
    print("Poisson%     : mean relative Poisson residual (lower = more physical)")
    print("Size KB      : model weight file size")
    print("Raw data est.: ~50 MB (200k particles x 7 floats x 4 bytes)")
    compression = 50_000 / list(results.values())[-1]['weight_size_kb']
    print(f"Compression  : ~{compression:.0f}x  (SIREN vs raw particles)")
    print("\nNote: Weighted errors give more importance to high-density regions")
    print("      where the physics is most important (inner halo).")
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("STAGE A — Neural Galactic Potential (Proof of Concept)")
    print(f"Device: {CFG['device']}")
    print("=" * 60)

    # Build NFW ground truth
    nfw = NFWProfile(rho_c=CFG['rho_c'], Rs=CFG['Rs'])

    # Gate tests — abort if any fail
    run_gate_tests(nfw)

    # Build dataset
    dataset, coloc, test_coords, test_rho, test_phi = build_dataset(nfw)

    # Train all three models
    histories = []
    labels    = []
    trained_models = {}

    # Model 1: ReLU baseline
    relu_model = build_relu_baseline(CFG['hidden_features'], CFG['hidden_layers'])
    hist_relu, _ = train_model("ReLU Baseline", relu_model, dataset, coloc, nfw,
                                use_physics=False)
    histories.append(hist_relu)
    labels.append("ReLU Baseline")
    trained_models["ReLU Baseline"] = relu_model

    # Model 2: SIREN, data only
    siren_data = SirenNetwork(
        hidden_features=CFG['hidden_features'],
        hidden_layers=CFG['hidden_layers'],
        omega_0=CFG['omega_0'],
    )
    hist_siren, _ = train_model("SIREN (data only)", siren_data, dataset, coloc, nfw,
                                 use_physics=False)
    histories.append(hist_siren)
    labels.append("SIREN (data only)")
    trained_models["SIREN (data only)"] = siren_data

    # Model 3: SIREN + Poisson (full PINN)
    siren_pinn = SirenNetwork(
        hidden_features=CFG['hidden_features'],
        hidden_layers=CFG['hidden_layers'],
        omega_0=CFG['omega_0'],
    )
    hist_pinn, _ = train_model("SIREN + Poisson (PINN)", siren_pinn, dataset, coloc, nfw,
                                use_physics=True)
    histories.append(hist_pinn)
    labels.append("SIREN + Poisson (PINN)")
    trained_models["SIREN + Poisson (PINN)"] = siren_pinn

    # Evaluate all three
    print("\n" + "=" * 60)
    print("PHASE 3 — Evaluation")
    print("=" * 60)
    results = {}
    for name, model in trained_models.items():
        print(f"\nEvaluating: {name}")
        metrics = evaluate_model(
            model, dataset, test_coords, test_rho, test_phi, nfw, CFG['device'])
        results[name] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v:,}")

    # Print benchmark table
    print_benchmark_table(results)

    # Save results to JSON
    results_path = os.path.join(CFG['output_dir'], 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate all plots
    print("\n" + "=" * 60)
    print("PHASE 4 — Generating Plots")
    print("=" * 60)
    plot_training_curves(histories, labels, CFG['output_dir'])
    plot_density_comparison(trained_models, dataset, nfw, CFG['output_dir'], CFG['device'])
    plot_radial_profile(trained_models, dataset, nfw, CFG['output_dir'], CFG['device'])
    plot_poisson_residual(siren_pinn, dataset, nfw, CFG['output_dir'], CFG['device'])

    print("\n" + "=" * 60)
    print("STAGE A COMPLETE")
    print(f"All outputs saved to: {CFG['output_dir']}/")
    print("Files:")
    for f in sorted(os.listdir(CFG['output_dir'])):
        fpath = os.path.join(CFG['output_dir'], f)
        if os.path.isfile(fpath):
            print(f"  {f}  ({os.path.getsize(fpath)/1024:.1f} KB)")
    print("=" * 60)


if __name__ == '__main__':
    main()

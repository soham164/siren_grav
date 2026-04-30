"""
Resume Stage A training from saved checkpoints or start fresh if none exist.
This script checks for existing checkpoints and resumes training if found.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_stage_a import *

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in a model directory."""
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint_step_*.pt'))
    if not checkpoints:
        # Check for final checkpoint
        final_ckpt = os.path.join(model_dir, 'checkpoint_final.pt')
        if os.path.exists(final_ckpt):
            return final_ckpt
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoints[-1]


def train_or_resume_model(name, model, dataset, coloc, nfw, use_physics=True):
    """Train a model or resume from checkpoint if it exists."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    out_dir = os.path.join(CFG['output_dir'], name.lower().replace(' ', '_'))
    os.makedirs(out_dir, exist_ok=True)
    
    # Check for existing checkpoint
    latest_ckpt = find_latest_checkpoint(out_dir)
    
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
    
    # Resume from checkpoint if found
    start_step = 0
    if latest_ckpt:
        print(f"\nFound checkpoint: {latest_ckpt}")
        print("Resuming training...")
        start_step = trainer.load_checkpoint(latest_ckpt)
        print(f"Resuming from step {start_step}")
    else:
        print("\nNo checkpoint found. Starting fresh training...")
    
    history = trainer.train(start_step=start_step)
    return history, trainer


def main_resume():
    print("=" * 60)
    print("STAGE A — Resume Training (with checkpoint support)")
    print(f"Device: {CFG['device']}")
    print("=" * 60)

    # Build NFW ground truth
    nfw = NFWProfile(rho_c=CFG['rho_c'], Rs=CFG['Rs'])

    # Check if we need to run gate tests
    print("\nSkipping gate tests (run run_stage_a.py if you need them)")

    # Build dataset
    dataset, coloc, test_coords, test_rho, test_phi = build_dataset(nfw)

    # Train all three models (with resume capability)
    histories = []
    labels = []
    trained_models = {}

    # Model 1: ReLU baseline
    relu_model = build_relu_baseline(CFG['hidden_features'], CFG['hidden_layers'])
    hist_relu, _ = train_or_resume_model("ReLU Baseline", relu_model, dataset, coloc, nfw,
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
    hist_siren, _ = train_or_resume_model("SIREN (data only)", siren_data, dataset, coloc, nfw,
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
    hist_pinn, _ = train_or_resume_model("SIREN + Poisson (PINN)", siren_pinn, dataset, coloc, nfw,
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
    print("=" * 60)


if __name__ == '__main__':
    main_resume()

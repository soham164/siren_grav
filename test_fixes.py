"""
Quick test to verify the fixes are working correctly.
Run this before full training to catch any issues.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.nfw import NFWProfile
from src.siren import SirenNetwork
from src.dataset import NFWDataset
from src.trainer import Trainer
from src.dataset import ColocationSampler

print("=" * 60)
print("Testing Fixes")
print("=" * 60)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

nfw = NFWProfile(rho_c=0.1, Rs=20.0)
dataset = NFWDataset(nfw, n_samples=1000, r_max=100.0, device=device, seed=42)
coloc = ColocationSampler(r_max=1.0, device=device, batch_size=512)

model = SirenNetwork(hidden_features=64, hidden_layers=2, omega_0=30.0)

trainer = Trainer(
    model=model,
    dataset=dataset,
    coloc_sampler=coloc,
    nfw=nfw,
    output_dir='outputs/test_fixes',
    device=device,
    learning_rate=5e-4,
    total_steps=100,
    warmup_steps=50,
    lambda_1=1.0,
    lambda_2_max=0.005,
    batch_size=256,
    coloc_size=256,
    log_every=25,
    save_every=50,
    G=1.0,
)

print("\n" + "=" * 60)
print("Test 1: Check Poisson loss scale")
print("=" * 60)

# Get a batch
coords = coloc.sample(256).to(device)
coords.requires_grad_(True)

# Compute Poisson loss
model.train()
loss_poisson = trainer._poisson_loss(coords)

print(f"Poisson loss value: {loss_poisson.item():.6f}")
print(f"Expected range: 0.01 - 100 (normalized scale)")

if 0.001 < loss_poisson.item() < 1000:
    print("✓ PASS: Poisson loss is in reasonable normalized range")
else:
    print("✗ FAIL: Poisson loss scale is wrong!")
    print("  If it's ~10^13, the normalization fix didn't work")

print("\n" + "=" * 60)
print("Test 2: Check lambda schedule")
print("=" * 60)

lambda_at_0 = trainer._get_lambda_2(0)
lambda_at_warmup = trainer._get_lambda_2(trainer.warmup_steps)
lambda_at_mid = trainer._get_lambda_2((trainer.warmup_steps + trainer.total_steps) // 2)
lambda_at_end = trainer._get_lambda_2(trainer.total_steps)

print(f"Lambda at step 0:       {lambda_at_0:.6f} (should be 0)")
print(f"Lambda at warmup end:   {lambda_at_warmup:.6f} (should be ~0)")
print(f"Lambda at mid-training: {lambda_at_mid:.6f}")
print(f"Lambda at end:          {lambda_at_end:.6f} (should be {trainer.lambda_2_max})")

if lambda_at_0 == 0 and abs(lambda_at_end - trainer.lambda_2_max) < 0.001:
    print("✓ PASS: Lambda schedule is correct")
else:
    print("✗ FAIL: Lambda schedule is wrong!")

print("\n" + "=" * 60)
print("Test 3: Check data loss scale")
print("=" * 60)

# Get a data batch
train_loader, _ = dataset.get_split(val_fraction=0.1)
coords_data, rho_true, phi_true = next(iter(train_loader))
coords_data = coords_data.to(device)
rho_true = rho_true.to(device)
phi_true = phi_true.to(device)

loss_data, _, _ = trainer._data_loss(coords_data, rho_true, phi_true)

print(f"Data loss value:    {loss_data.item():.6f}")
print(f"Poisson loss value: {loss_poisson.item():.6f}")
print(f"Ratio (Poisson/Data): {loss_poisson.item() / loss_data.item():.2f}")

if 0.01 < loss_poisson.item() / loss_data.item() < 100:
    print("✓ PASS: Losses are on similar scales (within 2 orders of magnitude)")
else:
    print("✗ FAIL: Losses are on very different scales!")
    print("  This will cause training instability")

print("\n" + "=" * 60)
print("Test 4: Quick training test (100 steps)")
print("=" * 60)

try:
    history = trainer.train(start_step=0)
    
    final_data_loss = history['loss_data'][-1]
    final_poisson_loss = history['loss_poisson'][-1]
    
    print(f"\nFinal data loss:    {final_data_loss:.6f}")
    print(f"Final Poisson loss: {final_poisson_loss:.6f}")
    
    if final_data_loss < 10 and final_poisson_loss < 1000:
        print("✓ PASS: Training completed without explosion")
    else:
        print("✗ FAIL: Losses exploded during training")
        
except Exception as e:
    print(f"✗ FAIL: Training crashed with error: {e}")

print("\n" + "=" * 60)
print("All tests complete!")
print("=" * 60)
print("\nIf all tests passed, you're ready to run the full training.")
print("Run: python experiments/run_stage_a.py")

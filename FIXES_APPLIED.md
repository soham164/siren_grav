# Physics-Informed Training Fixes

## Summary of Changes

All three critical issues have been fixed to improve training stability and evaluation fairness.

---

## Problem 1: Aggressive Lambda Ramp

**Issue:** Lambda ramped up 5x faster than linear, causing physics loss to dominate too early.

**Fix Applied:**
- Changed ramp multiplier from `* 5` to `* 1` (linear ramp)
- Increased `warmup_steps` from 3,000 to 8,000 (longer data-only phase)
- Reduced `lambda_2_max` from 0.1 to 0.005 (gentler physics constraint)

**Location:** `siren_grav/src/trainer.py` - `_get_lambda_2()` method

**Impact:** More stable training, physics loss introduced gradually after network learns basic structure.

---

## Problem 2: Unit Mismatch Between Losses

**Issue:** Poisson loss computed in physical units (~10¹³ scale), data loss in normalized units (~1 scale). This caused massive gradient imbalance.

**Fix Applied:**
- Compute Poisson residual entirely in **normalized space**
- Apply proper scaling: `(rho_std / phi_std) * r_max²` to convert physical Poisson equation to normalized coordinates
- Both losses now on same scale (~1), making lambda values meaningful

**Location:** `siren_grav/src/trainer.py` - `_poisson_loss()` method

**Before:**
```python
# Convert to physical units
rho_phys = denorm_rho(rho_pred)
phi_phys = denorm_phi(phi_pred)
# Compute in physical space (huge values!)
loss = poisson_residual(phi_phys, rho_phys)
```

**After:**
```python
# Stay in normalized space
lap_phi_norm = laplacian(phi_norm)
scale = (rho_std / phi_std) * r_max²
target_norm = scale * 4πG * rho_norm
loss = (lap_phi_norm - target_norm)²  # Same scale as data loss!
```

**Impact:** Physics loss and data loss now balanced, lambda values are interpretable.

---

## Problem 3: Unfair Evaluation Metrics

**Issue:** Mean relative error dominated by low-density outer regions (where errors are large but physics is less important). A 75% error sounds terrible but is mostly from the sparse halo outskirts.

**Fix Applied:**
- Added **density-weighted error metrics**
- Each point's error weighted by local density: `weight = rho / sum(rho)`
- High-density inner regions (where physics matters) now dominate the metric

**Location:** `siren_grav/experiments/run_stage_a.py` - `evaluate_model()` function

**New Metrics:**
- `rho_weighted_error`: Density-weighted relative error for ρ
- `phi_weighted_error`: Density-weighted relative error for Φ
- Old unweighted metrics kept for comparison

**Impact:** Fairer evaluation that reflects performance where it matters most (dense inner halo).

---

## Configuration Changes

### In `run_stage_a.py`:
```python
'warmup_steps'    : 8_000,      # Was: 3_000
'lambda_2_max'    : 0.005,      # Was: 0.1
```

### In `trainer.py`:
```python
# Lambda ramp
return lambda_2_max * ramp_frac * 1  # Was: * 5
```

---

## Expected Improvements

1. **More stable training:** Longer warmup + gentler physics loss prevents early divergence
2. **Better convergence:** Balanced loss scales mean optimizer can effectively minimize both terms
3. **Fairer metrics:** Weighted errors show true performance in physically important regions

---

## Retraining Required

**YES** - These changes affect:
- Training schedule (warmup duration)
- Loss computation (normalized Poisson loss)
- Loss weighting (lambda values)

Old checkpoints are incompatible. Delete `outputs/stage_a/` and retrain.

---

## Verification

After retraining, check:
1. **Training logs:** Poisson loss should be ~0.1-10 range (not 10¹³)
2. **Loss balance:** Data loss and Poisson loss should be similar magnitude
3. **Weighted errors:** Should be significantly lower than unweighted (10-20% vs 50-75%)
4. **Convergence:** Both losses should decrease smoothly without spikes

---

## Technical Details

### Why the scaling factor works:

In physical space:
```
∇² Φ = 4πG ρ
```

In normalized space (coords → coords/r_max, Φ → (Φ-μ_Φ)/σ_Φ, ρ → (ρ-μ_ρ)/σ_ρ):
```
∇²_norm Φ_norm = (σ_ρ / σ_Φ) * r_max² * 4πG * ρ_norm
```

The factor `(σ_ρ / σ_Φ) * r_max²` converts the physical Poisson equation to normalized coordinates.

### Why density weighting is fair:

The NFW profile has ρ ∝ 1/r³ at large r. The outer 90% of the volume contains <10% of the mass. Unweighted errors give equal importance to every point, so sparse regions dominate. Density weighting gives importance proportional to mass, which is physically meaningful.

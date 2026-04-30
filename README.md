# SIREN Galactic Potential — Stage A (Proof of Concept)

## What This Is
A Physics-Informed SIREN neural network that learns the gravitational potential
and density of a dark matter halo from coordinates alone, while satisfying
Poisson's equation as a hard physics constraint during training.

## Project Structure
```
siren_grav/
├── src/
│   ├── nfw.py          # Step 3 — Analytical NFW ground truth (density, potential, laplacian)
│   ├── siren.py        # Step 2 — SIREN network with correct initialization
│   ├── physics.py      # Step 4 — Laplacian via autograd (two methods, cross-verified)
│   ├── dataset.py      # Step 5 — Dataset sampler for NFW coordinates
│   └── trainer.py      # Step 6 — Full training loop with physics loss
├── tests/
│   ├── test_nfw.py         # Verify NFW functions against known values
│   ├── test_siren_init.py  # Verify SIREN initialization distribution
│   └── test_laplacian.py   # Cross-verify autograd vs finite differences
├── experiments/
│   └── run_stage_a.py  # Step 7 — Master experiment script (runs everything)
└── outputs/            # Plots, checkpoints, benchmark table saved here
```

## How to Run (Google Colab)
1. Upload this entire folder to Colab
2. Run: `!pip install torch numpy scipy matplotlib`
3. Run cells in order:
   - `!python tests/test_nfw.py`
   - `!python tests/test_siren_init.py`
   - `!python tests/test_laplacian.py`
   - `!python experiments/run_stage_a.py`
4. All plots and results saved to `outputs/`

## What Passes = Stage A Complete
- NFW analytical functions: error < 1e-6 vs known values
- SIREN initialization: activations uniform in [-1, 1] at all layers
- Laplacian: autograd vs finite diff < 1e-4 relative error
- SIREN (data only): density reconstruction error < 1%
- SIREN + Poisson: Poisson residual < 5% everywhere
- Compression ratio: weight file < 0.1 MB vs ~50 MB raw data

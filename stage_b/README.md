# SIREN Galactic Potential — Stage B (Real IllustrisTNG Data)

## What This Is
Stage B extends the Stage A proof-of-concept to real dark matter halos
from the IllustrisTNG cosmological simulation. The SIREN now learns
directly from actual N-body particle data, with a two-phase adaptive
PINN training that fixes the instability from Stage A.

## Project Structure
```
siren_grav_b/
├── src/
│   ├── units.py           — IllustrisTNG unit conversions (ckpc/h → kpc etc.)
│   ├── tng_loader.py      — TNG API downloader + HDF5 reader + halo processor
│   ├── field_estimator.py — KDE density + tree potential from particles
│   ├── models.py          — SIREN network (same architecture as Stage A)
│   └── trainer_b.py       — Two-phase PINN with NTK adaptive loss weighting
├── experiments/
│   └── run_stage_b.py     — Master script (runs everything end to end)
├── notebooks/
│   └── Stage_B_Colab.ipynb — Colab notebook (run this)
├── tests/
│   └── test_stage_b.py    — Gate tests before training
├── data/                  — Downloaded HDF5 halo files saved here
└── outputs/               — Trained models, plots, benchmark table
```

## How to Run

### Step 1 — Get a TNG API Key (free)
1. Register at https://www.tng-project.org/users/register/
2. Go to your profile page → copy your API key

### Step 2 — Open Colab
1. Upload `siren_grav_b/` folder to Google Colab
2. Open `notebooks/Stage_B_Colab.ipynb`
3. Enable GPU: Runtime → Change runtime type → T4
4. Paste your API key in Cell 2
5. Run all cells top to bottom

### Step 3 — Without API Key (mock data)
Leave the API key blank. The pipeline uses NFW-like mock halos
with realistic noise and ellipticity for pipeline testing.

## Key Differences From Stage A

| | Stage A | Stage B |
|---|---|---|
| Data source | Analytical NFW formula | Real IllustrisTNG particles |
| Density | Exact formula | KDE from particles |
| Potential | Exact formula | Tree-based summation |
| Halos | 1 synthetic | 5 real halos |
| PINN training | Manual lambda | NTK adaptive weighting |
| Error metric | Simple relative error | Density-weighted error |

## What Stage B Proves
- SIREN compresses real halo data (not just analytical formulas)
- Compression ratio holds across halos of different masses
- Two-phase training avoids the trivial solution trap
- Density-weighted error metric gives fair comparison

## Stage B Complete When
- Mean density error < 10% (density-weighted, across 5 halos)
- Mean compression ratio > 50x
- Poisson residual decreasing in Phase 2
- Results reproducible across all 5 halos

## Requirements
- Python 3.10+
- PyTorch 2.x
- numpy, scipy, matplotlib, h5py, requests, scikit-learn

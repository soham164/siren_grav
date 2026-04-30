# How to Resume Training After a Crash

## What I Fixed

1. **Fixed the gradient computation error** in `run_stage_a.py` that was causing the crash during evaluation
2. **Created a resume-capable training script** at `experiments/resume_stage_a.py`
3. **Modified the Trainer class** to support resuming from checkpoints

## The Problem

The original script crashed during evaluation because it tried to compute gradients (for the Laplacian) while having `@torch.no_grad()` decorator active. This is now fixed.

## Your Options

### Option 1: Start Fresh (Recommended - since no checkpoints exist yet)

Run the fixed original script:
```bash
python experiments/run_stage_a.py
```

This will:
- Run all gate tests
- Train all 3 models from scratch
- Save checkpoints every 10,000 steps
- Evaluate and generate plots

### Option 2: Use the Resume Script (For future crashes)

If training crashes in the future and you have checkpoints saved:
```bash
python experiments/resume_stage_a.py
```

This will:
- Skip gate tests
- Check for existing checkpoints in `outputs/stage_a/*/`
- Resume training from the latest checkpoint if found
- Otherwise start fresh training

## How Checkpoints Work

- Checkpoints are saved every **10,000 steps** (configurable via `save_every` in CFG)
- Location: `outputs/stage_a/<model_name>/checkpoint_step_XXXXXX.pt`
- Each checkpoint contains:
  - Model weights
  - Optimizer state
  - Training history
  - Current step number

## What Changed in the Code

1. **`experiments/run_stage_a.py`**: Fixed the `evaluate_model()` function to properly enable gradients during Laplacian computation
2. **`src/trainer.py`**: Added `start_step` parameter to `train()` method
3. **`experiments/resume_stage_a.py`**: New script that automatically finds and resumes from checkpoints

## Next Steps

Since you don't have any checkpoints yet (training crashed before the first checkpoint at step 10,000), you'll need to start fresh. But now:
- The evaluation bug is fixed
- Future crashes can be resumed from checkpoints
- You won't lose hours of training if it crashes again

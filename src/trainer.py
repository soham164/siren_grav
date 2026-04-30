"""
src/trainer.py  —  Step 6: Training Loop
=========================================
The Trainer class handles the complete training procedure for the
Physics-Informed SIREN. It manages:

    1. The warm-up schedule (data loss only first, then physics loss added)
    2. Separate logging of every loss component
    3. Checkpoint saving every N steps
    4. Validation after every epoch
    5. A clean summary at the end

Loss schedule:
    Steps 0     → warmup_steps :  L = L_data  only
    Steps warmup → total_steps :  L = lambda_1*L_data + lambda_2*L_Poisson

    This warm-up is critical. If we add the Poisson loss from step 0,
    it overwhelms the data loss before the network has learned anything,
    and training diverges. We let the network first learn a rough approximation
    of the potential from data, THEN enforce the physics constraint.

Loss definitions:
    L_data    = MSE(phi_pred, phi_true) + MSE(rho_pred, rho_true)
    L_Poisson = MSE(nabla^2 phi_pred,  4*pi*G * rho_pred)   [at colocation points]
    L_total   = lambda_1 * L_data + lambda_2 * L_Poisson
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.physics import laplacian_autograd, poisson_residual


class Trainer:
    """
    Full training loop for the Physics-Informed SIREN.

    Parameters
    ----------
    model          : SirenNetwork
    dataset        : NFWDataset — provides train/val loaders and norm stats
    coloc_sampler  : ColocationSampler — provides colocation batches
    nfw            : NFWProfile — for computing Poisson ground truth at coloc points
    output_dir     : str — where to save checkpoints and logs
    device         : str — 'cpu' or 'cuda'

    Training hyperparameters:
    learning_rate  : float — initial learning rate (default: 5e-4)
    total_steps    : int   — total gradient steps (default: 50_000)
    warmup_steps   : int   — steps before adding physics loss (default: 5_000)
    lambda_1       : float — weight for data loss (default: 1.0)
    lambda_2_max   : float — max weight for Poisson loss (default: 0.1)
    batch_size     : int   — training batch size (default: 4096)
    coloc_size     : int   — colocation batch size (default: 4096)
    log_every      : int   — log to console every N steps (default: 500)
    save_every     : int   — save checkpoint every N steps (default: 5000)
    G              : float — gravitational constant in simulation units (default: 1.0)
    """

    def __init__(self,
                 model,
                 dataset,
                 coloc_sampler,
                 nfw,
                 output_dir='outputs',
                 device='cpu',
                 learning_rate=5e-4,
                 total_steps=50_000,
                 warmup_steps=5_000,
                 lambda_1=1.0,
                 lambda_2_max=0.1,
                 batch_size=4096,
                 coloc_size=4096,
                 log_every=500,
                 save_every=5_000,
                 G=1.0):

        self.model         = model.to(device)
        self.dataset       = dataset
        self.coloc_sampler = coloc_sampler
        self.nfw           = nfw
        self.output_dir    = output_dir
        self.device        = device
        self.G             = G

        # Training hyperparameters
        self.lr            = learning_rate
        self.total_steps   = total_steps
        self.warmup_steps  = warmup_steps
        self.lambda_1      = lambda_1
        self.lambda_2_max  = lambda_2_max
        self.batch_size    = batch_size
        self.coloc_size    = coloc_size
        self.log_every     = log_every
        self.save_every    = save_every

        # Optimiser: Adam with weight decay for regularisation
        self.optimiser = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,   # small L2 regularisation — prevents overfitting
        )

        # LR scheduler: Cosine annealing
        # Smoothly reduces LR from lr to lr/100 over training
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=total_steps,
            eta_min=learning_rate / 100,
        )

        # Data loaders
        self.train_loader, self.val_loader = dataset.get_split(val_fraction=0.1)

        # History log — every loss component tracked separately
        self.history = {
            'step'         : [],
            'loss_data'    : [],
            'loss_poisson' : [],
            'loss_total'   : [],
            'val_loss_data': [],
            'lambda_2'     : [],
            'lr'           : [],
        }

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nTrainer ready:")
        print(f"  Model params : {model.count_parameters():,}")
        print(f"  Total steps  : {total_steps:,}")
        print(f"  Warmup steps : {warmup_steps:,}  (data only)")
        print(f"  Device       : {device}")
        print(f"  Output dir   : {output_dir}")

    # ── Lambda schedule ───────────────────────────────────────────────────────
    def _get_lambda_2(self, step):
        """
        Linearly ramp lambda_2 from 0 to lambda_2_max during the
        warmup-to-full-training transition.

        Steps 0 → warmup_steps:           lambda_2 = 0
        Steps warmup_steps → total_steps:  lambda_2 linearly → lambda_2_max
        """
        if step < self.warmup_steps:
            return 0.0
        ramp_frac = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps)
        return min(self.lambda_2_max, self.lambda_2_max * ramp_frac * 1)
        # Linear ramp to max

    # ── Data loss ─────────────────────────────────────────────────────────────
    def _data_loss(self, coords, rho_true, phi_true):
        """
        MSE between network predictions and ground truth.
        Both Phi and Rho contribute to the data loss.
        """
        phi_pred, rho_pred = self.model(coords)
        loss_phi = ((phi_pred - phi_true) ** 2).mean()
        loss_rho = ((rho_pred - rho_true) ** 2).mean()
        return loss_phi + loss_rho, phi_pred, rho_pred

    # ── Poisson loss ──────────────────────────────────────────────────────────
    def _poisson_loss(self, coloc_coords):
        """
        Poisson residual loss at colocation points.

        The ground truth for the Laplacian at colocation points is
        4*pi*G*rho_pred (from the network's own rho output).

        This is "self-consistent": we are asking the network to make its
        predicted Phi consistent with its predicted rho via Poisson's equation.
        No external data needed at these points.
        
        IMPORTANT: We compute the Poisson residual in NORMALIZED space to match
        the scale of the data loss. This prevents the physics loss from dominating
        due to the large physical unit values.
        """
        # Network predictions at colocation points (already normalized)
        phi_pred, rho_pred = self.model(coloc_coords)

        # Compute Poisson residual in NORMALIZED space
        # We need the Laplacian of normalized phi w.r.t. normalized coords
        def phi_norm_fn(c_norm):
            p, _ = self.model(c_norm)
            return p  # Already normalized output
        
        # Laplacian in normalized space
        lap_phi_norm = laplacian_autograd(phi_norm_fn, coloc_coords)
        
        # Target: 4*pi*G*rho in normalized space
        # We need to scale the physical Poisson equation to normalized units
        # nabla^2 phi_norm = (rho_std / phi_std) * r_max^2 * 4*pi*G * rho_norm
        scale_factor = (self.dataset.rho_std / self.dataset.phi_std) * (self.dataset.coord_max ** 2)
        target_norm = scale_factor * 4.0 * np.pi * self.G * rho_pred
        
        # Residual in normalized space (same scale as data loss)
        residual = lap_phi_norm - target_norm
        loss = (residual ** 2).mean()
        
        return loss

    # ── Validation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self):
        """
        Run validation on the held-out val set.
        Returns: mean data loss on validation set.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for coords, rho_true, phi_true in self.val_loader:
            coords   = coords.to(self.device)
            rho_true = rho_true.to(self.device)
            phi_true = phi_true.to(self.device)

            phi_pred, rho_pred = self.model(coords)
            loss = ((phi_pred - phi_true) ** 2).mean() + \
                   ((rho_pred - rho_true) ** 2).mean()
            total_loss += loss.item()
            n_batches  += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    # ── Main training loop ────────────────────────────────────────────────────
    def train(self, start_step=0):
        """
        Run the full training loop.

        Parameters
        ----------
        start_step : int — step to start/resume from (default: 0 for fresh training)

        Returns
        -------
        history : dict — training history with all loss components
        """
        print("\n" + "=" * 60)
        if start_step > 0:
            print(f"Resuming training from step {start_step}")
        else:
            print("Training started")
        print("=" * 60)

        self.model.train()
        step       = start_step
        start_time = time.time()

        # Infinite data iterator (we sample more data than one epoch)
        def data_iter():
            while True:
                for batch in self.train_loader:
                    yield batch

        data_generator = data_iter()

        while step < self.total_steps:
            # ── Get data batch ────────────────────────────────────────────────
            coords, rho_true, phi_true = next(data_generator)
            coords   = coords.to(self.device)
            rho_true = rho_true.to(self.device)
            phi_true = phi_true.to(self.device)

            # ── Get current lambda schedule ───────────────────────────────────
            lambda_2 = self._get_lambda_2(step)

            # ── Zero gradients ────────────────────────────────────────────────
            self.optimiser.zero_grad()

            # ── Compute data loss ─────────────────────────────────────────────
            loss_data, _, _ = self._data_loss(coords, rho_true, phi_true)

            # ── Compute Poisson loss (only after warmup) ──────────────────────
            loss_poisson = torch.tensor(0.0, device=self.device)
            if lambda_2 > 0:
                coloc_coords = self.coloc_sampler.sample(self.coloc_size)
                coloc_coords = coloc_coords.to(self.device)
                loss_poisson = self._poisson_loss(coloc_coords)

            # ── Total loss ────────────────────────────────────────────────────
            loss_total = self.lambda_1 * loss_data + lambda_2 * loss_poisson

            # ── Backward pass and optimiser step ─────────────────────────────
            loss_total.backward()

            # Gradient clipping: prevents explosions during physics loss phase
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimiser.step()
            self.scheduler.step()

            step += 1

            # ── Logging ───────────────────────────────────────────────────────
            if step % self.log_every == 0 or step == 1:
                val_loss = self._validate()
                elapsed  = time.time() - start_time
                current_lr = self.scheduler.get_last_lr()[0]

                self.history['step'].append(step)
                self.history['loss_data'].append(loss_data.item())
                self.history['loss_poisson'].append(loss_poisson.item())
                self.history['loss_total'].append(loss_total.item())
                self.history['val_loss_data'].append(val_loss)
                self.history['lambda_2'].append(lambda_2)
                self.history['lr'].append(current_lr)

                phase = "warmup" if step <= self.warmup_steps else "physics"
                print(f"  [{phase}] step={step:6d} | "
                      f"data={loss_data.item():.4f} | "
                      f"poisson={loss_poisson.item():.4f} | "
                      f"total={loss_total.item():.4f} | "
                      f"val={val_loss:.4f} | "
                      f"λ2={lambda_2:.4f} | "
                      f"t={elapsed:.0f}s")

            # ── Checkpoint saving ─────────────────────────────────────────────
            if step % self.save_every == 0:
                self._save_checkpoint(step)

        # ── Final checkpoint ──────────────────────────────────────────────────
        self._save_checkpoint(step, name='final')

        # ── Save history ──────────────────────────────────────────────────────
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 60)

        return self.history

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def _save_checkpoint(self, step, name=None):
        """Save model weights and optimiser state."""
        name = name or f'step_{step:06d}'
        path = os.path.join(self.output_dir, f'checkpoint_{name}.pt')
        torch.save({
            'step'            : step,
            'model_state'     : self.model.state_dict(),
            'optimiser_state' : self.optimiser.state_dict(),
            'history'         : self.history,
        }, path)
        # Print file size
        size_kb = os.path.getsize(path) / 1024
        print(f"  Checkpoint saved: {path}  ({size_kb:.1f} KB)")

    def load_checkpoint(self, path):
        """Load a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimiser.load_state_dict(ckpt['optimiser_state'])
        self.history = ckpt.get('history', self.history)
        print(f"Checkpoint loaded from: {path} (step {ckpt['step']})")
        return ckpt['step']


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.nfw     import NFWProfile
    from src.siren   import SirenNetwork
    from src.dataset import NFWDataset, ColocationSampler

    print("=" * 60)
    print("Trainer Self-Test (100 steps, small model)")
    print("=" * 60)

    nfw   = NFWProfile(rho_c=0.1, Rs=20.0)
    model = SirenNetwork(hidden_features=64, hidden_layers=2)
    ds    = NFWDataset(nfw, n_samples=10_000, r_max=100.0)
    coloc = ColocationSampler(batch_size=512)

    trainer = Trainer(
        model=model,
        dataset=ds,
        coloc_sampler=coloc,
        nfw=nfw,
        output_dir='outputs/test_run',
        total_steps=200,
        warmup_steps=100,
        lambda_2_max=0.01,
        log_every=50,
        save_every=100,
    )

    history = trainer.train()

    print(f"\nFinal data loss     : {history['loss_data'][-1]:.6f}")
    print(f"Final Poisson loss  : {history['loss_poisson'][-1]:.6f}")
    print(f"Final total loss    : {history['loss_total'][-1]:.6f}")
    print(f"Loss decreased      : "
          f"{'YES' if history['loss_total'][-1] < history['loss_total'][0] else 'CHECK'}")

    print("\n" + "=" * 60)
    print("Trainer self-test complete.")
    print("=" * 60)

"""
src/trainer_b.py  —  Stage B PINN Trainer with Adaptive Loss Weighting
=======================================================================
This fixes the core PINN instability from Stage A using the approach
from Wang et al. 2021 "Understanding and Mitigating Gradient Pathologies
in Physics-Informed Neural Networks."

The fix: NTK (Neural Tangent Kernel) adaptive weighting.

The problem in Stage A:
    The gradient of L_Poisson w.r.t. weights is much larger than
    the gradient of L_data. So the optimiser follows the Poisson
    gradient and ignores the data loss — causing the network to
    find a trivial Poisson solution (constant Phi, constant rho)
    rather than the physically correct one.

The fix — adaptive lambda:
    At every K steps, measure:
        mean_data    = mean(|grad L_data|)     over all weights
        mean_physics = mean(|grad L_Poisson|)  over all weights
    Set:
        lambda_2 = mean_data / mean_physics

    This automatically balances the two loss terms so they contribute
    equally to each weight update. No manual tuning needed.

Two-phase training:
    Phase 1 (pre-train): Data loss only, 10k steps.
                         Network learns the shape of the density field.
    Phase 2 (fine-tune): Add Poisson loss with adaptive weighting.
                         Network adjusts Phi to be physically consistent.
    
    Starting Phase 2 from a pre-trained checkpoint avoids the trivial
    solution trap — the network already has a good data-fitting solution,
    and the physics loss only needs to make small adjustments.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from torch.optim.lr_scheduler import CosineAnnealingLR


# ── Laplacian (same as Stage A physics.py) ────────────────────────────────────
def laplacian_autograd(phi_fn, coords):
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)
    phi      = phi_fn(coords)
    grad_phi = torch.autograd.grad(
        phi, coords,
        grad_outputs=torch.ones_like(phi),
        create_graph=True, retain_graph=True,
    )[0]
    lap = torch.zeros(coords.shape[0], 1, device=coords.device)
    for d in range(3):
        lap[:, 0] += torch.autograd.grad(
            grad_phi[:, d].sum(), coords,
            create_graph=True, retain_graph=True,
        )[0][:, d]
    return lap


class AdaptivePINNTrainer:
    """
    Two-phase PINN trainer with NTK adaptive loss weighting.

    Parameters
    ----------
    model          : SirenNetwork
    dataset        : RealHaloDataset (or NFWDataset for testing)
    coloc_sampler  : ColocationSampler
    output_dir     : str
    device         : str

    Phase 1 hyperparams:
    pretrain_steps : int — steps with data loss only (default 10_000)
    pretrain_lr    : float — learning rate for pre-training (default 5e-4)

    Phase 2 hyperparams:
    finetune_steps : int — steps with physics loss (default 40_000)
    finetune_lr    : float — learning rate for fine-tuning (default 1e-4)
    adapt_every    : int — recompute adaptive lambda every N steps (default 100)
    lambda_1       : float — fixed weight for data loss (default 1.0)
    lambda_2_init  : float — initial physics weight (default 0.0, set adaptively)
    coloc_size     : int — colocation batch size (default 1024)

    Physics params:
    G              : float — gravitational constant (sim units, default 1.0)
    """

    def __init__(self,
                 model,
                 dataset,
                 coloc_sampler,
                 output_dir='outputs/stage_b',
                 device='cpu',
                 pretrain_steps=10_000,
                 pretrain_lr=5e-4,
                 finetune_steps=40_000,
                 finetune_lr=1e-4,
                 adapt_every=200,
                 lambda_1=1.0,
                 coloc_size=1024,
                 G=1.0,
                 log_every=500,
                 save_every=10_000):

        self.model         = model.to(device)
        self.dataset       = dataset
        self.coloc_sampler = coloc_sampler
        self.device        = device
        self.output_dir    = output_dir
        self.G             = G

        self.pretrain_steps  = pretrain_steps
        self.pretrain_lr     = pretrain_lr
        self.finetune_steps  = finetune_steps
        self.finetune_lr     = finetune_lr
        self.adapt_every     = adapt_every
        self.lambda_1        = lambda_1
        self.lambda_2        = 0.0          # set adaptively
        self.coloc_size      = coloc_size
        self.log_every       = log_every
        self.save_every      = save_every

        # Pre-compute the Poisson scale factor (normalised space)
        # nabla^2_norm phi_norm = scale * rho_phys
        self.poisson_scale = (
            (dataset.coord_max ** 2) / dataset.phi_std.item()
        ) * 4.0 * np.pi * G
        print(f"\nPoisson scale factor: {self.poisson_scale:.4f}")

        os.makedirs(output_dir, exist_ok=True)

        self.history = {
            'phase'          : [],
            'step'           : [],
            'loss_data'      : [],
            'loss_poisson'   : [],
            'loss_total'     : [],
            'val_loss'       : [],
            'lambda_2'       : [],
            'lr'             : [],
        }

        print(f"\nAdaptivePINNTrainer:")
        print(f"  Phase 1 (pre-train): {pretrain_steps:,} steps @ lr={pretrain_lr}")
        print(f"  Phase 2 (fine-tune): {finetune_steps:,} steps @ lr={finetune_lr}")
        print(f"  Adaptive lambda every {adapt_every} steps")
        print(f"  Device: {device}")

    # ── Data helpers ─────────────────────────────────────────────────────────
    def _data_loss(self, coords, rho_true, phi_true):
        phi_pred, rho_pred = self.model(coords)
        return (((phi_pred - phi_true)**2).mean() +
                ((rho_pred - rho_true)**2).mean()), phi_pred, rho_pred

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        total, n = 0.0, 0
        for coords, rho_t, phi_t in val_loader:
            coords = coords.to(self.device)
            rho_t  = rho_t.to(self.device)
            phi_t  = phi_t.to(self.device)
            p, r   = self.model(coords)
            total += (((p - phi_t)**2).mean() + ((r - rho_t)**2).mean()).item()
            n     += 1
        self.model.train()
        return total / max(n, 1)

    # ── Normalised Poisson loss ───────────────────────────────────────────────
    def _poisson_loss(self, coloc_coords):
        """
        Poisson residual in normalised space.
        Both LHS and RHS are dimensionless and O(1).
        """
        phi_norm, rho_norm = self.model(coloc_coords)

        # Convert rho back to physical (M_sun/kpc^3)
        rho_phys = (rho_norm * self.dataset.rho_std +
                    self.dataset.rho_mean)

        # Laplacian of normalised phi w.r.t. normalised coords
        lap = laplacian_autograd(
            lambda c: self.model.forward_phi_only(c),
            coloc_coords,
        )

        # Normalised Poisson RHS
        rhs      = self.poisson_scale * rho_phys
        residual = lap - rhs
        return (residual ** 2).mean(), residual

    # ── NTK adaptive lambda ───────────────────────────────────────────────────
    def _compute_adaptive_lambda(self, coords, rho_true, phi_true, coloc_coords):
        """
        Compute adaptive lambda_2 using gradient magnitude matching.

        lambda_2 = mean|grad_w L_data| / mean|grad_w L_poisson|

        This ensures both losses contribute equally to weight updates.
        Called every adapt_every steps during Phase 2.
        """
        self.model.zero_grad()

        # Gradient of data loss
        loss_d, _, _ = self._data_loss(coords, rho_true, phi_true)
        loss_d.backward(retain_graph=False)
        grad_data = torch.cat([
            p.grad.detach().abs().flatten()
            for p in self.model.parameters()
            if p.grad is not None
        ]).mean().item()

        self.model.zero_grad()

        # Gradient of Poisson loss
        loss_p, _ = self._poisson_loss(coloc_coords)
        loss_p.backward(retain_graph=False)
        grad_phys = torch.cat([
            p.grad.detach().abs().flatten()
            for p in self.model.parameters()
            if p.grad is not None
        ]).mean().item()

        self.model.zero_grad()

        if grad_phys < 1e-12:
            return self.lambda_2   # avoid division by zero early in training

        new_lambda = grad_data / (grad_phys + 1e-12)

        # Clip to reasonable range — prevent lambda from going to zero or infinity
        new_lambda = float(np.clip(new_lambda, 1e-4, 10.0))
        return new_lambda

    # ── Infinite data iterator ────────────────────────────────────────────────
    @staticmethod
    def _inf_iter(loader):
        while True:
            for batch in loader:
                yield batch

    # ── Phase 1: Pre-training (data loss only) ────────────────────────────────
    def pretrain(self):
        print("\n" + "=" * 60)
        print("PHASE 1 — Pre-training (data loss only)")
        print("=" * 60)

        train_loader, val_loader = self.dataset.get_split(
            val_fraction=0.1, batch_size=4096)

        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.pretrain_lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(
            optimiser, T_max=self.pretrain_steps, eta_min=self.pretrain_lr/100)

        data_gen   = self._inf_iter(train_loader)
        start_time = time.time()
        self.model.train()

        for step in range(1, self.pretrain_steps + 1):
            coords, rho_t, phi_t = next(data_gen)
            coords = coords.to(self.device)
            rho_t  = rho_t.to(self.device)
            phi_t  = phi_t.to(self.device)

            optimiser.zero_grad()
            loss, _, _ = self._data_loss(coords, rho_t, phi_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            if step % self.log_every == 0:
                val_loss = self._validate(val_loader)
                lr_now   = scheduler.get_last_lr()[0]
                elapsed  = time.time() - start_time
                self.history['phase'].append('pretrain')
                self.history['step'].append(step)
                self.history['loss_data'].append(loss.item())
                self.history['loss_poisson'].append(0.0)
                self.history['loss_total'].append(loss.item())
                self.history['val_loss'].append(val_loss)
                self.history['lambda_2'].append(0.0)
                self.history['lr'].append(lr_now)
                print(f"  [pretrain] step={step:6d} | "
                      f"data={loss.item():.5f} | val={val_loss:.5f} | "
                      f"t={elapsed:.0f}s")

            if step % self.save_every == 0:
                self._save(f"pretrain_step_{step:06d}")

        self._save("pretrain_final")
        print(f"\nPhase 1 complete. Final data loss: {loss.item():.5f}")
        return self

    # ── Phase 2: Fine-tuning (data + physics loss) ────────────────────────────
    def finetune(self):
        print("\n" + "=" * 60)
        print("PHASE 2 — Fine-tuning (data + adaptive Poisson loss)")
        print("=" * 60)

        train_loader, val_loader = self.dataset.get_split(
            val_fraction=0.1, batch_size=2048)

        # Lower learning rate for fine-tuning — preserves pre-trained solution
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.finetune_lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(
            optimiser, T_max=self.finetune_steps, eta_min=self.finetune_lr/100)

        data_gen   = self._inf_iter(train_loader)
        start_time = time.time()
        self.model.train()

        for step in range(1, self.finetune_steps + 1):
            coords, rho_t, phi_t = next(data_gen)
            coords = coords.to(self.device)
            rho_t  = rho_t.to(self.device)
            phi_t  = phi_t.to(self.device)

            coloc = self.coloc_sampler.sample(self.coloc_size).to(self.device)

            # ── Adaptive lambda update ────────────────────────────────────────
            if step % self.adapt_every == 1:
                self.lambda_2 = self._compute_adaptive_lambda(
                    coords, rho_t, phi_t, coloc)

            # ── Forward + loss ────────────────────────────────────────────────
            optimiser.zero_grad()
            loss_d, _, _  = self._data_loss(coords, rho_t, phi_t)

            coloc_fresh   = self.coloc_sampler.sample(self.coloc_size).to(self.device)
            loss_p, _     = self._poisson_loss(coloc_fresh)

            loss_total    = self.lambda_1 * loss_d + self.lambda_2 * loss_p

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimiser.step()
            scheduler.step()

            if step % self.log_every == 0:
                val_loss = self._validate(val_loader)
                lr_now   = scheduler.get_last_lr()[0]
                elapsed  = time.time() - start_time
                self.history['phase'].append('finetune')
                self.history['step'].append(self.pretrain_steps + step)
                self.history['loss_data'].append(loss_d.item())
                self.history['loss_poisson'].append(loss_p.item())
                self.history['loss_total'].append(loss_total.item())
                self.history['val_loss'].append(val_loss)
                self.history['lambda_2'].append(self.lambda_2)
                self.history['lr'].append(lr_now)
                print(f"  [finetune] step={step:6d} | "
                      f"data={loss_d.item():.5f} | "
                      f"poisson={loss_p.item():.5f} | "
                      f"λ2={self.lambda_2:.4f} | "
                      f"val={val_loss:.5f} | t={elapsed:.0f}s")

            if step % self.save_every == 0:
                self._save(f"finetune_step_{step:06d}")

        self._save("finetune_final")
        self._save_history()
        print(f"\nPhase 2 complete.")
        return self

    def train(self):
        """Run both phases in sequence."""
        self.pretrain()
        self.finetune()
        return self.history

    def _save(self, name):
        path = os.path.join(self.output_dir, f"ckpt_{name}.pt")
        torch.save({
            'model_state': self.model.state_dict(),
            'history'    : self.history,
            'lambda_2'   : self.lambda_2,
        }, path)
        kb = os.path.getsize(path) / 1024
        print(f"  Saved: {path}  ({kb:.0f} KB)")

    def _save_history(self):
        path = os.path.join(self.output_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ── Colocation sampler (same as Stage A) ─────────────────────────────────────
class ColocationSampler:
    def __init__(self, r_max=1.0, device='cpu', batch_size=1024):
        self.r_max      = r_max
        self.device     = device
        self.batch_size = batch_size

    def sample(self, n=None):
        n = n or self.batch_size
        pts, collected = [], 0
        while collected < n:
            b      = torch.FloatTensor(n * 2, 3).uniform_(-self.r_max, self.r_max)
            inside = (b**2).sum(1) <= self.r_max**2
            b      = b[inside]
            pts.append(b)
            collected += b.shape[0]
        return torch.cat(pts)[:n].requires_grad_(True)

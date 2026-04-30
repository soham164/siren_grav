"""
src/models.py  —  SIREN Network (Stage B)
==========================================
Same SIREN architecture from Stage A. Copied here so Stage B is
self-contained. One addition: the pre-training → fine-tuning workflow
that avoids the trivial solution trap.
"""

import torch
import torch.nn as nn
import numpy as np


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0     = omega_0
        self.in_features = in_features
        self.linear      = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = np.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            nn.init.uniform_(self.linear.bias, -np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNetwork(nn.Module):
    def __init__(self, in_features=3, hidden_features=256,
                 hidden_layers=4, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

        layers = [SirenLayer(in_features, hidden_features, omega_0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden_features, hidden_features, omega_0))
        self.backbone = nn.Sequential(*layers)

        self.head_phi        = nn.Linear(hidden_features, 1)
        self.head_rho_linear = nn.Linear(hidden_features, 1)
        self.softplus        = nn.Softplus(beta=10)

        for head in [self.head_phi, self.head_rho_linear]:
            b = np.sqrt(6 / hidden_features) / omega_0
            nn.init.uniform_(head.weight, -b, b)

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

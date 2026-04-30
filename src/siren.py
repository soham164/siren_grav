"""
src/siren.py  —  Step 2: SIREN Network Implementation
======================================================
SIREN = Sinusoidal Representation Network (Sitzmann et al., NeurIPS 2020)

The key ideas:
    1. Replace ReLU with sin(omega_0 * x) as the activation function
    2. Use a specific weight initialization that ensures activations span
       the full sine wave range from the very first forward pass
    3. Because sin is infinitely differentiable, the network can be
       supervised on ANY order derivative — including the Laplacian
       we need for Poisson's equation

Architecture for this project:
    Input:  (x, y, z) — 3D coordinates in the galaxy
    Hidden: 4 SIREN layers, 256 units each
    Output head 1 (Phi):  linear activation  → gravitational potential
    Output head 2 (Rho):  Softplus activation → density (always > 0)

Why two heads?
    We could derive rho from Phi via Poisson's equation.
    But predicting both separately and penalising their inconsistency
    gives faster, more stable training. The physics loss forces them
    to agree with each other during training.
"""

import torch
import torch.nn as nn
import numpy as np


# ── Single SIREN layer ────────────────────────────────────────────────────────
class SirenLayer(nn.Module):
    """
    One layer of a SIREN network.

    Forward pass: y = sin(omega_0 * (W @ x + b))

    The weight initialization is the critical part.
    From Sitzmann et al. 2020, Supplementary Material:

        For the FIRST layer:
            W ~ Uniform(-1/fan_in, +1/fan_in)

        For ALL OTHER layers:
            W ~ Uniform(-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0)

    This initialization ensures that at the start of training, the distribution
    of activations sin(omega_0 * W @ x + b) is approximately uniform over [-1, 1].
    Without this, the network collapses to either a constant or a single frequency.

    Parameters
    ----------
    in_features  : int   — number of input features
    out_features : int   — number of output features (neurons in this layer)
    omega_0      : float — frequency multiplier. Controls the base frequency
                           of the representation. Default: 30.0 (from paper)
    is_first     : bool  — True only for the first layer (different init)
    """

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0   = omega_0
        self.is_first  = is_first
        self.in_features = in_features

        # Standard linear layer (no bias activation — we apply sin manually)
        self.linear = nn.Linear(in_features, out_features)

        # Apply the critical Sitzmann initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Apply the exact initialization from Sitzmann et al. 2020.
        This is NOT the same as PyTorch's default (Kaiming uniform).
        Using the wrong initialization is the most common SIREN bug.
        """
        with torch.no_grad():
            if self.is_first:
                # First layer: weights in [-1/fan_in, +1/fan_in]
                # This ensures the input to sin() spans roughly [-pi, pi]
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: weights scaled by 1/omega_0
                # This keeps activation distributions consistent across layers
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

            # Bias initialization: uniform in [-pi, pi]
            # This adds phase diversity across neurons
            nn.init.uniform_(self.linear.bias, -np.pi, np.pi)

    def forward(self, x):
        """
        Apply linear transformation then sine activation.
        The omega_0 multiplier is applied INSIDE the sine.
        """
        return torch.sin(self.omega_0 * self.linear(x))


# ── Full SIREN network ────────────────────────────────────────────────────────
class SirenNetwork(nn.Module):
    """
    Full SIREN network for learning galactic potentials.

    Takes 3D coordinates (x, y, z) and outputs:
        Phi : gravitational potential (unbounded, usually negative)
        Rho : mass density (always positive, guaranteed by Softplus)

    Parameters
    ----------
    in_features   : int   — input dimension. 3 for (x,y,z) coordinates.
    hidden_features: int  — neurons per hidden layer. 256 is standard.
    hidden_layers : int   — number of hidden SIREN layers. 4 recommended.
    omega_0       : float — base frequency. 30.0 from the paper.
    """

    def __init__(self,
                 in_features=3,
                 hidden_features=256,
                 hidden_layers=4,
                 omega_0=30.0):
        super().__init__()

        self.omega_0 = omega_0

        # ── Build the SIREN backbone ──────────────────────────────────────────
        # First layer (is_first=True uses different initialization)
        layers = [SirenLayer(in_features, hidden_features,
                             omega_0=omega_0, is_first=True)]

        # Remaining hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden_features, hidden_features,
                                     omega_0=omega_0, is_first=False))

        self.backbone = nn.Sequential(*layers)

        # ── Output head 1: Gravitational potential Phi ───────────────────────
        # Linear activation — Phi can be any real number (usually negative)
        self.head_phi = nn.Linear(hidden_features, 1)
        nn.init.uniform_(self.head_phi.weight,
                         -np.sqrt(6.0 / hidden_features) / omega_0,
                          np.sqrt(6.0 / hidden_features) / omega_0)

        # ── Output head 2: Mass density Rho ──────────────────────────────────
        # Softplus activation: log(1 + exp(x)) — always positive and smooth.
        # Physical: density must be >= 0 everywhere. This enforces it architecturally.
        self.head_rho_linear = nn.Linear(hidden_features, 1)
        nn.init.uniform_(self.head_rho_linear.weight,
                         -np.sqrt(6.0 / hidden_features) / omega_0,
                          np.sqrt(6.0 / hidden_features) / omega_0)
        self.softplus = nn.Softplus(beta=10)  # Sharp Softplus ≈ ReLU but smooth

    def forward(self, coords):
        """
        Forward pass.

        Parameters
        ----------
        coords : tensor of shape (N, 3)
            Normalised (x, y, z) coordinates in [-1, 1].

        Returns
        -------
        phi : tensor of shape (N, 1) — gravitational potential
        rho : tensor of shape (N, 1) — mass density (always > 0)
        """
        features = self.backbone(coords)
        phi = self.head_phi(features)
        rho = self.softplus(self.head_rho_linear(features))
        return phi, rho

    def forward_phi_only(self, coords):
        """
        Return only Phi. Used during Laplacian computation
        to avoid computing rho unnecessarily.
        """
        features = self.backbone(coords)
        return self.head_phi(features)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def weight_file_size_kb(self):
        """Estimate weight file size in KB (float32 = 4 bytes per parameter)."""
        return self.count_parameters() * 4 / 1024


# ── Initialization verifier ───────────────────────────────────────────────────
def verify_siren_initialization(model, n_test=10_000, verbose=True):
    """
    CRITICAL verification: check that the SIREN initialization is correct.

    After building a SIREN, before any training, pass random inputs through
    and check that activations at each layer are approximately uniform in [-1, 1].

    If the initialization is wrong:
        - Activations will cluster near 0 (collapsed)
        - Or saturate at ±1 (gradient vanishing/exploding)
        - Training will fail silently or produce garbage

    Parameters
    ----------
    model   : SirenNetwork instance
    n_test  : number of random inputs to test with
    verbose : print results

    Returns
    -------
    bool : True if all layers pass, False otherwise
    """
    model.eval()
    x = torch.FloatTensor(n_test, 3).uniform_(-1, 1)

    results = []
    all_pass = True

    with torch.no_grad():
        # Pass through backbone layer by layer, check activation distribution
        h = x
        for i, layer in enumerate(model.backbone):
            h = layer(h)

            mean = h.mean().item()
            std  = h.std().item()
            frac_saturated = ((h.abs() > 0.95)).float().mean().item()

            # Criteria:
            # Mean should be near 0 (no bias in activation)
            # Std should be near 0.5-0.7 (spread across sine wave range)
            # Saturated fraction < 20% (not clipped at ±1)
            mean_ok = abs(mean) < 0.15
            std_ok  = 0.3 < std < 0.85
            sat_ok  = frac_saturated < 0.25

            layer_pass = mean_ok and std_ok and sat_ok
            all_pass   = all_pass and layer_pass

            results.append({
                'layer': i + 1,
                'mean': mean,
                'std': std,
                'saturated_pct': frac_saturated * 100,
                'pass': layer_pass,
            })

            if verbose:
                status = "PASS" if layer_pass else "FAIL"
                print(f"  Layer {i+1}: mean={mean:+.3f}  std={std:.3f}  "
                      f"saturated={frac_saturated*100:.1f}%  [{status}]")

    return all_pass, results


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("SIREN Network Self-Test")
    print("=" * 60)

    # Build the network
    model = SirenNetwork(
        in_features=3,
        hidden_features=256,
        hidden_layers=4,
        omega_0=30.0
    )

    print(f"\nNetwork built:")
    print(f"  Total parameters : {model.count_parameters():,}")
    print(f"  Weight file size : {model.weight_file_size_kb():.1f} KB")

    # Test 1: Forward pass shape
    test_input = torch.randn(16, 3)
    phi, rho = model(test_input)
    print(f"\nTest 1 — Forward pass shape:")
    print(f"  Input shape : {test_input.shape}")
    print(f"  Phi shape   : {phi.shape}  (expected: [16, 1])")
    print(f"  Rho shape   : {rho.shape}  (expected: [16, 1])")
    shape_ok = phi.shape == (16, 1) and rho.shape == (16, 1)
    print(f"  Result      : {'PASS' if shape_ok else 'FAIL'}")

    # Test 2: Rho always positive (Softplus guarantee)
    rho_min = rho.min().item()
    print(f"\nTest 2 — Rho always positive (Softplus):")
    print(f"  Min rho: {rho_min:.6f}  {'PASS' if rho_min > 0 else 'FAIL'}")

    # Test 3: Gradient flows through the network (needed for Laplacian)
    coords = torch.randn(8, 3, requires_grad=True)
    phi_out, _ = model(coords)
    phi_out.sum().backward()
    grad_ok = coords.grad is not None and not coords.grad.isnan().any()
    print(f"\nTest 3 — Gradients flow correctly:")
    print(f"  Gradient exists and finite: {'PASS' if grad_ok else 'FAIL'}")

    # Test 4: Initialization distribution (the critical one)
    print(f"\nTest 4 — Initialization distribution (Sitzmann et al. 2020):")
    fresh_model = SirenNetwork()
    all_pass, _ = verify_siren_initialization(fresh_model, n_test=10_000, verbose=True)
    print(f"\n  Overall initialization: {'PASS' if all_pass else 'FAIL'}")

    # Test 5: omega_0 sensitivity
    print(f"\nTest 5 — omega_0 sensitivity check:")
    for omega in [10.0, 30.0, 50.0]:
        m = SirenNetwork(omega_0=omega)
        ok, _ = verify_siren_initialization(m, n_test=5_000, verbose=False)
        print(f"  omega_0={omega:5.1f}: {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print("SIREN self-test complete.")
    print("=" * 60)

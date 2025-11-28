"""
PyTorch implementation of GAIN (Generative Adversarial Imputation Nets)
for tabular numeric data with missing values (np.nan).

Ported from the original TensorFlow 1.x implementation by Yoon et al. (ICML 2018).
- Original gain(): 
- Normalization & rounding utils: 

Public API:
- GAINParams: hyperparameters dataclass
- train_gain_numeric(data_x, params, verbose=False) -> (generator, norm_params)
- gain_impute_numeric(generator, norm_params, data_x, params=None) -> imputed_data

All functions assume:
- data_x is a NumPy array of shape (N, D) with np.nan indicating missing entries.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
@dataclass
class GAINParams:
    batch_size: int = 128
    hint_rate: float = 0.9
    alpha: float = 100.0
    iterations: int = 10000
    learning_rate: float = 1e-3
    device: str = "cpu"


# ---------------------------------------------------------------------
# Normalization / renormalization / rounding (ported from utils.py)
# ---------------------------------------------------------------------
def _normalization(data: np.ndarray,
                   parameters: Optional[Dict[str, np.ndarray]] = None
                   ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Min-max normalization to [0, 1] per feature.
    If parameters is None: compute min/max from data.
    Otherwise: reuse given min/max (for test data).
    """
    data = data.copy()
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):
            col = norm_data[:, i]
            # If all missing, set to 0
            if np.all(np.isnan(col)):
                min_val[i] = 0.0
                max_val[i] = 1.0
                norm_data[:, i] = 0.0
                continue

            min_val[i] = np.nanmin(col)
            norm_data[:, i] = col - min_val[i]
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = {"min_val": min_val, "max_val": max_val}
    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]
        for i in range(dim):
            col = norm_data[:, i]
            col = col - min_val[i]
            col = col / (max_val[i] + 1e-6)
            norm_data[:, i] = col
        norm_parameters = parameters

    return norm_data, norm_parameters


def _renormalization(norm_data: np.ndarray,
                     norm_parameters: Dict[str, np.ndarray]) -> np.ndarray:
    """Undo min-max normalization."""
    norm_data = norm_data.copy()
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]
    return renorm_data


def _rounding(imputed_data: np.ndarray, data_x: np.ndarray) -> np.ndarray:
    """
    Round imputed data for quasi-categorical variables:
    if a feature has <20 unique non-missing values, round it.
    """
    imputed_data = imputed_data.copy()
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if temp.size == 0:
            continue
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data


# ---------------------------------------------------------------------
# Small samplers (ported from utils.py)
# ---------------------------------------------------------------------
def _binary_sampler(p: float, rows: int, cols: int) -> np.ndarray:
    """Sample Bernoulli(p) matrix."""
    return (np.random.uniform(0., 1., size=(rows, cols)) < p).astype(float)


def _uniform_sampler(low: float, high: float, rows: int, cols: int) -> np.ndarray:
    """Sample U(low, high) matrix."""
    return np.random.uniform(low, high, size=(rows, cols))


def _sample_batch_index(total: int, batch_size: int) -> np.ndarray:
    """Sample random batch indices."""
    total_idx = np.random.permutation(total)
    return total_idx[:batch_size]


# ---------------------------------------------------------------------
# GAIN networks
# ---------------------------------------------------------------------
class Generator(nn.Module):
    """
    Generator G(x, m):
      inputs: concat([x_tilde, m]) where x_tilde = m * x + (1-m) * z
      outputs: G_sample in [0,1]^D (sigmoid activation)
    """

    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim

        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x, m: (B, D)
        inputs = torch.cat([x, m], dim=1)  # (B, 2D)
        h = torch.relu(self.fc1(inputs))
        h = torch.relu(self.fc2(h))
        out = torch.sigmoid(self.fc3(h))  # (B, D)
        return out


class Discriminator(nn.Module):
    """
    Discriminator D(x_hat, h):
      inputs: concat([x_hat, h])
      outputs: D_prob in (0,1)^D (sigmoid activation)
    """

    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.dim = dim
        self.h_dim = h_dim

        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x_hat, h: (B, D)
        inputs = torch.cat([x_hat, h], dim=1)  # (B, 2D)
        h1 = torch.relu(self.fc1(inputs))
        h2 = torch.relu(self.fc2(h1))
        out = torch.sigmoid(self.fc3(h2))  # (B, D)
        return out


# ---------------------------------------------------------------------
# Training function (numeric data with NaNs)
# ---------------------------------------------------------------------
def train_gain_numeric(
    data_x: np.ndarray,
    params: GAINParams,
    verbose: bool = False,
) -> Tuple[Generator, Dict[str, np.ndarray]]:
    """
    Train GAIN on data_x with missing values (np.nan).

    Args:
        data_x: (N, D) NumPy array with np.nan indicating missing entries.
        params: GAINParams with hyperparameters.
        verbose: whether to print progress every ~1000 iterations.

    Returns:
        generator: trained Generator model (PyTorch nn.Module).
        norm_params: dict with 'min_val' and 'max_val' for renormalization.
    """
    device = params.device
    X = np.asarray(data_x, float)
    no, dim = X.shape

    # Mask matrix (1 if observed, 0 if missing)
    data_m = 1.0 - np.isnan(X)

    # Min-max normalization
    norm_data, norm_params = _normalization(X)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    # Hidden dimension (original code uses dim)
    h_dim = int(dim)

    # Models
    G = Generator(dim=dim, h_dim=h_dim).to(device)
    D = Discriminator(dim=dim, h_dim=h_dim).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=params.learning_rate)
    opt_D = torch.optim.Adam(D.parameters(), lr=params.learning_rate)

    batch_size = params.batch_size
    hint_rate = params.hint_rate
    alpha = params.alpha
    iters = params.iterations

    eps = 1e-8

    for it in range(iters):
        # Sample batch
        batch_idx = _sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]       # (B, D)
        M_mb = data_m[batch_idx, :]           # (B, D)

        # Sample random noise Z
        Z_mb = _uniform_sampler(0, 0.01, batch_size, dim)

        # Sample hint matrix H
        H_temp = _binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_temp

        # Combine observed X and noise Z
        X_tilde = M_mb * X_mb + (1.0 - M_mb) * Z_mb  # (B, D)

        # Convert to torch
        X_t = torch.from_numpy(X_tilde).float().to(device)
        M_t = torch.from_numpy(M_mb).float().to(device)
        H_t = torch.from_numpy(H_mb).float().to(device)

        # --------------------
        # Train Discriminator
        # --------------------
        G_sample = G(X_t, M_t)                      # (B, D)
        X_hat = M_t * X_t + (1.0 - M_t) * G_sample  # (B, D)
        D_prob = D(X_hat.detach(), H_t)             # (B, D)

        D_loss = -torch.mean(
            M_t * torch.log(D_prob + eps) +
            (1.0 - M_t) * torch.log(1.0 - D_prob + eps)
        )

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # ---------------
        # Train Generator
        # ---------------
        G_sample = G(X_t, M_t)
        X_hat = M_t * X_t + (1.0 - M_t) * G_sample
        D_prob = D(X_hat, H_t)

        G_loss_adv = -torch.mean((1.0 - M_t) * torch.log(D_prob + eps))

        # Reconstruction loss on observed entries
        MSE_loss = torch.sum((M_t * X_t - M_t * G_sample) ** 2) / (
            torch.sum(M_t) + eps
        )

        G_loss = G_loss_adv + alpha * MSE_loss

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if verbose and ((it + 1) % 1000 == 0 or it == 0 or it + 1 == iters):
            print(
                f"[GAIN] Iter {it+1}/{iters}  "
                f"D_loss={D_loss.item():.4f}  "
                f"G_loss={G_loss.item():.4f}  "
                f"MSE={MSE_loss.item():.4f}"
            )

    return G, norm_params


# ---------------------------------------------------------------------
# Imputation function
# ---------------------------------------------------------------------
def gain_impute_numeric(
    generator: Generator,
    norm_params: Dict[str, np.ndarray],
    data_x: np.ndarray,
    params: Optional[GAINParams] = None,
) -> np.ndarray:
    """
    Impute missing values in data_x using a trained GAIN generator.

    Args:
        generator: trained Generator (output of train_gain_numeric).
        norm_params: dict from train_gain_numeric for renormalization.
        data_x: (N, D) NumPy array with np.nan for missing entries.
        params: optional GAINParams to specify device, etc.
                If None, device='cpu' is assumed.

    Returns:
        imputed_data: (N, D) NumPy array with imputed values.
    """
    if params is None:
        device = "cpu"
    else:
        device = params.device

    X = np.asarray(data_x, float)
    no, dim = X.shape

    # Mask
    data_m = 1.0 - np.isnan(X)

    # Normalize with training parameters
    norm_data, _ = _normalization(X, norm_params)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    # Sample noise
    Z_mb = _uniform_sampler(0, 0.01, no, dim)
    X_tilde = data_m * norm_data_x + (1.0 - data_m) * Z_mb

    # To torch
    X_t = torch.from_numpy(X_tilde).float().to(device)
    M_t = torch.from_numpy(data_m).float().to(device)

    generator = generator.to(device)
    generator.eval()
    with torch.no_grad():
        G_sample = generator(X_t, M_t).cpu().numpy()

    # Combine observed + imputed in normalized space
    imputed_norm = data_m * norm_data_x + (1.0 - data_m) * G_sample

    # Renormalize to original scale
    imputed_data = _renormalization(imputed_norm, norm_params)

    # Rounding for quasi-categorical vars
    imputed_data = _rounding(imputed_data, data_x)

    return imputed_data

"""
Lightweight MisGAN-style GAN imputer in PyTorch for tabular numeric data.

- Uses a generator G(x_tilde, m) that outputs a full vector in [0,1]^D.
- Uses a discriminator D(x_masked, m) that outputs a scalar "real vs fake" score.
- Trains on masked real vs masked fake samples, using empirical masks.
- Adds an MSE reconstruction loss on observed entries, similar to GAIN.

Public API:
- MisGANLiteParams
- train_misgan_lite_numeric(data_x, params, verbose=False)
- misgan_lite_impute_numeric(generator, norm_params, data_x, params=None)

Assumptions:
- data_x is a numpy array (N, D) with np.nan indicating missing entries.
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
class MisGANLiteParams:
    batch_size: int = 128
    alpha: float = 100.0          # weight for reconstruction loss
    iterations: int = 10000
    learning_rate: float = 1e-3
    device: str = "cpu"


# ---------------------------------------------------------------------
# Normalization utilities (min-max per feature)
# ---------------------------------------------------------------------
def _normalization(
    data: np.ndarray,
    parameters: Optional[Dict[str, np.ndarray]] = None,
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
            if np.all(np.isnan(col)):
                # All-missing feature: fix range [0,1)
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


def _renormalization(
    norm_data: np.ndarray,
    norm_parameters: Dict[str, np.ndarray],
) -> np.ndarray:
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
# Small helpers
# ---------------------------------------------------------------------
def _uniform_sampler(low: float, high: float, rows: int, cols: int) -> np.ndarray:
    """Sample U(low, high) matrix."""
    return np.random.uniform(low, high, size=(rows, cols))


def _sample_batch_index(total: int, batch_size: int) -> np.ndarray:
    """Sample random batch indices."""
    total_idx = np.random.permutation(total)
    return total_idx[:batch_size]


# ---------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------
class MisGANLiteGenerator(nn.Module):
    """
    Generator G(x_tilde, m):
      - x_tilde: incomplete data with noise on missing entries
      - m: mask (1=observed, 0=missing)
      -> output: full normalized sample in [0,1]^D
    """

    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.dim = dim

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

    def forward(self, x_tilde: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x_tilde, m: (B, D)
        inp = torch.cat([x_tilde, m], dim=1)  # (B, 2D)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        out = torch.sigmoid(self.fc3(h))      # (B, D)
        return out


class MisGANLiteDiscriminator(nn.Module):
    """
    Discriminator D(x_masked, m):
      - x_masked: masked data (x * m)
      - m: mask (1=observed, 0=missing)
      -> output: scalar in (0,1) for each sample
    """

    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.dim = dim

        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_masked: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x_masked, m: (B, D)
        inp = torch.cat([x_masked, m], dim=1)  # (B, 2D)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        out = torch.sigmoid(self.fc3(h))       # (B, 1)
        return out


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train_misgan_lite_numeric(
    data_x: np.ndarray,
    params: MisGANLiteParams,
    verbose: bool = False,
) -> Tuple[MisGANLiteGenerator, Dict[str, np.ndarray]]:
    """
    Train MisGAN-lite on numeric data with NaNs.

    Args:
        data_x: (N, D) NumPy array with np.nan indicating missing entries.
        params: MisGANLiteParams.
        verbose: print progress every 1000 iterations.

    Returns:
        generator: trained MisGANLiteGenerator.
        norm_params: dict with 'min_val' and 'max_val'.
    """
    device = params.device
    X = np.asarray(data_x, float)
    no, dim = X.shape

    # Mask: 1 if observed, 0 if missing
    M = 1.0 - np.isnan(X)

    # Normalize
    norm_data, norm_params = _normalization(X)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    # Hidden dimension ~ dim
    h_dim = int(dim)

    G = MisGANLiteGenerator(dim=dim, h_dim=h_dim).to(device)
    D = MisGANLiteDiscriminator(dim=dim, h_dim=h_dim).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=params.learning_rate)
    opt_D = torch.optim.Adam(D.parameters(), lr=params.learning_rate)

    batch_size = params.batch_size
    alpha = params.alpha
    iters = params.iterations
    eps = 1e-8

    for it in range(iters):
        # Sample batch
        batch_idx = _sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]   # (B, D)
        M_mb = M[batch_idx, :]            # (B, D)

        # Noise for missing entries
        Z_mb = _uniform_sampler(0, 0.01, batch_size, dim)
        X_tilde_mb = M_mb * X_mb + (1.0 - M_mb) * Z_mb  # (B, D)

        # Torch tensors
        X_tilde_t = torch.from_numpy(X_tilde_mb).float().to(device)
        X_t = torch.from_numpy(X_mb).float().to(device)
        M_t = torch.from_numpy(M_mb).float().to(device)

        # ----------------
        # Train D
        # ----------------
        G_sample = G(X_tilde_t, M_t)                        # (B, D)
        X_hat = M_t * X_t + (1.0 - M_t) * G_sample          # (B, D)

        real_masked = M_t * X_t                             # (B, D)
        fake_masked = M_t * X_hat.detach()                  # (B, D)

        D_real = D(real_masked, M_t)                        # (B, 1)
        D_fake = D(fake_masked, M_t)                        # (B, 1)

        D_loss = -torch.mean(
            torch.log(D_real + eps) + torch.log(1.0 - D_fake + eps)
        )

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # ----------------
        # Train G
        # ----------------
        G_sample = G(X_tilde_t, M_t)
        X_hat = M_t * X_t + (1.0 - M_t) * G_sample

        fake_masked = M_t * X_hat
        D_fake = D(fake_masked, M_t)

        G_adv_loss = -torch.mean(torch.log(D_fake + eps))

        # Reconstruction on observed entries
        mse_num = torch.sum((M_t * X_t - M_t * G_sample) ** 2)
        mse_den = torch.sum(M_t) + eps
        MSE_loss = mse_num / mse_den

        G_loss = G_adv_loss + alpha * MSE_loss

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if verbose and ((it + 1) % 1000 == 0 or it == 1 or it + 1 == iters):
            print(
                f"[MisGAN-lite] Iter {it+1}/{iters}  "
                f"D_loss={D_loss.item():.4f}  "
                f"G_loss={G_loss.item():.4f}  "
                f"MSE={MSE_loss.item():.4f}"
            )

    return G, norm_params


# ---------------------------------------------------------------------
# Imputation function
# ---------------------------------------------------------------------
def misgan_lite_impute_numeric(
    generator: MisGANLiteGenerator,
    norm_params: Dict[str, np.ndarray],
    data_x: np.ndarray,
    params: Optional[MisGANLiteParams] = None,
) -> np.ndarray:
    """
    Impute missing values in data_x using a trained MisGAN-lite generator.

    Args:
        generator: trained MisGANLiteGenerator.
        norm_params: dict from training (_normalization).
        data_x: (N, D) NumPy array with np.nan for missing.
        params: optional MisGANLiteParams to get device.

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
    M = 1.0 - np.isnan(X)

    # Normalize with training params
    norm_data, _ = _normalization(X, norm_params)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    # Noise
    Z_mb = _uniform_sampler(0, 0.01, no, dim)
    X_tilde = M * norm_data_x + (1.0 - M) * Z_mb

    X_tilde_t = torch.from_numpy(X_tilde).float().to(device)
    M_t = torch.from_numpy(M).float().to(device)

    generator = generator.to(device)
    generator.eval()
    with torch.no_grad():
        G_sample = generator(X_tilde_t, M_t).cpu().numpy()

    # Combine in normalized space
    imputed_norm = M * norm_data_x + (1.0 - M) * G_sample

    # Renormalize
    imputed_data = _renormalization(imputed_norm, norm_params)

    # Round quasi-categorical
    imputed_data = _rounding(imputed_data, data_x)

    return imputed_data

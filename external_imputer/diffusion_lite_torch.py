# diffusion_torch.py
# Lightweight Gaussian diffusion imputer for numeric tabular data.
#
# NOTE: This is a simplified diffusion model:
#   - Unconditional diffusion (no explicit conditioning on mask)
#   - Small MLP denoiser (no Transformer)
#   - Clamping of observed entries during reverse diffusion
#
# Intended for use as an external generative imputer in validation_v2.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class DiffusionParams:
    """
    Hyperparameters for Diffusion-lite.

    These are what you will tune from validation_v2 via the
    build_diffusion_lite_external_imputer wrapper.
    """
    T: int = 50                  # number of diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 128
    time_dim: int = 64
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    n_samples_impute: int = 5    # diffusion runs per row for imputation

# ----------------------------------------------------------------------
# 1. Time embedding + small denoiser MLP
# ----------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """Simple sinusoidal + linear time embedding."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (batch,) integer timesteps in [0, T-1]
        Returns: (batch, embed_dim)
        """
        # normalize t to [0,1]
        device = t.device
        half_dim = self.embed_dim // 2
        # sinusoidal encoding
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(np.log(10000.0) / half_dim)
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros(emb.size(0), 1, device=device)], dim=1
            )
        return self.linear(emb)


class DenoiseMLP(nn.Module):
    """Small MLP denoiser: predicts noise epsilon from (x_t, t)."""

    def __init__(self, dim: int, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (batch, dim)
        t  : (batch,) integer timesteps
        """
        t_emb = self.time_mlp(t)  # (batch, time_dim)
        h = torch.cat([x_t, t_emb], dim=1)
        return self.net(h)


# ----------------------------------------------------------------------
# 2. Diffusion parameters + helpers
# ----------------------------------------------------------------------

@dataclass
class DiffusionParams:
    T: int = 50                  # number of diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 128
    time_dim: int = 64
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    n_samples_impute: int = 5    # number of diffusion runs per row


class DiffusionSchedule:
    """Precompute beta/alpha schedules and q(x_t|x_0) coefficients."""

    def __init__(self, T: int, beta_start: float, beta_end: float, device: torch.device):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas            # (T,)
        self.alphas = alphas          # (T,)
        self.alphas_cumprod = alphas_cumprod  # (T,)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
        )

        # For q(x_t | x_0) = sqrt(ac) * x0 + sqrt(1-ac) * eps
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # For reverse step p(x_{t-1} | x_t, x_0) approx
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (
            1.0 - self.alphas_cumprod
        )


# ----------------------------------------------------------------------
# 3. Training on numeric data
# ----------------------------------------------------------------------
def _mean_impute_numpy(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Column-wise mean imputation (for training only), robust to all-NaN columns.

    This mirrors the helper used in mcflow_torch / misgan_lite_torch so that
    all generative-lite imputers share the same preprocessing.
    """
    X_imp = X.copy()
    col_means = np.nanmean(X_imp, axis=0)

    # If a column is entirely NaN, nanmean returns NaN. For those columns
    # we just set the mean to 0.0 as a neutral fallback.
    nan_cols = np.isnan(col_means)
    if nan_cols.any():
        col_means[nan_cols] = 0.0

    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = np.take(col_means, inds[1])
    return X_imp, col_means


def train_diffusion_numeric(
    X: np.ndarray,
    params: DiffusionParams,
    verbose: bool = False,
) -> Tuple[DenoiseMLP, DiffusionParams, dict]:
    """
    Train a diffusion-lite model on numeric data with NaNs.

    X: (n_samples, n_features), NaNs allowed (for training we mean-impute).
    """
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    device = torch.device(params.device)

    X_imp, col_means = _mean_impute_numpy(X)
    n, d = X_imp.shape

    schedule = DiffusionSchedule(
        T=params.T,
        beta_start=params.beta_start,
        beta_end=params.beta_end,
        device=device,
    )

    model = DenoiseMLP(
        dim=d,
        hidden_dim=params.hidden_dim,
        time_dim=params.time_dim,
    ).to(device)

    x_tensor = torch.from_numpy(X_imp.astype(np.float32)).to(device)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    for epoch in range(1, params.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for (x0_batch,) in loader:
            bsz = x0_batch.size(0)

            # Sample random timestep t ∈ {0,...,T-1}
            t = torch.randint(
                low=0, high=schedule.T, size=(bsz,), device=device, dtype=torch.long
            )

            # Sample noise epsilon
            eps = torch.randn_like(x0_batch)

            # Compute x_t via closed form q(x_t | x_0)
            sqrt_ac = schedule.sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_om = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            x_t = sqrt_ac * x0_batch + sqrt_om * eps

            # Predict noise
            eps_pred = model(x_t, t)

            # Loss = E[ || eps - eps_pred ||^2 ]
            loss = ((eps - eps_pred) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        if verbose and (epoch % max(1, params.epochs // 10) == 0 or epoch == 1):
            avg_loss = total_loss / max(1, num_batches)
            print(f"[Diffusion-lite] Epoch {epoch}/{params.epochs}, loss={avg_loss:.4f}")

    aux = {
        "col_means": col_means,
        "dim": d,
        "schedule": schedule,
    }
    return model, params, aux


# ----------------------------------------------------------------------
# 4. Imputation with reverse diffusion + clamping
# ----------------------------------------------------------------------

@torch.no_grad()
def diffusion_impute_numeric(
    model: DenoiseMLP,
    params: DiffusionParams,
    aux: dict,
    X_incomplete: np.ndarray,
) -> np.ndarray:
    """
    Impute missing entries in X_incomplete.

    Strategy:
      - For each of n_samples_impute runs:
          * Start from x_T ~ N(0, I)
          * For t = T-1,...,0:
                - predict eps = model(x_t, t)
                - compute x_{t-1} with DDPM-style step
                - clamp observed entries to their true values
        Average over samples for missing entries.
    """
    device = torch.device(params.device)
    model.eval()

    X = X_incomplete.copy()
    n, d = X.shape
    assert d == aux["dim"]

    schedule: DiffusionSchedule = aux["schedule"]
    obs_mask = ~np.isnan(X)  # True where observed

    X_samples = np.zeros((params.n_samples_impute, n, d), dtype=np.float32)

    for k in range(params.n_samples_impute):
        # Start from pure noise at timestep T-1
        x_t = torch.randn(n, d, device=device)

        for t_step in reversed(range(schedule.T)):
            t = torch.full((n,), t_step, device=device, dtype=torch.long)
            eps_theta = model(x_t, t)  # predicted noise

            beta_t = schedule.betas[t_step]
            alpha_t = schedule.alphas[t_step]
            alpha_bar_t = schedule.alphas_cumprod[t_step]
            alpha_bar_prev = (
                schedule.alphas_cumprod_prev[t_step]
                if t_step > 0
                else torch.tensor(1.0, device=device)
            )
            # DDPM posterior mean
            # x0_hat ≈ (x_t - sqrt(1 - alpha_bar_t)*eps_theta) / sqrt(alpha_bar_t)
            sqrt_alpha_bar_t = alpha_bar_t**0.5
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t)**0.5
            x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t

            # mean of q(x_{t-1} | x_t, x0_hat)
            coef1 = (alpha_bar_prev**0.5 * beta_t) / (1 - alpha_bar_t)
            coef2 = ((1 - alpha_bar_prev) * alpha_t**0.5) / (1 - alpha_bar_t)
            mean = coef1 * x0_hat + coef2 * x_t

            if t_step > 0:
                var = schedule.posterior_variance[t_step]
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(var) * noise
            else:
                x_t = mean

            # Clamp observed entries
            x_np = x_t.cpu().numpy()
            x_np[obs_mask] = X[obs_mask]
            x_t = torch.from_numpy(x_np).to(device)

        X_samples[k] = x_t.cpu().numpy().astype(np.float32)

    # Average samples, but keep observed entries exact
    X_imp = X_samples.mean(axis=0)
    X_imp[obs_mask] = X[obs_mask]
    return X_imp

# mcflow_torch.py
# Lightweight RealNVP-based generative imputer inspired by MCFlow
#
# NOTE: This is a simplified version:
#   - Trains a standard RealNVP flow on mean-imputed data
#   - No copy-masking / conditional likelihood
#   - No latent-to-latent refinement network
#
# It is intended for numeric tabular data with NaNs.



# mcflow_torch.py  (patched)

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class MCFlowParams:
    """
    Hyperparameters for MCFlow-lite.

    These are the knobs you can tune from validation_v2 via the
    ExternalImputer wrapper (e.g. n_coupling_layers, hidden_dim, epochs).
    """
    latent_dim: Optional[int] = None  # kept for symmetry with VAE-based models
    hidden_dim: int = 128
    n_coupling_layers: int = 6
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    n_samples_impute: int = 10  # number of z samples per row for imputation


class CouplingNN(nn.Module):
    """Simple MLP used for scale/translate networks in RealNVP."""

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RealNVPFlow(nn.Module):
    """RealNVP-style normalizing flow with a list of binary masks."""

    def __init__(
        self,
        dim: int,
        n_coupling_layers: int = 6,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.device = device
        self.scale_limit = 2.0  # <- NEW: limit magnitude of s

        # Simple alternating masks: 1010..., 0101..., repeating
        masks = []
        base_mask = (torch.arange(dim) % 2).float()  # [0,1,0,1,...]
        for i in range(n_coupling_layers):
            if i % 2 == 0:
                masks.append(base_mask)
            else:
                masks.append(1.0 - base_mask)
        self.masks = nn.Parameter(torch.stack(masks), requires_grad=False)

        self.scale_nns = nn.ModuleList(
            [CouplingNN(dim, hidden_dim) for _ in range(n_coupling_layers)]
        )
        self.translate_nns = nn.ModuleList(
            [CouplingNN(dim, hidden_dim) for _ in range(n_coupling_layers)]
        )

        self.register_buffer("prior_mean", torch.zeros(dim))
        self.register_buffer("prior_logvar", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det = torch.zeros(x.size(0), device=x.device)
        for i in range(len(self.scale_nns)):
            mask = self.masks[i]
            z_masked = z * mask

            # robust but less restrictive scale: clip instead of tanh
            s_raw = self.scale_nns[i](z_masked) * (1 - mask)
            s = torch.clamp(s_raw, -self.scale_limit, self.scale_limit)

            t = self.translate_nns[i](z_masked) * (1 - mask)
            z = z_masked + (1 - mask) * (z * torch.exp(s) + t)
            log_det = log_det + s.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for i in reversed(range(len(self.scale_nns))):
            mask = self.masks[i]
            x_masked = x * mask

            s_raw = self.scale_nns[i](x_masked) * (1 - mask)
            s = torch.clamp(s_raw, -self.scale_limit, self.scale_limit)

            t = self.translate_nns[i](x_masked) * (1 - mask)
            x = x_masked + (1 - mask) * ((x - t) * torch.exp(-s))
        return x

    def _prior_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
            self.dim * np.log(2 * np.pi) + (z**2).sum(dim=1)
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_pz = self._prior_log_prob(z)
        return log_pz + log_det


@dataclass
class MCFlowParams:
    latent_dim: Optional[int] = None
    hidden_dim: int = 128
    n_coupling_layers: int = 6
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    n_samples_impute: int = 10


def _mean_impute_numpy(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column-wise mean imputation, robust to all-NaN columns."""
    X_imp = X.copy()
    col_means = np.nanmean(X_imp, axis=0)
    # handle all-NaN columns
    nan_cols = np.isnan(col_means)
    if nan_cols.any():
        col_means[nan_cols] = 0.0
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = np.take(col_means, inds[1])
    return X_imp, col_means


def train_mcflow_numeric(
    X: np.ndarray,
    params: MCFlowParams,
    verbose: bool = False,
) -> Tuple[RealNVPFlow, MCFlowParams, dict]:
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    device = torch.device(params.device)
    X_imp, col_means = _mean_impute_numpy(X)
    n, d = X_imp.shape

    flow = RealNVPFlow(
        dim=d,
        n_coupling_layers=params.n_coupling_layers,
        hidden_dim=params.hidden_dim,
        device=str(device),
    ).to(device)

    x_tensor = torch.from_numpy(X_imp.astype(np.float32)).to(device)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(flow.parameters(), lr=params.learning_rate)

    for epoch in range(1, params.epochs + 1):
        flow.train()
        total_loss = 0.0
        num_batches = 0

        for (batch_x,) in loader:
            opt.zero_grad()
            log_px = flow.log_prob(batch_x)
            loss = -log_px.mean()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        if verbose and (epoch % max(1, params.epochs // 10) == 0 or epoch == 1):
            print(f"[MCFlow-lite] Epoch {epoch}/{params.epochs}, NLL={total_loss/max(1,num_batches):.4f}")

    aux_state = {"col_means": col_means, "dim": d}
    return flow, params, aux_state


def mcflow_impute_numeric(
    flow: RealNVPFlow,
    params: MCFlowParams,
    aux_state: dict,
    X_incomplete: np.ndarray,
) -> np.ndarray:
    device = next(flow.parameters()).device
    flow.eval()

    X = X_incomplete.copy()
    n, d = X.shape
    assert d == aux_state["dim"]

    obs_mask = ~np.isnan(X)

    X_imp_samples = np.zeros((params.n_samples_impute, n, d), dtype=np.float32)
    for k in range(params.n_samples_impute):
        with torch.no_grad():
            z = torch.randn(n, d, device=device)
            x_sample = flow.inverse(z).cpu().numpy().astype(np.float32)

        x_sample[obs_mask] = X[obs_mask]
        # optional extra safety: replace inf with nan here
        x_sample[~np.isfinite(x_sample)] = np.nan
        X_imp_samples[k] = x_sample

    X_imp_mean = X_imp_samples.mean(axis=0)
    X_imp_mean[obs_mask] = X[obs_mask]
    return X_imp_mean

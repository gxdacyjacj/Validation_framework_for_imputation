"""
PyTorch implementation of a self-masking notMIWAE-style model
for tabular numeric data with missingness.

Interface is similar to miwae_torch.py:

- NotMIWAEParams: dataclass with hyperparameters
- train_notmiwae_numeric(X, params, verbose=False)
- notmiwae_impute_numeric(trained, params, X, L=10)

We assume:
- X is a numpy array of shape (N, D) with np.nan for missing.
- Only numeric features are passed here; categorical handled outside.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Hyperparameters (paper-level defaults, but easy to change)
# --------------------------------------------------------------------------
@dataclass
class NotMIWAEParams:
    latent_dim: int = 50        # latent z dimension
    hidden_dim: int = 100       # hidden size in encoder/decoder
    iw_samples: int = 10        # importance-weight samples during training
    epochs: int = 500           # training epochs
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"


# --------------------------------------------------------------------------
# Core model: encoder, decoder, self-masking missingness
# --------------------------------------------------------------------------
class NotMIWAEModel(nn.Module):
    def __init__(self, x_dim: int, params: NotMIWAEParams):
        super().__init__()
        self.x_dim = x_dim
        self.params = params
        h = params.hidden_dim
        z_dim = params.latent_dim

        # Encoder: input = [x_imp, mask] -> h -> (mu, logvar) of q(z|x)
        enc_in_dim = x_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(h, z_dim)
        self.enc_logvar = nn.Linear(h, z_dim)

        # Decoder: z -> h -> (x_mean, x_logvar)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.dec_mean = nn.Linear(h, x_dim)
        self.dec_logvar = nn.Linear(h, x_dim)

        # Self-masking parameters: p(missing=1 | x) = sigmoid(a + b * x)
        # We use x_mean from decoder as proxy for x.
        self.sm_a = nn.Parameter(torch.zeros(x_dim))
        self.sm_b = nn.Parameter(torch.zeros(x_dim))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def encode(self, x_imp: torch.Tensor, mask: torch.Tensor):
        """
        x_imp: (B, D) with missing filled (e.g., zeros)
        mask : (B, D) with 1=observed, 0=missing
        """
        inp = torch.cat([x_imp, mask], dim=-1)
        h = self.encoder(inp)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor):
        """
        z: (B, L, z_dim) or (B, z_dim)
        Return: mean, logvar shaped like z but last dim = x_dim
        """
        if z.dim() == 2:
            h = self.decoder(z)
            mean = self.dec_mean(h)
            logvar = self.dec_logvar(h)
        else:
            B, L, z_dim = z.shape
            z_flat = z.reshape(B * L, z_dim)
            h = self.decoder(z_flat)
            mean_flat = self.dec_mean(h)
            logvar_flat = self.dec_logvar(h)
            mean = mean_flat.view(B, L, self.x_dim)
            logvar = logvar_flat.view(B, L, self.x_dim)
        return mean, logvar

    def _log_normal(self, x, mean, logvar):
        """
        Elementwise diagonal Gaussian log-density.
        x, mean, logvar: same shape (..., D)
        Returns log-density per dimension (no sum over D).
        """
        return -0.5 * (
            (x - mean) ** 2 / torch.exp(logvar) + logvar + np.log(2 * np.pi)
        )

    def _log_p_z(self, z):
        """Standard normal prior over z."""
        return -0.5 * torch.sum(
            z**2 + np.log(2 * np.pi), dim=-1
        )

    def _log_q_z_given_x(self, z, mu, logvar):
        """Diagonal Gaussian q(z|x)."""
        return -0.5 * torch.sum(
            (z - mu) ** 2 / torch.exp(logvar) + logvar + np.log(2 * np.pi),
            dim=-1,
        )

    def _log_p_s_given_x(self, s_missing, x_mean):
        """
        Self-masking MNAR:
          s_missing = 1 if missing, 0 if observed.
          p(s_missing=1 | x) = sigmoid(a + b * x).
        s_missing, x_mean: (B, L, D) or (B, D)
        Returns sum over last dim.
        """
        logits = self.sm_a + self.sm_b * x_mean
        p_missing = torch.sigmoid(logits)
        # log Bernoulli for each dimension
        log_p = s_missing * torch.log(p_missing + 1e-8) + \
                (1.0 - s_missing) * torch.log(1.0 - p_missing + 1e-8)
        return torch.sum(log_p, dim=-1)

    # ------------------------------------------------------------------
    # Training loss: negative IWAE bound for MNAR
    # ------------------------------------------------------------------
    def iwae_loss(self, x, mask, L: int):
        """
        x:    (B, D) with NaNs already replaced (x_imp)
        mask: (B, D) 1=observed, 0=missing

        We treat:
          s_missing = 1 - mask   (1=missing)
        """
        device = x.device
        B, D = x.shape
        s_missing = 1.0 - mask

        # Encode to q(z|x)
        mu, logvar = self.encode(x, mask)  # (B, z_dim)
        z_dim = mu.shape[-1]

        # Sample z: shape (B, L, z_dim)
        eps = torch.randn(B, L, z_dim, device=device)
        std = torch.exp(0.5 * logvar)
        z = mu.unsqueeze(1) + eps * std.unsqueeze(1)

        # Decode: x_mean, x_logvar: (B, L, D)
        x_mean, x_logvar = self.decode(z)

        # Log p(x_obs | z): mask selects observed dims
        # Expand mask to (B, L, D)
        mask_exp = mask.unsqueeze(1).expand_as(x_mean) # (B, L, D)
        x_obs = x.unsqueeze(1).expand_as(x_mean) # (B, L, D)
        log_px_elem = self._log_normal(x_obs, x_mean, x_logvar)  # (B, L, D)
        # Zero-out contributions from missing entries
        log_p_x_given_z = torch.sum(log_px_elem * mask_exp, dim=-1)  # (B, L)

        # Log p(s | x): use x_mean as proxy for x
        s_missing_exp = s_missing.unsqueeze(1).expand_as(x_mean)
        log_p_s_given_x = self._log_p_s_given_x(s_missing_exp, x_mean)  # (B, L)

        # Log p(z)
        log_p_z = self._log_p_z(z)  # (B, L)

        # Log q(z|x)
        log_q_z_given_x = self._log_q_z_given_x(
            z, mu.unsqueeze(1).expand_as(z), logvar.unsqueeze(1).expand_as(z)
        )  # (B, L)

        # Importance weights
        log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x  # (B, L)

        # IWAE bound: log(1/L sum_k exp(log_w_k))
        m = torch.max(log_w, dim=1, keepdim=True)[0]
        log_iw = m.squeeze(1) + torch.log(torch.mean(torch.exp(log_w - m), dim=1) + 1e-8)
        # Negative bound (for minimization)
        loss = -torch.mean(log_iw)
        return loss

    # ------------------------------------------------------------------
    # Imputation: importance-weighted posterior mean of x
    # ------------------------------------------------------------------
    def impute(self, x_in: np.ndarray, L: int = 10) -> np.ndarray:
        """
        x_in: numpy array (N, D) with np.nan for missing.
        Returns: numpy array (N, D) with missing entries imputed.
        """
        device = next(self.parameters()).device
        X = np.asarray(x_in, float)
        mask = (~np.isnan(X)).astype(np.float32)
        X_imp = np.where(np.isnan(X), 0.0, X).astype(np.float32)

        X_t = torch.from_numpy(X_imp).to(device)
        M_t = torch.from_numpy(mask).to(device)

        self.eval()
        with torch.no_grad():
            N, D = X_t.shape
            # Encode
            mu, logvar = self.encode(X_t, M_t)
            z_dim = mu.shape[-1]

            # Sample z (N, L, z_dim)
            eps = torch.randn(N, L, z_dim, device=device)
            std = torch.exp(0.5 * logvar)
            z = mu.unsqueeze(1) + eps * std.unsqueeze(1)

            # Decode
            x_mean, x_logvar = self.decode(z)  # (N, L, D)

            # Self-masking log p(s|x)
            s_missing = 1.0 - M_t
            s_missing_exp = s_missing.unsqueeze(1).expand_as(x_mean)
            log_p_s_given_x = self._log_p_s_given_x(s_missing_exp, x_mean)  # (N, L)

            # Log p(z)
            log_p_z = self._log_p_z(z)  # (N, L)

            # Log q(z|x)
            log_q_z_given_x = self._log_q_z_given_x(
                z, mu.unsqueeze(1).expand_as(z), logvar.unsqueeze(1).expand_as(z)
            )

            # For imputation, we can ignore p(x_obs|z) since mask already constrains x
            # But to mimic notMIWAE more closely, we include it on observed entries:
            mask_exp = M_t.unsqueeze(1).expand_as(x_mean) # (N, L, D)
            x_obs = X_t.unsqueeze(1).expand_as(x_mean)    # (N, L, D)        
            log_px_elem = self._log_normal(x_obs, x_mean, x_logvar) # (N, L, D)
            log_p_x_given_z = torch.sum(log_px_elem * mask_exp, dim=-1) # (N, L)

            log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x  # (N, L)

            # Softmax over importance weights
            max_w, _ = torch.max(log_w, dim=1, keepdim=True)
            w = torch.exp(log_w - max_w)
            w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)  # (N, L)

            # Posterior mean of x: sum_k w_k * x_mean_k
            xm = torch.sum(w[:, :, None] * x_mean, dim=1)  # (N, D)

            X_hat = X_imp.copy()
            X_hat[np.isnan(X)] = xm.cpu().numpy()[np.isnan(X)]
        return X_hat


# --------------------------------------------------------------------------
# Public API: training and imputation helpers
# --------------------------------------------------------------------------
def _to_device(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device)


def train_notmiwae_numeric(
    X: np.ndarray,
    params: NotMIWAEParams,
    verbose: bool = False,
) -> Tuple[NotMIWAEModel, NotMIWAEParams]:
    """
    Train self-masking NotMIWAE on numeric data with NaNs.

    X: numpy array (N, D) with np.nan for missing.
    """
    device = params.device
    X = np.asarray(X, float)
    N, D = X.shape

    # Build mask and simple zero-imputation for encoder input
    mask = (~np.isnan(X)).astype(np.float32)
    X_imp = np.where(np.isnan(X), 0.0, X).astype(np.float32)

    X_t = torch.from_numpy(X_imp)
    M_t = torch.from_numpy(mask)

    model = NotMIWAEModel(x_dim=D, params=params).to(device)
    X_t = _to_device(X_t, device)
    M_t = _to_device(M_t, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    batch_size = params.batch_size
    iw_samples = params.iw_samples

    for epoch in range(1, params.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        nbatches = 0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_t[idx]
            mb = M_t[idx]

            loss = model.iwae_loss(xb, mb, L=iw_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            nbatches += 1

        if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == params.epochs):
            avg = total_loss / max(1, nbatches)
            print(f"[notMIWAE] Epoch {epoch}/{params.epochs}, loss={avg:.4f}")

    return model, params


def notmiwae_impute_numeric(
    trained: NotMIWAEModel,
    params: NotMIWAEParams,
    X: np.ndarray,
    L: int = 10,
) -> np.ndarray:
    """
    Impute missing entries in X using a trained notMIWAE model.

    X: numpy array (N, D) with np.nan for missing.
    Returns numpy array (N, D) with imputed values.
    """
    return trained.impute(X, L=L)

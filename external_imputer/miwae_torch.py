# external_imputer/miwae_torch.py

import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class MIWAEParams:
    latent_dim: int = 10       # dimension of z
    hidden_dim: int = 128      # hidden units in encoder/decoder
    iw_samples: int = 20       # K (importance samples) in the MIWAE bound
    epochs: int = 500          # training epochs
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MIWAEModel(nn.Module):
    """
    Simple MIWAE-style model with Student-t decoder.

    - Prior:      p(z) = N(0, I)
    - Encoder:    q(z|x_obs) = N(mu(x), diag(sigma^2(x)))
    - Decoder:    p(x|z) = StudentT(mu_x(z), scale_x(z), df_x(z)) (factorised over features)
    """

    def __init__(self, p: int, params: MIWAEParams):
        super().__init__()
        d = params.latent_dim
        h = params.hidden_dim
        self.p = p
        self.params = params

        # q(z | x)
        self.encoder = nn.Sequential(
            nn.Linear(p, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 2 * d),  # mean and log-variance
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 3 * p),  # mean, log-scale, log-df
        )



    # ------------------------------------------------------------------ #
    # Encoder / decoder helpers
    # ------------------------------------------------------------------ #
    def encode(self, x):
        out = self.encoder(x)
        d = self.params.latent_dim
        mu = out[..., :d]
        logvar = out[..., d:]
        return mu, logvar


    def decode(self, z):
        out = self.decoder(z)  # (..., 3p)
        p = self.p
        mean = out[..., :p]
        log_scale = out[..., p:2*p]
        log_df = out[..., 2*p:3*p]

        # ensure positivity / reasonable ranges
        scale = torch.nn.functional.softplus(log_scale) + 1e-3
        df = torch.nn.functional.softplus(log_df) + 3.0   # df > 3 (finite variance, etc.)

        return mean, scale, df

    # ------------------------------------------------------------------ #
    # MIWAE loss (IWAE bound with masking on observed entries)
    # ------------------------------------------------------------------ #
    def miwae_loss(self, x, mask):
        """
        x:    (batch, p) tensor, with *placeholder* values in missing positions
        mask: (batch, p) tensor, 1 = observed, 0 = missing
        """
        K = self.params.iw_samples
        batch_size, p = x.shape
        d = self.params.latent_dim
        device = x.device

        # q(z | x_obs)
        mu, logvar = self.encode(x)       # (batch, d)
        std = torch.exp(0.5 * logvar)     # (batch, d)

        # Sample K latents via reparameterisation: (K, batch, d)
        eps = torch.randn(K, batch_size, d, device=device)
        z = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (K, batch, d)

        # Prior p(z) = N(0, I)
        log_pz = -0.5 * torch.sum(z ** 2 + np.log(2 * np.pi), dim=-1)  # (K, batch)

        # Encoder density q(z | x)
        log_qz_x = -0.5 * torch.sum(
            ((z - mu.unsqueeze(0)) / std.unsqueeze(0)) ** 2
            + 2 * torch.log(std.unsqueeze(0))
            + np.log(2 * np.pi),
            dim=-1,
        )  # (K, batch)

        # Decode all z's
        z_flat = z.reshape(K * batch_size, d)
        mean_x, scale_x, df_x = self.decode(z_flat)  # (K*batch, p) each


        x_flat = x.repeat(K, 1)  # (K*batch, p)
        mask_flat = mask.repeat(K, 1)

        # build StudentT distribution per feature
        # We'll compute elementwise log_prob manually with StudentT:
        student = torch.distributions.StudentT(
            df=df_x,
            loc=mean_x,
            scale=scale_x
        )
        log_px_z_flat = student.log_prob(x_flat)  # (K*batch, p)

        log_px_z = torch.sum(log_px_z_flat * mask_flat, dim=-1)  # (K*batch,)
        log_px_z = log_px_z.view(K, batch_size)                 # (K, batch)

        # MIWAE / IWAE bound
        log_w = log_px_z + log_pz - log_qz_x           # (K, batch)
        log_w_max, _ = torch.max(log_w, dim=0, keepdim=True)  # stability
        w_normalised = torch.exp(log_w - log_w_max)    # (K, batch)
        log_avg_w = torch.log(torch.mean(w_normalised, dim=0)) + log_w_max.squeeze(0)

        loss = -torch.mean(log_avg_w)  # negative MIWAE bound
        return loss


# ---------------------------------------------------------------------- #
# Public training / imputation helpers (numeric-only)
# ---------------------------------------------------------------------- #
def train_miwae_numeric(
    X_np: np.ndarray,
    params: MIWAEParams | None = None,
    verbose: bool = True,
):
    """
    Train MIWAE on a *numeric-only* matrix with np.nan for missing entries.

    Returns a dict with:
      - "model" : trained MIWAEModel
      - "mean"  : per-column mean used for standardisation
      - "std"   : per-column std used for standardisation
    and the MIWAEParams object.
    """
    if params is None:
        params = MIWAEParams()
    device = params.device

    # mask and simple initial fill (column means)
    mask = ~np.isnan(X_np)
    col_means = np.nanmean(X_np, axis=0)
    X_filled = np.where(mask, X_np, col_means)

    # Standardise (so decoder roughly sees mean 0 / var 1)
    mean = np.mean(X_filled, axis=0)
    std = np.std(X_filled, axis=0)
    std[std == 0] = 1.0
    X_std = (X_filled - mean) / std

    X_tensor = torch.from_numpy(X_std.astype(np.float32)).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)

    n, p = X_np.shape
    model = MIWAEModel(p, params).to(device)
    optimiser = optim.Adam(model.parameters(), lr=params.learning_rate)

    n_batches = max(1, int(np.ceil(n / params.batch_size)))

    for epoch in range(params.epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * params.batch_size : (b + 1) * params.batch_size]
            xb = X_tensor[idx]
            mb = mask_tensor[idx]
            loss = model.miwae_loss(xb, mb)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"[MIWAE] Epoch {epoch+1}/{params.epochs}, loss={epoch_loss/n_batches:.4f}")

    model.eval()
    trained = {"model": model, "mean": mean, "std": std}
    return trained, params


def miwae_impute_numeric(
    trained: dict,
    params: MIWAEParams,
    X_np: np.ndarray,
    L: int = 10,
):
    """
    Impute missing values in a numeric-only matrix using a trained MIWAE model.

    - L: number of latent samples per row during imputation (averaged).
    """
    device = params.device
    model: MIWAEModel = trained["model"]
    mean = trained["mean"]
    std = trained["std"]

    mask = ~np.isnan(X_np)
    col_means = np.nanmean(X_np, axis=0)
    X_filled = np.where(mask, X_np, col_means)
    X_std = (X_filled - mean) / std

    X_tensor = torch.from_numpy(X_std.astype(np.float32)).to(device)
    n, p = X_np.shape

    with torch.no_grad():
        d = params.latent_dim
        K = L
        mu, logvar = model.encode(X_tensor)
        std_z = torch.exp(0.5 * logvar)
        eps = torch.randn(K, n, d, device=device)
        z = mu.unsqueeze(0) + std_z.unsqueeze(0) * eps  # (K, n, d)
        z_flat = z.reshape(K * n, d)
        mean_x, _, _ = model.decode(z_flat)
        mean_x = mean_x.view(K, n, p)
        x_rec = mean_x.mean(dim=0)  # (n, p)

        x_rec_np = x_rec.cpu().numpy()
        # undo standardisation
        x_rec_np = x_rec_np * std + mean

        # plug back observed values
        X_imputed = np.where(mask, X_np, x_rec_np)

    return X_imputed

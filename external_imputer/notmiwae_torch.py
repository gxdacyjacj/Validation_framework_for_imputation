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
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"

    # New: output distribution for p(x|z)
    # 'gauss'     -> diagonal Gaussian
    # 'studentt'  -> factorised Student-t
    # 'bern'      -> Bernoulli (for 0/1 features)
    out_dist: str = "gauss"

    # New: missingness process p(s|x)
    # 'selfmasking', 'selfmasking_known', 'linear', 'nonlinear'
    missing_process: str = "selfmasking"

    # New: encoder type
    # 'mlp'     -> your current encoder [x_imp, mask] -> MLP
    # 'perm_inv' -> permutation-invariant embedding as in the TF code
    encoder_type: str = "mlp"
    embedding_size: int = 20
    code_size: int = 20

    # New: KL / prior annealing hook (β schedule)
    # if kl_anneal_end_epoch > kl_anneal_start_epoch, we ramp β from 0->1
    kl_anneal_start_epoch: int = 0
    kl_anneal_end_epoch: int = 0


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

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder_type = params.encoder_type

        if self.encoder_type == "mlp":
            # Encoder: input = [x_imp, mask] -> h -> (mu, logvar)
            enc_in_dim = x_dim * 2
            self.encoder = nn.Sequential(
                nn.Linear(enc_in_dim, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
            )
            enc_out_dim = h

        elif self.encoder_type == "perm_inv":
            # Permutation-invariant embedding as in notMIWAE.permutation_invariant_embedding
            # E: (D, embedding_size)
            self.E = nn.Parameter(
                torch.randn(x_dim, params.embedding_size) * 0.01
            )
            # Maps [embedding; x_d] -> code_size
            self.perm_h = nn.Linear(params.embedding_size + 1, params.code_size)
            enc_out_dim = params.code_size

        else:
            raise ValueError(f"Unknown encoder_type {self.encoder_type}")

        self.enc_mu = nn.Linear(enc_out_dim, z_dim)
        self.enc_logvar = nn.Linear(enc_out_dim, z_dim)

        # -------------------------
        # Decoder core
        # -------------------------
        self.out_dist = params.out_dist

        self.decoder_core = nn.Sequential(
            nn.Linear(z_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        if self.out_dist == "gauss":
            self.dec_out = nn.Linear(h, 2 * x_dim)  # mean, logvar
        elif self.out_dist == "studentt":
            self.dec_out = nn.Linear(h, 3 * x_dim)  # mean, log_scale, log_df
        elif self.out_dist == "bern":
            self.dec_out = nn.Linear(h, x_dim)      # logits
        else:
            raise ValueError(f"Unknown out_dist {self.out_dist}")

        # -------------------------
        # Missingness process p(s|x)
        # -------------------------
        self.missing_process = params.missing_process

        if self.missing_process in ["selfmasking", "selfmasking_known"]:
            # elementwise a + b * x
            self.sm_a = nn.Parameter(torch.zeros(x_dim))
            self.sm_b = nn.Parameter(torch.zeros(x_dim))

        elif self.missing_process == "linear":
            # simple linear logits = Wx + b
            self.miss_linear = nn.Linear(x_dim, x_dim)

        elif self.missing_process == "nonlinear":
            self.miss_hid = nn.Linear(x_dim, h)
            self.miss_out = nn.Linear(h, x_dim)

        else:
            raise ValueError(
                "missing_process must be 'selfmasking', 'selfmasking_known', 'linear' or 'nonlinear'"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def encode(self, x_imp: torch.Tensor, mask: torch.Tensor):
        """
        x_imp: (B, D) with missing filled (e.g. zeros)
        mask : (B, D) with 1=observed, 0=missing
        """
        if self.encoder_type == "mlp":
            inp = torch.cat([x_imp, mask], dim=-1)  # (B, 2D)
            h = self.encoder(inp)                   # (B, h)
        else:  # perm_inv
            B, D = x_imp.shape
            # E: (D, embedding_size) -> broadcast to (B, D, embedding_size)
            E = self.E.unsqueeze(0).expand(B, D, -1)        # (B, D, E)
            # zero-out unobserved dims in E
            Es = mask.unsqueeze(-1) * E                     # (B, D, E)

            # concat embedding and x value: (B, D, E+1)
            Esx = torch.cat([Es, x_imp.unsqueeze(-1)], dim=-1)

            # flatten feature dimension: (B*D, E+1)
            Esxr = Esx.view(B * D, -1)

            # non-linear map h(s_d, x_d) -> (B*D, code_size)
            h_d = F.relu(self.perm_h(Esxr))

            # reshape back: (B, D, code_size)
            hr = h_d.view(B, D, -1)

            # zero-out unobserved dims again
            hz = mask.unsqueeze(-1) * hr                   # (B, D, code_size)

            # permutation-invariant aggregation (sum over feature dimension)
            h = hz.sum(dim=1)                              # (B, code_size)

        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar


    def decode(self, z: torch.Tensor):
        """
        z: (B, L, z_dim) or (B, z_dim)

        Returns:
        - if gauss:    (mean, logvar)
        - if studentt: (mean, scale, df)
        - if bern:     (logits,)
        """
        orig_shape = z.shape
        if z.dim() == 3:
            B, L, d = z.shape
            z_flat = z.view(B * L, d)
            reshape_back = True
        else:
            B, d = z.shape
            z_flat = z
            reshape_back = False

        h = self.decoder_core(z_flat)
        out = self.dec_out(h)

        D = self.x_dim

        if self.out_dist == "gauss":
            mean = out[:, :D]
            logvar = out[:, D:]
            if reshape_back:
                mean = mean.view(B, L, D)
                logvar = logvar.view(B, L, D)
            return mean, logvar

        elif self.out_dist == "studentt":
            mean = out[:, :D]
            log_scale = out[:, D:2*D]
            log_df = out[:, 2*D:]

            scale = F.softplus(log_scale) + 1e-3  # > 0
            df = F.softplus(log_df) + 3.0        # > 3, finite variance

            if reshape_back:
                mean = mean.view(B, L, D)
                scale = scale.view(B, L, D)
                df = df.view(B, L, D)
            return mean, scale, df

        else:  # bern
            logits = out
            if reshape_back:
                logits = logits.view(B, L, D)
            return logits,


    def _log_normal(self, x, mean, logvar):
        """
        Elementwise diagonal Gaussian log-density.
        x, mean, logvar: same shape (..., D)
        Returns log-density per dimension (no sum over D).
        """
        return -0.5 * (
            (x - mean) ** 2 / torch.exp(logvar) + logvar + np.log(2 * np.pi)
        )

    def _log_student_t(self, x, mean, scale, df):
        """
        Elementwise log-density of Student-t(df, loc=mean, scale).
        x, mean, scale, df: same shape (..., D)
        """
        # Using PyTorch's StudentT to avoid re-deriving formula
        dist = torch.distributions.StudentT(df=df, loc=mean, scale=scale)
        return dist.log_prob(x)

    def _log_bernoulli(self, x, logits):
        """
        Elementwise Bernoulli log-density given logits.
        x, logits: same shape (..., D), x in {0,1}
        """
        probs = torch.sigmoid(logits)
        eps = 1e-8
        return x * torch.log(probs + eps) + (1.0 - x) * torch.log(1.0 - probs + eps)

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

    def _logits_miss(self, x_mean):
        """
        Compute logits for p(s_missing=1 | x) for different missing_process.

        x_mean: (..., D) approximate x (decoder mean/sample)
        """
        if self.missing_process == "selfmasking":
            # logits = a + b * x
            return self.sm_a + self.sm_b * x_mean

        elif self.missing_process == "selfmasking_known":
            # constrain b >= 0 as in TF code (softplus)
            b_pos = F.softplus(self.sm_b)
            return self.sm_a + b_pos * x_mean

        elif self.missing_process == "linear":
            # logits = Wx + b (per-feature but can mix dims)
            return self.miss_linear(x_mean)

        elif self.missing_process == "nonlinear":
            h = torch.tanh(self.miss_hid(x_mean))
            return self.miss_out(h)

        else:
            raise RuntimeError("Invalid missing_process")

    def _log_p_s_given_x(self, s_missing, x_mean):
        """
        s_missing: (..., D) with 1 = missing, 0 = observed
        x_mean   : (..., D) proxy for x in p(s|x)
        Returns: summed log p(s | x) over D -> shape (...,)
        """
        logits = self._logits_miss(x_mean)
        p_missing = torch.sigmoid(logits)
        eps = 1e-8
        log_p = s_missing * torch.log(p_missing + eps) + \
                (1.0 - s_missing) * torch.log(1.0 - p_missing + eps)
        return torch.sum(log_p, dim=-1)


    # ------------------------------------------------------------------
    # Training loss: negative IWAE bound for MNAR
    # ------------------------------------------------------------------
    def iwae_loss(self, x, mask, L: int, beta: float = 1.0):
        """
        x:    (B, D) with NaNs already replaced (x_imp)
        mask: (B, D) 1=observed, 0=missing
        L:    number of importance samples
        beta: scaling for KL term (log_p_z - log_q_z_given_x); beta=1.0 gives
              the original notMIWAE objective; beta<1 can be used for KL annealing.

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

        # --------------------------------------------------------------
        # Decode and compute log p(x_obs | z) depending on out_dist
        # --------------------------------------------------------------
        if self.out_dist == "gauss":
            x_mean, x_logvar = self.decode(z)          # (B, L, D)
            mask_exp = mask.unsqueeze(1).expand_as(x_mean)
            x_obs = x.unsqueeze(1).expand_as(x_mean)
            log_px_elem = self._log_normal(x_obs, x_mean, x_logvar)  # (B, L, D)

        elif self.out_dist == "studentt":
            x_mean, x_scale, x_df = self.decode(z)     # (B, L, D)
            mask_exp = mask.unsqueeze(1).expand_as(x_mean)
            x_obs = x.unsqueeze(1).expand_as(x_mean)
            log_px_elem = self._log_student_t(x_obs, x_mean, x_scale, x_df)  # (B, L, D)

        elif self.out_dist == "bern":
            (logits,) = self.decode(z)                 # (B, L, D)
            mask_exp = mask.unsqueeze(1).expand_as(logits)
            x_obs = x.unsqueeze(1).expand_as(logits)
            log_px_elem = self._log_bernoulli(x_obs, logits)  # (B, L, D)

        else:
            raise RuntimeError(f"Unsupported out_dist: {self.out_dist}")

        # Zero-out contributions from missing entries, sum over D
        log_p_x_given_z = torch.sum(log_px_elem * mask_exp, dim=-1)  # (B, L)

        # --------------------------------------------------------------
        # Log p(s | x): use decoder mean/logits as proxy for x
        # --------------------------------------------------------------
        # We always use a continuous proxy x_mean for the missingness model.
        # For Bernoulli x, logits are transformed to probs in _log_p_s_given_x.
        if self.out_dist == "bern":
            # For p(s|x) it's reasonable to use sigmoid(logits) as "x_mean"
            x_proxy = torch.sigmoid(logits)
        else:
            x_proxy = x_mean

        s_missing_exp = s_missing.unsqueeze(1).expand_as(x_proxy)
        log_p_s_given_x = self._log_p_s_given_x(s_missing_exp, x_proxy)  # (B, L)

        # Log p(z)
        log_p_z = self._log_p_z(z)  # (B, L)

        # Log q(z|x)
        log_q_z_given_x = self._log_q_z_given_x(
            z, mu.unsqueeze(1).expand_as(z), logvar.unsqueeze(1).expand_as(z)
        )  # (B, L)

        # Importance weights
        # Original notMIWAE: log_w = log_p_x + log_p_s + log_p_z - log_q_z|x
        # Here we add beta on the KL term for optional KL annealing.
        log_w = (
            log_p_x_given_z
            + log_p_s_given_x
            + beta * (log_p_z - log_q_z_given_x)
        )  # (B, L)

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

        For Gaussian / Student-t decoders we return the mean of p(x|z).
        For Bernoulli decoders we return sigmoid(logits) (i.e. probabilities),
        which you can threshold externally if you need hard 0/1.
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

            # ----------------------------------------------------------
            # Decode and log p(x_obs | z) for importance weights
            # ----------------------------------------------------------
            if self.out_dist == "gauss":
                x_mean, x_logvar = self.decode(z)  # (N, L, D)
                mask_exp = M_t.unsqueeze(1).expand_as(x_mean)
                x_obs = X_t.unsqueeze(1).expand_as(x_mean)
                log_px_elem = self._log_normal(x_obs, x_mean, x_logvar)  # (N, L, D)
                x_recon = x_mean

            elif self.out_dist == "studentt":
                x_mean, x_scale, x_df = self.decode(z)  # (N, L, D)
                mask_exp = M_t.unsqueeze(1).expand_as(x_mean)
                x_obs = X_t.unsqueeze(1).expand_as(x_mean)
                log_px_elem = self._log_student_t(x_obs, x_mean, x_scale, x_df)  # (N, L, D)
                x_recon = x_mean

            elif self.out_dist == "bern":
                (logits,) = self.decode(z)  # (N, L, D)
                mask_exp = M_t.unsqueeze(1).expand_as(logits)
                x_obs = X_t.unsqueeze(1).expand_as(logits)
                log_px_elem = self._log_bernoulli(x_obs, logits)  # (N, L, D)
                x_recon = torch.sigmoid(logits)

            else:
                raise RuntimeError(f"Unsupported out_dist: {self.out_dist}")

            log_p_x_given_z = torch.sum(log_px_elem * mask_exp, dim=-1)  # (N, L)

            # Self-masking log p(s|x)
            s_missing = 1.0 - M_t
            if self.out_dist == "bern":
                x_proxy = x_recon
            else:
                x_proxy = x_recon  # = x_mean
            s_missing_exp = s_missing.unsqueeze(1).expand_as(x_proxy)
            log_p_s_given_x = self._log_p_s_given_x(s_missing_exp, x_proxy)  # (N, L)

            # Log p(z)
            log_p_z = self._log_p_z(z)  # (N, L)

            # Log q(z|x)
            log_q_z_given_x = self._log_q_z_given_x(
                z, mu.unsqueeze(1).expand_as(z), logvar.unsqueeze(1).expand_as(z)
            )

            # Importance weights (notMIWAE version)
            log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x  # (N, L)

            # Softmax over importance weights
            max_w, _ = torch.max(log_w, dim=1, keepdim=True)
            w = torch.exp(log_w - max_w)
            w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)  # (N, L)

            # Posterior mean of x under p(x|z) and weights w: sum_k w_k * E[x|z_k]
            xm = torch.sum(w[:, :, None] * x_recon, dim=1)  # (N, D)

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
    X_val: np.ndarray | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = False,
) -> Tuple[NotMIWAEModel, NotMIWAEParams]:
    """
    Train NotMIWAE on numeric data with NaNs.

    If X_val is not None, we track validation loss and (optionally) save
    the best model to checkpoint_path using torch.save(state_dict).
    """
    device = params.device
    X = np.asarray(X, float)
    N, D = X.shape

    # Train data mask and zero-imputation
    mask = (~np.isnan(X)).astype(np.float32)
    X_imp = np.where(np.isnan(X), 0.0, X).astype(np.float32)

    X_t = torch.from_numpy(X_imp)
    M_t = torch.from_numpy(mask)

    if X_val is not None:
        X_val = np.asarray(X_val, float)
        mask_val = (~np.isnan(X_val)).astype(np.float32)
        X_val_imp = np.where(np.isnan(X_val), 0.0, X_val).astype(np.float32)
        Xv_t = torch.from_numpy(X_val_imp)
        Mv_t = torch.from_numpy(mask_val)
    else:
        Xv_t = Mv_t = None

    model = NotMIWAEModel(x_dim=D, params=params).to(device)
    X_t = _to_device(X_t, device)
    M_t = _to_device(M_t, device)
    if Xv_t is not None:
        Xv_t = _to_device(Xv_t, device)
        Mv_t = _to_device(Mv_t, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    batch_size = params.batch_size
    iw_samples = params.iw_samples

    best_val = float("inf")
    best_state = None

    for epoch in range(1, params.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        nbatches = 0

        # KL annealing beta schedule
        if params.kl_anneal_end_epoch > params.kl_anneal_start_epoch:
            if epoch < params.kl_anneal_start_epoch:
                beta = 0.0
            elif epoch > params.kl_anneal_end_epoch:
                beta = 1.0
            else:
                t = (epoch - params.kl_anneal_start_epoch) / (
                    params.kl_anneal_end_epoch - params.kl_anneal_start_epoch
                )
                beta = float(t)
        else:
            beta = 1.0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_t[idx]
            mb = M_t[idx]

            loss = model.iwae_loss(xb, mb, L=iw_samples, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            nbatches += 1

        avg_train = total_loss / max(1, nbatches)

        # Validation
        if Xv_t is not None:
            model.eval()
            with torch.no_grad():
                val_loss = model.iwae_loss(Xv_t, Mv_t, L=iw_samples, beta=1.0).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                if checkpoint_path is not None:
                    torch.save(best_state, checkpoint_path)

            if verbose:
                print(
                    f"[notMIWAE] Epoch {epoch}/{params.epochs}, "
                    f"train={avg_train:.4f}, val={val_loss:.4f}, beta={beta:.3f}"
                )
        else:
            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == params.epochs):
                print(
                    f"[notMIWAE] Epoch {epoch}/{params.epochs}, "
                    f"train={avg_train:.4f}, beta={beta:.3f}"
                )

    # Load best checkpoint if we used validation
    if best_state is not None:
        model.load_state_dict(best_state)

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

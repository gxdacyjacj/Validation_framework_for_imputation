# hi_vae_torch.py
"""
PyTorch implementation of a simplified HI-VAE ("HI-VAE-lite") for mixed
continuous + categorical tabular data with missing values.

Design choices / simplifications
--------------------------------
- Single Gaussian latent z ~ N(0, I) (no discrete s mixture).
- Per-feature heterogeneous likelihoods:
    * continuous  : Gaussian
    * categorical : Categorical (softmax)
- Missing values indicated by np.nan in the input X.
- Categorical features are assumed to be integer-coded 0..K-1.
- For now we do not implement ordinal/count types (can be added later).

Public API
----------
- HIVAEParams          : hyperparameters (latent_dim, hidden_dim, iw_samples, ...)
- train_hivae_mixed    : fit HI-VAE-lite on a numpy array (mixed types)
- hivae_impute_mixed   : impute missing values using a trained model

These are meant to be called from validation_v2 via an ExternalImputer wrapper,
similar to MIWAE / notMIWAE / GAIN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class HIVAEParams:
    latent_dim: int = 10
    hidden_dim: int = 128
    iw_samples: int = 20         # K in importance-weighted ELBO
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    max_categories: int = 20     # for auto-detection of categoricals
    seed: int = 0


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class HIVAEModel(nn.Module):
    """
    Encoder-decoder VAE for mixed continuous + categorical features.

    - Encoder input: [X_enc, M] where X_enc is normalized numeric matrix and
      M is the binary mask (1=observed, 0=missing).
    - Latent z ~ N(mu, diag(exp(logvar))).
    - Decoder outputs:
        * cont_mean, cont_logvar for continuous features
        * a list of logits tensors for categorical features
    """

    def __init__(
        self,
        n_features: int,
        cont_indices: List[int],
        cat_indices: List[int],
        cat_n_classes: List[int],
        latent_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.n_features = n_features
        self.cont_indices = cont_indices
        self.cat_indices = cat_indices
        self.cat_n_classes = cat_n_classes

        self.n_cont = len(cont_indices)
        self.n_cat = len(cat_indices)
        self.latent_dim = latent_dim

        enc_input_dim = 2 * n_features  # X_enc + mask

        # Encoder
        self.enc_fc1 = nn.Linear(enc_input_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.n_cont > 0:
            self.dec_cont_mean = nn.Linear(hidden_dim, self.n_cont)
            self.dec_cont_logvar = nn.Linear(hidden_dim, self.n_cont)

        # one head per categorical feature
        self.dec_cat_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, n_cls) for n_cls in self.cat_n_classes]
        )

    # ----- core operations -------------------------------------------------

    def encode(self, x_enc: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x_enc, mask], dim=1)
        h = F.relu(self.enc_fc1(inp))
        h = F.relu(self.enc_fc2(h))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(
        self, z: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[torch.Tensor]]:
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))

        cont_mean = cont_logvar = None
        if self.n_cont > 0:
            cont_mean = self.dec_cont_mean(h)
            cont_logvar = self.dec_cont_logvar(h)

        cat_logits: List[torch.Tensor] = []
        for layer in self.dec_cat_layers:
            cat_logits.append(layer(h))

        return cont_mean, cont_logvar, cat_logits


# ---------------------------------------------------------------------------
# Utilities: feature-type inference & normalization
# ---------------------------------------------------------------------------

def _infer_feature_types(
    X: np.ndarray,
    max_categories: int,
) -> List[Dict[str, Any]]:
    """
    Infer feature types from numeric array X (assumed float / int).

    Rule:
      - if a column has at least 2 unique non-NaN values AND
        all unique values are (approximately) integers AND
        number of unique values <= max_categories
        -> treat as categorical with n_classes = n_unique.
      - otherwise treat as continuous.

    NOTE: assumes categorical values are already integer-coded.
    """
    n, d = X.shape
    types: List[Dict[str, Any]] = []

    for j in range(d):
        col = X[:, j]
        mask = ~np.isnan(col)
        col_obs = col[mask]
        if col_obs.size == 0:
            # degenerate column: treat as continuous; stats handled later
            types.append({"type": "continuous"})
            continue

        uniq = np.unique(col_obs)
        # check integer-ness
        is_int_like = np.allclose(uniq, np.round(uniq))
        if is_int_like and 2 <= uniq.size <= max_categories:
            n_classes = int(uniq.max()) + 1
            types.append(
                {"type": "categorical", "n_classes": n_classes}
            )
        else:
            types.append({"type": "continuous"})

    return types


def _build_feature_meta(
    X: np.ndarray,
    types: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build metadata: indices of continuous / categorical features, normalisation
    statistics, etc. Missing values in X are assumed to be np.nan.
    """
    n, d = X.shape
    cont_indices: List[int] = []
    cat_indices: List[int] = []
    cat_n_classes: List[int] = []
    cont_means: List[float] = []
    cont_stds: List[float] = []

    for j, t in enumerate(types):
        col = X[:, j]
        mask = ~np.isnan(col)
        col_obs = col[mask]

        if t["type"] == "categorical":
            cat_indices.append(j)
            cat_n_classes.append(int(t["n_classes"]))
        else:
            cont_indices.append(j)
            if col_obs.size == 0:
                m = 0.0
                s = 1.0
            else:
                m = float(col_obs.mean())
                s = float(col_obs.std())
                if s < 1e-6:
                    s = 1.0
            cont_means.append(m)
            cont_stds.append(s)

    feature_meta = {
        "types": types,
        "cont_indices": cont_indices,
        "cat_indices": cat_indices,
        "cat_n_classes": cat_n_classes,
        "cont_means": np.array(cont_means, dtype=np.float32),
        "cont_stds": np.array(cont_stds, dtype=np.float32),
    }
    return feature_meta


def _preprocess_X(
    X: np.ndarray,
    feature_meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given raw X with np.nan for missing, and feature_meta, return:

      - X_proc : zero-filled numeric matrix (for likelihood targets)
      - X_enc  : normalized version for encoder input
      - M      : binary mask (1=observed, 0=missing)

    Shapes: all (n, d).
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape

    M = (~np.isnan(X)).astype(np.float32)

    # zero-fill for internal use
    X_proc = X.copy()
    X_proc[~np.isfinite(X_proc)] = 0.0  # np.nan or inf -> 0

    types = feature_meta["types"]
    cont_idx = feature_meta["cont_indices"]
    cat_idx = feature_meta["cat_indices"]
    cont_means = feature_meta["cont_means"]
    cont_stds = feature_meta["cont_stds"]
    cat_n_classes = feature_meta["cat_n_classes"]

    X_enc = np.zeros_like(X_proc, dtype=np.float32)

    # continuous: z-score
    for k, j in enumerate(cont_idx):
        m = cont_means[k]
        s = cont_stds[k]
        X_enc[:, j] = (X_proc[:, j] - m) / s

    # categorical: scale to [0,1] by dividing by (K-1)
    for k, j in enumerate(cat_idx):
        K = max(1, cat_n_classes[k] - 1)
        X_enc[:, j] = X_proc[:, j] / float(K)

    # for missing entries, encoder sees 0 (and mask=0)
    X_enc[M == 0.0] = 0.0

    return X_proc, X_enc, M.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hivae_mixed(
    X: np.ndarray,
    params: HIVAEParams,
    feature_types: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], HIVAEParams]:
    """
    Train HI-VAE-lite on a (possibly incomplete) numeric array X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Mixed continuous + categorical; missing entries as np.nan.
        Categorical features must be integer-coded 0..K-1.
    params : HIVAEParams
        Hyperparameters (latent_dim, hidden_dim, iw_samples, epochs, ...)
    feature_types : optional list
        If provided, a list of dicts of length D, each with keys:
            - "type": "continuous" or "categorical"
            - if categorical: "n_classes": int
        If None, feature types are inferred automatically.

    Returns
    -------
    trained : dict
        Contains the trained model & feature metadata.
    params : HIVAEParams
        Same object (for API consistency with other external imputers).
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(
            "HIVAE-lite expects a numeric array. Please encode categoricals "
            "as integers 0..K-1 before calling train_hivae_mixed."
        )

    n, d = X.shape

    if feature_types is None:
        feature_types = _infer_feature_types(X, max_categories=params.max_categories)

    feature_meta = _build_feature_meta(X, feature_types)

    X_proc, X_enc, M = _preprocess_X(X, feature_meta)

    device = torch.device(params.device)
    model = HIVAEModel(
        n_features=d,
        cont_indices=feature_meta["cont_indices"],
        cat_indices=feature_meta["cat_indices"],
        cat_n_classes=feature_meta["cat_n_classes"],
        latent_dim=params.latent_dim,
        hidden_dim=params.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )

    batch_size = params.batch_size
    iw_samples = max(1, params.iw_samples)

    X_proc_t = torch.from_numpy(X_proc).to(device)
    X_enc_t = torch.from_numpy(X_enc).to(device)
    M_t = torch.from_numpy(M).to(device)

    n_batches = int(np.ceil(n / batch_size))
    log2pi = float(np.log(2.0 * np.pi))

    for epoch in range(params.epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = X_proc_t[idx]
            xencb = X_enc_t[idx]
            mb = M_t[idx]

            optimizer.zero_grad()

            mu, logvar = model.encode(xencb, mb)

            # log-weights for IWAE: shape (B, K)
            log_w_list = []

            for _ in range(iw_samples):
                z = model.reparameterize(mu, logvar)
                cont_mean, cont_logvar, cat_logits = model.decode(z)

                # log p(x | z) over observed entries only
                log_px = torch.zeros(z.size(0), device=device)

                # continuous part
                if model.n_cont > 0:
                    cont_idx = feature_meta["cont_indices"]
                    xb_cont = xb[:, cont_idx]
                    mb_cont = mb[:, cont_idx]

                    cm = cont_mean
                    clv = cont_logvar
                    var = torch.exp(clv) + 1e-6

                    # Gaussian log-likelihood per dim
                    log_probs = -0.5 * (
                        log2pi + clv + (xb_cont - cm) ** 2 / var
                    )
                    # sum over dims, mask missing
                    log_px_cont = torch.sum(log_probs * mb_cont, dim=1)
                    log_px = log_px + log_px_cont

                # categorical part
                for f, j in enumerate(feature_meta["cat_indices"]):
                    logits_f = cat_logits[f]              # (B, K_f)
                    mb_f = mb[:, j]                      # (B,)
                    xb_f = xb[:, j]                      # (B,)

                    # set targets for observed entries; missing targets will be ignored
                    xb_f_clean = xb_f.clone()
                    xb_f_clean[mb_f < 0.5] = 0.0
                    targets = xb_f_clean.long()

                    log_probs_f = F.log_softmax(logits_f, dim=1)
                    gathered = log_probs_f[torch.arange(z.size(0), device=device), targets]
                    log_px_cat_f = gathered * mb_f
                    log_px = log_px + log_px_cat_f

                # log p(z) and log q(z|x)
                log_pz = -0.5 * torch.sum(z ** 2 + log2pi, dim=1)
                # diagonal Gaussian q(z|x)
                log_qz = -0.5 * torch.sum(
                    ((z - mu) ** 2) / torch.exp(logvar) + logvar + log2pi, dim=1
                )

                log_w = log_px + log_pz - log_qz
                log_w_list.append(log_w)

            log_w_stack = torch.stack(log_w_list, dim=1)  # (B, K)
            # IWAE objective: log(1/K * sum_k exp(log w_k))
            m_max, _ = torch.max(log_w_stack, dim=1, keepdim=True)
            w_rel = torch.exp(log_w_stack - m_max)
            log_mean_w = m_max.squeeze(1) + torch.log(torch.mean(w_rel, dim=1)) - np.log(
                iw_samples
            )
            loss = -torch.mean(log_mean_w)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        if verbose and ((epoch + 1) % max(1, params.epochs // 10) == 0):
            print(f"[HI-VAE-lite] Epoch {epoch+1}/{params.epochs}, loss={epoch_loss/n:.4f}")

    trained = {
        "model": model,
        "feature_meta": feature_meta,
    }
    return trained, params


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

@torch.no_grad()
def hivae_impute_mixed(
    trained: Dict[str, Any],
    params: HIVAEParams,
    X: np.ndarray,
    L: int = 10,
) -> np.ndarray:
    """
    Impute missing values in X using a trained HI-VAE-lite model.

    Parameters
    ----------
    trained : dict
        Output of train_hivae_mixed: contains 'model' and 'feature_meta'.
    params : HIVAEParams
        Same hyperparameter object used in training (needed for device).
    X : np.ndarray, shape (n_samples, n_features)
        Incomplete data with np.nan for missing entries. Must have the same
        column order and coding as the training data.
    L : int, default=10
        Number of latent samples used to Monte Carlo-average predictions.

    Returns
    -------
    X_imp : np.ndarray
        Imputed data (same shape as X). Observed entries are preserved.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    feature_meta = trained["feature_meta"]
    model: HIVAEModel = trained["model"]
    model.eval()

    device = torch.device(params.device)
    model = model.to(device)

    X_proc, X_enc, M = _preprocess_X(X, feature_meta)
    X_proc_t = torch.from_numpy(X_proc).to(device)
    X_enc_t = torch.from_numpy(X_enc).to(device)
    M_t = torch.from_numpy(M).to(device)

    n, d = X.shape

    # encode once
    mu, logvar = model.encode(X_enc_t, M_t)

    # accumulators for predictions
    if model.n_cont > 0:
        cont_idx = feature_meta["cont_indices"]
        pred_cont_sum = torch.zeros(n, model.n_cont, device=device)
    else:
        cont_idx = []
        pred_cont_sum = None

    pred_cat_logits_sum: List[torch.Tensor] = [
        torch.zeros(n, K, device=device) for K in feature_meta["cat_n_classes"]
    ]

    for _ in range(max(1, L)):
        z = model.reparameterize(mu, logvar)
        cont_mean, _, cat_logits = model.decode(z)

        if model.n_cont > 0:
            pred_cont_sum += cont_mean

        for i, logits in enumerate(cat_logits):
            pred_cat_logits_sum[i] += logits

    L_eff = float(max(1, L))

    X_imp = X.copy().astype(np.float32)
    M_bool = M.astype(bool)

    # continuous: fill missing with mean over samples, un-normalised
    if model.n_cont > 0:
        cont_means = feature_meta["cont_means"]
        cont_stds = feature_meta["cont_stds"]
        cont_mean_avg = (pred_cont_sum / L_eff).cpu().numpy()

        for k, j in enumerate(cont_idx):
            # reverse z-score
            pred_vals = cont_mean_avg[:, k] * cont_stds[k] + cont_means[k]
            mask_missing = ~M_bool[:, j]
            X_imp[mask_missing, j] = pred_vals[mask_missing]

    # categorical: pick argmax over average softmax
    for f, j in enumerate(feature_meta["cat_indices"]):
        logits_avg = (pred_cat_logits_sum[f] / L_eff)
        probs = F.softmax(logits_avg, dim=1)
        pred_classes = torch.argmax(probs, dim=1).cpu().numpy().astype(np.float32)
        mask_missing = ~M_bool[:, j]
        X_imp[mask_missing, j] = pred_classes[mask_missing]

    return X_imp

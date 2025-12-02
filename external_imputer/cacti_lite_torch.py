"""
CACTI-lite: A lightweight CACTI-style masked autoencoder for tabular imputation.

Key ideas kept from CACTI:
- Features as tokens (Transformer encoder over columns).
- Self-supervised masked reconstruction objective.
- Copy masking: training masks copied from empirical mask patterns.
- Learnable per-feature context embeddings.

Simplifications vs full CACTI:
- No median truncation (MT-CM) yet.
- No text-based context encoder (just learnable embeddings per feature).
- Single TransformerEncoder (no separate decoder).
- Numeric features only (no categorical handling).

Public API:
- CACTILiteParams
- train_cacti_lite_numeric(data_x, params, verbose=False)
- cacti_lite_impute_numeric(model, norm_params, data_x, params=None)
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
class CACTILiteParams:
    d_model: int = 64           # token/hidden dimension
    n_heads: int = 4            # attention heads
    n_layers: int = 2           # transformer encoder layers
    dim_ff: int = 256           # feedforward dimension
    batch_size: int = 128
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
# Helpers
# ---------------------------------------------------------------------
def _sample_batch_index(total: int, batch_size: int) -> np.ndarray:
    total_idx = np.random.permutation(total)
    return total_idx[:batch_size]


# ---------------------------------------------------------------------
# Model: CACTI-lite masked autoencoder
# ---------------------------------------------------------------------
class CactiLiteMAE(nn.Module):
    """
    CACTI-lite masked autoencoder over features.

    Inputs:
      - x: (B, D) normalized values (zeros for NaNs/masked positions)
      - mask_in: (B, D) binary mask for encoder input (1=show value, 0=mask token)

    Outputs:
      - preds: (B, D) predicted normalized values for all features.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature context embedding (simplified context)
        self.context_emb = nn.Embedding(n_features, d_model)

        # Scalar value -> d_model embedding
        self.value_proj = nn.Linear(1, d_model)

        # Learned mask token (used when mask_in == 0)
        self.mask_token = nn.Parameter(torch.zeros(d_model))

        # Transformer encoder over tokens (features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: predict scalar value per feature
        self.out_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        if self.value_proj.bias is not None:
            nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask_in: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) normalized values (zeros where NaN originally)
        mask_in: (B, D) 1 = show value, 0 = mask token (for encoder input)

        Returns:
          preds: (B, D) normalized value predictions for all features.
        """
        B, D = x.shape
        device = x.device

        # Scalar -> embedding
        x_val = x.unsqueeze(-1)                    # (B, D, 1)
        val_emb = self.value_proj(x_val)           # (B, D, d_model)

        # Mask token embedding
        mask_tok = self.mask_token.view(1, 1, -1).to(device)   # (1, 1, d_model)
        mask_tok = mask_tok.expand(B, D, self.d_model)         # (B, D, d_model)

        mask_bool = mask_in.bool().unsqueeze(-1)               # (B, D, 1)
        token_val = torch.where(mask_bool, val_emb, mask_tok)  # (B, D, d_model)

        # Context embedding per feature index
        feat_idx = torch.arange(D, device=device).view(1, D)
        feat_idx = feat_idx.expand(B, D)                       # (B, D)
        ctx_emb = self.context_emb(feat_idx)                   # (B, D, d_model)

        tokens = token_val + ctx_emb                           # (B, D, d_model)

        # Transformer encoder: features as tokens
        h = self.encoder(tokens)                               # (B, D, d_model)

        # Predict scalar per token
        preds = self.out_proj(h).squeeze(-1)                   # (B, D)
        return preds


# ---------------------------------------------------------------------
# Training: CACTI-lite with copy masking
# ---------------------------------------------------------------------
def train_cacti_lite_numeric(
    data_x: np.ndarray,
    params: CACTILiteParams,
    verbose: bool = False,
) -> Tuple[CactiLiteMAE, Dict[str, np.ndarray]]:
    """
    Train CACTI-lite on numeric data with NaNs.

    Args:
        data_x: (N, D) array with np.nan for missing.
        params: CACTILiteParams.
        verbose: print every ~1000 iterations.

    Returns:
        model: trained CactiLiteMAE.
        norm_params: dict with min/max for renormalization.
    """
    device = params.device
    X = np.asarray(data_x, float)
    N, D = X.shape

    # True mask: 1 if observed, 0 if missing
    M_true = 1.0 - np.isnan(X)

    # Normalize
    norm_data, norm_params = _normalization(X)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    model = CactiLiteMAE(
        n_features=D,
        d_model=params.d_model,
        n_heads=params.n_heads,
        n_layers=params.n_layers,
        dim_ff=params.dim_ff,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    batch_size = params.batch_size
    iters = params.iterations
    eps = 1e-8

    for it in range(iters):
        # Sample batch
        batch_idx = _sample_batch_index(N, batch_size)
        X_mb = norm_data_x[batch_idx, :]   # (B, D)
        M_true_mb = M_true[batch_idx, :]  # (B, D)

        B = X_mb.shape[0]

        # Copy masking: for each i, copy mask from some j
        donor_idx = np.random.randint(0, N, size=B)
        M_train_mb = M_true[donor_idx, :]  # (B, D) training mask

        # Encoder input mask_in = M_train_mb (1=show value, 0=mask)
        mask_in = M_train_mb

        # Supervision mask: where original is observed AND we artificially mask
        sup_mask = (M_true_mb == 1.0) & (mask_in == 0.0)  # (B, D)

        if np.sum(sup_mask) == 0:
            # Nothing to learn from this batch; skip
            continue

        X_t = torch.from_numpy(X_mb).float().to(device)
        mask_in_t = torch.from_numpy(mask_in).float().to(device)
        sup_mask_t = torch.from_numpy(sup_mask.astype(float)).float().to(device)

        model.train()
        preds = model(X_t, mask_in_t)  # (B, D)

        # MSE only on supervised positions
        diff2 = (preds - X_t) ** 2
        num = torch.sum(diff2 * sup_mask_t)
        den = torch.sum(sup_mask_t) + eps
        loss = num / den

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and ((it + 1) % 1000 == 0 or it == 0 or it + 1 == iters):
            print(f"[CACTI-lite] Iter {it+1}/{iters}, loss={loss.item():.4f}")

    return model, norm_params


# ---------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------
def cacti_lite_impute_numeric(
    model: CactiLiteMAE,
    norm_params: Dict[str, np.ndarray],
    data_x: np.ndarray,
    params: Optional[CACTILiteParams] = None,
) -> np.ndarray:
    """
    Impute missing values in data_x using a trained CACTI-lite model.

    Args:
        model: trained CactiLiteMAE.
        norm_params: dict from training (_normalization).
        data_x: (N, D) array with np.nan for missing.
        params: optional CACTILiteParams to specify device.

    Returns:
        imputed_data: (N, D) array with imputed values.
    """
    if params is None:
        device = "cpu"
    else:
        device = params.device

    X = np.asarray(data_x, float)
    N, D = X.shape

    # True mask
    M_true = 1.0 - np.isnan(X)

    # Normalize using training params
    norm_data, _ = _normalization(X, norm_params)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0)

    # For inference, mask_in = M_true (we show observed, mask missing)
    mask_in = M_true

    X_t = torch.from_numpy(norm_data_x).float().to(device)
    mask_in_t = torch.from_numpy(mask_in).float().to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X_t, mask_in_t).cpu().numpy()  # normalized predictions

    # Combine: keep observed, fill missing with preds
    imputed_norm = M_true * norm_data_x + (1.0 - M_true) * preds

    # Renormalize
    imputed_data = _renormalization(imputed_norm, norm_params)

    # Rounding for quasi-categorical columns
    imputed_data = _rounding(imputed_data, data_x)

    return imputed_data

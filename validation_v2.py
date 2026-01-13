#%%
"""
validation_v2.py

Framework for validating imputation methods under different missing-data
mechanisms (MCAR, MAR, MNAR and paired variants) and different evaluation
criteria (direct error, downstream task, and pattern-based validation).

This script was originally developed for the experiments in the paper, so it
contains:
    1. missingness generators (MCAR / MAR / MNAR + paired versions)
    2. a mixed-type imputer wrapper (handles numeric + categorical)
    3. three validation criteria
    4. an experiment runner that splits data into train/val/test

To use it as a library:
    - import the functions/classes you need
    - provide your own pandas DataFrame X and target y
    - call `generate_evaluation_results(...)`

To reproduce the paper experiments, see the example in the
`if __name__ == "__main__":` block at the end.
"""

import os

# ---------------------------------------------------------------------------
# Limit BLAS/OMP threads so that parallel evaluations don't oversubscribe.
# This is helpful on HPC or when using joblib. Adjust/remove if not needed.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Standard libraries
# ---------------------------------------------------------------------------
import re
import pickle
import warnings
import random  # used in some earlier versions, can be removed if unused
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    root_mean_squared_error,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# models for downstream evaluation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# some imputers rely on this
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# optional external libraries used in the original experiments
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostError


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
    module="sklearn.utils.validation",
)

# -----------------------------------------------------------------------------
# Public API (exported symbols)
# -----------------------------------------------------------------------------
__all__ = [
    # Missingness generators
    "generate_MCAR",
    "generate_MAR",
    "generate_MNAR",
    "generate_paired_missingness",
    # Validators
    "validate_baseline_direct_error",
    "validate_proxy_complete_case",
    "validate_downstream_performance",
    # Imputers and wrappers
    "MixedImputer",
    "StandardizeBeforeImpute",
    "wrap_imputers_with_standardize",
    "build_imputers",
    # Experiments
    "run_one_dataset",
    "split_train_val_test",
    # Utilities (new)
    "build_processed",
    "flatten_results",
]
# ---------------------------------------------------------------------------
# Default model list used in the paper (can be overridden by the user)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_LIST = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "SVM": SVR(),
    "Neural Network": MLPRegressor(),
}

# ---------------------------------------------------------------------------
# NOTE ABOUT DATA LOADING
# ---------------------------------------------------------------------------
# In the original version, UCI datasets were fetched immediately on import:
#   from ucimlrepo import fetch_ucirepo
#   ...
# That makes the module less reusable and also requires internet.
# For the GitHub version we move dataset fetching into __main__ so that
# `import validation_v2` does NOT download anything.
# ---------------------------------------------------------------------------
# =============================================================================
# Missingness Generators
# =============================================================================
"""
The following functions implement the six masking procedures described in
Section 2.3 of the paper *“Decision-Making Criteria on Choosing Appropriate
Imputation Methods for Incomplete Dataset Prepared for Machine Learning”*.

Each function generates artificial missingness under a specific mechanism or
pattern, enabling controlled evaluation of imputation methods. Two levels of
concepts are represented:

1. **Mechanism** – governs *why* data become missing:
       - MCAR  (Missing Completely at Random)
       - MAR   (Missing At Random)
       - MNAR  (Missing Not At Random)

2. **Pattern** – governs *where* missingness occurs:
       - Unstructured pattern: independent missing entries
       - Structured pattern: paired or block-wise missingness

All functions take both the raw (unencoded) dataset and the preprocessed
(one-hot encoded) dataset as inputs so that missingness can be applied
consistently across numerical and categorical variables.
"""

import re
import numpy as np
import scipy.special


# -----------------------------------------------------------------------------
# MCAR — Missing Completely at Random
# -----------------------------------------------------------------------------
def generate_MCAR(X, X_processed, missing_rate, independent_feature_idx=None):
    """
    Generate unstructured missingness under the MCAR mechanism.

    Definition (Rubin, 1976):
        The probability of a value being missing is independent of both
        observed and unobserved data:  P(M | X_obs, X_mis) = P(M).

    Implementation summary:
        - A Bernoulli mask is sampled for each feature independently.
        - For categorical variables (expanded by one-hot encoding),
          all dummy columns of the same variable share the same mask
          to preserve internal consistency.

    Parameters
    ----------
    X : pandas.DataFrame
        Original data before encoding.
    X_processed : pandas.DataFrame
        One-hot encoded version of X.
    missing_rate : array-like of float
        Target missing rate per original feature.
    independent_feature_idx : any, optional
        Not used in MCAR (kept for interface consistency).

    Returns
    -------
    X_masked, Xp_masked, independent_feature_idx
        DataFrames with NaNs inserted.
    """
    n_samples, n_features = X.shape
    mask = np.zeros_like(X, dtype=bool)
    mask_p = np.zeros_like(X_processed, dtype=bool)

    for i, feat in enumerate(X.columns):
        rate = missing_rate[i]

        if X[feat].dtype == "object":  # categorical
            pattern = re.compile(rf"^{re.escape(feat)}(_.+)?$")
            cols = [c for c in X_processed.columns if pattern.match(c)]
            idx = [X_processed.columns.get_loc(c) for c in cols]
            feat_mask = np.random.random(n_samples) < rate
            mask_p[:, idx] = feat_mask[:, None]
            mask[:, i] = feat_mask
        else:  # numeric
            feat_mask = np.random.random(n_samples) < rate
            idx = X_processed.columns.get_loc(feat)
            mask_p[:, idx] = feat_mask
            mask[:, i] = feat_mask

    Xm = X.copy()
    Xm[mask] = np.nan
    Xpm = X_processed.copy()
    Xpm[mask_p] = np.nan
    return Xm.reset_index(drop=True), Xpm.reset_index(drop=True), independent_feature_idx


# -----------------------------------------------------------------------------
# MAR — Missing At Random  (FIXED: full independent_feature_idx & no self-picks)
# -----------------------------------------------------------------------------
def generate_MAR(X, X_processed, missing_rate, independent_feature_idx=None):
    """
    Generate unstructured missingness under the MAR mechanism.

    Fixes:
    - Build a full independent_feature_idx once (length = n_features).
    - For every dependent feature i, choose a predictor j != i.
    - If an index array is provided, use it as-is (must satisfy j != i).
    - Return the complete independent_feature_idx actually used.

    Implementation (unchanged otherwise):
    - For each dependent feature i, derive missingness probability from its
      predictor feature j via a scaled sigmoid, then rescale to match the
      target missing rate for feature i.
    - If the predictor is categorical (one-hot expanded in X_processed),
      use a weighted combination of its dummy columns.
    - For categorical dependents, mask all their dummy columns together.
    """


    n_samples, n_features = X.shape
    if len(missing_rate) != n_features:
        raise ValueError("Length of missing_rate must equal number of features.")

    # ---- build or validate the dependency array (no self-picks) --------------
    if independent_feature_idx is None:
        indep_idx = np.empty(n_features, dtype=int)
        for i in range(n_features):
            # pick uniformly from all features except itself
            choices = np.delete(np.arange(n_features), i)
            indep_idx[i] = np.random.choice(choices)
    else:
        indep_idx = np.asarray(independent_feature_idx, dtype=int)
        if indep_idx.shape[0] != n_features:
            raise ValueError("independent_feature_idx must have length = n_features.")
        # ensure no self-picks; if found, reassign randomly (excluding i)
        for i in range(n_features):
            if indep_idx[i] == i:
                choices = np.delete(np.arange(n_features), i)
                indep_idx[i] = np.random.choice(choices)

    mask = np.zeros_like(X, dtype=bool)
    mask_p = np.zeros_like(X_processed, dtype=bool)

    for i in range(n_features):
        dep_name = X.columns[i]
        # columns in processed data for the DEPENDENT feature (to be masked)
        if X[dep_name].dtype == "object":
            pat_dep = re.compile(rf"^{re.escape(dep_name)}(_.+)?$")
            dep_cols = [c for c in X_processed.columns if pat_dep.match(c)]
            dep_idx_p = [X_processed.columns.get_loc(c) for c in dep_cols]
        else:
            dep_cols = [dep_name]
            dep_idx_p = [X_processed.columns.get_loc(dep_name)]

        # predictor (independent) feature j for this dependent i
        j = int(indep_idx[i])
        indep_name = X.columns[j]

        # build predictor signal z in [0,1] (weighted dummies for categorical)
        if X[indep_name].dtype == "object":
            pat_ind = re.compile(rf"^{re.escape(indep_name)}(_.+)?$")
            ind_cols = [c for c in X_processed.columns if pat_ind.match(c)]
            ind_vals = X_processed.loc[:, ind_cols].to_numpy()
            weights = np.random.random(len(ind_cols))
            z = np.dot(ind_vals, weights)
        else:
            z = X[indep_name].to_numpy()

        # normalise to [0,1] robustly
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        if zmax - zmin == 0:
            z_norm = np.full_like(z, 0.5, dtype=float)
        else:
            z_norm = (z - zmin) / (zmax - zmin)

        # sigmoid, then rescale to target rate for feature i
        probs = scipy.special.expit(4.0 * (z_norm - 0.5))
        target = float(missing_rate[i])
        mean_prob = probs.mean() if probs.size else 0.0
        if mean_prob > 0:
            probs = probs * (target / mean_prob)
        probs = np.clip(probs, 0.0, 1.0)

        feat_mask = (np.random.random(n_samples) < probs)
        mask[:, i] = feat_mask
        mask_p[:, dep_idx_p] = feat_mask[:, None]

    Xm = X.copy()
    Xm[mask] = np.nan
    Xpm = X_processed.copy()
    Xpm[mask_p] = np.nan

    # Return the COMPLETE indep_idx so callers can reuse it for VAL/TEST/C1.2
    return Xm.reset_index(drop=True), Xpm.reset_index(drop=True), indep_idx

# -----------------------------------------------------------------------------
# MNAR — Missing Not At Random
# -----------------------------------------------------------------------------
def generate_MNAR(X, X_processed, missing_rate, independent_feature_idx=None):
    """
    Generate unstructured missingness under the MNAR mechanism.

    Definition:
        The probability of missingness depends on the *unobserved* value itself:
            P(M_j | X_obs, X_mis) = P(M_j | X_j).

    Implementation summary:
        - Each feature acts as its own predictor.
        - Missingness probability is derived from the feature’s own values via
          a scaled sigmoid transformation, as in Fig. 3.
        - For categorical variables, all dummy columns are masked together.

    Returns
    -------
    X_masked, Xp_masked, independent_feature_idx
    """
    n_samples, n_features = X.shape
    if len(missing_rate) != n_features:
        raise ValueError("Length of missing_rate must equal number of features.")
    mask = np.zeros_like(X, dtype=bool)
    mask_p = np.zeros_like(X_processed, dtype=bool)

    for i, feat in enumerate(X.columns):
        if X[feat].dtype == "object":
            pat = re.compile(rf"^{re.escape(feat)}(_.+)?$")
            cols = [c for c in X_processed.columns if pat.match(c)]
            idx = [X_processed.columns.get_loc(c) for c in cols]
            vals = X_processed.loc[:, cols].to_numpy()
            weights = np.random.random(len(cols))
            z = np.dot(vals, weights)
        else:
            idx = [X_processed.columns.get_loc(feat)]
            z = X.iloc[:, i].to_numpy()

        z = (z - z.min()) / (z.max() - z.min() + 1e-12)
        probs = scipy.special.expit(4.0 * (z - 0.5))
        probs *= missing_rate[i] / np.mean(probs)
        probs = np.clip(probs, 0, 1)

        feat_mask = np.random.random(n_samples) < probs
        mask[:, i] = feat_mask
        mask_p[:, idx] = feat_mask[:, None]

    Xm = X.copy()
    Xm[mask] = np.nan
    Xpm = X_processed.copy()
    Xpm[mask_p] = np.nan
    return Xm.reset_index(drop=True), Xpm.reset_index(drop=True), independent_feature_idx


# -----------------------------------------------------------------------------
# Structured Pattern Generator (Pairwise Masking)
# -----------------------------------------------------------------------------
def generate_paired_missingness(
    X,
    Xm,
    X_processed,
    Xp_masked,
    pair_feature_map=None,
):
    """
    Generate structured-pattern missingness by imposing *pairwise* masking.

    Definition (Section 2.3.2):
        In structured-pattern mechanisms, missingness in one feature implies
        missingness in related features. This reflects correlated experimental
        tests where several measurements are jointly absent.

    Implementation summary:
        - Given an existing masked dataset (Xm, Xp_masked) generated under an
          unstructured mechanism (MCAR/MAR/MNAR),
          copy the missingness pattern from selected "source" features
          to designated "paired" features.
        - The mapping `pair_feature_map` defines which features depend on which.

    Parameters
    ----------
    X, X_processed : pandas.DataFrame
        Original and preprocessed data.
    Xm, Xp_masked : pandas.DataFrame
        Versions already containing unstructured missingness.
    pair_feature_map : array-like or None
        Length = n_features. If entry[i] = j, then feature i copies the mask
        from feature j. If entry[i] is NaN, feature i retains its own mask.

    Returns
    -------
    X_masked_pair, Xp_masked_pair
    """
    if pair_feature_map is None:
        return Xm, Xp_masked

    pair_feature_map = np.asarray(pair_feature_map)
    old_mask = Xm.isna().to_numpy()
    old_mask_p = Xp_masked.isna().to_numpy()
    new_mask = np.zeros_like(old_mask)
    new_mask_p = np.zeros_like(old_mask_p)

    for i, feat in enumerate(X.columns):
        if np.isnan(pair_feature_map[i]):
            new_mask[:, i] = old_mask[:, i]
            pat = re.compile(rf"^{re.escape(feat)}(_.+)?$")
            idx = [X_processed.columns.get_loc(c) for c in X_processed.columns if pat.match(c)]
            new_mask_p[:, idx] = old_mask_p[:, idx]
            continue

        dep_on = int(pair_feature_map[i])
        new_mask[:, i] = old_mask[:, dep_on]

        pat_i = re.compile(rf"^{re.escape(feat)}(_.+)?$")
        idx_i = [X_processed.columns.get_loc(c) for c in X_processed.columns if pat_i.match(c)]
        pat_j = re.compile(rf"^{re.escape(X.columns[dep_on])}(_.+)?$")
        idx_j = [X_processed.columns.get_loc(c) for c in X_processed.columns if pat_j.match(c)]
        new_mask_p[:, idx_i] = old_mask_p[:, idx_j[0]][:, None]

    Xm_new = X.copy()
    Xm_new[new_mask] = np.nan
    Xp_new = X_processed.copy()
    Xp_new[new_mask_p] = np.nan
    return Xm_new.reset_index(drop=True), Xp_new.reset_index(drop=True)

def build_processed(X: pd.DataFrame) -> pd.DataFrame:
    """
    Build a processed frame with numeric columns untouched and categorical
    columns one-hot encoded. Works for:
      - pure numeric datasets,
      - pure categorical datasets,
      - mixed.
    """
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if X[c].dtype == "object"]

    if len(cat) == 0:
        # pure numeric
        return X[num].copy()

    if len(num) == 0:
        # pure categorical
        return pd.get_dummies(
            X[cat], prefix=cat, prefix_sep="_", dummy_na=False
        )

    # mixed: numeric + one-hot categoricals
    Xp = pd.concat(
        [
            X[num].copy(),
            pd.get_dummies(X[cat], prefix=cat, prefix_sep="_", dummy_na=False),
        ],
        axis=1,
    )
    return Xp
#%%
# =============================================================================
# Validation Criteria (Sec. 2.2, Fig. 1)
# =============================================================================
"""
This section implements the three validation criteria used in the paper:

- Baseline criterion (Sec. 2.2.1): direct error vs hidden truth (test set).
- Criterion 1 (Sec. 2.2.2): proxy error on a complete-case subset (validation).
  * 1.1: MCAR proxy
  * 1.2: mechanism-aligned proxy
- Criterion 2 (Sec. 2.2.3): downstream task performance (validation).

Conventions:
- Numerical features are scored with RMSE (after Z-score normalisation).
- Categorical features are scored with classification error = 1 - accuracy.
- Final score per imputer is the mean across features present in the split.
- For Criterion 2, the final score per imputer is the mean across models
  (Linear Regression, Random Forest, SVM, MLP by default).
"""

# =============================================================================
# Validation Criteria (detailed outputs preserved)
# =============================================================================
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# --- helpers from before (reused) --------------------------------------------
def _split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    return num_cols, cat_cols

def _zscore_fit(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for c in cols:
        m = df[c].mean()
        s = df[c].std(ddof=0) or 1.0
        stats[c] = (m, s)
    return stats

def _zscore_apply(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (m, s) in stats.items():
        out[c] = (out[c] - m) / s
    return out

def _align_train_val_columns(
    Xtr: pd.DataFrame, Xva: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure TRAIN and VAL matrices have exactly the same dummy-encoded columns
    in the same order.

    - Any column present in TRAIN but missing in VAL is added to VAL and filled with 0.
    - Any extra column in VAL (not seen in TRAIN) is dropped.
    - Columns are then reordered so that Xva.columns == Xtr.columns.

    This is needed because one-hot encoding on TRAIN and VAL may create
    slightly different sets of dummy columns when some categories are
    missing in one split.
    """
    Xtr = Xtr.copy()
    Xva = Xva.copy()

    # 1) add missing train columns to val as zeros
    for col in Xtr.columns:
        if col not in Xva.columns:
            Xva[col] = 0.0

    # 2) drop extra val columns that were not seen in train
    extra_cols = [c for c in Xva.columns if c not in Xtr.columns]
    if extra_cols:
        Xva.drop(columns=extra_cols, inplace=True)

    # 3) reorder val columns to match train
    Xva = Xva[Xtr.columns]

    return Xtr, Xva


def _compute_feature_error_numeric(y_true, y_pred) -> float:
    """
    Numeric feature error = RMSE, ignoring any non-finite entries
    produced by a misbehaving imputer.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    # keep only finite positions
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return np.nan

    mse = mean_squared_error(yt[mask], yp[mask])
    return float(np.sqrt(mse))

def _compute_feature_error_categorical(y_true, y_pred) -> float:
    yt = pd.Series(y_true).reset_index(drop=True)
    yp = pd.Series(y_pred).reset_index(drop=True)

    mask = yt.notna() & yp.notna()
    if not mask.any():
        return np.nan

    # mismatch rate = 1 - accuracy (robust to mixed dtypes)
    return float((yt[mask].astype(str) != yp[mask].astype(str)).mean())

def _errors_full_and_missing_only(
    X_true: pd.DataFrame,
    X_imp: pd.DataFrame,
    X_masked: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return two dicts:
      - per_feature_full:   error over all available rows (where both values exist)
      - per_feature_missing_only: error computed only on rows that were masked as missing
    """
    X_true = X_true.reset_index(drop=True)
    X_imp = X_imp.reset_index(drop=True)
    X_masked = X_masked.reset_index(drop=True)

    num_cols, cat_cols = _split_feature_types(X_true)
    per_feature_full, per_feature_missing = {}, {}

    # numeric
    for c in num_cols:
        # full-column error (only where both values are available)
        mask_full = X_true[c].notna() & X_imp[c].notna()
        if mask_full.any():
            per_feature_full[c] = _compute_feature_error_numeric(
                X_true.loc[mask_full, c], X_imp.loc[mask_full, c]
            )
        # missing-only error (compare only positions originally masked)
        mask_missing = X_masked[c].isna() & X_true[c].notna() & X_imp[c].notna()
        if mask_missing.any():
            per_feature_missing[c] = _compute_feature_error_numeric(
                X_true.loc[mask_missing, c], X_imp.loc[mask_missing, c]
            )

    # categorical
    for c in cat_cols:
        mask_full = X_true[c].notna() & X_imp[c].notna()
        if mask_full.any():
            per_feature_full[c] = _compute_feature_error_categorical(
                X_true.loc[mask_full, c], X_imp.loc[mask_full, c]
            )
        mask_missing = X_masked[c].isna() & X_true[c].notna() & X_imp[c].notna()
        if mask_missing.any():
            per_feature_missing[c] = _compute_feature_error_categorical(
                X_true.loc[mask_missing, c], X_imp.loc[mask_missing, c]
            )

    return per_feature_full, per_feature_missing

def _mean_or_nan(d: Dict[str, float]) -> float:
    return float(np.mean(list(d.values()))) if d else np.nan


def _detect_task_type(y: pd.Series) -> str:
    """
    For this study, all six benchmark datasets are treated as regression problems
    """
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    return "regression"

# -------------------------------------------------------------------------
# Baseline criterion — detailed per-feature outputs (full & missing-only)
# -------------------------------------------------------------------------
def validate_baseline_direct_error(
    *,
    X_test_complete: pd.DataFrame,    # ground-truth test (fully observed)
    X_test_masked: pd.DataFrame,      # test with introduced NaNs
    imputer_dict: Dict[str, object],  # {name: sklearn-style imputer}
    fit_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    num_cols: Optional[List[str]] = None,
    zstats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Baseline criterion: direct reconstruction error on the TEST set.

    Parameters
    ----------
    X_test_complete : fully observed test DataFrame (ground truth)
    X_test_masked   : same shape as X_test_complete, but with introduced NaNs
    imputer_dict    : {name: fitted-imputer}
    fit_data        : (X_train, y_train), used only as a fallback to
                      compute Z-score stats if num_cols/zstats are not provided.
    num_cols        : optional list of numeric feature names (from TRAIN)
    zstats          : optional dict {col: (mean, std)} from TRAIN

    Returns
    -------
    out[name] = {
        "per_feature_full":         {feature: error},
        "per_feature_missing_only": {feature: error},
        "mean_full":                float,
        "mean_missing_only":        float,
    }
    """
    # Prefer explicit stats if given; otherwise fall back to fit_data
    if num_cols is None or zstats is None:
        if fit_data is None:
            raise ValueError(
                "Either (num_cols, zstats) or fit_data must be provided "
                "to validate_baseline_direct_error."
            )
        X_train, _ = fit_data
        num_cols, _ = _split_feature_types(X_train)
        zstats = _zscore_fit(X_train, num_cols)

    out = {}
    for name, imputer in imputer_dict.items():
        # imputer is assumed ALREADY FITTED on X_train
        X_imp = imputer.transform(X_test_masked.copy())
        X_imp = pd.DataFrame(X_imp, columns=X_test_masked.columns)

        # Guard against inf/-inf from any imputer
        X_imp.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Z-score numeric in both truth & imputed using TRAIN stats
        X_true_std = X_test_complete.copy()
        X_imp_std = X_imp.copy()
        X_true_std[num_cols] = _zscore_apply(X_test_complete[num_cols], zstats)
        X_imp_std[num_cols]  = _zscore_apply(X_imp[num_cols], zstats)

        per_full, per_missing = _errors_full_and_missing_only(
            X_true_std, X_imp_std, X_test_masked
        )
        out[name] = {
            "per_feature_full": per_full,
            "per_feature_missing_only": per_missing,
            "mean_full": _mean_or_nan(per_full),
            "mean_missing_only": _mean_or_nan(per_missing),
        }
    return out

# -------------------------------------------------------------------------
# Criterion 1 — detailed per-feature outputs (full & missing-only)
# -------------------------------------------------------------------------
def validate_proxy_complete_case(
    *,
    imputer_dict: Dict[str, object],
    prebuilt_complete_case: pd.DataFrame,
    prebuilt_masked: pd.DataFrame,
    fit_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    num_cols: Optional[List[str]] = None,
    zstats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Criterion 1 (both 1.1 and 1.2), using prebuilt complete-case subset (Xcc)
    and its masked counterpart (Xcc_masked).

    Parameters
    ----------
    imputer_dict          : {name: fitted imputer}
    prebuilt_complete_case: Xcc (rows with no missing values before masking)
    prebuilt_masked       : Xcc_masked (after introducing mask)
    fit_data              : optional (X_train, y_train) fallback for Z-score stats
    num_cols              : optional list of numeric feature names from TRAIN
    zstats                : optional dict {col: (mean, std)} from TRAIN

    Returns
    -------
    out[name] = {
        "per_feature_full":         {feature: error},
        "per_feature_missing_only": {feature: error},
        "mean_full":                float,
        "mean_missing_only":        float,
    }
    """
    Xcc = prebuilt_complete_case.copy()
    Xcc_masked = prebuilt_masked.copy()

    if Xcc.empty:
        return {
            name: {
                "per_feature_full": {},
                "per_feature_missing_only": {},
                "mean_full": np.nan,
                "mean_missing_only": np.nan,
            }
            for name in imputer_dict
        }

    # Prefer explicit stats if given; otherwise fall back to fit_data
    if num_cols is None or zstats is None:
        if fit_data is None:
            raise ValueError(
                "Either (num_cols, zstats) or fit_data must be provided "
                "to validate_proxy_complete_case."
            )
        X_train, _ = fit_data
        num_cols, _ = _split_feature_types(X_train)
        zstats = _zscore_fit(X_train, num_cols)

    out = {}
    for name, imputer in imputer_dict.items():
        # imputer is assumed ALREADY FITTED on X_train
        X_imp = imputer.transform(Xcc_masked.copy())
        X_imp = pd.DataFrame(X_imp, columns=Xcc_masked.columns)

        # Guard against inf/-inf from any imputer
        X_imp.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Z-score numeric using TRAIN stats
        X_true_std = Xcc.copy()
        X_imp_std = X_imp.copy()
        X_true_std[num_cols] = _zscore_apply(Xcc[num_cols], zstats)
        X_imp_std[num_cols]  = _zscore_apply(X_imp[num_cols], zstats)

        per_full, per_missing = _errors_full_and_missing_only(
            X_true_std, X_imp_std, Xcc_masked
        )
        out[name] = {
            "per_feature_full": per_full,
            "per_feature_missing_only": per_missing,
            "mean_full": _mean_or_nan(per_full),
            "mean_missing_only": _mean_or_nan(per_missing),
        }
    return out

# -------------------------------------------------------------------------
# Criterion 2 — per-model downstream performance
# -------------------------------------------------------------------------
def validate_downstream_performance(
    *,
    X_train_incomplete: pd.DataFrame,
    y_train: pd.Series,
    X_val_incomplete: pd.DataFrame,
    y_val: pd.Series,
    imputer_dict: Dict[str, object],
    model_dict: Optional[Dict[str, object]] = None,
    num_cols: Optional[List[str]] = None,
    x_zstats: Optional[Dict[str, Tuple[float, float]]] = None,
    y_stats: Optional[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Criterion 2: downstream predictive performance on a validation set.

    IMPORTANT:
    ----------
    - Imputers in `imputer_dict` are assumed to be ALREADY FITTED
      (e.g. on Xm_train in run_one_dataset). Here we *only* call
      `transform`, never `fit` or `fit_transform`.

    - If `x_zstats` and `num_cols` are provided, numeric features in
      both TRAIN and VAL are Z-scored using TRAIN stats (from Xm_train).

    - If `y_stats` is provided and the task is regression, y is also
      Z-scored using TRAIN mean/std, and RMSE is reported in *std-y* units.
    """
    if model_dict is None:
        models = DEFAULT_MODEL_LIST
    else:
        models = model_dict

    # Decide regression vs classification (your 6 datasets are regression,
    # but we keep the general path)
    y_tr = pd.Series(y_train).reset_index(drop=True)
    y_va = pd.Series(y_val).reset_index(drop=True)
    task = _detect_task_type(y_tr)

    # ----- scale y (regression) ---------------------------------------------
    if task == "regression" and y_stats is not None:
        mean_y, std_y = y_stats
        if std_y == 0 or not np.isfinite(std_y):
            # Degenerate: fall back to unscaled y
            y_tr_scaled = y_tr.astype(float)
            y_va_scaled = y_va.astype(float)
        else:
            y_tr_scaled = (y_tr.astype(float) - mean_y) / std_y
            y_va_scaled = (y_va.astype(float) - mean_y) / std_y
    else:
        y_tr_scaled = y_tr
        y_va_scaled = y_va

    out: Dict[str, Dict[str, float]] = {}

    for imp_name, imputer in imputer_dict.items():
        # 1) USE PRE-FITTED IMPUTER: transform only
        Xtr_imp = imputer.transform(X_train_incomplete.copy())
        Xva_imp = imputer.transform(X_val_incomplete.copy())
        Xtr_imp = pd.DataFrame(Xtr_imp, columns=X_train_incomplete.columns)
        Xva_imp = pd.DataFrame(Xva_imp, columns=X_val_incomplete.columns)

        # 2) Optional: Z-score numeric columns BEFORE dummy encoding
        if num_cols is not None and x_zstats is not None and len(num_cols) > 0:
            Xtr_imp[num_cols] = _zscore_apply(Xtr_imp[num_cols], x_zstats)
            Xva_imp[num_cols] = _zscore_apply(Xva_imp[num_cols], x_zstats)

        # 3) One-hot encode categoricals
        Xtr_num = build_processed(Xtr_imp)
        Xva_num = build_processed(Xva_imp)

        # 4) Align columns (same dummy set and order)
        Xtr_num, Xva_num = _align_train_val_columns(Xtr_num, Xva_num)

        # 5) Train/eval downstream models
        per_model: Dict[str, float] = {}
        for mname, model in models.items():
            # Fresh copy of the model
            params = getattr(model, "get_params", lambda: {})()
            mdl = model.__class__(**params)

            if task == "regression":
                mdl.fit(Xtr_num, y_tr_scaled)
                yhat = mdl.predict(Xva_num)
                err = float(root_mean_squared_error(y_va_scaled, yhat))
            else:
                mdl.fit(Xtr_num, y_tr)
                if hasattr(mdl, "predict"):
                    ypred = mdl.predict(Xva_num)
                    err = float(1.0 - accuracy_score(y_va, ypred))
                else:
                    yhat = mdl.predict_proba(Xva_num)[:, 1]
                    ypred = (yhat > 0.5).astype(y_va.dtype)
                    err = float(1.0 - accuracy_score(y_va, ypred))

            per_model[mname] = err

        out[imp_name] = {
            "per_model": per_model,
            "mean": _mean_or_nan(per_model),
        }

    return out





#%%
# =============================================================================
# MixedImputer: unified interface for numerical and categorical imputation
# =============================================================================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostError
import pandas as pd
import numpy as np
import random


# =============================================================================
# Imputers & “standardise-before-impute” wrapper
# =============================================================================
from sklearn.base import BaseEstimator, TransformerMixin

class BaseImputer:
    """
    Simple interface that every imputer (internal or external) must follow:

    - fit(X)
    - transform(X)
    - fit_transform(X)

    This class allows external imputers (VAE/MIWAE/GAIN) to be plugged
    into the framework without modifying MixedImputer.
    """
    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class StandardizeBeforeImpute(BaseEstimator, TransformerMixin):
    """
    Wrap any sklearn-style imputer so that numeric features are Z-scored
    *before* imputation and inverse-transformed afterward.

    - Only numeric columns are standardised; categorical columns are passed as-is.
    - Mean/Mode → unchanged in effect after inverse transform (mean becomes mean).
    - KNN/PMM often benefit from scaling; tree/boosting models tolerate it.
    - Returned DataFrame preserves original column names/order.
    """
    def __init__(self, base_imputer):
        self.base_imputer = base_imputer
        self.num_cols_ = None
        self.cat_cols_ = None
        self.zstats_ = None

    def fit(self, X, y=None):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.num_cols_, self.cat_cols_ = _split_feature_types(Xdf)
        # fit Z-stats on numeric only
        self.zstats_ = _zscore_fit(Xdf, self.num_cols_)
        # standardise a copy for imputer.fit
        Xstd = Xdf.copy()
        if self.num_cols_:
            Xstd[self.num_cols_] = _zscore_apply(Xstd[self.num_cols_], self.zstats_)
        self.base_imputer.fit(Xstd, y)
        return self

    def transform(self, X):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.num_cols_ + self.cat_cols_)
        Xstd = Xdf.copy()
        if self.num_cols_:
            Xstd[self.num_cols_] = _zscore_apply(Xstd[self.num_cols_], self.zstats_)
        Ximp = self.base_imputer.transform(Xstd)
        Ximp = pd.DataFrame(Ximp, columns=Xdf.columns)
        # inverse standardisation for numeric columns
        if self.num_cols_:
            out = Ximp.copy()
            for c, (m, s) in self.zstats_.items():
                out[c] = out[c] * s + m
            return out
        return Ximp

def wrap_imputers_with_standardize(imputer_dict: Dict[str, object]) -> Dict[str, object]:
    """
    Return a new dict where every imputer is wrapped by StandardizeBeforeImpute.
    Use this dict in the experiments so all imputations benefit from scaling.
    """
    return {name: StandardizeBeforeImpute(imp) for name, imp in imputer_dict.items()}

def _get_dummies_with_nan(X_cat: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categoricals but keep NaNs as NaNs in the dummy block.
    For rows where original category is NaN, set all corresponding dummy
    columns to NaN so downstream imputers can see missingness.
    """
    X_dummy = pd.get_dummies(X_cat, dummy_na=False)

    for c in X_cat.columns:
        na_mask = X_cat[c].isna()
        if not na_mask.any():
            continue
        cols = [cc for cc in X_dummy.columns if cc.startswith(f"{c}_")]
        if cols:
            X_dummy.loc[na_mask, cols] = np.nan

    return X_dummy

class MixedImputer(BaseEstimator, TransformerMixin):
    """
    MixedImputer
    ------------
    A unified wrapper for imputing datasets containing both numerical and
    categorical features, supporting multiple algorithms described in the
    paper *“Decision-Making Criteria on Choosing Appropriate Imputation Methods
    for Incomplete Dataset Prepared for Machine Learning”* (Sec. 2.4).

    Each imputer instance can specify distinct strategies for numerical and
    categorical variables, e.g. `"Mean_Mode"`, `"KNN_KNN"`, `"LGBM_LGBM"`, etc.

    Supported method codes (case-insensitive):
      • "Mean", "Median"     → univariate numeric replacements
      • "Mode"               → univariate categorical replacement
      • "Random"             → simple hot-deck (random donor)
      • "KNN"                → k-Nearest Neighbour (scikit-learn KNNImputer)
      • "PMM"                → Predictive Mean Matching via IterativeImputer
      • "RF"                 → Random-Forest regressor in IterativeImputer
      • "LGBM"               → LightGBM regressor in IterativeImputer
      • "CB"                 → CatBoost regressor in IterativeImputer
      • "Lin"                → Linear (BayesianRidge) model in IterativeImputer

    Notes
    -----
    - For "Random" (hot-deck), if a column becomes fully missing in the
      current subset, we fall back to donors taken from the TRAIN data
      at fit time (stored in numeric_fallback_ / categorical_fallback_).
    """

    def __init__(
        self,
        method: str = "Mean_Mode",
        mice_iters: int = 10,
        rf_n_estimators: int = 200,
        knn_k: int = 5,
        tree_jobs: int = 1,
        random_state: int = 0,
    ):
        self.method = method
        self.mice_iters = mice_iters
        self.rf_n_estimators = rf_n_estimators
        self.knn_k = knn_k
        self.tree_jobs = tree_jobs
        self.random_state = random_state

        # attributes initialised in fit
        self.num_cols_ = None
        self.cat_cols_ = None
        self.num_imputer_ = None
        self.num_method_ = None
        self.cat_imputer_ = None
        self.cat_method_ = None
        self.cat_dummy_cols_ = None

        # fallback donor pools (from TRAIN data)
        self.numeric_fallback_ = {}
        self.categorical_fallback_ = {}

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _split_features(self, X: pd.DataFrame):
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        return num_cols, cat_cols



    def _fit_numeric_imputer(self, X: pd.DataFrame):
        """Return fitted numeric imputer based on selected method."""
        num_method = self.method.split("_")[0].upper()

        if num_method == "MEAN":
            imp = SimpleImputer(strategy="mean")
        elif num_method == "MEDIAN":
            imp = SimpleImputer(strategy="median")
        elif num_method == "RANDOM":
            # placeholder; we handle RANDOM manually in transform
            imp = SimpleImputer(strategy="constant", fill_value=np.nan)
        elif num_method == "KNN":
            imp = KNNImputer(n_neighbors=self.knn_k)
        elif num_method == "PMM":
            imp = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.mice_iters,
                sample_posterior=True,
                random_state=self.random_state,
            )
        elif num_method == "RF":
            imp = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=self.rf_n_estimators,
                    n_jobs=self.tree_jobs,
                    random_state=self.random_state,
                ),
                max_iter=self.mice_iters,
                random_state=self.random_state,
            )
        elif num_method == "LGBM":
            imp = IterativeImputer(
                estimator=LGBMRegressor(
                    n_estimators=self.rf_n_estimators,
                    n_jobs=self.tree_jobs,
                    random_state=self.random_state,
                    verbose=-1,
                ),
                max_iter=self.mice_iters,
                random_state=self.random_state,
            )
        elif num_method == "CB":
            # First attempt: CatBoostRegressor (no file writing, good for parallel runs)
            estimator = CatBoostRegressor(
                iterations=self.rf_n_estimators,
                thread_count=self.tree_jobs,
                verbose=False,
                random_seed=self.random_state,
                allow_writing_files=False,
            )
            imp = IterativeImputer(
                estimator=estimator,
                max_iter=self.mice_iters,
                random_state=self.random_state,
            )
        elif num_method == "LIN":
            imp = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.mice_iters,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown numeric imputation method: {num_method}")

        try:
            imp.fit(X)
        except CatBoostError as e:
            # Typical CatBoost failure here is "All train targets are equal"
            # or similar issues during IterativeImputer's per-column fits.
            # Fall back to a RandomForest-based IterativeImputer, which
            # can handle constant targets without crashing.
            if num_method == "CB":
                estimator = RandomForestRegressor(
                    n_estimators=self.rf_n_estimators,
                    n_jobs=self.tree_jobs,
                    random_state=self.random_state,
                )
                imp = IterativeImputer(
                    estimator=estimator,
                    max_iter=self.mice_iters,
                    random_state=self.random_state,
                )
                imp.fit(X)
            else:
                # If CatBoostError arises for non-CB methods (unlikely),
                # re-raise so we can see it.
                raise
        return imp, num_method

    def _fit_categorical_imputer(self, X: pd.DataFrame):
        """Return fitted categorical imputer based on selected method."""
        cat_method = self.method.split("_")[-1].upper()

        if cat_method in ("MODE", "MEAN"):
            imp = SimpleImputer(strategy="most_frequent")
            imp.fit(X)
        elif cat_method == "RANDOM":
            imp = None  # handled manually in transform
        else:
            # reuse numeric-style imputers on dummies
            X_dummy = _get_dummies_with_nan(X)
            imp, _ = self._fit_numeric_imputer(X_dummy)
        return imp, cat_method

    # -------------------------------------------------------------------------
    # sklearn API
    # -------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.num_cols_, self.cat_cols_ = self._split_features(X)

        # (re)initialise fallback donor pools
        self.numeric_fallback_ = {}
        self.categorical_fallback_ = {}

        for c in self.num_cols_:
            vals = X[c].dropna().values
            if vals.size > 0:
                self.numeric_fallback_[c] = vals

        for c in self.cat_cols_:
            vals = X[c].dropna().values
            if vals.size > 0:
                self.categorical_fallback_[c] = vals

        if self.num_cols_:
            self.num_imputer_, self.num_method_ = self._fit_numeric_imputer(X[self.num_cols_])
        else:
            self.num_imputer_, self.num_method_ = None, None

        if self.cat_cols_:
            self.cat_imputer_, self.cat_method_ = self._fit_categorical_imputer(X[self.cat_cols_])
            if self.cat_method_ not in ("RANDOM", "MODE", "MEAN"):
                X_dummy = _get_dummies_with_nan(X[self.cat_cols_])
                self.cat_dummy_cols_ = list(X_dummy.columns)
            else:
                self.cat_dummy_cols_ = None
        else:
            self.cat_imputer_, self.cat_method_ = None, None
            self.cat_dummy_cols_ = None

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # ----- numeric side ---------------------------------------------------
        if self.num_cols_ and self.num_imputer_:
            if self.num_method_ == "RANDOM":
                for c in self.num_cols_:
                    na_mask = X[c].isna()
                    if not na_mask.any():
                        continue  # nothing to impute

                    # donors from current subset
                    observed = X.loc[~na_mask, c].dropna().values

                    # if no donors in this subset, fallback to TRAIN donors
                    if observed.size == 0:
                        observed = self.numeric_fallback_.get(c, None)
                        if observed is None or len(observed) == 0:
                            # no information at all; leave NaNs as-is
                            continue

                    X.loc[na_mask, c] = np.random.choice(
                        observed,
                        size=na_mask.sum(),
                        replace=True,
                    )
            else:
                X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])

        # ----- categorical side ----------------------------------------------
        if self.cat_cols_:
            if self.cat_method_ == "RANDOM":
                for c in self.cat_cols_:
                    na_mask = X[c].isna()
                    if not na_mask.any():
                        continue

                    observed = X.loc[~na_mask, c].dropna().values

                    if observed.size == 0:
                        observed = self.categorical_fallback_.get(c, None)
                        if observed is None or len(observed) == 0:
                            continue  # leave NaNs; no donors anywhere

                    X.loc[na_mask, c] = np.random.choice(
                        observed,
                        size=na_mask.sum(),
                        replace=True,
                    )
            elif self.cat_method_ in ("MODE", "MEAN"):
                # use a fitted most_frequent imputer on TRAIN stats
                # (self.cat_imputer_ was fit in _fit_categorical_imputer)
                X[self.cat_cols_] = self.cat_imputer_.transform(X[self.cat_cols_])
            else:
                # use fitted iterative-style imputer on dummy-coded columns
                X_dummy = _get_dummies_with_nan(X[self.cat_cols_])

                # align dummy columns to those seen at fit time
                if getattr(self, "cat_dummy_cols_", None) is not None:
                    for col in self.cat_dummy_cols_:
                        if col not in X_dummy.columns:
                            X_dummy[col] = 0.0
                    X_dummy = X_dummy[self.cat_dummy_cols_]

                X_imp_dummy = self.cat_imputer_.transform(X_dummy)
                X_imp_dummy = pd.DataFrame(X_imp_dummy, columns=X_dummy.columns)

                # decode back to labels (robust to ties/all-zero)
                for c in self.cat_cols_:
                    cat_cols = [col for col in X_imp_dummy.columns if col.startswith(c + "_")]
                    if not cat_cols:
                        continue
                    block = X_imp_dummy[cat_cols].copy()

                    maxvals = block.max(axis=1)
                    tie_mask = (block.eq(maxvals, axis=0).sum(axis=1) > 1) | (maxvals == 0)

                    if tie_mask.any():
                        fallback_col = block.mean().idxmax()
                        block.loc[tie_mask, :] = 0.0
                        block.loc[tie_mask, fallback_col] = 1.0

                    X[c] = block.idxmax(axis=1).str.replace(c + "_", "", regex=False)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# =============================================================================
# Imputer factory for paper experiments
# =============================================================================
def build_imputers(
    *,
    internal_imputer: list | None = ("Mean/Mode", "Hot-Deck", "KNN", "LightGBM", "CatBoost", "PMM"),
    mice_iters: int = 10,
    rf_n_estimators: int = 200,
    knn_k: int = 5,
    tree_jobs: int = 1,
    random_state: int = 0,
    external_imputer: list | None = None,
) -> Dict[str, object]:
    """
    Return a dictionary of imputers built from MixedImputer.

    - By default `internal_imputer` includes all six built-in methods.
    - If the caller explicitly passes internal_imputer=None and external_imputer=None
      a ValueError is raised (requires at least one imputer source).
    """
    # If caller explicitly set both to None -> error
    if internal_imputer is None and external_imputer is None:
        raise ValueError("At least one of internal_imputer or external_imputer must be provided.")

    # canonical set used to expand "all" if requested
    all_internal = ["Mean/Mode", "Hot-Deck", "KNN", "LightGBM", "CatBoost", "PMM"]

    # Normalize requested list
    requested = list(internal_imputer) if internal_imputer is not None else []
    if isinstance(requested, str):
        requested = [requested]

    shared_args = dict(
        mice_iters=mice_iters,
        rf_n_estimators=rf_n_estimators,
        knn_k=knn_k,
        tree_jobs=tree_jobs,
        random_state=random_state,
    )

    imps: Dict[str, object] = {}

    def _norm(name: str) -> str:
        return str(name).lower().replace(" ", "").replace("-", "").replace("_", "").replace("/", "")

    # expand "all" token if present
    if any(_norm(x) == "all" for x in requested):
        requested = all_internal

    for name in requested:
        n = _norm(name)
        if n in ("meanmode", "mean", "meanmode"):
            imps["Mean/Mode"] = MixedImputer(method="Mean_Mode", **shared_args)
        elif n in ("hotdeck", "random", "hot-deck"):
            imps["Hot-Deck"] = MixedImputer(method="Random_Random", **shared_args)
        elif n in ("knn", "knnimputer"):
            imps["KNN"] = MixedImputer(method="KNN_KNN", **shared_args)
        elif n in ("lightgbm", "lgbm", "lightgbmregressor"):
            imps["LightGBM"] = MixedImputer(method="LGBM_LGBM", **shared_args)
        elif n in ("catboost", "cb", "catboostregressor"):
            imps["CatBoost"] = MixedImputer(method="CB_CB", **shared_args)
        elif n in ("pmm", "predictivemeanmatching"):
            imps["PMM"] = MixedImputer(method="PMM_PMM", **shared_args)
        else:
            raise ValueError(f"Unknown internal imputer requested: {name}")

    # 2) optionally add external imputers (e.g., MIWAE)
    if external_imputer is not None:
        raise NotImplementedError("External imputers are not implemented in this code snippet.")

    return imps
#%%
# =============================================================================
# Experiments (splits, seeding, runners)
# =============================================================================
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- seeding -----------------------------------------------------------
BASE_SEED = 42

def set_random_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)

# ---------- splits ------------------------------------------------------------
def split_train_val_test(
    X: pd.DataFrame, y: Optional[pd.Series] = None, *, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    3:1:1 split (train:val:test = 0.6:0.2:0.2 by default).
    We first carve out TEST, then split remaining into TRAIN/VAL.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    rel_val = val_size / (1.0 - test_size)  # fraction of the remainder
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=rel_val, random_state=random_state + 17
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------- mask config -------------------------------------------------------
MaskName = Literal["MCAR", "MAR", "MNAR", "MCAR_pair", "MAR_pair", "MNAR_pair"]

def _unstructured_mask_fn(name: MaskName):
    if name in ("MCAR", "MCAR_pair"):
        return generate_MCAR
    if name in ("MAR", "MAR_pair"):
        return generate_MAR
    if name in ("MNAR", "MNAR_pair"):
        return generate_MNAR
    raise ValueError(f"Unknown mask name: {name}")

def _is_paired(name: MaskName) -> bool:
    return name.endswith("_pair")

# ---------- main runner for one dataset --------------------------------------
# ---------- main runner for one dataset --------------------------------------
def run_one_dataset(
    *,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    mask_names: List[MaskName],
    missing_rates: List[float],
    n_repeats: int,
    imputer_dict: Dict[str, object],
    use_standardize_before_impute: bool = True,
    model_dict: Optional[Dict[str, object]] = None,
    checkpoint_dir: Optional[str] = None,
    dataset_label: Optional[str] = None,
    resume: bool = True,
    progress_callback=None,
) -> Dict:

    """
    Execute the full evaluation on a single dataset across:
      - mask_names: any of ["MCAR","MAR","MNAR","MCAR_pair","MAR_pair","MNAR_pair"]
      - missing_rates: e.g., [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
      - n_repeats: e.g., 100

    New behaviour (for long runs / HPC use):
    ----------------------------------------
    - If `checkpoint_dir` and `dataset_label` are given, the function will:
        * after EVERY repetition, overwrite a checkpoint file
          "ckpt_{dataset_label}.pkl" under checkpoint_dir
          containing the partial results.
        * also write a small text file
          "progress_{dataset_label}.txt" with the latest
          (mask, rate, rep / n_repeats) status.
    - If `resume=True` and a checkpoint exists, it will:
        * load existing results,
        * skip any (mask, rate, rep) combinations that are already present,
        * continue from the first missing combination.

    Random seed logic:
    ------------------
    - Instead of using the same seed for all mask/rate at a given rep,
      we now use a deterministic per-combination seed:
          seed = BASE_SEED + 1000*rep + 10*mask_idx + rate_idx
      where mask_idx and rate_idx are the indices in mask_names/missing_rates.
    - This makes each (mask, rate, rep) fully reproducible and independent
      of whether you pause/resume or change the order of execution.
    """

    # choose imputers (optionally wrap with standardise-before-impute)
    imputers = wrap_imputers_with_standardize(imputer_dict) if use_standardize_before_impute else imputer_dict

    # ---------------------------------------------------------------
    # Set up checkpoint paths (if requested)
    # ---------------------------------------------------------------
    ckpt_path = None
    progress_path = None
    if checkpoint_dir is not None and dataset_label is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{dataset_label}.pkl")
        progress_path = os.path.join(checkpoint_dir, f"progress_{dataset_label}.txt")

    # ---------------------------------------------------------------
    # Load previous partial results if resume=True and checkpoint exists
    # ---------------------------------------------------------------
    results: Dict = {}
    if resume and ckpt_path is not None and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "rb") as f:
                saved = pickle.load(f)
            results = saved.get("results", {})
            print(f"[run_one_dataset] Resuming from checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[run_one_dataset] Failed to load checkpoint ({e}), starting fresh.")
            results = {}
    else:
        results = {}

    # Helper: check whether a given (mask_name, rate, rep) is already done
    def _already_done(mask_name, rate, rep) -> bool:
        if mask_name not in results:
            return False
        if rate not in results[mask_name]:
            return False
        return rep in results[mask_name][rate]

    # Helper: save checkpoint after each repetition
    def _save_checkpoint():
        if ckpt_path is None:
            return
        state = dict(
            results=results,
            mask_names=mask_names,
            missing_rates=missing_rates,
            n_repeats=n_repeats,
        )
        with open(ckpt_path, "wb") as f:
            pickle.dump(state, f)

    # Helper: update progress text file
    def _update_progress(mask_name, rate, rep):
        if progress_path is None:
            return
        msg = (
            f"dataset={dataset_label} | "
            f"mask={mask_name} | rate={rate:.2f} | rep={rep+1}/{n_repeats}\n"
        )
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                f.write(msg)
        except OSError:
            pass

    # Precompute index lookups for deterministic per-combination seeds
    mask_index = {name: i for i, name in enumerate(mask_names)}
    rate_index = {r: i for i, r in enumerate(missing_rates)}

    # ---------------------------------------------------------------
    # Main loops
    # ---------------------------------------------------------------
    for mask_name in mask_names:
        if mask_name not in results:
            results[mask_name] = {}
        for rate in missing_rates:
            if rate not in results[mask_name]:
                results[mask_name][rate] = {}

            for rep in range(n_repeats):
                # Skip if already done in a previous run (checkpoint)
                if _already_done(mask_name, rate, rep):
                    continue

                # Deterministic seed per (mask, rate, rep)
                midx = mask_index[mask_name]
                ridx = rate_index[rate]
                combo_seed = BASE_SEED + 1000 * rep + 10 * midx + ridx
                set_random_seed(combo_seed)

                # ---- split: use FULL X, y (oracle), but only Xm_train is "visible"
                X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                    X, y, test_size=0.2, val_size=0.2, random_state=combo_seed
                )

                # ---- build missing-rate vector per original feature ----------
                miss_vec = np.full(X.shape[1], rate, dtype=float)

                # ---- generate unstructured missingness on TRAIN/VAL/TEST ----
                gen = _unstructured_mask_fn(mask_name.replace("_pair", ""))

                Xp_train = build_processed(X_train)
                Xp_val   = build_processed(X_val)
                Xp_test  = build_processed(X_test)

                Xm_train, Xpm_train, indep_idx = gen(
                    X_train, Xp_train, miss_vec, independent_feature_idx=None
                )
                Xm_val,   Xpm_val,   _ = gen(
                    X_val, Xp_val, miss_vec, independent_feature_idx=indep_idx
                )
                Xm_test,  Xpm_test,  _ = gen(
                    X_test, Xp_test, miss_vec, independent_feature_idx=indep_idx
                )

                # ---- apply "paired" pattern if requested --------------------
                if _is_paired(mask_name):
                    nfeat = X_train.shape[1]
                    half = max(1, nfeat // 2)
                    pair_map = np.full(nfeat, np.nan)
                    for i in range(half, nfeat):
                        pair_map[i] = np.random.randint(0, half)

                    Xm_train, Xpm_train = generate_paired_missingness(
                        X_train, Xm_train, Xp_train, Xpm_train, pair_map
                    )
                    Xm_val, Xpm_val = generate_paired_missingness(
                        X_val, Xm_val, Xp_val, Xpm_val, pair_map
                    )
                    Xm_test, Xpm_test = generate_paired_missingness(
                        X_test, Xm_test, Xp_test, Xpm_test, pair_map
                    )

                # ---- NOW: fit imputers ONCE on incomplete Xm_train for ALL criteria ----
                for imp in imputers.values():
                    imp.fit(Xm_train)

                # ---- TRAIN-BASED SCALERS FROM *MASKED* TRAIN DATA -----------
                # Only use information that is practically available (Xm_train)
                num_cols_train, _ = _split_feature_types(Xm_train)
                x_zstats_train = _zscore_fit(Xm_train, num_cols_train)

                # y scaling (regression output), based on y_train
                if y_train is not None and pd.api.types.is_numeric_dtype(y_train):
                    mean_y_train = float(y_train.mean())
                    std_y_train = float(y_train.std(ddof=0)) or 1.0
                    y_stats_train = (mean_y_train, std_y_train)
                else:
                    y_stats_train = None
                print(y_stats_train)

                # ---- Baseline (test set; direct error vs truth) --------------
                baseline_out = validate_baseline_direct_error(
                    X_test_complete=X_test,
                    X_test_masked=Xm_test,
                    imputer_dict=imputers,
                    # scaler based on Xm_train (available knowledge)
                    num_cols=num_cols_train,
                    zstats=x_zstats_train,
                )

                # ---- Criterion 1.1 (MCAR proxy on VAL complete-case) --------
                Xcc = Xm_val.dropna(axis=0, how="any")   # complete-case subset from *masked* VAL
                if Xcc.shape[0] == 0:
                    c11_out = {
                        name: {
                            "per_feature_full": {},
                            "per_feature_missing_only": {},
                            "mean_full": np.nan,
                            "mean_missing_only": np.nan,
                        }
                        for name in imputers
                    }
                else:
                    Xccp = build_processed(Xcc)
                    Xm_cc, Xpm_cc, _ = generate_MCAR(
                        Xcc, Xccp, miss_vec, independent_feature_idx=None
                    )
                    c11_out = validate_proxy_complete_case(
                        imputer_dict=imputers,
                        prebuilt_complete_case=Xcc,
                        prebuilt_masked=Xm_cc,
                        num_cols=num_cols_train,
                        zstats=x_zstats_train,
                    )

                # ---- Criterion 1.2 (mechanism-aligned proxy on VAL) ---------
                Xcc = Xm_val.dropna(axis=0, how="any")
                if Xcc.shape[0] == 0:
                    c12_out = {
                        name: {
                            "per_feature_full": {},
                            "per_feature_missing_only": {},
                            "mean_full": np.nan,
                            "mean_missing_only": np.nan,
                        }
                        for name in imputers
                    }
                else:
                    Xccp = build_processed(Xcc)
                    Xm_cc, Xpm_cc, _ = gen(
                        Xcc, Xccp, miss_vec, independent_feature_idx=indep_idx
                    )
                    if _is_paired(mask_name):
                        Xm_cc, Xpm_cc = generate_paired_missingness(
                            Xcc, Xm_cc, Xccp, Xpm_cc, pair_map
                        )

                    c12_out = validate_proxy_complete_case(
                        imputer_dict=imputers,
                        prebuilt_complete_case=Xcc,
                        prebuilt_masked=Xm_cc,
                        num_cols=num_cols_train,
                        zstats=x_zstats_train,
                    )

                # ---- Criterion 2 (downstream on VAL) -------------------------
                c2_out = validate_downstream_performance(
                    X_train_incomplete=Xm_train,
                    y_train=y_train if y_train is not None else pd.Series(np.zeros(len(X_train))),
                    X_val_incomplete=Xm_val,
                    y_val=y_val if y_val is not None else pd.Series(np.zeros(len(X_val))),
                    imputer_dict=imputers,
                    model_dict=model_dict,
                    num_cols=num_cols_train,
                    x_zstats=x_zstats_train,
                    y_stats=y_stats_train,
                )

                results[mask_name][rate][rep] = dict(
                    baseline=baseline_out,
                    criterion1_mcar=c11_out,
                    criterion1_mechanism=c12_out,
                    criterion2=c2_out,
                )

                # Optional external progress callback (e.g. per-block .txt)
                if progress_callback is not None:
                    progress_callback(mask_name, rate, rep + 1, n_repeats)

                _update_progress(mask_name, rate, rep)
                _save_checkpoint()


    return results

#%%
# -----------------------------------------------------------------------------
# Utility: flatten nested results → long DataFrame
# -----------------------------------------------------------------------------
def flatten_results(results: dict) -> pd.DataFrame:
    """
    Convert nested 'run_one_dataset' results into a long-form DataFrame.

    Output columns:
      mask, rate, rep, criterion, scope, imputer, feature, error
    """
    rows = []
    for mask_name, by_rate in results.items():
        for rate, by_rep in by_rate.items():
            for rep, blocks in by_rep.items():
                # Baseline & Criterion 1.x: per-feature errors
                for crit_key in ("baseline", "criterion1_mcar", "criterion1_mechanism"):
                    crit = blocks.get(crit_key, {})
                    for imp, d in crit.items():
                        for scope in ("per_feature_full", "per_feature_missing_only"):
                            feats = d.get(scope, {})
                            for feat, err in feats.items():
                                rows.append(
                                    dict(
                                        mask=mask_name,
                                        rate=rate,
                                        rep=rep,
                                        criterion=crit_key,
                                        scope=scope,
                                        imputer=imp,
                                        feature=feat,
                                        error=float(err),
                                    )
                                )
                # Criterion 2: per-model errors
                c2 = blocks.get("criterion2", {})
                for imp, d in c2.items():
                    per_model = d.get("per_model", {})
                    for model, err in per_model.items():
                        rows.append(
                            dict(
                                mask=mask_name,
                                rate=rate,
                                rep=rep,
                                criterion="criterion2",
                                scope="per_model",
                                imputer=imp,
                                feature=model,
                                error=float(err),
                            )
                        )
    return pd.DataFrame(rows)
#%%
# ---------- example __main__ --------------------------------------------------
if __name__ == "__main__":
    # Example usage (replace with your real dataset + imputers)
    # X, y = <load your DataFrame and target Series here>
    # imputer_dict = <your six imputers: mean/mode, hot-deck, KNN, PMM, LGBM, CatBoost>

    # mask_names = ["MCAR", "MAR", "MNAR", "MCAR_pair", "MAR_pair", "MNAR_pair"]
    # rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    # res = run_one_dataset(
    #     X=X,
    #     y=y,
    #     mask_names=mask_names,
    #     missing_rates=rates,
    #     n_repeats=100,
    #     imputer_dict=wrap_imputers_with_standardize(imputer_dict),
    #     use_standardize_before_impute=False,   # already wrapped above
    # )
    # # Save results for your figure scripts
    # import pickle
    # with open("results_dataset.pkl", "wb") as f:
    #     pickle.dump(res, f)
    pass
# %%

#%%
# =============================================================================
# plot_for_validation_v2.py
# =============================================================================
"""
Utilities for loading and aggregating validation results for plotting.

This script is designed to work with the block-level outputs produced by
`run_one_block.py`, which saves files of the form:

    results/<DatasetName>/
        dataset<id>_<DatasetName>_mask-<mask>_rate-<rate>.pkl

Each .pkl file contains a dictionary with keys:
    - 'dataset_id'
    - 'dataset_name'
    - 'mask_name'
    - 'missing_rate'
    - 'n_repeats'
    - 'results'   (nested dict returned by `run_one_dataset` in validation_v2)

We use `validation_v2.flatten_results(...)` to convert each nested result
into a long-form DataFrame with columns:

    mask, rate, rep, criterion, scope, imputer, feature, error

and then add `dataset_id` and `dataset_name` so that all blocks can be
concatenated.

The plotting functions for Fig. 5–7 and the appendix figures will build
on top of these long-form DataFrames.
"""

# =============================================================================
# Imports
# =============================================================================
import os
import glob
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List

import numpy as np
import pandas as pd

from validation_v2 import flatten_results

# =============================================================================
# Configuration: dataset order, masks, and rates (paper convention)
# =============================================================================

# Paper order for the six benchmark datasets
DATASET_ORDER: List[tuple[int, str]] = [
    (1, "Concrete"),
    (2, "Composite"),
    (3, "Steel"),
    (4, "Energy"),
    (5, "Student"),
    (6, "Wine"),
]

# Names of missingness mechanisms used in the experiments
VALID_MASK_NAMES: List[str] = [
    "MCAR",
    "MAR",
    "MNAR",
    "MCAR_pair",
    "MAR_pair",
    "MNAR_pair",
]

# Missing rates used in the experiments
MISSING_RATES: List[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# =============================================================================
# Block-level loaders
# =============================================================================

def _load_block_file(path: str) -> pd.DataFrame:
    """
    Load a single block .pkl file and return a long-form DataFrame.

    Parameters
    ----------
    path : str
        Path to a file produced by `run_one_block.py`.

    Returns
    -------
    df : pd.DataFrame
        Long-form DataFrame with columns
        (mask, rate, rep, criterion, scope, imputer, feature, error)
        plus dataset_id and dataset_name.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    results_nested = payload["results"]  # dict: results[mask][rate][rep]
    df = flatten_results(results_nested)

    # Attach dataset metadata
    df["dataset_id"] = int(payload.get("dataset_id", np.nan))
    df["dataset_name"] = str(payload.get("dataset_name", "unknown"))

    return df


def load_dataset_results(results_root: str, dataset_name: str) -> pd.DataFrame:
    """
    Load and merge all block-level results for a single dataset.

    Parameters
    ----------
    results_root : str
        Root directory where the 'results' folders live (e.g. "results").
    dataset_name : str
        Dataset name / subfolder name, e.g. "Concrete", "Composite", ...

    Returns
    -------
    df : pd.DataFrame
        Long-form DataFrame with one row per
        (mask, rate, rep, criterion, scope, imputer, feature).
    """
    ds_dir = os.path.join(results_root, dataset_name.replace(" ", "_"))
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

    # Only pick up the block files; ignore backup/ and progress .txt files.
    pattern = os.path.join(ds_dir, "dataset*_mask-*_rate-*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No block files found under {ds_dir!r}")

    frames: List[pd.DataFrame] = []
    for path in files:
        frames.append(_load_block_file(path))

    df = pd.concat(frames, ignore_index=True)

    # Enforce some consistent dtypes for later grouping
    df["mask"] = df["mask"].astype("category")
    df["criterion"] = df["criterion"].astype("category")
    df["scope"] = df["scope"].astype("category")
    df["imputer"] = df["imputer"].astype("category")

    # Ensure rate and rep are numeric
    df["rate"] = df["rate"].astype(float)
    df["rep"] = df["rep"].astype(int)

    return df


def load_all_flat_results(results_root: str = "results") -> Dict[str, pd.DataFrame]:
    """
    Convenience helper: load all six datasets in paper order.

    Parameters
    ----------
    results_root : str, default "results"
        Root directory created by `run_parallel_blocks.py` /
        `run_one_block.py`.

    Returns
    -------
    out : dict[str, pd.DataFrame]
        Mapping from dataset name to long-form DataFrame.
        Keys: "Concrete", "Composite", "Steel", "Energy", "Student", "Wine".
    """
    out: Dict[str, pd.DataFrame] = {}
    for ds_id, name in DATASET_ORDER:
        df = load_dataset_results(results_root, name)
        # sanity: overwrite dataset_id with our configured id (in case)
        df["dataset_id"] = ds_id
        out[name] = df
    return out
#%%
results_root = "results"
flat_results = load_all_flat_results(results_root)
#%%
flat_results['Concrete']['scope'].unique()
#%%
# reconstrcute the flat_results to following structure
# flat_results is a dict, with keys representing dataset names
# under each key is a dataframe containing following columns:
# mask (missingness mask), rate (missing rate), rep (repetition index),
# criterion (baseline or other criteria), scope (overall or per-feature),
# imputer (imputer name), feature (feature name for criterion 1.1, 1.2 and baseline, and name of model for criterion 2),
# error (imputation error). all these columns will be reconstructed following the old structure:
# each key in the flat_results save as a list
# so results_dict_1 represents the results of dataset Concrete, 2 Composite, 3 Steel, 4 Energy, 5 Student, 6 Wine
# each list has two layers of index:
# first layer is missing rate and repitition, 6 missing rates and 100 repititions
# 0-99 is 5% missing rate, 100-199 is 10% missing rate, ..., 500-599 is 30% missing rate
# second layer is about criterion and error calculation method
# 0,1 is baseline, 2,3 is criterion 2, 4,5 is criterion 1.1, 6,7 is criterion 1.2
# even number 0,2,4,6 is overall error, odd number 1,3,5,7 is error on missing entries only
# under each combination of indexes is the dataframe for error
# each dataframe has 36 rows, representing 6 imputers and 6 missingness masks
# so 0-5 is Mean/Mode, Hot_deck, KNN, CB, PMM, LGBM under MCAR mask
# 6-11 is same 6 imputers under MAR mask, 12-17 is same 6 imputers under MNAR mask
# 18-23 is MCAR_pair mask, 24-29 is MAR_pair mask, 30-35 is MNAR_pair mask
# index name is the imputer name + mask name, columns are the feature names for criterion 1.1, 1.2 and baseline
# and name of model for criterion 2
def reconstruct_old_structure(flat_results: Dict[str, pd.DataFrame]) -> Dict[str, List[List[pd.DataFrame]]]:
    results_dict: Dict[str, List[List[pd.DataFrame]]] = {}

    for ds_name, df in flat_results.items():
        # Initialize the nested list structure
        nested_list: List[List[pd.DataFrame]] = [
            [None for _ in range(8)]  # 8 combinations of criterion and scope
            for _ in range(6 * 100)   # 6 missing rates * 100 repetitions
        ]

        for rate_idx, rate in enumerate(MISSING_RATES):
            for rep in range(100):
                base_idx = rate_idx * 100 + rep

                for crit_scope_idx, (criterion, scope) in enumerate([
                    ("baseline", "per_feature_full"),
                    ("baseline", "per_feature_missing_only"),
                    ("criterion_2", "per_feature_full"),
                    ("criterion_2", "per_feature_missing_only"),
                    ("criterion_1.1", "per_feature_full"),
                    ("criterion_1.1", "per_feature_missing_only"),
                    ("criterion_1.2", "per_feature_full"),
                    ("criterion_1.2", "per_feature_missing_only"),
                ]):
                    # Filter the DataFrame for the current combination
                    df_filtered = df[
                        (df["rate"] == rate) &
                        (df["rep"] == rep) &
                        (df["criterion"] == criterion) &
                        (df["scope"] == scope)
                    ]

                    # Pivot the DataFrame to have imputers + masks as index and features/models as columns
                    df_pivot = df_filtered.pivot_table(
                        index=["imputer", "mask"],
                        columns="feature",
                        values="error"
                    )

                    # Flatten MultiIndex columns if necessary
                    if isinstance(df_pivot.columns, pd.MultiIndex):
                        df_pivot.columns = df_pivot.columns.get_level_values(0)

                    # Store in the nested list
                    nested_list[base_idx][crit_scope_idx] = df_pivot

        results_dict[ds_name] = nested_list

    return results_dict
#%%
results_dict = reconstruct_old_structure(flat_results)
#%%
results_dict['Concrete'][0][1]
#%%
# =============================================================================
# Aggregation for Figure 5: overall numeric / categorical errors
# =============================================================================

from datasets import load_all_datasets  # uses the local datasets.py


def get_feature_types():
    """
    Return a mapping: dataset_name -> {'num': [...], 'cat': [...]}
    based on dtypes from datasets.py.
    """
    X_list, y_list, names = load_all_datasets()
    feat_types = {}
    for X, nm in zip(X_list, names):
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        feat_types[nm] = dict(num=num_cols, cat=cat_cols)
    return feat_types


def aggregate_for_figure5(flat_results):
    """
    Build four DataFrames needed for Figure 5:

    - df_mean, df_std: overall RMSE (numeric features only)
      index = datasets, columns = imputers
    - df_mean_cat, df_std_cat: overall categorical error
      (1 - accuracy, categorical features only)
      index = ['Student', 'Energy'], columns = imputers

    We use:
      * criterion == 'baseline'
      * scope == 'per_feature_missing_only'
      * average over all masks, rates, repetitions and features
        (restricted to numeric / categorical sets per dataset).
    """
    feat_types = get_feature_types()

    # Consistent dataset / imputer order
    dataset_order = ["Concrete", "Composite", "Steel", "Energy", "Student", "Wine"]
    # Use the same imputer order as in the paper / old code
    imputer_order = ["Mean_Mode", "Hot_deck", "KNN", "CB", "PMM", "LGBM"]

    # Check which imputers actually exist in the results
    some_df = next(iter(flat_results.values()))
    present_imputers = set(some_df["imputer"].unique())
    imputer_order = [imp for imp in imputer_order if imp in present_imputers]

    df_mean = pd.DataFrame(index=dataset_order, columns=imputer_order, dtype=float)
    df_std  = pd.DataFrame(index=dataset_order, columns=imputer_order, dtype=float)

    df_mean_cat = pd.DataFrame(index=["Student", "Energy"],
                               columns=imputer_order, dtype=float)
    df_std_cat  = pd.DataFrame(index=["Student", "Energy"],
                               columns=imputer_order, dtype=float)

    for ds_name in dataset_order:
        df_ds = flat_results[ds_name]
        num_cols = feat_types[ds_name]["num"]
        cat_cols = feat_types[ds_name]["cat"]

        # common baseline filter: per-feature error on *missing* entries
        base_mask = (
            (df_ds["criterion"] == "baseline")
            & (df_ds["scope"] == "per_feature_missing_only")
        )

        for imp in imputer_order:
            df_imp = df_ds[base_mask & (df_ds["imputer"] == imp)]

            # ---------- numeric ----------
            if num_cols:
                vals_num = df_imp[df_imp["feature"].isin(num_cols)]["error"].values
                if len(vals_num) > 0:
                    df_mean.loc[ds_name, imp] = float(np.mean(vals_num))
                    df_std.loc[ds_name, imp]  = float(np.std(vals_num, ddof=1))
                else:
                    df_mean.loc[ds_name, imp] = np.nan
                    df_std.loc[ds_name, imp]  = np.nan
            else:
                df_mean.loc[ds_name, imp] = np.nan
                df_std.loc[ds_name, imp]  = np.nan

            # ---------- categorical (only for Student + Energy) ----------
            if ds_name in ("Student", "Energy") and cat_cols:
                vals_cat = df_imp[df_imp["feature"].isin(cat_cols)]["error"].values
                if len(vals_cat) > 0:
                    df_mean_cat.loc[ds_name, imp] = float(np.mean(vals_cat))
                    df_std_cat.loc[ds_name, imp]  = float(np.std(vals_cat, ddof=1))
                else:
                    df_mean_cat.loc[ds_name, imp] = np.nan
                    df_std_cat.loc[ds_name, imp]  = np.nan

    return df_mean, df_std, df_mean_cat, df_std_cat

# =============================================================================
# Figure 5: overall imputation error (numeric vs categorical)
# =============================================================================

def plot_figure5(df_mean, df_std, df_mean_cat, df_std_cat,
                 outdir="figures", fname="figure5_overall.pdf"):
    """
    Re-implementation of the old Figure 5 using the new flat results.

    Left panel  : numeric features (all six datasets)
    Right panel : categorical features (Student + Energy only)

    Error bars are ±1 std across masks, rates, repetitions and features.
    """
    os.makedirs(outdir, exist_ok=True)

    datasets_num = df_mean.index.tolist()          # 6 datasets
    datasets_cat = df_mean_cat.index.tolist()      # ['Student', 'Energy']
    imputers     = df_mean.columns.tolist()

    # x positions for datasets
    x_num = np.arange(len(datasets_num))
    x_cat = np.arange(len(datasets_cat))

    # marker / style per imputer (roughly matching the old figure)
    plot_imputer_marker = {
        "Mean_Mode": dict(marker="o",  linestyle="--"),
        "Hot_deck": dict(marker="s",  linestyle="--"),
        "KNN":       dict(marker="^", linestyle="--"),
        "CB":        dict(marker="v", linestyle="-"),
        "PMM":       dict(marker="<", linestyle="-"),
        "LGBM":      dict(marker=">", linestyle="-"),
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=False
    )

    # -----------------------------
    # Left: numeric features
    # -----------------------------
    for imp in imputers:
        style = plot_imputer_marker.get(imp, dict(marker="o", linestyle="-"))
        means = df_mean[imp].values
        stds  = df_std[imp].values
        ax1.errorbar(
            x_num, means, yerr=stds,
            label=imp,
            capsize=3,
            **style,
        )

    ax1.set_xticks(x_num)
    ax1.set_xticklabels(datasets_num, rotation=45, ha="right")
    ax1.set_ylabel("RMSE on missing entries (std-y)")
    ax1.set_title("Numerical features")

    # -----------------------------
    # Right: categorical features
    # -----------------------------
    for imp in imputers:
        style = plot_imputer_marker.get(imp, dict(marker="o", linestyle="-"))
        means = df_mean_cat[imp].values
        stds  = df_std_cat[imp].values
        ax2.errorbar(
            x_cat, means, yerr=stds,
            label=imp,
            capsize=3,
            **style,
        )

    ax2.set_xticks(x_cat)
    ax2.set_xticklabels(datasets_cat, rotation=45, ha="right")
    ax2.set_ylabel("1 − accuracy on missing entries")
    ax2.set_title("Categorical features")

    # Put legend under the two panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_path = os.path.join(outdir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Figure 5] Saved to: {out_path}")

# =============================================================================
# Simple summary + Figure 5 when run as a script
# =============================================================================

def main():
    results_root = "results"  # top-level results folder

    # 1) Load flat results for all datasets
    flat_results = load_all_flat_results(results_root)

    # 2) Print a quick sanity summary (what you already saw)
    for name, df in flat_results.items():
        nrows = len(df)
        n_masks = df["mask"].nunique()
        n_rates = df["rate"].nunique()
        n_reps = df["rep"].nunique()
        n_imps = df["imputer"].nunique()
        print(
            f"{name:<9} -> {nrows:7d} rows | "
            f"masks= {n_masks:<2d} | "
            f"rates= {n_rates:<2d} | "
            f"reps={n_reps:<3d} | "
            f"imputers= {n_imps:<2d}"
        )

    # 3) Aggregate for Figure 5
    df_mean, df_std, df_mean_cat, df_std_cat = aggregate_for_figure5(flat_results)

    # Optional: quick echo of the summary tables
    print("\n[Figure 5] Numeric RMSE (means):")
    print(df_mean.round(4))
    print("\n[Figure 5] Categorical error (means):")
    print(df_mean_cat.round(4))

    # 4) Plot and save Figure 5
    plot_figure5(df_mean, df_std, df_mean_cat, df_std_cat,
                 outdir="figures", fname="figure5_overall.pdf")


if __name__ == "__main__":
    main()

# %%
results_dict['Concrete'][0][1].loc[('CatBoost', 'MAR')]
# %%
results_dict['Concrete'][0][7]
# %%
flat_results['Concrete'][flat_results['Concrete']['criterion']=='criterion_2'].head()
# %%

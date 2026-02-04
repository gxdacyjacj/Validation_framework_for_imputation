"""
plot_for_validation_v2.py

End-to-end script to:

  • load block-level results from `run_one_block.py`,
  • aggregate them into long-form DataFrames,
  • generate Figures 5, 6 and 7, and
  • export the appendix tables for Figures 5 and 6.

Expected directory layout (relative to this script):

  results/
      Concrete/
          dataset1_Concrete_mask-*_rate-*.pkl
      Composite/
      Steel/
      Energy/
      Student/
      Wine/

  datasets.py    (provides `load_all_datasets()`)

This script assumes the same dataset order and criterion / scope names
as used in validation_v2.
"""

# =============================================================================
# Imports
# =============================================================================

import glob
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


from datasets import load_all_datasets
from validation_v2 import flatten_results
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# Configuration
# =============================================================================

# Dataset order used throughout the paper
DATASET_ORDER: List[Tuple[int, str]] = [
    (1, "Concrete"),
    (2, "Composite"),
    (3, "Steel"),
    (4, "Energy"),
    (5, "Student"),
    (6, "Wine"),
]

# Names of missingness mechanisms
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

# Imputer display order
METHOD_ORDER = ["Mean_Mode", "Hot_deck", "KNN", "CB", "PMM", "LGBM"]

# Locations for input results and exported outputs
RESULTS_ROOT = "results"
OUTPUT_ROOT = "figures_and_tables"

# Global feature lists (initialised in main())
CAT_FEATURES: Dict[str, List[str]] = {}
NUM_FEATURES_FIG5: Dict[str, List[str]] = {}
NUM_FEATURES_FOR_FIGS: Dict[str, List[str]] = {}

# =============================================================================
# I. Loading and flattening block-level results
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
        Long-form DataFrame with columns:
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
        Root directory where the 'results' folders live.
    dataset_name : str
        Dataset name / subfolder name, e.g. "Concrete".

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

    df["rate"] = df["rate"].astype(float)
    df["rep"] = df["rep"].astype(int)

    return df


def load_all_flat_results(results_root: str = RESULTS_ROOT) -> Dict[str, pd.DataFrame]:
    """
    Load all six datasets in paper order and return a dict mapping
    dataset name -> long-form DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    for ds_id, name in DATASET_ORDER:
        df = load_dataset_results(results_root, name)
        df["dataset_id"] = ds_id  # enforce consistent id
        out[name] = df
    return out


# =============================================================================
# II. Dataset feature information
# =============================================================================


def build_feature_lists(X_list: List[pd.DataFrame]) -> Tuple[
    Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]
]:
    """
    Build feature-type dictionaries from the dataset list returned by
    `load_all_datasets()`.

    Returns
    -------
    cat_features : dict[str, list[str]]
        Categorical features for datasets that have them (Energy, Student).
    num_features_fig5 : dict[str, list[str]]
        Feature sets used for Figure 5:
          - for Energy/Student: all non-categorical columns,
          - otherwise: numeric columns.
    num_features_figs : dict[str, list[str]]
        Numeric feature sets used for Figure 6.
    """
    cat_features: Dict[str, List[str]] = {}
    num_features_fig5: Dict[str, List[str]] = {}
    num_features_figs: Dict[str, List[str]] = {}

    id_to_X = {ds_id: X_list[ds_id - 1] for ds_id, _ in DATASET_ORDER}

    for ds_id, name in DATASET_ORDER:
        X = id_to_X[ds_id]

        cats = X.select_dtypes(include=["object"]).columns.tolist()
        if name in ("Energy", "Student"):
            cat_features[name] = cats

        # For Figure 5: remove categoricals if present, otherwise just numeric
        if cats:
            num_cols_fig5 = X.columns.difference(cats).tolist()
        else:
            num_cols_fig5 = X.select_dtypes(include=[np.number]).columns.tolist()
        num_features_fig5[name] = num_cols_fig5

        # For Figure 6: purely numeric
        num_cols_figs = X.select_dtypes(include=[np.number]).columns.tolist()
        num_features_figs[name] = num_cols_figs

    return cat_features, num_features_fig5, num_features_figs


# =============================================================================
# III. Common helpers
# =============================================================================


def _get_scope_label_for_missing(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask selecting 'error on missing entries only' rows.

    The preferred label is 'per_feature_missing_only'. If that label
    is not present, we fall back to a generic search for 'missing'.
    """
    scope = df["scope"].astype(str).str.lower()

    if "per_feature_missing_only" in scope.unique():
        return scope == "per_feature_missing_only"

    mask = scope.str.contains("missing", case=False)
    if not mask.any():
        return pd.Series(True, index=df.index)

    return mask


def _get_baseline_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting baseline criterion rows."""
    crit_str = df["criterion"].astype(str).str.lower()
    candidates = ["baseline", "base", "ground_truth"]
    mask = crit_str.isin(candidates)
    if not mask.any():
        raise ValueError("Could not find baseline criterion in 'criterion' column.")
    return mask


def _imputer_display_name(raw_name: str) -> str:
    """
    Map raw imputer code name to the display label used in the paper.
    """
    name = raw_name.lower()
    if "mean" in name or "mode" in name:
        return "Mean_Mode"
    if "deck" in name:
        return "Hot_deck"
    if "knn" in name:
        return "KNN"
    if "cat" in name:
        return "CB"
    if "lgbm" in name or "lightgbm" in name or "lgb" in name:
        return "LGBM"
    if "pmm" in name or "bayesridge" in name or "iterative" in name:
        return "PMM"
    return raw_name


# =============================================================================
# IV. Figure 5: Baseline error at 5% MAR_pair
# =============================================================================


def prepare_fig5_from_flat_results(
    flat_results: Dict[str, pd.DataFrame],
    rate: float = 0.05,
    mask_name: str = "MAR_pair",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate long-form `flat_results` into the wide tables required for Fig. 5.

    Parameters
    ----------
    flat_results : dict[str, DataFrame]
        Mapping dataset name -> long-form DF with columns
        (mask, rate, rep, criterion, scope, imputer, feature, error, ...).
    rate : float, default 0.05
        Missing rate to be visualised (5% in the paper).
    mask_name : str, default "MAR_pair"
        Masking procedure to be visualised.

    Returns
    -------
    df_mean_num, df_std_num, df_mean_cat, df_std_cat
    """
    datasets_num = ["Concrete", "Composite", "Steel", "Energy", "Student", "Wine"]
    datasets_cat = ["Student", "Energy"]

    df_mean = pd.DataFrame(index=datasets_num, columns=METHOD_ORDER, dtype=float)
    df_std = pd.DataFrame(index=datasets_num, columns=METHOD_ORDER, dtype=float)
    df_mean_cat = pd.DataFrame(index=datasets_cat, columns=METHOD_ORDER, dtype=float)
    df_std_cat = pd.DataFrame(index=datasets_cat, columns=METHOD_ORDER, dtype=float)

    for ds_name in datasets_num:
        df_ds = flat_results[ds_name]

        mask_ok = df_ds["mask"].astype(str) == mask_name
        rate_ok = np.isclose(df_ds["rate"].astype(float), rate)
        df_sub = df_ds.loc[mask_ok & rate_ok].copy()
        if df_sub.empty:
            raise ValueError(
                f"No rows found for dataset={ds_name}, mask={mask_name}, rate={rate}"
            )

        # Keep only baseline criterion + "missing only" scope
        baseline_mask = _get_baseline_mask(df_sub)
        missing_scope_mask = _get_scope_label_for_missing(df_sub)
        df_sub = df_sub.loc[baseline_mask & missing_scope_mask].copy()

        # ----- numerical features -----
        num_feats = NUM_FEATURES_FIG5[ds_name]
        df_num = df_sub[df_sub["feature"].isin(num_feats)].copy()
        if not df_num.empty:
            grp = (
                df_num.groupby(["imputer", "rep"], observed=True)["error"]
                .mean()
                .reset_index()
            )
            summary = grp.groupby("imputer", observed=True)["error"].agg(["mean", "std"])

            for raw_imp, row in summary.iterrows():
                label = _imputer_display_name(str(raw_imp))
                if label not in df_mean.columns:
                    continue
                df_mean.loc[ds_name, label] = row["mean"]
                df_std.loc[ds_name, label] = row["std"]

        # ----- categorical features (Student & Energy only) -----
        if ds_name in CAT_FEATURES:
            cat_feats = CAT_FEATURES[ds_name]
            df_cat = df_sub[df_sub["feature"].isin(cat_feats)].copy()
            if not df_cat.empty:
                grp_cat = (
                    df_cat.groupby(["imputer", "rep"], observed=True)["error"]
                    .mean()
                    .reset_index()
                )
                summary_cat = grp_cat.groupby("imputer", observed=True)["error"].agg(
                    ["mean", "std"]
                )

                for raw_imp, row in summary_cat.iterrows():
                    label = _imputer_display_name(str(raw_imp))
                    if label not in df_mean_cat.columns:
                        continue
                    df_mean_cat.loc[ds_name, label] = row["mean"]
                    df_std_cat.loc[ds_name, label] = row["std"]

    return df_mean, df_std, df_mean_cat, df_std_cat


def make_figure5(flat_results: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Create Figure 5: baseline performance for numerical and categorical
    features at 5% MAR_pair missingness.
    """
    df_mean, df_std, df_mean_cat, df_std_cat = prepare_fig5_from_flat_results(
        flat_results, rate=0.05, mask_name="MAR_pair"
    )

    datasets_num = ["Concrete", "Composite", "Steel", "Energy", "Student", "Wine"]
    datasets_cat = ["Student", "Energy"]
    all_datasets = datasets_num + datasets_cat

    x_locs = np.arange(len(all_datasets))
    methods = [m for m in METHOD_ORDER if m in df_mean.columns]
    colors = plt.get_cmap("tab10").colors

    def plot_imputer_marker(ax, x, y, yerr, color, **kwargs):
        return ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            ms=5,
            capsize=3,
            lw=1.0,
            color=color,
            **kwargs,
        )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    n_methods = len(methods)
    dodge_width = 0.5
    offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n_methods)

    # numerical part (RMSE)
    for i, method in enumerate(methods):
        x_num = x_locs[: len(datasets_num)] + offsets[i]
        y_num = df_mean.loc[datasets_num, method].astype(float).values
        yerr_num = df_std.loc[datasets_num, method].astype(float).values
        plot_imputer_marker(ax1, x_num, y_num, yerr_num, color=colors[i])

    # categorical part (classification error)
    for i, method in enumerate(methods):
        x_cat = x_locs[len(datasets_num) :] + offsets[i]
        y_cat = df_mean_cat.loc[datasets_cat, method].astype(float).values
        yerr_cat = df_std_cat.loc[datasets_cat, method].astype(float).values
        plot_imputer_marker(ax2, x_cat, y_cat, yerr_cat, color=colors[i])

    ax1.set_xticks(x_locs)
    ax1.set_xticklabels(all_datasets)
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Mean RMSE")
    ax2.set_ylabel("Mean Classification Error")

    ax1.set_ylim(0.0, 2.0)
    ax2.set_ylim(0.0, 1.0)

    # dashed vertical lines between datasets
    for x in np.arange(len(all_datasets) - 1) + 0.5:
        ax1.axvline(x, color="black", linestyle="--", linewidth=0.8, zorder=0)

    # solid line between numerical and categorical
    sep_x = len(datasets_num) - 0.5
    ax1.axvline(sep_x, color="black", linestyle="-", linewidth=1.2, zorder=1)

    # --- header bars (axes coordinates) ---
    xmin, xmax = ax1.get_xlim()
    to_ax = lambda x: (x - xmin) / (xmax - xmin)  # data-x -> axes fraction

    # numerical spans datasets [0 .. len(datasets_num)-1]
    x0_num = to_ax(xmin)
    x1_num = to_ax(sep_x)

    # categorical spans datasets [len(datasets_num) .. end]
    x0_cat = to_ax(sep_x)
    x1_cat = to_ax(xmax)

    # bar geometry (in axes coords)
    bar_y = 1          # top of axes
    bar_h = 0.055
    pad = 0.00            # small gap around separator

    # rectangles
    ax1.add_patch(
        Rectangle(
            (x0_num + pad, bar_y), (x1_num - x0_num) - 2 * pad, bar_h,
            transform=ax1.transAxes, facecolor="gray", edgecolor="black",
            lw=1, clip_on=False, zorder=10
        )
    )
    ax1.add_patch(
        Rectangle(
            (x0_cat + pad, bar_y), (x1_cat - x0_cat) - 2 * pad, bar_h,
            transform=ax1.transAxes, facecolor="gray", edgecolor="black",
            lw=1, clip_on=False, zorder=10
        )
    )

    # white labels centered in each bar
    ax1.text(
        (x0_num + x1_num) / 2, bar_y + bar_h / 2, "Numerical",
        transform=ax1.transAxes, ha="center", va="center",
        color="white", fontsize=11, zorder=11
    )
    ax1.text(
        (x0_cat + x1_cat) / 2, bar_y + bar_h / 2, "Categorical",
        transform=ax1.transAxes, ha="center", va="center",
        color="white", fontsize=11, zorder=11
    )


    # legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            markersize=6,
            label=method,
        )
        for i, method in enumerate(methods)
    ]

    ax1.legend(
        handles=legend_handles,
        ncol=3,
        loc="upper left",
        frameon=True,
        fontsize=9,
        facecolor="white",
        framealpha=1,
    )

    fig.tight_layout()
    return fig


# =============================================================================
# V. Figure 6: Criteria vs baseline for one mask/rate
# =============================================================================


def _criterion_mask(df: pd.DataFrame, kind: str) -> pd.Series:
    """
    Map logical criterion labels to the actual strings in the `criterion` column.
    """
    c = df["criterion"].astype(str).str.lower().str.strip()

    if kind == "baseline":
        return c == "baseline"
    if kind == "c11":
        return c == "criterion1_mcar"
    if kind == "c12":
        return c == "criterion1_mechanism"
    if kind == "c2":
        return c == "criterion2"

    raise ValueError(f"Unknown kind={kind!r}")


def prepare_fig6_df_for_dataset(
    df_ds: pd.DataFrame,
    dataset_name: str,
    rate: float = 0.05,
    mask_name: str = "MAR_pair",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build df_mean / df_std for one dataset for Figure 6.

    Rows:  Baseline, Criterion 1.1, Criterion 1.2, Criterion 2
    Cols:  Mean_Mode, Hot_deck, KNN, CB, PMM, LGBM

    Definitions:
      • Baseline / Criterion 1.1 / 1.2:
          criterion in {baseline, criterion1_mcar, criterion1_mechanism},
          scope == "per_feature_missing_only",
          averaged over all numeric features and repetitions.

      • Criterion 2:
          criterion == "criterion2",
          scope == "per_model",
          averaged over models and repetitions.

    For each (imputer, criterion) we remove 3σ outliers across repetitions
    before computing the final mean and std.
    """
    df = df_ds.copy()

    mask_ok = df["mask"].astype(str) == mask_name
    rate_ok = np.isclose(df["rate"].astype(float), rate)
    df = df.loc[mask_ok & rate_ok]
    if df.empty:
        raise ValueError(f"No rows for {dataset_name} at mask={mask_name}, rate={rate}")

    rows = ["Baseline", "Criterion 1.1", "Criterion 1.2", "Criterion 2"]
    cols = METHOD_ORDER
    df_mean = pd.DataFrame(index=rows, columns=cols, dtype=float)
    df_std = pd.DataFrame(index=rows, columns=cols, dtype=float)

    num_feats = NUM_FEATURES_FOR_FIGS[dataset_name]

    def _fill_row(row_name: str, crit_value: str, scope_value: str, numeric_only: bool):
        """
        Compute mean/std for a single criterion row, with 3σ outlier removal.
        """
        df_c = df[(df["criterion"] == crit_value) & (df["scope"] == scope_value)].copy()
        if df_c.empty:
            return

        if numeric_only:
            df_c = df_c[df_c["feature"].isin(num_feats)].copy()
            if df_c.empty:
                return

        grp = (
            df_c.groupby(["imputer", "rep"], observed=True)["error"]
            .mean()
            .reset_index()
        )

        for imp_raw, sub in grp.groupby("imputer", observed=True):
            label = _imputer_display_name(str(imp_raw))
            if label not in cols:
                continue

            vals = sub["error"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                df_mean.loc[row_name, label] = np.nan
                df_std.loc[row_name, label] = np.nan
                continue

            mu = np.mean(vals)
            sigma = np.std(vals)

            if sigma == 0:
                cleaned = vals
            else:
                thresh = 2.0 * sigma
                cleaned = vals[np.abs(vals - mu) < thresh]

            if cleaned.size < vals.size:
                print(
                    f"[{dataset_name} - {row_name} - {label}] "
                    f"Removed {vals.size - cleaned.size} outlier(s)"
                )

            if cleaned.size == 0 or np.isnan(cleaned).any():
                df_mean.loc[row_name, label] = np.nan
                df_std.loc[row_name, label] = np.nan
            else:
                df_mean.loc[row_name, label] = float(np.mean(cleaned))
                df_std.loc[row_name, label] = float(np.std(cleaned))

    _fill_row("Baseline", "baseline", "per_feature_missing_only", True)
    _fill_row("Criterion 1.1", "criterion1_mcar", "per_feature_missing_only", True)
    _fill_row("Criterion 1.2", "criterion1_mechanism", "per_feature_missing_only", True)
    _fill_row("Criterion 2", "criterion2", "per_model", False)

    return df_mean, df_std


def plot_fig6_panel(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    title: str | None = None,
    ylim_rmse: Tuple[float, float] | None = None,
    ylim_pred: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (6.5, 4.5),
    ax1=None,
) -> plt.Figure:
    """
    Draw a single Figure-6 style panel (no rank table).
    """
    from matplotlib.lines import Line2D

    criteria = df_mean.index.tolist()
    methods = [m for m in METHOD_ORDER if m in df_mean.columns]
    colors = plt.cm.tab10.colors

    created_fig = ax1 is None
    if created_fig:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig = ax1.figure

    ax2 = ax1.twinx()

    x = np.arange(len(criteria))
    n_methods = len(methods)
    dodge_width = 0.5
    offsets = np.linspace(-dodge_width / 2, dodge_width / 2, n_methods)

    if ylim_rmse is None:
        vals_rmse = df_mean.loc[["Baseline", "Criterion 1.1", "Criterion 1.2"], methods].to_numpy(
            dtype=float
        )
        ymin = 0.0
        ymax = np.nanmax(vals_rmse) * 1.1 if np.isfinite(np.nanmax(vals_rmse)) else 1.0
        ylim_rmse = (ymin, ymax)

    if ylim_pred is None:
        vals_pred = df_mean.loc[["Criterion 2"], methods].to_numpy(dtype=float)
        vmin = np.nanmin(vals_pred)
        vmax = np.nanmax(vals_pred)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            ylim_pred = (0.0, 1.0)
        else:
            ylim_pred = (vmin * 0.98, vmax * 1.02)

    ax1.set_ylim(*ylim_rmse)
    ax2.set_ylim(*ylim_pred)

    ax1.set_ylabel("Mean RMSE")
    ax2.set_ylabel("Prediction Error")
    ax1.set_xlabel("Criteria")

    is_c2 = np.array(criteria) == "Criterion 2"
    not_c2 = ~is_c2

    for i, method in enumerate(methods):
        xi = x + offsets[i]
        yi = df_mean.loc[:, method].values.astype(float)
        yerr = df_std.loc[:, method].values.astype(float)

        ax1.errorbar(
            xi[not_c2],
            yi[not_c2],
            yerr=yerr[not_c2],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            capsize=5,
            lw=1.0,
        )

        ax2.errorbar(
            xi[is_c2],
            yi[is_c2],
            yerr=yerr[is_c2],
            fmt="none",
            ecolor=colors[i],
            capsize=5,
            lw=1.0,
        )
        ax2.scatter(xi[is_c2], yi[is_c2], marker="x", color=colors[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(criteria)

    for k in range(len(criteria) - 1):
        ax1.axvline(k + 0.5, color="black", linestyle="--", linewidth=0.7, zorder=0)

    if title is not None:
        ax1.set_title(title)

    # legend for imputers
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            linestyle="None",
            label=methods[i],
            markersize=6,
        )
        for i in range(len(methods))
    ]
    legend1 = ax1.legend(
        handles=method_handles,
        loc="upper left",
        ncol=3,
        framealpha=1,
        bbox_to_anchor=(0.0, 1.01),
    )

    metric_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="Mean RMSE"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="Prediction Error"),
    ]
    legend2 = ax1.legend(
        handles=metric_handles,
        loc="upper right",
        framealpha=1,
        bbox_to_anchor=(1.0, 1.0),
    )
    ax1.add_artist(legend1)

    if created_fig:
        fig.tight_layout()

    return fig


def assemble_fig6_grid(
    panel_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ylims_rmse: Dict[str, Tuple[float, float]] | None = None,
    ylims_pred: Dict[str, Tuple[float, float]] | None = None,
    panel_figsize: Tuple[float, float] = (6.5, 4.5),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Assemble six Figure-6 panels into a 3x2 grid.
    """
    dataset_order = ["Concrete", "Composite", "Steel", "Energy", "Student", "Wine"]

    caption_map = {
        "Concrete": "(a) Concrete compressive strength",
        "Composite": "(b) Composite material",
        "Steel": "(c) Steel strength",
        "Energy": "(d) Energy efficiency",
        "Student": "(e) Student performance",
        "Wine": "(f) Wine quality",
    }

    if ylims_rmse is None:
        ylims_rmse = {}
    if ylims_pred is None:
        ylims_pred = {}

    nrows, ncols = 3, 2
    panel_w, panel_h = panel_figsize
    fig_w = panel_w * ncols
    fig_h = panel_h * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes_flat = axes.flatten()

    for i, ds_name in enumerate(dataset_order):
        ax = axes_flat[i]
        df_mean, df_std = panel_data[ds_name]

        ylim_r = ylims_rmse.get(ds_name)
        ylim_p = ylims_pred.get(ds_name)

        plot_fig6_panel(df_mean, df_std, title=None, ylim_rmse=ylim_r, ylim_pred=ylim_p, ax1=ax)

        caption = caption_map.get(ds_name, ds_name)
        fontsize = ax.xaxis.label.get_size()
        ax.text(
            0.5,
            -0.15,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fontsize,
        )

    for j in range(len(dataset_order), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.97,
        bottom=0.06,
        wspace=0.25,
        hspace=0.25,
    )

    return fig, axes


def make_figure6(flat_results: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Build all six panels and assemble them into the Figure 6 grid.
    """
    panels = {
        "Concrete": prepare_fig6_df_for_dataset(flat_results["Concrete"], "Concrete", rate=0.05, mask_name="MAR_pair"),
        "Composite": prepare_fig6_df_for_dataset(
            flat_results["Composite"], "Composite", rate=0.05, mask_name="MAR_pair"
        ),
        "Steel": prepare_fig6_df_for_dataset(flat_results["Steel"], "Steel", rate=0.05, mask_name="MAR_pair"),
        "Energy": prepare_fig6_df_for_dataset(flat_results["Energy"], "Energy", rate=0.05, mask_name="MAR_pair"),
        "Student": prepare_fig6_df_for_dataset(flat_results["Student"], "Student", rate=0.05, mask_name="MAR_pair"),
        "Wine": prepare_fig6_df_for_dataset(flat_results["Wine"], "Wine", rate=0.05, mask_name="MAR_pair"),
    }

    # Y-limits tuned to match the original figure
    ylims_rmse = {
        "Concrete": (0.0, 2.0),
        "Composite": (0.0, 2.0),
        "Steel": (0.0, 2.0),
        "Energy": (0.0, 2.0),
        "Student": (0.0, 2.0),
        "Wine": (0.0, 2.0),
    }
    ylims_pred = {
        "Concrete": (0.40, 0.65),
        "Composite": (0.95, 1.15),
        "Steel": (0.35, 0.85),
        "Energy": (0.10, 0.45),
        "Student": (0.70, 1.10),
        "Wine": (0.75, 0.90),
    }

    fig, _ = assemble_fig6_grid(
        panels,
        ylims_rmse=ylims_rmse,
        ylims_pred=ylims_pred,
        panel_figsize=(6.5, 4.5),
    )
    return fig


# =============================================================================
# VI. Appendix tables for Figure 5
# =============================================================================


def _fmt_mean_std(mean: float, std: float, ndigits: int = 2) -> str:
    """Format as 'X.XX ± Y.YY'; return empty string if NaN."""
    if pd.isna(mean) or pd.isna(std):
        return ""
    return f"{mean:.{ndigits}f} ± {std:.{ndigits}f}"


def build_fig5_appendix_tables(
    flat_results: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build long-form tables for Appendix (Figure 5 style) across all
    missing rates, masks and datasets.

    Returns
    -------
    df_num : DataFrame
        Numerical variables.
    df_cat : DataFrame
        Categorical variables (Student, Energy only).
    """
    records_num = []
    records_cat = []

    for rate in MISSING_RATES:
        for mask in VALID_MASK_NAMES:
            df_mean, df_std, df_mean_cat, df_std_cat = prepare_fig5_from_flat_results(
                flat_results, rate=rate, mask_name=mask
            )

            # numerical datasets
            for dataset in df_mean.index:
                for method in METHOD_ORDER:
                    if method not in df_mean.columns:
                        continue
                    mean_val = df_mean.loc[dataset, method]
                    std_val = df_std.loc[dataset, method]
                    if pd.isna(mean_val):
                        continue
                    records_num.append(
                        {
                            "mask": mask,
                            "rate": rate,
                            "dataset": dataset,
                            "variable_type": "numerical",
                            "method": method,
                            "mean_rmse": float(mean_val),
                            "std_rmse": float(std_val),
                        }
                    )

            # categorical datasets
            for dataset in df_mean_cat.index:
                for method in METHOD_ORDER:
                    if method not in df_mean_cat.columns:
                        continue
                    mean_val = df_mean_cat.loc[dataset, method]
                    std_val = df_std_cat.loc[dataset, method]
                    if pd.isna(mean_val):
                        continue
                    records_cat.append(
                        {
                            "mask": mask,
                            "rate": rate,
                            "dataset": dataset,
                            "variable_type": "categorical",
                            "method": method,
                            "mean_error": float(mean_val),
                            "std_error": float(std_val),
                        }
                    )

    df_num = pd.DataFrame.from_records(records_num)
    df_cat = pd.DataFrame.from_records(records_cat)
    return df_num, df_cat


ROW_ORDER_FIG5 = [
    ("Concrete", "numerical"),
    ("Composite", "numerical"),
    ("Steel", "numerical"),
    ("Energy", "numerical"),
    ("Student", "numerical"),
    ("Wine", "numerical"),
    ("Energy", "categorical"),
    ("Student", "categorical"),
]


def build_fig5_block_table(df_num: pd.DataFrame, df_cat: pd.DataFrame, mask: str, rate: float) -> pd.DataFrame:
    """
    Build one appendix block for Figure 5, e.g. "MCAR -- 5% missing rate".

    Returns a DataFrame:

        Dataset | Data type | Mean_Mode | Hot_deck | KNN | CB | PMM | LGBM
    """
    rows = []

    for dataset, vtype in ROW_ORDER_FIG5:
        if vtype == "numerical":
            sub = df_num[
                (df_num["mask"] == mask)
                & (df_num["rate"] == rate)
                & (df_num["dataset"] == dataset)
            ]
            mean_col = "mean_rmse"
            std_col = "std_rmse"
            dtype_label = "Numerical"
        else:
            sub = df_cat[
                (df_cat["mask"] == mask)
                & (df_cat["rate"] == rate)
                & (df_cat["dataset"] == dataset)
            ]
            mean_col = "mean_error"
            std_col = "std_error"
            dtype_label = "Categorical"

        row = {"Dataset": dataset, "Data type": dtype_label}
        if sub.empty:
            for m in METHOD_ORDER:
                row[m] = ""
            rows.append(row)
            continue

        for m in METHOD_ORDER:
            sub_m = sub[sub["method"] == m]
            if sub_m.empty:
                row[m] = ""
            else:
                mean_val = sub_m[mean_col].iloc[0]
                std_val = sub_m[std_col].iloc[0]
                row[m] = _fmt_mean_std(mean_val, std_val, ndigits=2)
        rows.append(row)

    return pd.DataFrame(rows)


def write_fig5_appendix_csvs(df_num: pd.DataFrame, df_cat: pd.DataFrame) -> None:
    """Write 36 CSV tables for Appendix 1 (Figure 5)."""
    out_dir = os.path.join(OUTPUT_ROOT, "Appendix1")
    os.makedirs(out_dir, exist_ok=True)

    for mask in VALID_MASK_NAMES:
        for rate in MISSING_RATES:
            block = build_fig5_block_table(df_num, df_cat, mask, rate)
            path = os.path.join(out_dir, f"fig5_table_{mask}_rate{int(rate * 100)}.csv")
            block.to_csv(path, index=False)


# =============================================================================
# VII. Appendix tables for Figure 6
# =============================================================================


def build_fig6_appendix_table(flat_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build long-form table for Appendix (Figure 6 style) across all
    missing rates, masks and datasets.
    """
    records = []

    for rate in MISSING_RATES:
        for mask in VALID_MASK_NAMES:
            for ds_id, ds_name in DATASET_ORDER:
                df_ds = flat_results[ds_name]
                df_mean, df_std = prepare_fig6_df_for_dataset(
                    df_ds, dataset_name=ds_name, rate=rate, mask_name=mask
                )

                for criterion in df_mean.index:
                    metric_type = "rmse" if criterion != "Criterion 2" else "prediction_error"

                    for method in METHOD_ORDER:
                        if method not in df_mean.columns:
                            continue
                        mean_val = df_mean.loc[criterion, method]
                        std_val = df_std.loc[criterion, method]
                        if pd.isna(mean_val):
                            continue

                        records.append(
                            {
                                "mask": mask,
                                "rate": rate,
                                "dataset": ds_name,
                                "criterion": criterion,
                                "metric": metric_type,
                                "method": method,
                                "mean_error": float(mean_val),
                                "std_error": float(std_val),
                            }
                        )

    return pd.DataFrame.from_records(records)


CRITERION_ORDER = [
    "Baseline",
    "Criterion 1.1",
    "Criterion 1.2",
    "Criterion 2",
]

DATASET_NAME_ORDER = [name for _, name in DATASET_ORDER]


def build_fig6_block_table(
    df_all: pd.DataFrame,
    mask: str,
    rate: float,
    ndigits: int = 2,
) -> pd.DataFrame:
    """
    Build one appendix block for Figure 6, e.g. "MCAR -- 5% Missing rate".

    Each dataset contributes 4 rows (Baseline, 1.1, 1.2, 2).
    """
    rows = []

    sub_mask_rate = df_all[(df_all["mask"] == mask) & (df_all["rate"] == rate)]

    for dataset in DATASET_NAME_ORDER:
        sub_ds = sub_mask_rate[sub_mask_rate["dataset"] == dataset]
        if sub_ds.empty:
            for crit in CRITERION_ORDER:
                row = {"Dataset": dataset, "Validation criteria": crit}
                for m in METHOD_ORDER:
                    row[m] = ""
                rows.append(row)
            continue

        for crit in CRITERION_ORDER:
            sub_crit = sub_ds[sub_ds["criterion"] == crit]
            row = {"Dataset": dataset, "Validation criteria": crit}
            for m in METHOD_ORDER:
                sub_m = sub_crit[sub_crit["method"] == m]
                if sub_m.empty:
                    row[m] = ""
                else:
                    mean_val = sub_m["mean_error"].iloc[0]
                    std_val = sub_m["std_error"].iloc[0]
                    row[m] = _fmt_mean_std(mean_val, std_val, ndigits=ndigits)
            rows.append(row)

    return pd.DataFrame(rows)


def write_fig6_appendix_csvs(df_all: pd.DataFrame) -> None:
    """Write 36 CSV tables for Appendix 2 (Figure 6)."""
    out_dir = os.path.join(OUTPUT_ROOT, "Appendix2")
    os.makedirs(out_dir, exist_ok=True)

    for mask in VALID_MASK_NAMES:
        for rate in MISSING_RATES:
            block = build_fig6_block_table(df_all, mask, rate, ndigits=2)
            path = os.path.join(out_dir, f"fig6_table_{mask}_rate{int(rate * 100)}.csv")
            block.to_csv(path, index=False)


# =============================================================================
# VIII. Figure 7: winner counts heatmaps
# =============================================================================


def _spearman_corr(s1: pd.Series, s2: pd.Series) -> float:
    """
    Compute Spearman rank correlation between two 1D Series.
    Returns np.nan if not computable.
    """
    common = s1.index.intersection(s2.index)
    if len(common) < 2:
        return np.nan

    r1 = s1.loc[common].astype(float).values
    r2 = s2.loc[common].astype(float).values

    if np.isnan(r1).any() or np.isnan(r2).any():
        return np.nan

    r1_rank = pd.Series(r1).rank(method="average").values
    r2_rank = pd.Series(r2).rank(method="average").values

    x = r1_rank - r1_rank.mean()
    y = r2_rank - r2_rank.mean()
    denom = x.std(ddof=0) * y.std(ddof=0)
    if denom == 0:
        return np.nan
    return float((x * y).mean() / denom)


def _build_rank_vector_for_setting(
    df_ds: pd.DataFrame,
    mask_name: str,
    rate: float,
    rep: int,
    crit_kind: str,  # "baseline", "c11", "c12", "c2"
) -> pd.Series | None:
    """
    For a fixed (mask, rate, repetition, criterion-kind) build a vector of
    mean ranks over features/models.

    - baseline / c11 / c12:
        scope == "per_feature_missing_only",
        rank imputers per feature and average ranks across features.

    - c2 (Criterion 2):
        scope == "per_model",
        compute Fano factor per model (var / mean across imputers),
        keep models with sufficient discrimination, then rank imputers
        within the selected models and average ranks.
    """
    if crit_kind in ("baseline", "c11", "c12"):
        scope_val = "per_feature_missing_only"
    elif crit_kind == "c2":
        scope_val = "per_model"
    else:
        raise ValueError(f"Unknown crit_kind={crit_kind!r}")

    sub = df_ds[
        (df_ds["mask"] == mask_name)
        & (df_ds["rate"] == rate)
        & (df_ds["rep"] == rep)
        & (df_ds["scope"] == scope_val)
    ].copy()
    if sub.empty:
        return None

    sub = sub[_criterion_mask(sub, crit_kind)]
    if sub.empty:
        return None

    mat = sub.pivot_table(
        index="imputer",
        columns="feature",  # for c2 this stores model names
        values="error",
        aggfunc="mean",
    )

    mat = mat.dropna(axis=1, how="all")
    if mat.empty or mat.shape[0] < 2:
        return None

    if crit_kind in ("baseline", "c11", "c12"):
        rank_mat = mat.rank(axis=0, method="average", ascending=True)
        mean_rank = rank_mat.mean(axis=1)
        if mean_rank.isna().all():
            return None
        return mean_rank

    # Criterion 2: Fano filtering
    FANO_THRESHOLD = 1e-4

    fanos = []
    col_names = list(mat.columns)
    for col in col_names:
        vals = mat[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            fanos.append(np.nan)
        else:
            fanos.append(np.var(vals) / np.mean(vals))

    fanos = np.array(fanos, dtype=float)
    if np.all(np.isnan(fanos)):
        return None

    if np.all(fanos < FANO_THRESHOLD):
        max_idx = np.nanargmax(fanos)
        keep_cols = [col_names[max_idx]]
    else:
        keep_cols = [
            col for col, f in zip(col_names, fanos) if (not np.isnan(f)) and (f >= FANO_THRESHOLD)
        ]

    if not keep_cols:
        return None

    mat_sel = mat[keep_cols].copy()
    mat_sel = mat_sel.dropna(axis=1, how="all")
    if mat_sel.empty:
        return None

    rank_mat = mat_sel.rank(axis=0, method="average", ascending=True)
    mean_rank = rank_mat.mean(axis=1)
    if mean_rank.isna().all():
        return None

    return mean_rank


def count_winner_flat_for_dataset(
    df_ds: pd.DataFrame,
    dataset_name: str,
    n_reps: int = 100,
    mask_names: List[str] | None = None,
    missing_rates: List[float] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Count how often each practical criterion (1.1, 1.2, 2) achieves the
    highest Spearman rank correlation with the baseline.

    Returns a dict:
        winners['Criterion 1.1'], winners['Criterion 1.2'], winners['Criterion 2']
    where each value is a DataFrame indexed by mask, with columns = missing rates.
    """
    if mask_names is None:
        mask_names = VALID_MASK_NAMES
    if missing_rates is None:
        missing_rates = MISSING_RATES

    crit_labels = ["Criterion 1.1", "Criterion 1.2", "Criterion 2"]
    winners = {
        label: pd.DataFrame(0, index=mask_names, columns=missing_rates, dtype=int)
        for label in crit_labels
    }

    crit_kind_map = {
        "Criterion 1.1": "c11",
        "Criterion 1.2": "c12",
        "Criterion 2": "c2",
    }

    for mask_name in mask_names:
        for rate in missing_rates:
            for rep in range(n_reps):
                base_rank = _build_rank_vector_for_setting(
                    df_ds, mask_name, rate, rep, crit_kind="baseline"
                )
                if base_rank is None:
                    continue

                rhos = {}
                for label in crit_labels:
                    k = crit_kind_map[label]
                    r_vec = _build_rank_vector_for_setting(df_ds, mask_name, rate, rep, crit_kind=k)
                    if r_vec is None:
                        rhos[label] = np.nan
                        continue
                    rhos[label] = _spearman_corr(base_rank, r_vec)

                valid = {k: v for k, v in rhos.items() if not np.isnan(v)}
                if not valid:
                    continue

                max_rho = max(valid.values())
                for label, rho in valid.items():
                    if np.isclose(rho, max_rho, atol=1e-12):
                        winners[label].loc[mask_name, rate] += 1

    return winners


def count_winner_flat_all(
    flat_results: Dict[str, pd.DataFrame],
    n_reps: int = 100,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Apply `count_winner_flat_for_dataset` to all datasets.
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds_name, df_ds in flat_results.items():
        out[ds_name] = count_winner_flat_for_dataset(df_ds, dataset_name=ds_name, n_reps=n_reps)
    return out

import pickle

WINNERS_CACHE_PATH = os.path.join(OUTPUT_ROOT, "winners_all_cache.pkl")

def get_winners_all_cached(
    flat_results: Dict[str, pd.DataFrame],
    n_reps: int = 100,
    cache_path: str = WINNERS_CACHE_PATH,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load winners_all from cache if present; otherwise compute and save.

    Cached object contains full results for:
      - Criterion 1.1
      - Criterion 1.2
      - Criterion 2
    for all datasets, masks, rates.
    """
    if (not force_recompute) and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            winners_all = pickle.load(f)
        return winners_all

    winners_all = count_winner_flat_all(flat_results, n_reps=n_reps)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(winners_all, f)

    return winners_all


CRITERIA_FOR_FIG7 = ["Criterion 1.1", "Criterion 1.2", "Criterion 2"]


def plot_figure7_heatmaps(
    winners_all: Dict[str, Dict[str, pd.DataFrame]],
    dataset_names: List[str],
    n_reps: int = 100,
    figsize: Tuple[float, float] = (10, 12),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Generate Figure 7-style heatmaps:

        - 3 datasets (rows)
        - 3 criteria (columns)
        - one colourbar per dataset row
        - dataset caption under each row
    """
    assert len(dataset_names) == 3, "Exactly 3 datasets are required."

    MASK_DISPLAY_NAMES = {
        "MCAR": "MCAR",
        "MAR": "MAR",
        "MNAR": "MNAR",
        "MCAR_pair": "MCAR-p",
        "MAR_pair": "MAR-p",
        "MNAR_pair": "MNAR-p",
    }

    masks = VALID_MASK_NAMES
    rates = MISSING_RATES
    n_rows = len(dataset_names)

    fig, axes = plt.subplots(n_rows, len(CRITERIA_FOR_FIG7), figsize=figsize)
    axes = np.array(axes)

    vmin, vmax = 0, n_reps

    caption_map = {
        "Concrete": "(a) Concrete compressive strength",
        "Composite": "(b) Composite material",
        "Steel": "(c) Steel strength",
        "Energy": "(d) Energy efficiency",
        "Student": "(e) Student performance",
        "Wine": "(f) Wine quality",
    }

    fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.08, wspace=0.15, hspace=0.30)

    for row, ds_name in enumerate(dataset_names):
        last_im = None

        for col, crit_label in enumerate(CRITERIA_FOR_FIG7):
            ax = axes[row, col]

            win_df = winners_all[ds_name][crit_label].reindex(index=masks, columns=rates)
            data = win_df.values.astype(float)

            im = ax.imshow(
                data,
                origin="upper",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap="YlGnBu",
                interpolation="nearest",
            )
            last_im = im

            # y-axis
            ax.set_yticks(np.arange(len(masks)))
            if col == 0:
                ax.set_yticklabels(
                    [MASK_DISPLAY_NAMES[m] for m in masks],
                    rotation=90,
                    fontsize=8,
                    va="center",
                    ha="center",
                )
                ax.set_ylabel("Missing mechanism", fontsize=10)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)

            # x-axis
            ax.set_xticks(np.arange(len(rates)))
            ax.set_xticklabels([f"{r:.2f}" for r in rates], fontsize=8)
            ax.set_xlabel("Missing ratio", fontsize=10)

            # annotate counts (black or white depending on background)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    norm_val = (val - vmin) / (vmax - vmin)
                    cell_color = im.cmap(norm_val)
                    luminance = (
                        0.299 * cell_color[0]
                        + 0.587 * cell_color[1]
                        + 0.114 * cell_color[2]
                    )
                    text_color = "black" if luminance > 0.55 else "white"
                    ax.text(
                        j,
                        i,
                        f"{int(val)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                    )

            if row == 0:
                ax.set_title(crit_label)

        # colourbar for this dataset row
        row_left = axes[row, 0].get_position().x0
        row_right = axes[row, -1].get_position().x1
        row_bottom = min(ax.get_position().y0 for ax in axes[row])
        row_top = max(ax.get_position().y1 for ax in axes[row])
        row_height = row_top - row_bottom

        cax = fig.add_axes([row_right + 0.01, row_bottom, 0.02, row_height])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label(f"Number of wins (out of {n_reps})")

        # dataset caption
        ax_mid = axes[row, 1]
        bbox_mid = ax_mid.get_position()
        cap_x = 0.5 * (bbox_mid.x0 + bbox_mid.x1)
        cap_y = row_bottom - 0.04

        fig.text(
            cap_x,
            cap_y,
            caption_map.get(ds_name, ds_name),
            ha="center",
            va="top",
            fontsize=ax_mid.xaxis.label.get_size(),
        )

    return fig, axes



def draw_region_boundaries(
    ax,
    data: np.ndarray,
    threshold: float = 50,
    color_lo: str = "green",   # < threshold  (C1 dominates)
    color_hi: str = "#e53e3e",     # >= threshold (C2 dominates)
    lw_outer: float = 3.5,
    lw_split: float = 0.2,
    eps: float = 0.03,         # offset in data coords (~cell units)
    zorder: int = 6,
):
    """
    Overlay region boundaries on an imshow heatmap.

    Assumes imshow is called with default pixel-aligned coordinates, i.e.
    cell (i,j) is centered at (j,i) and its edges are at j±0.5, i±0.5.
    Works with origin='upper' too (geometry still holds).

    Boundary rules:
      - Outside boundary: single thick line in the cell's region color
      - Between different regions: two parallel thinner lines (one per region),
        offset slightly to each side of the shared edge.
    """
    data = np.asarray(data)
    nrows, ncols = data.shape
    hi = data >= threshold  # True => C2 wins more (red)

    def c(is_hi: bool) -> str:
        return color_hi if is_hi else color_lo

    # ---------- outer boundary ----------
    for i in range(nrows):
        for j in range(ncols):
            col = c(bool(hi[i, j]))

            xL, xR = j - 0.5, j + 0.5
            yT, yB = i - 0.5, i + 0.5

            if j == 0:         # left edge
                ax.plot([xL, xL], [yT, yB], color=col, lw=lw_outer, zorder=zorder)
            if j == ncols - 1:  # right edge
                ax.plot([xR, xR], [yT, yB], color=col, lw=lw_outer, zorder=zorder)
            if i == 0:         # top edge
                ax.plot([xL, xR], [yT, yT], color=col, lw=lw_outer, zorder=zorder)
            if i == nrows - 1:  # bottom edge
                ax.plot([xL, xR], [yB, yB], color=col, lw=lw_outer, zorder=zorder)

    # ---------- internal boundaries (vertical) ----------
    # boundary between (i,j) and (i,j+1) lies at x = j+0.5
    for i in range(nrows):
        for j in range(ncols - 1):
            a = bool(hi[i, j])
            b = bool(hi[i, j + 1])
            if a == b:
                continue

            x = j + 0.5
            yT, yB = i - 0.5, i + 0.5

            # draw two parallel lines, offset to each side of the boundary
            # side offset: left cell uses x - eps, right cell uses x + eps
            ax.plot([x - eps, x - eps], [yT, yB], color=c(a), lw=lw_split, zorder=zorder)
            ax.plot([x + eps, x + eps], [yT, yB], color=c(b), lw=lw_split, zorder=zorder)

    # ---------- internal boundaries (horizontal) ----------
    # boundary between (i,j) and (i+1,j) lies at y = i+0.5
    for i in range(nrows - 1):
        for j in range(ncols):
            a = bool(hi[i, j])
            b = bool(hi[i + 1, j])
            if a == b:
                continue

            y = i + 0.5
            xL, xR = j - 0.5, j + 0.5

            # top cell uses y - eps, bottom cell uses y + eps
            ax.plot([xL, xR], [y - eps, y - eps], color=c(a), lw=lw_split, zorder=zorder)
            ax.plot([xL, xR], [y + eps, y + eps], color=c(b), lw=lw_split, zorder=zorder)

def add_convex_corner_fixes(segs, hi, eps, c, lw, zorder=9, box_scale=1.0):
    """
    Add tiny square-outline connectors at convex corners only.
    Much more robust than an "L" against AA/capstyle artifacts.

    box_scale=1.0 -> square half-size = eps
    box_scale=1.2 -> slightly larger connector (often nicer)
    """
    hi = np.asarray(hi, dtype=bool)
    nrows, ncols = hi.shape

    s = eps * float(box_scale)

    for i in range(nrows - 1):
        for j in range(ncols - 1):
            xv = j + 0.5
            yv = i + 0.5

            TL = bool(hi[i, j])
            TR = bool(hi[i, j + 1])
            BL = bool(hi[i + 1, j])
            BR = bool(hi[i + 1, j + 1])

            cnt = TL + TR + BL + BR
            if cnt not in (1, 3):
                continue  # no convex corner

            # cnt==1 => True side convex; cnt==3 => False side convex
            if cnt == 1:
                side_val = True
                single = {(0,0): TL, (0,1): TR, (1,0): BL, (1,1): BR}
            else:
                side_val = False
                single = {(0,0): not TL, (0,1): not TR, (1,0): not BL, (1,1): not BR}

            quad = None
            for (qi, qj), is_single in single.items():
                if is_single:
                    quad = (qi, qj)
                    break
            if quad is None:
                continue

            col = c(1-side_val)
            qi, qj = quad

            # outward directions from the vertex depend on quadrant of the single cell
            # TL: left+up, TR: right+up, BL: left+down, BR: right+down
            dx = -1 if qj == 1 else +1
            dy = -1 if qi == 1 else +1

            # square center is shifted outward from the vertex
            cx = xv + dx * s
            cy = yv + dy * s

            segs.append((cx, cy - s, cx, cy + s, col, lw))
            segs.append((cx - s, cy, cx + s, cy, col, lw))

    return segs

def draw_region_boundaries_strict(
    ax,
    hi_mask: np.ndarray,          # bool array (nrows, ncols): True = C2, False = C1
    color_lo="green",             # C1 boundary color
    color_hi="red",               # C2 boundary color
    lw_outer=3.0,                 # outer boundary linewidth
    lw_split=2.0,                 # internal split linewidth
    eps=0.035,                    # offset for split lines (in data units)
    zorder=50,
    merge=True,                   # merge consecutive segments into long lines
):
    """
    Draw strict region boundaries aligned to cell edges.

    - Outer boundary: one line, colored by the adjacent cell's class.
    - Inner boundary (between hi/lo): two parallel lines, one on each side,
      perfectly trimmed to the boundary segment length (no overhang).
    """

    hi = np.asarray(hi_mask, dtype=bool)
    nrows, ncols = hi.shape

    def c(v: bool):
        return color_hi if v else color_lo

    # ---------- helpers to build & (optionally) merge segments ----------
    # Each segment defined by: (x0, y0, x1, y1, color, lw)
    segs = []

    # boundary exists where two adjacent cells differ, or cell vs outside
    # We'll construct per GRID EDGE (not per cell) so no duplicates.

    # ---- OUTER boundary edges ----
    # Left and right outer vertical edges
    for i in range(nrows):
        y0, y1 = i - 0.5, i + 0.5

        # left edge adjacent to (i,0)
        segs.append((-0.5, y0, -0.5, y1, c(hi[i, 0]), lw_outer))
        # right edge adjacent to (i,ncols-1)
        segs.append((ncols - 0.5, y0, ncols - 0.5, y1, c(hi[i, ncols - 1]), lw_outer))

    # Top and bottom outer horizontal edges
    for j in range(ncols):
        x0, x1 = j - 0.5, j + 0.5

        # top edge adjacent to (0,j)
        segs.append((x0, -0.5, x1, -0.5, c(hi[0, j]), lw_outer))
        # bottom edge adjacent to (nrows-1,j)
        segs.append((x0, nrows - 0.5, x1, nrows - 0.5, c(hi[nrows - 1, j]), lw_outer))

    # ---- INTERNAL boundaries: vertical (between j and j+1) ----
    # For each vertical grid line x = j+0.5, add a split segment where hi differs
    for j in range(ncols - 1):
        x = j + 0.5
        for i in range(nrows):
            a = hi[i, j]
            b = hi[i, j + 1]
            if a == b:
                continue
            y0, y1 = i - 0.5, i + 0.5
            # left side line (belongs to cell a)
            segs.append((x - eps, y0, x - eps, y1, c(a), lw_split))
            # right side line (belongs to cell b)
            segs.append((x + eps, y0, x + eps, y1, c(b), lw_split))

    # ---- INTERNAL boundaries: horizontal (between i and i+1) ----
    # For each horizontal grid line y = i+0.5, add a split segment where hi differs
    for i in range(nrows - 1):
        y = i + 0.5
        for j in range(ncols):
            a = hi[i, j]
            b = hi[i + 1, j]
            if a == b:
                continue
            x0, x1 = j - 0.5, j + 0.5
            # top side line (belongs to cell a) => y - eps
            segs.append((x0, y - eps, x1, y - eps, c(a), lw_split))
            # bottom side line (belongs to cell b) => y + eps
            segs.append((x0, y + eps, x1, y + eps, c(b), lw_split))

    # ---------- OPTIONAL merge: join consecutive collinear segments ----------
    # This removes tiny seams and makes corners cleaner.
    if merge:
        # bucket by (orientation, fixed_coord, color, lw)
        # vertical: x fixed, horizontal: y fixed
        buckets = {}
        for x0, y0, x1, y1, col, lw in segs:
            if np.isclose(x0, x1):  # vertical
                key = ("v", round(float(x0), 6), col, float(lw))
                a, b = sorted([float(y0), float(y1)])
                buckets.setdefault(key, []).append((a, b))
            else:  # horizontal
                key = ("h", round(float(y0), 6), col, float(lw))
                a, b = sorted([float(x0), float(x1)])
                buckets.setdefault(key, []).append((a, b))

        merged = []
        for key, intervals in buckets.items():
            intervals.sort()
            cur_a, cur_b = intervals[0]
            for a, b in intervals[1:]:
                # contiguous or touching
                if a <= cur_b + 1e-9:
                    cur_b = max(cur_b, b)
                else:
                    merged.append((key, cur_a, cur_b))
                    cur_a, cur_b = a, b
            merged.append((key, cur_a, cur_b))

        segs2 = []
        for (orient, fixed, col, lw), a, b in merged:
            if orient == "v":
                x = fixed
                segs2.append((x, a, x, b, col, lw))
            else:
                y = fixed
                segs2.append((a, y, b, y, col, lw))
        segs = segs2

    # ---------- fix convex corners ----------
    segs = add_convex_corner_fixes(segs, hi, eps=eps, c=c, lw=lw_split)

    # ---------- draw via LineCollection (clean joins/caps) ----------
    # Group by (color,lw) so LineCollection can be created efficiently
    groups = {}
    for x0, y0, x1, y1, col, lw in segs:
        groups.setdefault((col, lw), []).append([(x0, y0), (x1, y1)])

    for (col, lw), lines in groups.items():
        lc = LineCollection(
            lines,
            colors=[col],
            linewidths=lw,
            zorder=zorder,
            capstyle="butt",     # prevents little protruding caps
            joinstyle="miter",   # crisp right-angle corners
        )
        ax.add_collection(lc)

    return ax


def region_boundary_legend_handles(
    color_lo="green",
    color_hi="#e53e3e",
    lw=3,
):
    """
    Create legend handles for region-boundary outlines.
    """
    h_lo = Rectangle(
        (0, 0), 1, 1,
        fill=False,
        edgecolor=color_lo,
        linewidth=lw,
        label="Criterion 2 performs better"
    )

    h_hi = Rectangle(
        (0, 0), 1, 1,
        fill=False,
        edgecolor=color_hi,
        linewidth=lw,
        label="Criterion 1 performs better"
    )

    return [h_hi, h_lo]

def plot_figure7_c2_only_2x3(
    winners_all,
    dataset_names,
    n_reps=100,
    figsize=(14, 7.5),
    cmap="YlGnBu",
    aspect=1.1,          # <1 makes heatmap shorter (more rectangular)
    caption_y=-0.16,      # closer to subplot
    tick_fs=8,
    label_fs=9,
    annot_fs=8,
):
    """
    Plot Criterion 2 wins only, in a 2x3 layout.
    Each row has its own colorbar (without shrinking the 3rd column).
    """

    assert len(dataset_names) == 6, "Need 6 datasets for a 2x3 layout."

    masks = VALID_MASK_NAMES
    rates = MISSING_RATES
    vmin, vmax = 0, n_reps

    caption_map = {
        "Concrete": "(a) Concrete compressive strength",
        "Composite": "(b) Composite material",
        "Steel": "(c) Steel strength",
        "Energy": "(d) Energy efficiency",
        "Student": "(e) Student performance",
        "Wine": "(f) Wine quality",
    }
    MASK_DISPLAY = {
        "MCAR": "MCAR",
        "MAR": "MAR",
        "MNAR": "MNAR",
        "MCAR_pair": "MCAR_p",
        "MAR_pair": "MAR_p",
        "MNAR_pair": "MNAR_p",
    }

    # ---- GridSpec: 2 rows × (3 plots + 1 thin cbar column) ----
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.055],   # last col reserved for cbar
        wspace=0.2,
        hspace=0.1
    )

    axes = np.empty((2, 3), dtype=object)
    cbar_axes = [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])]

    # store one mappable per row for row-wise colorbar
    row_mappable = [None, None]

    for row in range(2):
        for col in range(3):
            idx = row * 3 + col
            ds_name = dataset_names[idx]

            ax = fig.add_subplot(gs[row, col])
            axes[row, col] = ax

            win_df = winners_all[ds_name]["Criterion 2"].reindex(index=masks, columns=rates)
            win_c1 = n_reps - win_df
            data = win_c1.values.astype(float)

            im = ax.imshow(
                data,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                aspect=aspect,
                interpolation="nearest",
            )

            hi_mask = data >= 50
            draw_region_boundaries_strict(ax, hi_mask, lw_outer=8, lw_split=2.5, eps=0.035)

            # remember one im per row (any of the three works; they share vmin/vmax/cmap)
            if row_mappable[row] is None:
                row_mappable[row] = im

            # ---- x axis ----
            ax.set_xticks(np.arange(len(rates)))
            ax.set_xticklabels([f"{r:.2f}" for r in rates], fontsize=tick_fs)
            ax.set_xlabel("Missing ratio", fontsize=label_fs)

            # ---- y axis ----
            ax.set_yticks(np.arange(len(masks)))
            if col == 0:
                ax.set_yticklabels([MASK_DISPLAY[m] for m in masks],
                                   fontsize=tick_fs, rotation=90,
                                   va="center", ha="center")
                ax.set_ylabel("Missing mechanism", fontsize=label_fs)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            # ---- caption below ----
            ax.text(
                0.5, caption_y,
                caption_map.get(ds_name, ds_name),
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=label_fs
            )

            # ---- annotate values with adaptive text color ----
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                #    norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                #    rgba = im.cmap(norm_val)
                #    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                #    txt_color = "black" if luminance > 0.55 else "white"
                    txt_color = "white" if val >= 50 else "black"
                    ax.text(j, i, f"{int(val)}",
                            ha="center", va="center",
                            fontsize=annot_fs, color=txt_color)

    # ---- row-wise colorbars (use reserved cbar axes; NO shrinking) ----
    for row in range(2):
        cax = cbar_axes[row]
        cbar = fig.colorbar(row_mappable[row], cax=cax)
        cbar.set_label(f"Number of Criterion 1 wins (out of {n_reps})", fontsize=label_fs)
        cbar.ax.tick_params(labelsize=tick_fs)

    # Tighten margins a bit
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.97)

    # --- align each row's colorbar height to the row's heatmap axes ---
    fig.canvas.draw()  # ensure positions are finalized

    for row in range(2):
        ref_ax = axes[row, 2]      # rightmost heatmap in the row
        cax = cbar_axes[row]       # the row colorbar axis

        ref_pos = ref_ax.get_position()
        cax_pos = cax.get_position()

        gap = 0.01   # <<< controls ONLY the space to colorbar (try 0.004–0.008)

        cax.set_position([
            ref_pos.x1 + gap,      # move cbar closer to 3rd column
            ref_pos.y0,
            cax_pos.width,
            ref_pos.height
        ])
    # ---- region-boundary legend (figure-level) ----
    legend_handles = region_boundary_legend_handles(

    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.99),
        handlelength=3.0,
        handleheight=1.2,
        columnspacing=2.6,
    )
    return fig, axes


def build_c11_vs_c12_aggregated_table(
    winners_all: dict,
    dataset_names: list,
    criterion_11: str = "Criterion 1.1",
    criterion_12: str = "Criterion 1.2",
    include_mcar: bool = True,
    mcar_label: str = "MCAR",
    ratio_as_percent: bool = True,
    round_ratio: int = 1,
) -> pd.DataFrame:
    """
    Build an aggregated comparison table between Criterion 1.1 and Criterion 1.2.

    Parameters
    ----------
    winners_all : dict
        winners_all[dataset][criterion] -> pd.DataFrame
        DataFrame index: mask procedures, columns: missing rates,
        values: number of wins (out of n_reps).
    dataset_names : list[str]
        Datasets to include, in order.
    criterion_11, criterion_12 : str
        Keys used in winners_all for C1.1 and C1.2.
    include_mcar : bool
        If False, drop the MCAR row before aggregation.
    mcar_label : str
        The exact row label for MCAR in the winners tables.
    ratio_as_percent : bool
        If True, show ratio in percent (0-100). Otherwise in [0, 1].
    round_ratio : int
        Decimal places for the ratio column.

    Returns
    -------
    pd.DataFrame
        Columns:
        ['Dataset', 'Criterion 1.1 wins', 'Criterion 1.2 wins', 'Criterion 1.2 win ratio']
    """

    rows = []

    for ds in dataset_names:
        if ds not in winners_all:
            raise KeyError(f"Dataset '{ds}' not found in winners_all.")

        if criterion_11 not in winners_all[ds]:
            raise KeyError(f"'{criterion_11}' not found for dataset '{ds}'.")
        if criterion_12 not in winners_all[ds]:
            raise KeyError(f"'{criterion_12}' not found for dataset '{ds}'.")

        df11 = winners_all[ds][criterion_11].copy()
        df12 = winners_all[ds][criterion_12].copy()

        # Align to common index/columns (safety)
        common_index = df11.index.intersection(df12.index)
        common_cols = df11.columns.intersection(df12.columns)
        df11 = df11.loc[common_index, common_cols]
        df12 = df12.loc[common_index, common_cols]

        # Optionally exclude MCAR row
        if not include_mcar and (mcar_label in common_index):
            df11 = df11.drop(index=mcar_label)
            df12 = df12.drop(index=mcar_label)

        c11_wins = float(np.nansum(df11.to_numpy(dtype=float)))
        c12_wins = float(np.nansum(df12.to_numpy(dtype=float)))
        denom = c11_wins + c12_wins

        if denom <= 0:
            ratio = np.nan
        else:
            ratio = (c12_wins / denom) * (100.0 if ratio_as_percent else 1.0)

        rows.append({
            "Dataset": ds,
            "Criterion 1.1 wins": int(round(c11_wins)),
            "Criterion 1.2 wins": int(round(c12_wins)),
            "Criterion 1.2 win ratio": (round(ratio, round_ratio) if np.isfinite(ratio) else np.nan),
        })

    out = pd.DataFrame(rows)

    # Optional: nicer formatting if percent
    if ratio_as_percent:
        out["Criterion 1.2 win ratio"] = out["Criterion 1.2 win ratio"].map(
            lambda x: f"{x:.{round_ratio}f}%" if pd.notna(x) else ""
        )

    return out


# =============================================================================
# IX. Main entry point
# =============================================================================


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Load datasets to build feature lists
    X_list, y_list, name_list = load_all_datasets()
    global CAT_FEATURES, NUM_FEATURES_FIG5, NUM_FEATURES_FOR_FIGS
    CAT_FEATURES, NUM_FEATURES_FIG5, NUM_FEATURES_FOR_FIGS = build_feature_lists(X_list)

    # Load flat results
    flat_results = load_all_flat_results(RESULTS_ROOT)
    '''
    # Figure 5
    fig5 = make_figure5(flat_results)
    fig5_path = os.path.join(OUTPUT_ROOT, "Figure5.png")
    fig5.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # Figure 6
    fig6 = make_figure6(flat_results)
    fig6_path = os.path.join(OUTPUT_ROOT, "Figure6.png")
    fig6.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close(fig6)

    # Appendix tables for Figure 5
    df_fig5_num, df_fig5_cat = build_fig5_appendix_tables(flat_results)
    write_fig5_appendix_csvs(df_fig5_num, df_fig5_cat)

    # Appendix tables for Figure 6
    df_fig6_all = build_fig6_appendix_table(flat_results)
    write_fig6_appendix_csvs(df_fig6_all)
'''
    # Winner counts and Figure 7
    winners_all = get_winners_all_cached(flat_results, n_reps=100)

    fig7a, _ = plot_figure7_heatmaps(
        winners_all,
        dataset_names=["Concrete", "Composite", "Steel"],
        n_reps=100,
        figsize=(10, 12),
    )
    fig7a_path = os.path.join(OUTPUT_ROOT, "Figure7_Concrete_Composite_Steel.png")
    fig7a.savefig(fig7a_path, dpi=300, bbox_inches="tight")
    plt.close(fig7a)

    fig7b, _ = plot_figure7_heatmaps(
        winners_all,
        dataset_names=["Energy", "Student", "Wine"],
        n_reps=100,
        figsize=(10, 12),
    )
    fig7b_path = os.path.join(OUTPUT_ROOT, "Figure7_Energy_Student_Wine.png")
    fig7b.savefig(fig7b_path, dpi=300, bbox_inches="tight")
    plt.close(fig7b)
    fig7c, _ = plot_figure7_c2_only_2x3(
        winners_all,
        dataset_names=["Concrete", "Composite", "Steel", "Energy", "Student", "Wine"],
        n_reps=100,
        figsize=(10, 8),
    )
    fig7_path = os.path.join(OUTPUT_ROOT, "Figure7_C2_only_3x2.png")
    fig7c.savefig(fig7_path, dpi=300, bbox_inches="tight")
    plt.close(fig7c)
'''
    tbl_inc = build_c11_vs_c12_aggregated_table(
        winners_all,
        dataset_names=["Concrete","Composite","Steel","Energy","Student","Wine"],
        include_mcar=True,
    )
    tbl_inc_path = os.path.join(OUTPUT_ROOT, "C11_vs_C12_aggregated_including_MCAR.csv")
    tbl_inc.to_csv(tbl_inc_path, index=False)

    tbl_exc = build_c11_vs_c12_aggregated_table(
        winners_all,
        dataset_names=["Concrete","Composite","Steel","Energy","Student","Wine"],
        include_mcar=False,
    )
    tbl_exc_path = os.path.join(OUTPUT_ROOT, "C11_vs_C12_aggregated_excluding_MCAR.csv")
    tbl_exc.to_csv(tbl_exc_path, index=False)
'''
if __name__ == "__main__":
    main()

# %%

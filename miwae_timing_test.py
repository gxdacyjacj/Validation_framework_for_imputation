# miwae_timing_test.py
"""
Timing + performance comparison between internal imputers and MIWAE.

- One dataset (index dataset_idx)
- One mask:      "MCAR"
- One rate:      0.10
- 10 repetitions

For each imputer we:
  - build a dict with only that imputer
  - call run_one_dataset()
  - measure wall-clock time
  - compute mean BASELINE error (per-feature RMSE on missing-only positions)
"""

import importlib
import time

import numpy as np

import validation_v2 as v2
from datasets import load_all_datasets  # same helper used in smoke_test

# Reload to ensure latest edits
v2 = importlib.reload(v2)


def main():
    # ---------------------------
    # 1) Load datasets
    # ---------------------------
    X_list, y_list, names = load_all_datasets()
    print("Available datasets:", names)

    # Choose dataset for timing/performance
    dataset_idx = 2  # Concrete dataset
    X = X_list[dataset_idx]
    y = y_list[dataset_idx]
    name = names[dataset_idx]
    print(f"\nUsing dataset {dataset_idx} -> {name}")

    # ---------------------------
    # 2) Config for timing test
    # ---------------------------
    mask_names = ["MAR_pair"]
    missing_rates = [0.15]
    n_repeats = 5

    # Build base imputers (including MIWAE)
    imputers = v2.build_imputers(
        internal_imputer=["KNN"],
        random_state=v2.BASE_SEED,
        external_imputer=["notMIWAE"],
    )

    imputer_names = list(imputers.keys())
    print("\nImputers to time:", imputer_names)

    timings = {}
    mean_baseline_errors = {}  # mean baseline error (RMSE) per imputer

    # ---------------------------
    # 3) Run each imputer separately
    # ---------------------------
    for imp_name in imputer_names:
        print(f"\n=== Running timing for imputer: {imp_name} ===")
        single_imputer_dict = {imp_name: imputers[imp_name]}

        start = time.time()
        results = v2.run_one_dataset(
            X=X,
            y=y,
            mask_names=mask_names,
            missing_rates=missing_rates,
            n_repeats=n_repeats,
            imputer_dict=single_imputer_dict,
            use_standardize_before_impute=True,
            model_dict=None,           # default downstream models
            checkpoint_dir=None,
            dataset_label=None,
            resume=False,
        )
        elapsed = time.time() - start

        timings[imp_name] = elapsed

        # --------- compute mean BASELINE error from results ---------
        df = v2.flatten_results(results)
        # We only look at baseline, missing-only per-feature errors
        df_base = df[
            (df["criterion"] == "baseline")
            & (df["scope"] == "per_feature_missing_only")
        ]
        if len(df_base) > 0:
            mean_err = float(df_base["error"].mean())
        else:
            mean_err = float("nan")
        mean_baseline_errors[imp_name] = mean_err

        print(f"Imputer {imp_name} finished in {elapsed:.1f} seconds.")
        print(f"  Mean BASELINE error (missing-only RMSE) = {mean_err:.4f}")

    # ---------------------------
    # 4) Write timing + performance summary to txt
    # ---------------------------
    out_lines = []
    out_lines.append("Timing + performance summary (Baseline-based)\n")
    out_lines.append(f"Dataset used: {name}\n")
    out_lines.append("Mask: MCAR, Missing rate: 0.10, Repeats: 10\n\n")

    out_lines.append("Timings (seconds):\n")
    out_lines.append(f"{'Imputer':12s} {'Time (s)':>10s}\n")
    out_lines.append("-" * 30 + "\n")
    for imp_name in imputer_names:
        t = timings[imp_name]
        out_lines.append(f"{imp_name:12s}: {t:8.1f} s\n")

    out_lines.append("\nMean BASELINE error (missing-only RMSE):\n")
    out_lines.append(f"{'Imputer':12s} {'Mean error':>12s}\n")
    out_lines.append("-" * 30 + "\n")
    for imp_name in imputer_names:
        e = mean_baseline_errors[imp_name]
        out_lines.append(f"{imp_name:12s}: {e:12.4f}\n")

    outpath = "miwae_timing_results_baseline.txt"
    with open(outpath, "w") as f:
        f.writelines(out_lines)

    print("\n======== Timing + performance summary (Baseline) ========")
    for line in out_lines:
        print(line, end="")
    print(f"\nResults written to: {outpath}")


if __name__ == "__main__":
    main()

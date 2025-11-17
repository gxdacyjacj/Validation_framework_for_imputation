# run_all_experiments.py
"""
Run the full validation_v2 experiments on all six datasets.

Features:
- Dataset-by-dataset execution (each dataset saved separately).
- Parallel across datasets (n_jobs = 6).
- Progress bar over datasets.
- Uses the same mask names, missing rates, and repetitions as in the paper.
"""

import os
import time
import pickle
import traceback

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import numpy as np

from datasets import load_all_datasets
import validation_v2 as v2


# -------------------------------------------------------------------------
# Experiment configuration (match the paper / old code)
# -------------------------------------------------------------------------

# Mask mechanisms to evaluate
MASK_NAMES = ["MCAR", "MAR", "MNAR", "MCAR_pair", "MAR_pair", "MNAR_pair"]

# Missing rates used in the paper
MISSING_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Number of repetitions per (dataset, mask, rate)
N_REPEATS = 100

# Parallelism
N_JOBS = 6  # as requested

# Output directory for results
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# Helper: build imputers once (no wrapping here; we use run_one_dataset's flag)
# -------------------------------------------------------------------------

def build_imputers_for_experiments():
    """
    Build the 6 imputers used in the paper:
      - Mean/mode
      - Simple hot-deck
      - kNN
      - LightGBM
      - CatBoost
      - PMM

    Uses the correct argument names from validation_v2.build_imputers
    (notably: knn_k, NOT knn_n_neighbors).
    """
    imputers = v2.build_imputers(
        mice_iters=20,          # you can adjust if your final paper uses a different value
        rf_n_estimators=200,
        knn_k=5,                # <- correct name
        tree_jobs=-1,
        random_state=v2.BASE_SEED,
    )
    return imputers


# -------------------------------------------------------------------------
# Single-job wrapper: run one dataset and save its results
# -------------------------------------------------------------------------

def run_single_dataset_job(dataset_id, X, y, name, imputers):
    """
    Run validation_v2.run_one_dataset on a single dataset and save the result.

    Returns a dict summarising the outcome (for logging in main()).
    """
    job_start = time.time()
    result_path = os.path.join(
        OUT_DIR, f"results_dataset{dataset_id}_{name}.pkl"
    )

    try:
        res = v2.run_one_dataset(
            X=X,
            y=y,
            mask_names=MASK_NAMES,
            missing_rates=MISSING_RATES,
            n_repeats=N_REPEATS,
            imputer_dict=imputers,
            use_standardize_before_impute=True,  # our agreed default
            model_dict=None,                     # use DEFAULT_MODEL_LIST inside validation_v2
        )

        # Save per-dataset result
        with open(result_path, "wb") as f:
            pickle.dump(
                dict(
                    dataset_id=dataset_id,
                    dataset_name=name,
                    mask_names=MASK_NAMES,
                    missing_rates=MISSING_RATES,
                    n_repeats=N_REPEATS,
                    results=res,
                ),
                f,
            )

        elapsed = time.time() - job_start
        return dict(
            dataset_id=dataset_id,
            name=name,
            success=True,
            error=None,
            traceback=None,
            elapsed_seconds=elapsed,
            result_path=result_path,
        )

    except Exception as e:
        elapsed = time.time() - job_start
        return dict(
            dataset_id=dataset_id,
            name=name,
            success=False,
            error=str(e),
            traceback=traceback.format_exc(),
            elapsed_seconds=elapsed,
            result_path=None,
        )


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def main():
    print("Loading all datasets...")
    X_list, y_list, names = load_all_datasets()
    n_datasets = len(names)
    print(f"Found {n_datasets} datasets:", names)

    print("\nBuilding imputers once for all datasets...")
    imputers = build_imputers_for_experiments()
    print("Imputers:", list(imputers.keys()))

    # Prepare jobs: one per dataset
    jobs = []
    for dataset_id, (X, y, name) in enumerate(
        zip(X_list, y_list, names), start=1
    ):
        jobs.append((dataset_id, X, y, name))

    print(f"\nStarting parallel execution with n_jobs={N_JOBS} ...")

    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_single_dataset_job)(dataset_id, X, y, name, imputers)
        for dataset_id, X, y, name in tqdm(
            jobs, desc="Datasets", unit="dataset"
        )
    )

    print("\n=== Summary ===")
    for r in results:
        if r["success"]:
            print(
                f"[OK] Dataset {r['dataset_id']}: {r['name']} "
                f"in {r['elapsed_seconds']:.1f}s -> {r['result_path']}"
            )
        else:
            print(
                f"[FAIL] Dataset {r['dataset_id']}: {r['name']} "
                f"in {r['elapsed_seconds']:.1f}s"
            )
            print("    Error:", r["error"])
            # If you want full traceback, uncomment:
            # print(r["traceback"])

    n_ok = sum(r["success"] for r in results)
    print(f"\nFinished. {n_ok}/{n_datasets} datasets completed successfully.")


if __name__ == "__main__":
    main()

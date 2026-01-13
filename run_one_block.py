# run_one_block.py
"""
Run a *single* (dataset, mask, missing_rate) block from the full experiment.

This is designed so you can split the full workload into many small jobs, for example:
  - 6 masks × 6 missing rates = 36 blocks per dataset
  - run different blocks on different machines

Each job:
  - loads all datasets (so we can pick the right one by id)
  - builds the imputers (same as in run_all_experiments.py)
  - calls validation_v2.run_one_dataset with:
        mask_names    = [chosen_mask]
        missing_rates = [chosen_rate]
  - saves the resulting nested dict to a per-block .pkl file under:

        results/<dataset_name>/dataset{ID}_{dataset_name}_mask-{MASK}_rate-{XXpct}.pkl

You can either:
  - call the CLI:

        python run_one_block.py --dataset-id 1 --mask MCAR --rate 0.10 --n-repeats 100

  - or import and call run_block(...) from another script.
"""

import os
import time
import pickle
import argparse

from datasets import load_all_datasets
import importlib



VALID_MASK_NAMES = ["MCAR", "MAR", "MNAR", "MCAR_pair", "MAR_pair", "MNAR_pair"]


def build_imputers_for_block(vmod):
    """
    Same imputers as in run_all_experiments.py:
      - Mean/mode
      - Simple hot-deck
      - kNN
      - LightGBM
      - CatBoost
      - PMM
    """
    imputers = vmod.build_imputers(
        mice_iters=20,
        rf_n_estimators=200,
        knn_k=5,
        tree_jobs=-1,
        random_state=vmod.BASE_SEED,
    )
    return imputers


def run_block(
    dataset_id: int,
    mask_name: str,
    rate: float,
    validator: str = "v2",
    n_repeats: int = 100,
    out_root: str = "results",
) -> str:
    """
    Core function to run a single (dataset, mask, rate) block.
    ...
    """
    # 1) Load datasets and pick the requested one
    X_list, y_list, names = load_all_datasets()
    n_datasets = len(names)
    if not (1 <= dataset_id <= n_datasets):
        raise ValueError(
            f"dataset_id must be between 1 and {n_datasets}, got {dataset_id}."
        )

    ds_idx = dataset_id - 1
    X = X_list[ds_idx]
    y = y_list[ds_idx]
    ds_name = names[ds_idx]
    safe_name = str(ds_name).replace(" ", "_")

    print(f"[Block] Dataset {dataset_id}/{n_datasets}: {ds_name}")
    print(f"        mask={mask_name}, rate={rate}, repeats={n_repeats}")


    # Choose validation module
    if validator not in {"v2"}:
        raise ValueError(f"validator must be 'v2', got {validator!r}")
    vmod = importlib.import_module("validation_v2")

    # 2) Build imputers
    print("        Building imputers …")
    imputers = build_imputers_for_block(vmod)
    print("        Imputers:", list(imputers.keys()))

    # 3) Prepare output + progress paths: results/<dataset_name>/...
    rate_str = f"{int(round(rate * 100))}pct"
    out_dir = os.path.join(out_root, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"dataset{dataset_id}_{safe_name}_mask-{mask_name}_rate-{rate_str}.pkl",
    )
    progress_path = os.path.join(
        out_dir,
        f"progress_mask-{mask_name}_rate-{rate_str}.txt",
    )

    print(f"        Output   -> {out_path}")
    print(f"        Progress -> {progress_path}")

    # 4) Define per-repetition progress callback
    def progress_callback(m_name, m_rate, rep_idx, total_reps):
        # one-line status; overwrite file each time
        msg = f"mask={m_name} | rate={m_rate:.2f} | rep={rep_idx}/{total_reps}\n"
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                f.write(msg)
        except OSError:
            # Don't kill the job if updating the txt fails
            pass

        # Optional: also print occasionally to terminal
        if rep_idx == 1 or rep_idx == total_reps or (rep_idx % 10 == 0):
            print("        " + msg.strip())

    # 5) Run run_one_dataset restricted to this single (mask, rate)
    t0 = time.time()
    res = vmod.run_one_dataset(
        X=X,
        y=y,
        mask_names=[mask_name],
        missing_rates=[rate],
        n_repeats=n_repeats,
        imputer_dict=imputers,
        use_standardize_before_impute=True,
        model_dict=None,               # use DEFAULT_MODEL_LIST
        # new: pass block-level progress callback
        progress_callback=progress_callback,
    )
    elapsed = time.time() - t0

    # 6) Save result
    payload = dict(
        dataset_id=dataset_id,
        dataset_name=ds_name,
        mask_name=mask_name,
        missing_rate=rate,
        n_repeats=n_repeats,
        results=res,  # nested dict: results[mask][rate][rep]
    )
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"        Done in {elapsed/3600.0:.2f} hours.")
    print(f"        Saved to: {out_path}")

    # Clean up progress file once this block is finished
    try:
        if os.path.exists(progress_path):
            os.remove(progress_path)
    except OSError:
        # If removal fails for some reason, just ignore it
        pass

    return out_path



# ---------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single (dataset, mask, rate) block from validation_v2."
    )

    parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help=(
            "Dataset index in paper order: 1=Concrete, 2=Composite, 3=Steel, "
            "4=Energy, 5=Student, 6=Wine."
        ),
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        choices=VALID_MASK_NAMES,
        help="Missingness mechanism (e.g. MCAR, MAR, MNAR, MCAR_pair, ...).",
    )
    parser.add_argument(
        "--validator",
        type=str,
        default="v2",
        choices=["v2"],
        help="Which validation implementation to use: v2 (default: v2).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Missing rate (e.g. 0.05, 0.10, 0.15, 0.20, 0.25, 0.30).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=100,
        help="Number of repetitions for this (dataset, mask, rate) block (default: 100).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="results",
        help="Root directory to save block-level results (default: results).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_block(
        dataset_id=args.dataset_id,
        mask_name=args.mask,
        rate=args.rate,
        n_repeats=args.n_repeats,
        out_root=args.out_root,
        validator=args.validator,
    )


if __name__ == "__main__":
    main()

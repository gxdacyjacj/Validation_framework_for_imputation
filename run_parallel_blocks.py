# run_parallel_blocks.py
"""
Run ALL remaining (mask, rate) blocks for a single dataset in parallel.

- Checks which results already exist under results/<dataset_name>/...
- Only runs the missing blocks.
- Uses joblib.Parallel with n_jobs workers to run run_one_block.run_block(...).

Typical usage on a big machine:
    python run_parallel_blocks.py --dataset-id 1 --n-jobs 18 --n-repeats 100

You can stop the script at any time. Next time you run it, it will skip blocks
that already have a .pkl file in the dataset's results folder.
"""

# run_parallel_blocks.py

import os
from itertools import product
from joblib import Parallel, delayed

from datasets import load_all_datasets
from run_one_block import run_block, VALID_MASK_NAMES

# Default rates if user does not override
DEFAULT_MISSING_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]



def block_output_path(
    dataset_id: int, dataset_name: str, mask_name: str, rate: float, out_root: str, validator: str = "v2"
) -> str:
    """Construct the output path for a given (dataset, mask, rate)."""
    safe_name = str(dataset_name).replace(" ", "_")
    rate_str = f"{int(round(rate * 100))}pct"
    out_dir = os.path.join(out_root, safe_name)
    fname = f"dataset{dataset_id}_{safe_name}_mask-{mask_name}_rate-{rate_str}.pkl"
    return os.path.join(out_dir, fname)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all missing (mask, rate) blocks for one dataset in parallel."
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
        "--n-jobs",
        type=int,
        required=True,
        help="Number of parallel workers (e.g., 6 on your PC, 18 on a 64-core machine).",
    )
    parser.add_argument(
        "--validator",
        type=str,
        default="v2",
        choices=["v2"],
        help="Which validation implementation to use: v2 (default: v2).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=100,
        help="Number of repetitions per block (must match your paper setup; default 100).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="results",
        help="Root folder for results (default: results).",
    )
    parser.add_argument(
        "--rates",
        type=str,
        default=None,
        help=(
            "Comma-separated list of missing rates to run for this job, "
            "e.g. '0.05' or '0.10,0.15,0.20'. "
            "If omitted, uses all default rates "
            "[0.05,0.10,0.15,0.20,0.25,0.30]."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1) Load dataset names to identify the dataset folder
    # -------------------------------------------------------------
    X_list, y_list, names = load_all_datasets()
    n_datasets = len(names)
    if not (1 <= args.dataset_id <= n_datasets):
        raise ValueError(
            f"dataset-id must be between 1 and {n_datasets}, got {args.dataset_id}."
        )

    ds_name = names[args.dataset_id - 1]
    print(f"Dataset {args.dataset_id}: {ds_name}")

    # -------------------------------------------------------------
    # 2) Decide which missing rates this job will handle
    # -------------------------------------------------------------
    if args.rates is None:
        missing_rates = DEFAULT_MISSING_RATES
    else:
        try:
            missing_rates = [float(x) for x in args.rates.split(",")]
        except ValueError:
            raise ValueError(
                f"Could not parse --rates '{args.rates}'. "
                "Use e.g. --rates '0.05' or --rates '0.05,0.10,0.15'."
            )

    print("Missing rates for this job:", missing_rates)

    # 3) Enumerate all blocks for these rates
    all_blocks = list(product(VALID_MASK_NAMES, missing_rates))
    remaining_blocks = []

    for mask_name, rate in all_blocks:
        out_path = block_output_path(
            dataset_id=args.dataset_id,
            dataset_name=ds_name,
            mask_name=mask_name,
            rate=rate,
            out_root=args.out_root,
            validator=args.validator,
        )
        if os.path.exists(out_path):
            print(f"[SKIP] {mask_name}, rate={rate:.2f} (found {out_path})")
        else:
            print(f"[TODO] {mask_name}, rate={rate:.2f}")
            remaining_blocks.append((mask_name, rate))

    if not remaining_blocks:
        print("\nAll blocks for these rates are already done. Nothing to run.")
        return

    print(f"\nTotal blocks: {len(all_blocks)}, remaining: {len(remaining_blocks)}")
    print(f"Running remaining blocks with n_jobs={args.n_jobs} ...\n")

    # -------------------------------------------------------------
    # 3) Run remaining blocks in parallel
    # -------------------------------------------------------------
    def _run(mask_name, rate):
        return run_block(
            dataset_id=args.dataset_id,
            mask_name=mask_name,
            rate=rate,
            n_repeats=args.n_repeats,
            out_root=args.out_root,
            validator=args.validator,
        )

    Parallel(n_jobs=args.n_jobs)(
        delayed(_run)(mask_name, rate) for mask_name, rate in remaining_blocks
    )

    print("\nAll remaining blocks for this dataset are now complete (or attempted).")


if __name__ == "__main__":
    main()

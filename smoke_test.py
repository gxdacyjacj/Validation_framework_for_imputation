#%%
import importlib
import time

import validation_v2 as v2
from datasets import load_all_datasets

from joblib import Parallel, delayed
from tqdm.auto import tqdm  # nice progress bar in Jupyter

v2 = importlib.reload(v2)

# ---------------------------
# 1) Load datasets (paper order)
# ---------------------------
X_list, y_list, names = load_all_datasets()
print("Datasets:", names)

# ---------------------------
# 2) Config for smoke test
# ---------------------------
mask_names    = ["MCAR", "MAR"]   # light subset
missing_rates = [0.10]            # one rate
n_repeats     = 2                 # small repeats

# build imputers ONCE; run_one_dataset will apply standardisation if needed
imputers = v2.build_imputers(
    mice_iters=5,          # lighter than full run
    rf_n_estimators=50,
    knn_k=5,
    tree_jobs=1,
    random_state=0,
)

# ---------------------------
# 3) Helper for a single dataset
# ---------------------------
def run_smoke_for_dataset(dataset_id, X, y, name):
    """Run a small validation on one dataset and return summary + full results."""
    res = v2.run_one_dataset(
        X=X,
        y=y,
        mask_names=mask_names,
        missing_rates=missing_rates,
        n_repeats=n_repeats,
        imputer_dict=imputers,
        use_standardize_before_impute=True,  # only place standardisation is applied
        model_dict=None,                     # use DEFAULT_MODEL_LIST
    )

    flat = v2.flatten_results(res)
    summary = flat.groupby(["criterion", "imputer"])["error"].mean()

    return dataset_id, name, res, summary

# ---------------------------
# 4) Parallel execution across 6 datasets
# ---------------------------
jobs = [
    (dataset_id, X, y, name)
    for dataset_id, (X, y, name) in enumerate(zip(X_list, y_list, names), start=1)
]

start_time = time.time()

parallel_outputs = Parallel(n_jobs=6, backend="loky")(
    delayed(run_smoke_for_dataset)(dataset_id, X, y, name)
    for (dataset_id, X, y, name) in tqdm(jobs, desc="Smoke test datasets")
)

elapsed = time.time() - start_time

# ---------------------------
# 5) Collect and print summaries
# ---------------------------
all_results = {}
for dataset_id, name, res, summary in parallel_outputs:
    all_results[dataset_id] = res
    print(f"\n=== Dataset {dataset_id}: {name} ===")
    print(summary)

print(f"\nParallel smoke test finished in {elapsed:.1f} seconds.")

# %%

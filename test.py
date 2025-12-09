#%%
import numpy as np
import pandas as pd

import datasets               # your datasets.py
import validation_v2 as v2    # your validation_v2.py


#%% -------------------------------------------------------------------
# 1) Choose dataset + mask combo to debug
# ---------------------------------------------------------------------
X_list, y_list, names = datasets.load_all_datasets()

ds_id = 0          # 0=Concrete, 1=Composite, ..., 5=Wine
mask_name = "MCAR"
rate = 0.05
rep = 0            # which repetition to debug

X = X_list[ds_id].copy()
y = y_list[ds_id].copy()
name = names[ds_id]

print(f"Dataset: {name}, shape={X.shape}")
print(f"Mask: {mask_name}, rate={rate}, rep={rep}")

#%% -------------------------------------------------------------------
# 2) Reproduce the seed used in run_one_dataset
# --------------------------------------------------------------------
mask_order = ["MCAR", "MAR", "MNAR", "MCAR_pair", "MAR_pair", "MNAR_pair"]
rate_order = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

midx = mask_order.index(mask_name)
ridx = rate_order.index(rate)

combo_seed = v2.BASE_SEED + 1000 * rep + 10 * midx + ridx
print("combo_seed =", combo_seed)
v2.set_random_seed(combo_seed)

#%% -------------------------------------------------------------------
# 3) Train / Val / Test split (same helper as run_one_dataset)
# --------------------------------------------------------------------
X_train, X_val, X_test, y_train, y_val, y_test = v2.split_train_val_test(
    X, y, test_size=0.2, val_size=0.2, random_state=combo_seed
)
print("Split shapes:",
      "train", X_train.shape,
      "val",   X_val.shape,
      "test",  X_test.shape)

#%% -------------------------------------------------------------------
# 4) Build processed versions & generate missingness (incl. paired)
# --------------------------------------------------------------------
miss_vec = np.full(X.shape[1], rate, dtype=float)

base_name = mask_name.replace("_pair", "")
gen = v2._unstructured_mask_fn(base_name)

Xp_train = v2.build_processed(X_train)
Xp_val   = v2.build_processed(X_val)
Xp_test  = v2.build_processed(X_test)

Xm_train, Xpm_train, indep_idx = gen(
    X_train, Xp_train, miss_vec, independent_feature_idx=None
)
Xm_val,   Xpm_val,   _ = gen(
    X_val, Xp_val, miss_vec, independent_feature_idx=indep_idx
)
Xm_test,  Xpm_test,  _ = gen(
    X_test, Xp_test, miss_vec, independent_feature_idx=indep_idx
)

# Paired missingness, if needed
if v2._is_paired(mask_name):
    nfeat = X_train.shape[1]
    half = max(1, nfeat // 2)
    pair_map = np.full(nfeat, np.nan)
    for i in range(half, nfeat):
        pair_map[i] = np.random.randint(0, half)

    Xm_train, Xpm_train = v2.generate_paired_missingness(
        X_train, Xm_train, Xp_train, Xpm_train, pair_map
    )
    Xm_val, Xpm_val = v2.generate_paired_missingness(
        X_val, Xm_val, Xp_val, Xpm_val, pair_map
    )
    Xm_test, Xpm_test = v2.generate_paired_missingness(
        X_test, Xm_test, Xp_test, Xpm_test, pair_map
    )

print("\nMissingness on TEST (per feature):")
print(Xm_test.isna().mean())

#%% -------------------------------------------------------------------
# 5) Build imputers, wrap with StandardizeBeforeImpute, and fit ONCE
#    on Xm_train (exactly as in run_one_dataset)
# --------------------------------------------------------------------
base_imputers = v2.build_imputers(
    internal_imputer=["Mean/Mode", "Hot-Deck", "KNN", "LightGBM", "CatBoost", "PMM"],
    mice_iters=5,          # smaller for debugging
    rf_n_estimators=50,
    knn_k=5,
    tree_jobs=1,
    random_state=combo_seed,
)

imputers = v2.wrap_imputers_with_standardize(base_imputers)

for name_imp, imp in imputers.items():
    imp.fit(Xm_train)

print("\nBuilt & fitted imputers:", list(imputers.keys()))

#%% -------------------------------------------------------------------
# 6) TRAIN-based scalers from Xm_train (practical information only)
# --------------------------------------------------------------------
num_cols_train, _ = v2._split_feature_types(Xm_train)
x_zstats_train = v2._zscore_fit(Xm_train, num_cols_train)

if y_train is not None and pd.api.types.is_numeric_dtype(y_train):
    mean_y_train = float(y_train.mean())
    std_y_train = float(y_train.std(ddof=0)) or 1.0
    y_stats_train = (mean_y_train, std_y_train)
else:
    y_stats_train = None

print("\nnum_cols_train:", num_cols_train)
print("y_stats_train:", y_stats_train)

#%% -------------------------------------------------------------------
# 7) Baseline: direct error on TEST (uses Xm_train stats)
# --------------------------------------------------------------------
baseline_out = v2.validate_baseline_direct_error(
    X_test_complete=X_test,
    X_test_masked=Xm_test,
    imputer_dict=imputers,
    num_cols=num_cols_train,
    zstats=x_zstats_train,
)
print("\n[Baseline] keys (imputers):", baseline_out.keys())
one_imp_name = next(iter(baseline_out))
print("[Baseline] example for", one_imp_name, ":", baseline_out[one_imp_name])

#%% -------------------------------------------------------------------
# 8) Criterion 1.1: MCAR proxy on complete-case subset (from Xm_val)
# --------------------------------------------------------------------
X_val_cc = Xm_val.dropna(how="any").copy()
if X_val_cc.shape[0] > 0:
    X_val_ccp = v2.build_processed(X_val_cc)
    Xm_cc, Xpm_cc, _ = v2.generate_MCAR(
        X_val_cc, X_val_ccp, miss_vec, independent_feature_idx=None
    )

    c11_out = v2.validate_proxy_complete_case(
        imputer_dict=imputers,
        prebuilt_complete_case=X_val_cc,
        prebuilt_masked=Xm_cc,
        num_cols=num_cols_train,
        zstats=x_zstats_train,
    )
else:
    c11_out = {
        name: {
            "per_feature_full": {},
            "per_feature_missing_only": {},
            "mean_full": np.nan,
            "mean_missing_only": np.nan,
        }
        for name in imputers
    }

print("\n[C1.1] MCAR proxy – keys (imputers):", c11_out.keys())
print("[C1.1] example for", one_imp_name, ":", c11_out[one_imp_name])

#%% -------------------------------------------------------------------
# 9) Criterion 1.2: mechanism-aligned proxy on VAL (from Xm_val)
# --------------------------------------------------------------------
X_val_cc2 = Xm_val.dropna(how="any").copy()
if X_val_cc2.shape[0] > 0:
    X_val_ccp2 = v2.build_processed(X_val_cc2)
    Xm_cc2, Xpm_cc2, _ = gen(
        X_val_cc2, X_val_ccp2, miss_vec, independent_feature_idx=indep_idx
    )
    if v2._is_paired(mask_name):
        Xm_cc2, Xpm_cc2 = v2.generate_paired_missingness(
            X_val_cc2, Xm_cc2, X_val_ccp2, Xpm_cc2, pair_map
        )

    c12_out = v2.validate_proxy_complete_case(
        imputer_dict=imputers,
        prebuilt_complete_case=X_val_cc2,
        prebuilt_masked=Xm_cc2,
        num_cols=num_cols_train,
        zstats=x_zstats_train,
    )
else:
    c12_out = {
        name: {
            "per_feature_full": {},
            "per_feature_missing_only": {},
            "mean_full": np.nan,
            "mean_missing_only": np.nan,
        }
        for name in imputers
    }

print("\n[C1.2] Mechanism proxy – keys (imputers):", c12_out.keys())
print("[C1.2] example for", one_imp_name, ":", c12_out[one_imp_name])

#%% -------------------------------------------------------------------
# 10) Criterion 2: downstream performance on VAL
# --------------------------------------------------------------------
c2_out = v2.validate_downstream_performance(
    X_train_incomplete=Xm_train,
    y_train=y_train,
    X_val_incomplete=Xm_val,
    y_val=y_val,
    imputer_dict=imputers,
    model_dict=None,          # use DEFAULT_MODEL_LIST
    num_cols=num_cols_train,
    x_zstats=x_zstats_train,
    y_stats=y_stats_train,
)
print("\n[C2] Downstream – keys (imputers):", c2_out.keys())
print("[C2] example for", one_imp_name, ":", c2_out[one_imp_name])

#%% -------------------------------------------------------------------
# 11) Pack results exactly like run_one_dataset for this combo
# --------------------------------------------------------------------
result_one_combo = dict(
    baseline=baseline_out,
    criterion1_mcar=c11_out,
    criterion1_mechanism=c12_out,
    criterion2=c2_out,
)

# convenience slices for interactive inspection
# %%
result_one_combo["criterion2"]["CatBoost"]["per_model"]
# %%
result_one_combo["baseline"]["CatBoost"]["per_feature_missing_only"]
# %%
np.array(
    list(result_one_combo["criterion1_mcar"]["KNN"]["per_feature_missing_only"].values())
) - np.array(
    list(result_one_combo["criterion1_mcar"]["Mean/Mode"]["per_feature_missing_only"].values())
)
# %%
np.array(
    list(result_one_combo["criterion2"]["CatBoost"]["per_model"].values())
) - np.array(
    list(result_one_combo["criterion2"]["Mean/Mode"]["per_model"].values())
)
# %%
result_one_combo
# %%

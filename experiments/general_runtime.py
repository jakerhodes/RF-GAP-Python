# experiments/general_runtime.py

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forestkernel import ForestKernel
from experiments.baselines.rfgap import RFGAP
from experiments.baselines.naive_dense import original_proximity_dense_from_forest
from experiments.runtime_utils import (
    kernel_percent_nnz,
    load_dataset_pair,
    log_progress,
    resolve_dataset_paths_from_base_names,
    safe_timed_call,
    stratified_cap_subset,
    stratified_subset,
    timed_call,
)


# ---------------------------------------------------------------------
# CONFIG
# Edit directly here
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Use base dataset names only.
# Resolution order for each base name:
#   1) <base>_train.parquet + <base>_test.parquet
#   2) <base>.parquet
DATASET_NAMES = [
    "celegans",
    "covertype",
    "nsl_kdd+",
    "pathmnist_28",
    "pbmc",
    "sign_mnist",
    "tissuemnist_28",
    "tv_news_combined",
    "zilionis",
]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RUN_DIR / "general_runtime_results_multi_seed.csv"
OUT_PARQUET = RUN_DIR / "general_runtime_results_multi_seed.parquet"
PROGRESS_LOG = RUN_DIR / "general_runtime_progress.log"

# Multiple seed rounds
SEEDS = [44, 578, 9, 912, 345]

# dataprep options
LABEL_COL_IDX = 0
SCALE = "standardize"
GLOBAL_TRANSFORM = False
DROP_MISSING_Y = True
VERBOSE_DATAPREP = False

# Cap the training pool before building fractional subsets.
# If len(X_train_pool) > TRAIN_POOL_CAP, first stratified-subsample to TRAIN_POOL_CAP.
APPLY_TRAIN_POOL_CAP = True
TRAIN_POOL_CAP = 100_000

# subset schedule
TRAIN_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# underlying forest config
MODEL_TYPE = "rf"
BASE_MODEL_KWARGS = {
    "bootstrap": True,
    "n_jobs": -1,
}
if MODEL_TYPE == "xgb":
    BASE_MODEL_KWARGS["max_depth"] = 0

# methods to benchmark
KERNEL_METHODS = ["original", "oob", "gap"]
LEGACY_METHODS = {
    "original": "original",
    "oob": "oob",
    "gap": "rfgap",
}

# Optional baselines
INCLUDE_LEGACY_BASELINE = False
INCLUDE_NAIVE_BASELINE = False

# if a full kernel build fails, just record the failure and continue
RUN_FULL_KERNEL = True


# ---------------------------------------------------------------------
# Script-local utilities
# ---------------------------------------------------------------------
def get_model_kwargs_for_seed(seed: int) -> dict:
    model_kwargs = dict(BASE_MODEL_KWARGS)
    model_kwargs["random_state"] = seed
    return model_kwargs


def instantiate_fk(kernel_method: str, seed: int) -> ForestKernel:
    return ForestKernel(
        prediction_type="classification",
        kernel_method=kernel_method,
        model_type=MODEL_TYPE,
        **get_model_kwargs_for_seed(seed),
    )


def instantiate_legacy(prox_method: str, seed: int) -> RFGAP:
    return RFGAP(
        prediction_type="classification",
        prox_method=prox_method,
        matrix_type="sparse",
        triangular=False,
        non_zero_diagonal=(prox_method == "rfgap"),
        force_symmetric=False,
        **get_model_kwargs_for_seed(seed),
    )


def flush_results(rows: list[dict]) -> None:
    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUT_CSV, index=False)
    df_results.to_parquet(OUT_PARQUET, index=False)


def append_and_flush(rows: list[dict], row: dict) -> None:
    rows.append(row)
    flush_results(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dataset_groups = resolve_dataset_paths_from_base_names(DATA_DIR, DATASET_NAMES)
    fractions = TRAIN_FRACTIONS
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", PROGRESS_LOG)
    log_progress(f"Run directory: {RUN_DIR}", PROGRESS_LOG)
    log_progress(f"Resolved datasets: {sorted(dataset_groups.keys())}", PROGRESS_LOG)
    log_progress(f"Train fractions: {fractions}", PROGRESS_LOG)
    log_progress(f"Seeds: {SEEDS}", PROGRESS_LOG)
    log_progress(f"Apply train pool cap: {APPLY_TRAIN_POOL_CAP}", PROGRESS_LOG)
    log_progress(f"Train pool cap: {TRAIN_POOL_CAP}", PROGRESS_LOG)
    log_progress(f"Include legacy baseline: {INCLUDE_LEGACY_BASELINE}", PROGRESS_LOG)
    log_progress(f"Include naive baseline: {INCLUDE_NAIVE_BASELINE}", PROGRESS_LOG)
    log_progress(f"CSV output: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Parquet output: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress(f"Progress log: {PROGRESS_LOG}", PROGRESS_LOG)

    for dataset_name, paths in dataset_groups.items():
        log_progress(f"=== DATASET: {dataset_name} ===", PROGRESS_LOG)

        for seed in SEEDS:
            log_progress(f">>> SEED: {seed}", PROGRESS_LOG)

            try:
                X_train_pool, X_test, y_train_pool, y_test, meta = load_dataset_pair(
                    dataset_name=dataset_name,
                    paths=paths,
                    seed=seed,
                    label_col_idx=LABEL_COL_IDX,
                    scale=SCALE,
                    global_transform=GLOBAL_TRANSFORM,
                    drop_missing_y=DROP_MISSING_Y,
                    verbose_dataprep=VERBOSE_DATAPREP,
                )
            except Exception as e:
                log_progress(
                    f"Failed to load dataset {dataset_name} with seed {seed}: {e}",
                    PROGRESS_LOG,
                )
                continue

            original_train_pool_size = len(y_train_pool)

            if APPLY_TRAIN_POOL_CAP:
                X_train_pool, y_train_pool = stratified_cap_subset(
                    X_train_pool,
                    y_train_pool,
                    max_train_size=TRAIN_POOL_CAP,
                    seed=seed,
                )

            capped_train_pool_size = len(y_train_pool)

            log_progress(
                f"Loaded {dataset_name}: train_pool={X_train_pool.shape}, "
                f"test={X_test.shape}, predefined_split={meta['predefined_split']}, "
                f"original_train_pool_size={original_train_pool_size}, "
                f"capped_train_pool_size={capped_train_pool_size}",
                PROGRESS_LOG,
            )

            for split_id, frac in enumerate(fractions, start=1):
                log_progress(
                    f"--- split {split_id}/{len(fractions)} | frac={frac:.2f} | seed={seed} ---",
                    PROGRESS_LOG,
                )

                subset_seed = seed + split_id
                X_sub, y_sub = stratified_subset(
                    X_train_pool,
                    y_train_pool,
                    frac=frac,
                    seed=subset_seed,
                )

                n_sub = len(y_sub)
                log_progress(f"Subset shape: {X_sub.shape}", PROGRESS_LOG)

                # ---------------------------------------------------------
                # Shared forest fit, done once
                # ---------------------------------------------------------
                fk = instantiate_fk(kernel_method="original", seed=seed)

                _, forest_fit_time, forest_fit_mem = timed_call(
                    fk.fit_forest,
                    X_sub,
                    y_sub,
                )

                y_pred_forest, forest_pred_time, forest_pred_mem = timed_call(
                    fk.predict_forest,
                    X_test,
                )
                forest_acc = accuracy_score(y_test, y_pred_forest)

                append_and_flush(rows, {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "original_train_pool_size": original_train_pool_size,
                    "capped_train_pool_size": capped_train_pool_size,
                    "train_pool_cap_applied": capped_train_pool_size < original_train_pool_size,
                    "train_pool_cap": TRAIN_POOL_CAP if APPLY_TRAIN_POOL_CAP else np.nan,
                    "include_legacy_baseline": INCLUDE_LEGACY_BASELINE,
                    "include_naive_baseline": INCLUDE_NAIVE_BASELINE,
                    "split_id": split_id,
                    "train_fraction": frac,
                    "n_train_subset": n_sub,
                    "n_test": len(y_test),
                    "method_family": "shared_forest",
                    "method_name": "rf_fit_once",
                    "forest_fit_time_s": forest_fit_time,
                    "forest_fit_peak_mb": forest_fit_mem,
                    "forest_test_predict_time_s": forest_pred_time,
                    "forest_test_predict_peak_mb": forest_pred_mem,
                    "forest_test_acc": forest_acc,
                    "status": "ok",
                    "error": "",
                })

                log_progress(
                    f"Shared forest done | dataset={dataset_name} | seed={seed} | "
                    f"split={split_id} | fit_time={forest_fit_time:.3f}s | test_acc={forest_acc:.4f}",
                    PROGRESS_LOG,
                )

                # ---------------------------------------------------------
                # ForestKernel methods: rebuild cache only
                # ---------------------------------------------------------
                for km in KERNEL_METHODS:
                    log_progress(f"ForestKernel | {km}", PROGRESS_LOG)

                    _, cache_time, cache_mem = timed_call(
                        fk.build_kernel_cache,
                        kernel_method=km,
                    )

                    _, q_time, q_mem = timed_call(fk.get_train_query_map)

                    if RUN_FULL_KERNEL:
                        K_fk, k_time, k_mem, k_status, k_error = safe_timed_call(fk.get_kernel)
                        k_percent_nnz = kernel_percent_nnz(K_fk) if k_status == "ok" else np.nan
                    else:
                        k_time, k_mem, k_status, k_error = np.nan, np.nan, "skipped", ""
                        k_percent_nnz = np.nan

                    y_pred_kp, kp_time, kp_mem = timed_call(
                        fk.kernel_predict,
                        X_test,
                    )
                    kp_acc = accuracy_score(y_test, y_pred_kp)

                    append_and_flush(rows, {
                        "run_id": RUN_ID,
                        "dataset": dataset_name,
                        "seed": seed,
                        "predefined_split": meta["predefined_split"],
                        "original_train_pool_size": original_train_pool_size,
                        "capped_train_pool_size": capped_train_pool_size,
                        "train_pool_cap_applied": capped_train_pool_size < original_train_pool_size,
                        "train_pool_cap": TRAIN_POOL_CAP if APPLY_TRAIN_POOL_CAP else np.nan,
                        "include_legacy_baseline": INCLUDE_LEGACY_BASELINE,
                        "include_naive_baseline": INCLUDE_NAIVE_BASELINE,
                        "split_id": split_id,
                        "train_fraction": frac,
                        "n_train_subset": n_sub,
                        "n_test": len(y_test),
                        "method_family": "forestkernel",
                        "method_name": km,
                        "forest_fit_time_s": forest_fit_time,
                        "forest_fit_peak_mb": forest_fit_mem,
                        "cache_build_time_s": cache_time,
                        "cache_build_peak_mb": cache_mem,
                        "q_build_time_s": q_time,
                        "q_build_peak_mb": q_mem,
                        "full_kernel_time_s": k_time,
                        "full_kernel_peak_mb": k_mem,
                        "kernel_percent_nnz": k_percent_nnz,
                        "forest_test_acc": forest_acc,
                        "kernel_predict_time_s": kp_time,
                        "kernel_predict_peak_mb": kp_mem,
                        "kernel_predict_test_acc": kp_acc,
                        "status": k_status,
                        "error": k_error,
                    })

                    log_progress(
                        f"ForestKernel done | dataset={dataset_name} | seed={seed} | split={split_id} | "
                        f"method={km} | cache={cache_time:.3f}s | q={q_time:.3f}s | "
                        f"kernel={k_time if not np.isnan(k_time) else 'nan'} | "
                        f"%nnz={k_percent_nnz if not np.isnan(k_percent_nnz) else 'nan'} | "
                        f"kp_acc={kp_acc:.4f} | status={k_status}",
                        PROGRESS_LOG,
                    )

                # ---------------------------------------------------------
                # Legacy baselines: same fitted forest
                # ---------------------------------------------------------
                if INCLUDE_LEGACY_BASELINE:
                    for km in KERNEL_METHODS:
                        legacy_name = LEGACY_METHODS[km]
                        log_progress(f"Legacy | {legacy_name}", PROGRESS_LOG)

                        legacy = instantiate_legacy(legacy_name, seed=seed)
                        legacy.set_forest(fk.forest_, y=y_sub)

                        _, legacy_cache_time, legacy_cache_mem = timed_call(
                            legacy.build_proximity_cache,
                            X_sub,
                        )

                        if RUN_FULL_KERNEL:
                            K_legacy, legacy_k_time, legacy_k_mem, legacy_status, legacy_error = safe_timed_call(
                                legacy.get_proximities
                            )
                            legacy_percent_nnz = (
                                kernel_percent_nnz(K_legacy) if legacy_status == "ok" else np.nan
                            )
                        else:
                            legacy_k_time, legacy_k_mem, legacy_status, legacy_error = np.nan, np.nan, "skipped", ""
                            legacy_percent_nnz = np.nan

                        append_and_flush(rows, {
                            "run_id": RUN_ID,
                            "dataset": dataset_name,
                            "seed": seed,
                            "predefined_split": meta["predefined_split"],
                            "original_train_pool_size": original_train_pool_size,
                            "capped_train_pool_size": capped_train_pool_size,
                            "train_pool_cap_applied": capped_train_pool_size < original_train_pool_size,
                            "train_pool_cap": TRAIN_POOL_CAP if APPLY_TRAIN_POOL_CAP else np.nan,
                            "include_legacy_baseline": INCLUDE_LEGACY_BASELINE,
                            "include_naive_baseline": INCLUDE_NAIVE_BASELINE,
                            "split_id": split_id,
                            "train_fraction": frac,
                            "n_train_subset": n_sub,
                            "n_test": len(y_test),
                            "method_family": "legacy_rfgap",
                            "method_name": legacy_name,
                            "forest_fit_time_s": forest_fit_time,
                            "forest_fit_peak_mb": forest_fit_mem,
                            "cache_build_time_s": legacy_cache_time,
                            "cache_build_peak_mb": legacy_cache_mem,
                            "full_kernel_time_s": legacy_k_time,
                            "full_kernel_peak_mb": legacy_k_mem,
                            "kernel_percent_nnz": legacy_percent_nnz,
                            "forest_test_acc": forest_acc,
                            "status": legacy_status,
                            "error": legacy_error,
                        })

                        log_progress(
                            f"Legacy done | dataset={dataset_name} | seed={seed} | split={split_id} | "
                            f"method={legacy_name} | cache={legacy_cache_time:.3f}s | "
                            f"kernel={legacy_k_time if not np.isnan(legacy_k_time) else 'nan'} | "
                            f"%nnz={legacy_percent_nnz if not np.isnan(legacy_percent_nnz) else 'nan'} | "
                            f"status={legacy_status}",
                            PROGRESS_LOG,
                        )

                # ---------------------------------------------------------
                # Naive dense original proximity: same fitted forest
                # ---------------------------------------------------------
                if INCLUDE_NAIVE_BASELINE:
                    log_progress("Naive | original_dense", PROGRESS_LOG)

                    _, naive_time, naive_mem, naive_status, naive_error = safe_timed_call(
                        original_proximity_dense_from_forest,
                        fk.forest_,
                        X_sub,
                    )

                    append_and_flush(rows, {
                        "run_id": RUN_ID,
                        "dataset": dataset_name,
                        "seed": seed,
                        "predefined_split": meta["predefined_split"],
                        "original_train_pool_size": original_train_pool_size,
                        "capped_train_pool_size": capped_train_pool_size,
                        "train_pool_cap_applied": capped_train_pool_size < original_train_pool_size,
                        "train_pool_cap": TRAIN_POOL_CAP if APPLY_TRAIN_POOL_CAP else np.nan,
                        "include_legacy_baseline": INCLUDE_LEGACY_BASELINE,
                        "include_naive_baseline": INCLUDE_NAIVE_BASELINE,
                        "split_id": split_id,
                        "train_fraction": frac,
                        "n_train_subset": n_sub,
                        "n_test": len(y_test),
                        "method_family": "naive_dense",
                        "method_name": "original_dense",
                        "forest_fit_time_s": forest_fit_time,
                        "forest_fit_peak_mb": forest_fit_mem,
                        "full_kernel_time_s": naive_time,
                        "full_kernel_peak_mb": naive_mem,
                        "forest_test_acc": forest_acc,
                        "status": naive_status,
                        "error": naive_error,
                    })

                    log_progress(
                        f"Naive done | dataset={dataset_name} | seed={seed} | split={split_id} | "
                        f"time={naive_time if not np.isnan(naive_time) else 'nan'} | status={naive_status}",
                        PROGRESS_LOG,
                    )

    flush_results(rows)

    log_progress(f"Saved results to: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Saved results to: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress("Done.", PROGRESS_LOG)


if __name__ == "__main__":
    main()
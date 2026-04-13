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

def run_fk_full_pipeline(
    fk: ForestKernel,
    X_sub,
    y_sub,
    X_test,
    y_test,
    kernel_method: str,
):
    t0 = time.perf_counter()
    fk.fit_forest(X_sub, y_sub)
    forest_fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_forest = fk.predict_forest(X_test)
    forest_pred_time = time.perf_counter() - t0
    forest_acc = accuracy_score(y_test, y_pred_forest)

    t0 = time.perf_counter()
    fk.build_kernel_cache(kernel_method=kernel_method)
    cache_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fk.get_train_query_map()
    q_time = time.perf_counter() - t0

    if RUN_FULL_KERNEL:
        t0 = time.perf_counter()
        K_fk = fk.get_kernel()
        k_time = time.perf_counter() - t0
        k_percent_nnz = kernel_percent_nnz(K_fk)
    else:
        k_time = np.nan
        k_percent_nnz = np.nan

    t0 = time.perf_counter()
    y_pred_kp = fk.kernel_predict(X_test)
    kp_time = time.perf_counter() - t0
    kp_acc = accuracy_score(y_test, y_pred_kp)

    return {
        "forest_fit_time_s": forest_fit_time,
        "forest_test_predict_time_s": forest_pred_time,
        "forest_test_acc": forest_acc,
        "cache_build_time_s": cache_time,
        "q_build_time_s": q_time,
        "full_kernel_time_s": k_time,
        "kernel_percent_nnz": k_percent_nnz,
        "kernel_predict_time_s": kp_time,
        "kernel_predict_test_acc": kp_acc,
    }


def run_legacy_full_pipeline(
    legacy: RFGAP,
    X_sub,
):
    t0 = time.perf_counter()
    legacy.build_proximity_cache(X_sub)
    cache_time = time.perf_counter() - t0

    if RUN_FULL_KERNEL:
        t0 = time.perf_counter()
        K_legacy = legacy.get_proximities()
        k_time = time.perf_counter() - t0
        k_percent_nnz = kernel_percent_nnz(K_legacy)
    else:
        k_time = np.nan
        k_percent_nnz = np.nan

    return {
        "cache_build_time_s": cache_time,
        "full_kernel_time_s": k_time,
        "kernel_percent_nnz": k_percent_nnz,
    }

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
                # ForestKernel methods
                # ---------------------------------------------------------
                for km in KERNEL_METHODS:
                    log_progress(f"ForestKernel | {km}", PROGRESS_LOG)
                
                    fk = instantiate_fk(kernel_method=km, seed=seed)
                
                    pipeline_out, pipeline_time, pipeline_peak_mb, pipeline_status, pipeline_error = safe_timed_call(
                        run_fk_full_pipeline,
                        fk,
                        X_sub,
                        y_sub,
                        X_test,
                        y_test,
                        km,
                    )
                
                    if pipeline_status == "ok":
                        forest_fit_time = pipeline_out["forest_fit_time_s"]
                        forest_pred_time = pipeline_out["forest_test_predict_time_s"]
                        forest_acc = pipeline_out["forest_test_acc"]
                        cache_time = pipeline_out["cache_build_time_s"]
                        q_time = pipeline_out["q_build_time_s"]
                        k_time = pipeline_out["full_kernel_time_s"]
                        k_percent_nnz = pipeline_out["kernel_percent_nnz"]
                        kp_time = pipeline_out["kernel_predict_time_s"]
                        kp_acc = pipeline_out["kernel_predict_test_acc"]
                    else:
                        forest_fit_time = np.nan
                        forest_pred_time = np.nan
                        forest_acc = np.nan
                        cache_time = np.nan
                        q_time = np.nan
                        k_time = np.nan
                        k_percent_nnz = np.nan
                        kp_time = np.nan
                        kp_acc = np.nan
                
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
                        "forest_test_predict_time_s": forest_pred_time,
                        "forest_test_acc": forest_acc,
                        "cache_build_time_s": cache_time,
                        "q_build_time_s": q_time,
                        "full_kernel_time_s": k_time,
                        "kernel_percent_nnz": k_percent_nnz,
                        "kernel_predict_time_s": kp_time,
                        "kernel_predict_test_acc": kp_acc,
                        "pipeline_peak_mb": pipeline_peak_mb,
                        "status": pipeline_status,
                        "error": pipeline_error,
                    })
                
                    log_progress(
                        f"ForestKernel done | dataset={dataset_name} | seed={seed} | split={split_id} | "
                        f"method={km} | fit={forest_fit_time if not np.isnan(forest_fit_time) else 'nan'}s | "
                        f"cache={cache_time if not np.isnan(cache_time) else 'nan'}s | "
                        f"q={q_time if not np.isnan(q_time) else 'nan'}s | "
                        f"kernel={k_time if not np.isnan(k_time) else 'nan'}s | "
                        f"kp={kp_time if not np.isnan(kp_time) else 'nan'}s | "
                        f"peak_mb={pipeline_peak_mb if not np.isnan(pipeline_peak_mb) else 'nan'} | "
                        f"%nnz={k_percent_nnz if not np.isnan(k_percent_nnz) else 'nan'} | "
                        f"forest_acc={forest_acc if not np.isnan(forest_acc) else 'nan'} | "
                        f"kp_acc={kp_acc if not np.isnan(kp_acc) else 'nan'} | "
                        f"status={pipeline_status}",
                        PROGRESS_LOG,
                    )

                # ---------------------------------------------------------
                # Legacy baselines: same fitted forest
                # ---------------------------------------------------------
                if INCLUDE_LEGACY_BASELINE:
                    for km in KERNEL_METHODS:
                        legacy_name = LEGACY_METHODS[km]
                        log_progress(f"Legacy | {legacy_name}", PROGRESS_LOG)
                
                        # Reuse the forest fitted in the ForestKernel pipeline above
                        legacy = instantiate_legacy(legacy_name, seed=seed)
                        legacy.set_forest(fk.forest_, y=y_sub)
                
                        legacy_out, legacy_pipeline_time, legacy_peak_mb, legacy_status, legacy_error = safe_timed_call(
                            run_legacy_full_pipeline,
                            legacy,
                            X_sub,
                        )
                
                        if legacy_status == "ok":
                            legacy_cache_time = legacy_out["cache_build_time_s"]
                            legacy_k_time = legacy_out["full_kernel_time_s"]
                            legacy_percent_nnz = legacy_out["kernel_percent_nnz"]
                        else:
                            legacy_cache_time = np.nan
                            legacy_k_time = np.nan
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
                            "forest_test_predict_time_s": forest_pred_time,
                            "forest_test_acc": forest_acc,
                            "cache_build_time_s": legacy_cache_time,
                            "q_build_time_s": np.nan,
                            "full_kernel_time_s": legacy_k_time,
                            "kernel_percent_nnz": legacy_percent_nnz,
                            "kernel_predict_time_s": np.nan,
                            "kernel_predict_test_acc": np.nan,
                            "pipeline_peak_mb": legacy_peak_mb,
                            "status": legacy_status,
                            "error": legacy_error,
                        })
                
                        log_progress(
                            f"Legacy done | dataset={dataset_name} | seed={seed} | split={split_id} | "
                            f"method={legacy_name} | cache={legacy_cache_time if not np.isnan(legacy_cache_time) else 'nan'}s | "
                            f"kernel={legacy_k_time if not np.isnan(legacy_k_time) else 'nan'}s | "
                            f"peak_mb={legacy_peak_mb if not np.isnan(legacy_peak_mb) else 'nan'} | "
                            f"%nnz={legacy_percent_nnz if not np.isnan(legacy_percent_nnz) else 'nan'} | "
                            f"status={legacy_status}",
                            PROGRESS_LOG,
                        )

                # ---------------------------------------------------------
                # Naive dense original proximity: same fitted forest
                # ---------------------------------------------------------
                if INCLUDE_NAIVE_BASELINE:
                    log_progress("Naive | original_dense", PROGRESS_LOG)
                
                    _, naive_time, naive_peak_mb, naive_status, naive_error = safe_timed_call(
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
                        "forest_test_predict_time_s": forest_pred_time,
                        "forest_test_acc": forest_acc,
                        "cache_build_time_s": np.nan,
                        "q_build_time_s": np.nan,
                        "full_kernel_time_s": naive_time,
                        "kernel_percent_nnz": np.nan,
                        "kernel_predict_time_s": np.nan,
                        "kernel_predict_test_acc": np.nan,
                        "pipeline_peak_mb": naive_peak_mb,
                        "status": naive_status,
                        "error": naive_error,
                    })
                
                    log_progress(
                        f"Naive done | dataset={dataset_name} | seed={seed} | split={split_id} | "
                        f"time={naive_time if not np.isnan(naive_time) else 'nan'}s | "
                        f"peak_mb={naive_peak_mb if not np.isnan(naive_peak_mb) else 'nan'} | "
                        f"status={naive_status}",
                        PROGRESS_LOG,
                    )

    flush_results(rows)

    log_progress(f"Saved results to: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Saved results to: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress("Done.", PROGRESS_LOG)


if __name__ == "__main__":
    main()
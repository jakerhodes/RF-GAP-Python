from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forestkernel import ForestKernel
from experiments.runtime_utils import (
    kernel_percent_nnz,
    load_dataset_pair,
    log_progress,
    resolve_dataset_paths_from_base_names,
    safe_timed_call,
)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Used only for dataset ablation
DATASET_ABLATION_DATASET_NAMES = [
    "higgs",
    "susy",
    "epsilon",
    "airlines",
    "celegans",
    "covertype",
    # "nsl_kdd+",
    "pathmnist_28",
    "pbmc",
    "sign_mnist",
    "tissuemnist_28",
    "tv_news_combined",
    "zilionis",
]

# Used for all non-dataset ablations
FIXED_ABLATION_DATASET_NAMES = [
    "airlines",
    # "pathmnist_28",
    # "tv_news_combined",
    "covertype",
]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"{RUN_ID}_ablation"
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [44, 578, 9, 912, 345]

LABEL_COL_IDX = 0
DROP_MISSING_Y = True
VERBOSE_DATAPREP = False

# ---------------------------------------------------------
# Train subset sizes used for scaling curves.
# Sizes are generated per dataset from a global minimum power
# of two, then doubled until the largest power of two below
# full size. The exact full size is appended if needed.
# ---------------------------------------------------------
MIN_POW = 14   # 2**14 = 16384

RUN_DATASET_ABLATION = True
RUN_KERNEL_METHOD_ABLATION = True
RUN_MODEL_TYPE_ABLATION = True
RUN_MAX_DEPTH_ABLATION = True
RUN_MIN_SAMPLES_LEAF_ABLATION = True

RUN_FULL_KERNEL = True


# ---------------------------------------------------------------------
# ABLATION SETTINGS
# ---------------------------------------------------------------------

# 1) dataset ablation
DATASET_ABLATION_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": "dataset_ablation",
        "ablation_cfg": {"bootstrap": True},
    }
]

# 2) kernel method ablation
KERNEL_METHOD_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "original",
        "ablation_name": "kernel_method=original",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "rf",
        "kernel_method": "kerf",
        "ablation_name": "kernel_method=kerf",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "rf",
        "kernel_method": "oob",
        "ablation_name": "kernel_method=oob",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": "kernel_method=gap",
        "ablation_cfg": {"bootstrap": True},
    },
]

# 3) model type ablation
MODEL_TYPE_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": "model_type=rf",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "et",
        "kernel_method": "gap",
        "ablation_name": "model_type=et",
        "ablation_cfg": {"bootstrap": True},
    },
    # {
    #     "model_type": "xgb",
    #     "kernel_method": "original",
    #     "ablation_name": "model_type=xgb",
    #     "ablation_cfg": {"max_depth": 0},
    # },
]

# 4) max depth ablation
MAX_DEPTH_VALUES = [10, 20, 30, None]
MAX_DEPTH_FIXED_CFG = {
    "bootstrap": True,
    "min_samples_leaf": 1,
}
MAX_DEPTH_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": f"max_depth={value}",
        "ablation_cfg": {"max_depth": value, **MAX_DEPTH_FIXED_CFG},
    }
    for value in MAX_DEPTH_VALUES
]

# 5) min samples leaf ablation
MIN_SAMPLES_LEAF_VALUES = [1, 5, 10, 20]
MIN_SAMPLES_LEAF_FIXED_CFG = {
    "bootstrap": True,
    "max_depth": None,
}
MIN_SAMPLES_LEAF_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": f"min_samples_leaf={value}",
        "ablation_cfg": {"min_samples_leaf": value, **MIN_SAMPLES_LEAF_FIXED_CFG},
    }
    for value in MIN_SAMPLES_LEAF_VALUES
]


# ---------------------------------------------------------------------
# Script-local utilities
# ---------------------------------------------------------------------
def make_output_paths(mode_name: str) -> dict[str, Path]:
    mode_dir = RUN_DIR / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": mode_dir,
        "csv": mode_dir / f"{mode_name}_results.csv",
        "parquet": mode_dir / f"{mode_name}_results.parquet",
        "log": mode_dir / f"{mode_name}_progress.log",
    }


def flush_results(rows: list[dict], out_csv: Path, out_parquet: Path) -> None:
    df_results = pd.DataFrame(rows)
    df_results.to_csv(out_csv, index=False)
    df_results.to_parquet(out_parquet, index=False)


def append_and_flush(rows: list[dict], row: dict, out_csv: Path, out_parquet: Path) -> None:
    rows.append(row)
    flush_results(rows, out_csv, out_parquet)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def ceil_log2_int(n: int) -> int:
    if n <= 1:
        return 0
    return int(np.ceil(np.log2(n)))


def floor_log2_int(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive.")
    return int(np.floor(np.log2(n)))


def make_train_size_grid(
    n_max: int,
    min_pow: int = MIN_POW,
) -> tuple[list[int], int | None, int | None]:
    """
    Build a per-dataset power-of-two grid from 2**min_pow upward.
    Append n_max if it is not already a power of two.
    """
    if n_max <= 0:
        raise ValueError("n_max must be positive.")

    k_max = floor_log2_int(n_max)

    if min_pow > k_max:
        return [n_max], None, None

    sizes = [2 ** k for k in range(min_pow, k_max + 1)]

    if sizes[-1] != n_max:
        sizes.append(n_max)

    sizes = sorted(set(sizes))
    return sizes, min_pow, k_max


def sample_train_subset_size(
    X: np.ndarray,
    y: np.ndarray,
    train_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_size >= len(y):
        return X, y

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=seed,
    )
    idx, _ = next(splitter.split(X, y))
    return X[idx], y[idx]


def instantiate_fk(
    model_type: str,
    kernel_method: str,
    seed: int,
    model_kwargs: dict[str, object],
) -> ForestKernel:
    kwargs = dict(model_kwargs)

    if model_type in {"rf", "et"}:
        kwargs.setdefault("n_jobs", -1)
    elif model_type == "xgb":
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("device", "cuda")

    kwargs["random_state"] = seed

    return ForestKernel(
        prediction_type="classification",
        kernel_method=kernel_method,
        model_type=model_type,
        **kwargs,
    )


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


# ---------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------
def run_one_ablation_mode(
    mode_name: str,
    settings: list[dict[str, object]],
    dataset_groups: dict[str, dict[str, Path | None]],
) -> None:
    paths = make_output_paths(mode_name)
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", paths["log"])
    log_progress(f"Mode: {mode_name}", paths["log"])
    log_progress(f"Run directory: {paths['dir']}", paths["log"])
    log_progress(f"Resolved datasets: {sorted(dataset_groups.keys())}", paths["log"])
    log_progress(f"MIN_POW: {MIN_POW}", paths["log"])
    log_progress("Grid type: powers_of_two_plus_full_size", paths["log"])
    log_progress(f"Seeds: {SEEDS}", paths["log"])
    log_progress("Scale: None", paths["log"])
    log_progress("Global transform: False", paths["log"])
    log_progress(f"Number of settings: {len(settings)}", paths["log"])
    log_progress(f"CSV output: {paths['csv']}", paths["log"])
    log_progress(f"Parquet output: {paths['parquet']}", paths["log"])
    log_progress(f"Progress log: {paths['log']}", paths["log"])

    for setting in settings:
        log_progress(
            f"Setting prepared | model_type={setting['model_type']} | "
            f"kernel_method={setting['kernel_method']} | "
            f"name={setting['ablation_name']} | cfg={setting['ablation_cfg']}",
            paths["log"],
        )

    for dataset_name, dataset_paths in dataset_groups.items():
        log_progress(f"=== DATASET: {dataset_name} ===", paths["log"])
        log_progress(
            f"Dataprep scheme | dataset={dataset_name} | scale=None | global_transform=False",
            paths["log"],
        )

        for seed in SEEDS:
            log_progress(f">>> SEED: {seed}", paths["log"])

            try:
                X_train_pool, X_test, y_train_pool, y_test, meta = load_dataset_pair(
                    dataset_name=dataset_name,
                    paths=dataset_paths,
                    seed=seed,
                    label_col_idx=LABEL_COL_IDX,
                    scale=None,
                    global_transform=False,
                    drop_missing_y=DROP_MISSING_Y,
                    verbose_dataprep=VERBOSE_DATAPREP,
                )
            except Exception as e:
                log_progress(
                    f"Failed to load dataset {dataset_name} with seed {seed}: {e}",
                    paths["log"],
                )
                continue

            available_train_size = len(y_train_pool)
            train_sizes, k_min, k_max = make_train_size_grid(
                n_max=available_train_size,
                min_pow=MIN_POW,
            )

            log_progress(
                f"Loaded {dataset_name}: train_pool={X_train_pool.shape}, "
                f"test={X_test.shape}, predefined_split={meta['predefined_split']}, "
                f"available_train_size={available_train_size}",
                paths["log"],
            )
            log_progress(f"Dataset-specific k_min: {k_min}", paths["log"])
            log_progress(f"Dataset-specific k_max: {k_max}", paths["log"])
            log_progress(f"Train sizes: {train_sizes}", paths["log"])

            for size_id, train_size in enumerate(train_sizes, start=1):
                log_progress(
                    f"--- size {size_id}/{len(train_sizes)} | "
                    f"train_size={train_size} | seed={seed} ---",
                    paths["log"],
                )

                subset_seed = seed + size_id
                X_sub, y_sub = sample_train_subset_size(
                    X_train_pool,
                    y_train_pool,
                    train_size=train_size,
                    seed=subset_seed,
                )

                n_sub = len(y_sub)
                log_progress(f"Subset shape: {X_sub.shape}", paths["log"])

                for ablation_id, setting in enumerate(settings, start=1):
                    model_type = setting["model_type"]
                    kernel_method = setting["kernel_method"]
                    ablation_name = setting["ablation_name"]
                    ablation_cfg = setting["ablation_cfg"]

                    log_progress(
                        f"Ablation {ablation_id}/{len(settings)} | "
                        f"model_type={model_type} | kernel_method={kernel_method} | "
                        f"{ablation_name}",
                        paths["log"],
                    )

                    fk = instantiate_fk(
                        model_type=model_type,
                        kernel_method=kernel_method,
                        seed=seed,
                        model_kwargs=ablation_cfg,
                    )

                    pipeline_out, pipeline_time, pipeline_peak_mb, pipeline_status, pipeline_error = safe_timed_call(
                        run_fk_full_pipeline,
                        fk,
                        X_sub,
                        y_sub,
                        X_test,
                        y_test,
                        kernel_method,
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

                    row = {
                        "run_id": RUN_ID,
                        "ablation_mode": mode_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "predefined_split": meta["predefined_split"],
                        "scale": None,
                        "global_transform": False,
                        "model_type": model_type,
                        "kernel_method": kernel_method,
                        "ablation_id": ablation_id,
                        "ablation_name": ablation_name,
                        "ablation_cfg": str(ablation_cfg),
                        "available_train_size": available_train_size,
                        "min_pow": MIN_POW,
                        "dataset_k_min": k_min,
                        "dataset_k_max": k_max,
                        "size_id": size_id,
                        "requested_train_size": train_size,
                        "is_power_of_two_size": is_power_of_two(train_size),
                        "log2_requested_train_size": np.log2(train_size),
                        "n_train_subset": n_sub,
                        "n_test": len(y_test),
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
                    }
                    append_and_flush(rows, row, paths["csv"], paths["parquet"])

                    log_progress(
                        f"Done | dataset={dataset_name} | seed={seed} | "
                        f"train_size={train_size} | n_train={n_sub} | "
                        f"model_type={model_type} | kernel_method={kernel_method} | "
                        f"ablation={ablation_name} | "
                        f"fit={forest_fit_time if not np.isnan(forest_fit_time) else 'nan'}s | "
                        f"cache={cache_time if not np.isnan(cache_time) else 'nan'}s | "
                        f"q={q_time if not np.isnan(q_time) else 'nan'}s | "
                        f"kernel={k_time if not np.isnan(k_time) else 'nan'}s | "
                        f"kp={kp_time if not np.isnan(kp_time) else 'nan'}s | "
                        f"peak_mb={pipeline_peak_mb if not np.isnan(pipeline_peak_mb) else 'nan'} | "
                        f"%nnz={k_percent_nnz if not np.isnan(k_percent_nnz) else 'nan'} | "
                        f"forest_acc={forest_acc if not np.isnan(forest_acc) else 'nan'} | "
                        f"kp_acc={kp_acc if not np.isnan(kp_acc) else 'nan'} | "
                        f"status={pipeline_status}",
                        paths["log"],
                    )

    flush_results(rows, paths["csv"], paths["parquet"])
    log_progress(f"Saved results to: {paths['csv']}", paths["log"])
    log_progress(f"Saved results to: {paths['parquet']}", paths["log"])
    log_progress("Done.", paths["log"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dataset_ablation_groups = resolve_dataset_paths_from_base_names(
        DATA_DIR,
        DATASET_ABLATION_DATASET_NAMES,
    )
    fixed_ablation_groups = resolve_dataset_paths_from_base_names(
        DATA_DIR,
        FIXED_ABLATION_DATASET_NAMES,
    )

    if RUN_DATASET_ABLATION:
        run_one_ablation_mode(
            mode_name="dataset",
            settings=DATASET_ABLATION_SETTINGS,
            dataset_groups=dataset_ablation_groups,
        )

    if RUN_KERNEL_METHOD_ABLATION:
        run_one_ablation_mode(
            mode_name="kernel_method",
            settings=KERNEL_METHOD_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )

    if RUN_MODEL_TYPE_ABLATION:
        run_one_ablation_mode(
            mode_name="model_type",
            settings=MODEL_TYPE_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )

    if RUN_MAX_DEPTH_ABLATION:
        run_one_ablation_mode(
            mode_name="max_depth",
            settings=MAX_DEPTH_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )

    if RUN_MIN_SAMPLES_LEAF_ABLATION:
        run_one_ablation_mode(
            mode_name="min_samples_leaf",
            settings=MIN_SAMPLES_LEAF_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )


if __name__ == "__main__":
    main()
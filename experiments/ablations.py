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
    timed_call,
)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Used only for dataset ablation
DATASET_ABLATION_DATASET_NAMES = [
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

# Used for all non-dataset ablations
FIXED_ABLATION_DATASET_NAMES = [
    "pathmnist_28",
    "tv_news_combined",
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

# Default dataprep fallback
DEFAULT_SCALE = "standardize"
DEFAULT_GLOBAL_TRANSFORM = False

# ---------------------------------------------------------
# Per-dataset dataprep scheme
# ---------------------------------------------------------
DATASET_DATAPREP = {
    "pathmnist_28": {"scale": "normalize", "global_transform": True},
    "sign_mnist": {"scale": "normalize", "global_transform": True},
    "tissuemnist_28": {"scale": "normalize", "global_transform": True},
}

# ---------------------------------------------------------
# Train fractions used for subset scaling curves.
# Each dataset uses fractions of its own available train set.
# ---------------------------------------------------------
TRAIN_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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


def sample_train_subset_fraction(
    X: np.ndarray,
    y: np.ndarray,
    frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if frac >= 1.0:
        return X, y

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=frac,
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


def get_dataprep_kwargs(dataset_name: str) -> dict[str, object]:
    cfg = DATASET_DATAPREP.get(dataset_name, {})
    return {
        "scale": cfg.get("scale", DEFAULT_SCALE),
        "global_transform": cfg.get("global_transform", DEFAULT_GLOBAL_TRANSFORM),
    }


# ---------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------
def run_one_ablation_mode(
    mode_name: str,
    settings: list[dict[str, object]],
    dataset_groups: dict[str, dict[str, Path | None]],
    train_fractions: list[float],
) -> None:
    paths = make_output_paths(mode_name)
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", paths["log"])
    log_progress(f"Mode: {mode_name}", paths["log"])
    log_progress(f"Run directory: {paths['dir']}", paths["log"])
    log_progress(f"Resolved datasets: {sorted(dataset_groups.keys())}", paths["log"])
    log_progress(f"Train fractions: {train_fractions}", paths["log"])
    log_progress(f"Seeds: {SEEDS}", paths["log"])
    log_progress(f"Default scale: {DEFAULT_SCALE}", paths["log"])
    log_progress(f"Default global_transform: {DEFAULT_GLOBAL_TRANSFORM}", paths["log"])
    log_progress(f"Per-dataset dataprep: {DATASET_DATAPREP}", paths["log"])
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

        dataprep_kwargs = get_dataprep_kwargs(dataset_name)
        scale = dataprep_kwargs["scale"]
        global_transform = dataprep_kwargs["global_transform"]

        log_progress(
            f"Dataprep scheme | dataset={dataset_name} | "
            f"scale={scale} | global_transform={global_transform}",
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
                    scale=scale,
                    global_transform=global_transform,
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

            log_progress(
                f"Loaded {dataset_name}: train_pool={X_train_pool.shape}, "
                f"test={X_test.shape}, predefined_split={meta['predefined_split']}, "
                f"available_train_size={available_train_size}",
                paths["log"],
            )

            for frac_id, train_fraction in enumerate(train_fractions, start=1):
                log_progress(
                    f"--- fraction {frac_id}/{len(train_fractions)} | "
                    f"train_fraction={train_fraction:.2f} | seed={seed} ---",
                    paths["log"],
                )

                subset_seed = seed + frac_id
                X_sub, y_sub = sample_train_subset_fraction(
                    X_train_pool,
                    y_train_pool,
                    frac=train_fraction,
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

                    _, cache_time, cache_mem = timed_call(
                        fk.build_kernel_cache,
                        kernel_method=kernel_method,
                    )

                    _, q_time, q_mem = timed_call(
                        fk.get_train_query_map
                    )

                    if RUN_FULL_KERNEL:
                        K_fk, k_time, k_mem, k_status, k_error = safe_timed_call(
                            fk.get_kernel
                        )
                        k_percent_nnz = (
                            kernel_percent_nnz(K_fk) if k_status == "ok" else np.nan
                        )
                    else:
                        k_time, k_mem, k_status, k_error = np.nan, np.nan, "skipped", ""
                        k_percent_nnz = np.nan

                    y_pred_kp, kp_time, kp_mem = timed_call(
                        fk.kernel_predict,
                        X_test,
                    )
                    kp_acc = accuracy_score(y_test, y_pred_kp)

                    row = {
                        "run_id": RUN_ID,
                        "ablation_mode": mode_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "predefined_split": meta["predefined_split"],
                        "scale": scale,
                        "global_transform": global_transform,
                        "model_type": model_type,
                        "kernel_method": kernel_method,
                        "ablation_id": ablation_id,
                        "ablation_name": ablation_name,
                        "ablation_cfg": str(ablation_cfg),
                        "available_train_size": available_train_size,
                        "frac_id": frac_id,
                        "train_fraction": train_fraction,
                        "n_train_subset": n_sub,
                        "n_test": len(y_test),
                        "forest_fit_time_s": forest_fit_time,
                        "forest_fit_peak_mb": forest_fit_mem,
                        "forest_test_predict_time_s": forest_pred_time,
                        "forest_test_predict_peak_mb": forest_pred_mem,
                        "forest_test_acc": forest_acc,
                        "cache_build_time_s": cache_time,
                        "cache_build_peak_mb": cache_mem,
                        "q_build_time_s": q_time,
                        "q_build_peak_mb": q_mem,
                        "full_kernel_time_s": k_time,
                        "full_kernel_peak_mb": k_mem,
                        "kernel_percent_nnz": k_percent_nnz,
                        "kernel_predict_time_s": kp_time,
                        "kernel_predict_peak_mb": kp_mem,
                        "kernel_predict_test_acc": kp_acc,
                        "status": k_status,
                        "error": k_error,
                    }
                    append_and_flush(rows, row, paths["csv"], paths["parquet"])

                    log_progress(
                        f"Done | dataset={dataset_name} | seed={seed} | "
                        f"train_fraction={train_fraction:.2f} | n_train={n_sub} | "
                        f"model_type={model_type} | kernel_method={kernel_method} | "
                        f"ablation={ablation_name} | fit={forest_fit_time:.3f}s | "
                        f"cache={cache_time:.3f}s | q={q_time:.3f}s | "
                        f"kernel={k_time if not np.isnan(k_time) else 'nan'} | "
                        f"%nnz={k_percent_nnz if not np.isnan(k_percent_nnz) else 'nan'} | "
                        f"forest_acc={forest_acc:.4f} | kp_acc={kp_acc:.4f} | "
                        f"status={k_status}",
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
            train_fractions=TRAIN_FRACTIONS,
        )

    if RUN_KERNEL_METHOD_ABLATION:
        run_one_ablation_mode(
            mode_name="kernel_method",
            settings=KERNEL_METHOD_SETTINGS,
            dataset_groups=fixed_ablation_groups,
            train_fractions=TRAIN_FRACTIONS,
        )

    if RUN_MODEL_TYPE_ABLATION:
        run_one_ablation_mode(
            mode_name="model_type",
            settings=MODEL_TYPE_SETTINGS,
            dataset_groups=fixed_ablation_groups,
            train_fractions=TRAIN_FRACTIONS,
        )

    if RUN_MAX_DEPTH_ABLATION:
        run_one_ablation_mode(
            mode_name="max_depth",
            settings=MAX_DEPTH_SETTINGS,
            dataset_groups=fixed_ablation_groups,
            train_fractions=TRAIN_FRACTIONS,
        )

    if RUN_MIN_SAMPLES_LEAF_ABLATION:
        run_one_ablation_mode(
            mode_name="min_samples_leaf",
            settings=MIN_SAMPLES_LEAF_SETTINGS,
            dataset_groups=fixed_ablation_groups,
            train_fractions=TRAIN_FRACTIONS,
        )


if __name__ == "__main__":
    main()
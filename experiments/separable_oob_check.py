# prox_equivalence_grid_oob_count_ratio.py
#
# Sweep over:
#   - train fractions
#   - number of trees
#   - multiple seeds
#
# For each setting:
#   - fit shared forest
#   - extract OOB masks
#   - compare LeafEncoder OOB mask vs legacy OOB mask
#   - compute the proposition ratio
#
# Proposition quantity:
#   R_ij = S(i,j) / ( S(i) S(j) / T )
#
# recorded over off-diagonal pairs with S(i,j) > 0.
#
# This avoids building the full kernel matrix.

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import dataprep
from forestkernel import LeafEncoder
from experiments.baselines.rfgap import RFGAP


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

DATASET_NAME = "sign_mnist"
LABEL_COL_IDX = 0
SEEDS = [44, 578, 9, 912, 345]

TRAIN_FRACTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
N_ESTIMATORS_GRID = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

HOLDOUT_TEST_SIZE = 0.2

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"{RUN_ID}_oob_count_ratio_grid"
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RUN_DIR / "oob_count_ratio_grid_results.csv"
OUT_PARQUET = RUN_DIR / "oob_count_ratio_grid_results.parquet"
PROGRESS_LOG = RUN_DIR / "progress.log"


# ---------------------------------------------------------------------
# Logging / saving
# ---------------------------------------------------------------------
def log_progress(msg: str, path: Path) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def flush_results(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)


def append_and_flush(rows: list[dict], row: dict) -> None:
    rows.append(row)
    flush_results(rows)


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def load_dataset_train_test(
    dataset_name: str,
    label_col_idx: int,
    seed: int,
):
    path_full = DATA_DIR / f"{dataset_name}.parquet"
    path_train = DATA_DIR / f"{dataset_name}_train.parquet"
    path_test = DATA_DIR / f"{dataset_name}_test.parquet"

    if path_train.exists() and path_test.exists():
        df_train = pd.read_parquet(path_train)
        df_test = pd.read_parquet(path_test)

        X_train, y_train = dataprep(
            df_train,
            label_col_idx=label_col_idx,
            scale=None,
            global_transform=False,
            drop_missing_y=True,
            verbose=False,
        )
        X_test, y_test = dataprep(
            df_test,
            label_col_idx=label_col_idx,
            scale=None,
            global_transform=False,
            drop_missing_y=True,
            verbose=False,
        )
        return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

    if path_full.exists():
        df = pd.read_parquet(path_full)
        X, y = dataprep(
            df,
            label_col_idx=label_col_idx,
            scale=None,
            global_transform=False,
            drop_missing_y=True,
            verbose=False,
        )
        X = np.asarray(X)
        y = np.asarray(y)

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=HOLDOUT_TEST_SIZE,
            random_state=seed,
        )
        idx_train, idx_test = next(sss.split(X, y))
        return X[idx_train], X[idx_test], y[idx_train], y[idx_test]

    raise FileNotFoundError(
        "Could not find dataset parquet. Expected one of:\n"
        f"  {path_train}\n"
        f"  {path_test}\n"
        f"  {path_full}"
    )


def stratified_subsample_indices(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if not (0 < frac <= 1):
        raise ValueError("frac must be in (0, 1].")

    n = len(y)
    if frac >= 1.0:
        return np.arange(n)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=frac,
        random_state=seed,
    )
    idx, _ = next(sss.split(np.zeros((n, 1)), y))
    return np.sort(idx)


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------
def build_fk(
    n_estimators: int,
    seed: int,
) -> LeafEncoder:
    return LeafEncoder(
        forest=RandomForestClassifier(
            n_estimators=n_estimators,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed,
        ),
        weight_scheme="oob",
    )


def build_legacy(
    n_estimators: int,
    seed: int,
) -> RFGAP:
    return RFGAP(
        prediction_type="classification",
        prox_method="oob",
        matrix_type="dense",
        triangular=False,
        random_state=seed,
        n_estimators=n_estimators,
        bootstrap=True,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------
# OOB count statistics
# ---------------------------------------------------------------------
def get_fk_oob_mask(fk: LeafEncoder) -> np.ndarray:
    if not hasattr(fk, "cache_") or fk.cache_ is None:
        raise ValueError("LeafEncoder cache is missing. Did you call _build_cache()?")
    if not hasattr(fk.cache_, "oob_mask"):
        raise ValueError("LeafEncoder cache has no attribute 'oob_mask'.")
    return np.asarray(fk.cache_.oob_mask, dtype=np.int8)


def get_legacy_oob_mask(legacy, X_train: np.ndarray) -> np.ndarray:
    return np.asarray(legacy.get_oob_indices(X_train), dtype=np.int8)


def summarize_oob_mask(oob_mask: np.ndarray) -> dict[str, float]:
    """
    oob_mask: shape (n, T), entries in {0,1}
    """
    n, T = oob_mask.shape

    s_i = oob_mask.sum(axis=1).astype(float)              # shape (n,)
    s_ij = (oob_mask @ oob_mask.T).astype(float)          # shape (n,n)

    off_diag_mask = ~np.eye(n, dtype=bool)
    positive_mask = (s_ij > 0) & off_diag_mask

    num_positive_pairs = int(np.sum(positive_mask))
    frac_positive_pairs = num_positive_pairs / (n * (n - 1)) if n > 1 else np.nan

    if num_positive_pairs == 0:
        return {
            "oob_n": int(n),
            "oob_T": int(T),
            "oob_mean_Si": float(np.mean(s_i)),
            "oob_std_Si": float(np.std(s_i)),
            "oob_min_Si": float(np.min(s_i)),
            "oob_max_Si": float(np.max(s_i)),
            "oob_num_positive_pairs": 0,
            "oob_frac_positive_pairs": frac_positive_pairs,
            "oob_ratio_mean": np.nan,
            "oob_ratio_median": np.nan,
            "oob_ratio_std": np.nan,
            "oob_ratio_min": np.nan,
            "oob_ratio_max": np.nan,
            "oob_ratio_mean_abs_dev_from_1": np.nan,
            "oob_ratio_median_abs_dev_from_1": np.nan,
        }

    denom = (s_i[:, None] * s_i[None, :]) / T
    ratios = s_ij[positive_mask] / denom[positive_mask]

    return {
        "oob_n": int(n),
        "oob_T": int(T),
        "oob_mean_Si": float(np.mean(s_i)),
        "oob_std_Si": float(np.std(s_i)),
        "oob_min_Si": float(np.min(s_i)),
        "oob_max_Si": float(np.max(s_i)),
        "oob_num_positive_pairs": num_positive_pairs,
        "oob_frac_positive_pairs": frac_positive_pairs,
        "oob_ratio_mean": float(np.mean(ratios)),
        "oob_ratio_median": float(np.median(ratios)),
        "oob_ratio_std": float(np.std(ratios)),
        "oob_ratio_min": float(np.min(ratios)),
        "oob_ratio_max": float(np.max(ratios)),
        "oob_ratio_mean_abs_dev_from_1": float(np.mean(np.abs(ratios - 1.0))),
        "oob_ratio_median_abs_dev_from_1": float(np.median(np.abs(ratios - 1.0))),
    }


def compare_oob_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"OOB mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")

    diff = mask_a != mask_b
    num_diff = int(np.sum(diff))
    total = diff.size
    frac_diff = num_diff / total if total > 0 else np.nan

    row_diff = np.sum(np.abs(mask_a - mask_b), axis=1)

    return {
        "oob_mask_same_shape": True,
        "oob_mask_num_diff_entries": num_diff,
        "oob_mask_frac_diff_entries": frac_diff,
        "oob_mask_mean_row_abs_diff": float(np.mean(row_diff)),
        "oob_mask_max_row_abs_diff": float(np.max(row_diff)),
        "oob_mask_all_equal": bool(num_diff == 0),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", PROGRESS_LOG)
    log_progress(f"Run directory: {RUN_DIR}", PROGRESS_LOG)
    log_progress(f"Dataset: {DATASET_NAME}", PROGRESS_LOG)
    log_progress(f"Seeds: {SEEDS}", PROGRESS_LOG)
    log_progress(f"Train fractions: {TRAIN_FRACTIONS}", PROGRESS_LOG)
    log_progress(f"N estimators grid: {N_ESTIMATORS_GRID}", PROGRESS_LOG)

    for seed in SEEDS:
        log_progress(f"=== SEED {seed} ===", PROGRESS_LOG)

        X_train_full, X_test, y_train_full, y_test = load_dataset_train_test(
            dataset_name=DATASET_NAME,
            label_col_idx=LABEL_COL_IDX,
            seed=seed,
        )

        log_progress(
            f"Loaded data | train_full={X_train_full.shape} | test={X_test.shape}",
            PROGRESS_LOG,
        )

        for train_frac in TRAIN_FRACTIONS:
            sub_idx = stratified_subsample_indices(y_train_full, train_frac, seed)
            X_train = X_train_full[sub_idx]
            y_train = y_train_full[sub_idx]

            log_progress(
                f"Seed={seed} | train fraction={train_frac:.2f} | "
                f"n_train={len(y_train)} | n_test={len(y_test)}",
                PROGRESS_LOG,
            )

            for n_estimators in N_ESTIMATORS_GRID:
                log_progress(
                    f"Seed={seed} | n_estimators={n_estimators}",
                    PROGRESS_LOG,
                )

                # Shared forest through LeafEncoder
                fk = build_fk(n_estimators=n_estimators, seed=seed)

                t0 = time.perf_counter()
                fk._fit_forest(X_train, y_train)
                forest_fit_time_s = time.perf_counter() - t0

                # Build FK OOB cache to expose fk.cache_.oob_mask
                t0 = time.perf_counter()
                fk._build_cache()
                fk_oob_cache_time_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                fk_oob_mask = get_fk_oob_mask(fk)
                fk_oob_extract_time_s = time.perf_counter() - t0

                # Legacy wrapper, reusing the same fitted forest
                legacy = build_legacy(
                    n_estimators=n_estimators,
                    seed=seed,
                )
                legacy.set_forest(fk.forest_.estimator, y=y_train)

                t0 = time.perf_counter()
                legacy_oob_mask = get_legacy_oob_mask(legacy, X_train)
                legacy_oob_extract_time_s = time.perf_counter() - t0

                fk_stats = summarize_oob_mask(fk_oob_mask)
                legacy_stats = summarize_oob_mask(legacy_oob_mask)
                cmp_stats = compare_oob_masks(fk_oob_mask, legacy_oob_mask)

                row = {
                    "run_id": RUN_ID,
                    "dataset": DATASET_NAME,
                    "seed": seed,
                    "train_fraction": train_frac,
                    "n_estimators": n_estimators,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "forest_fit_time_s": forest_fit_time_s,
                    "fk_oob_cache_time_s": fk_oob_cache_time_s,
                    "fk_oob_extract_time_s": fk_oob_extract_time_s,
                    "legacy_oob_extract_time_s": legacy_oob_extract_time_s,

                    # FK proposition quantity
                    "fk_oob_n": fk_stats["oob_n"],
                    "fk_oob_T": fk_stats["oob_T"],
                    "fk_oob_mean_Si": fk_stats["oob_mean_Si"],
                    "fk_oob_std_Si": fk_stats["oob_std_Si"],
                    "fk_oob_min_Si": fk_stats["oob_min_Si"],
                    "fk_oob_max_Si": fk_stats["oob_max_Si"],
                    "fk_oob_num_positive_pairs": fk_stats["oob_num_positive_pairs"],
                    "fk_oob_frac_positive_pairs": fk_stats["oob_frac_positive_pairs"],
                    "fk_oob_ratio_mean": fk_stats["oob_ratio_mean"],
                    "fk_oob_ratio_median": fk_stats["oob_ratio_median"],
                    "fk_oob_ratio_std": fk_stats["oob_ratio_std"],
                    "fk_oob_ratio_min": fk_stats["oob_ratio_min"],
                    "fk_oob_ratio_max": fk_stats["oob_ratio_max"],
                    "fk_oob_ratio_mean_abs_dev_from_1": fk_stats["oob_ratio_mean_abs_dev_from_1"],
                    "fk_oob_ratio_median_abs_dev_from_1": fk_stats["oob_ratio_median_abs_dev_from_1"],

                    # Legacy proposition quantity
                    "legacy_oob_n": legacy_stats["oob_n"],
                    "legacy_oob_T": legacy_stats["oob_T"],
                    "legacy_oob_mean_Si": legacy_stats["oob_mean_Si"],
                    "legacy_oob_std_Si": legacy_stats["oob_std_Si"],
                    "legacy_oob_min_Si": legacy_stats["oob_min_Si"],
                    "legacy_oob_max_Si": legacy_stats["oob_max_Si"],
                    "legacy_oob_num_positive_pairs": legacy_stats["oob_num_positive_pairs"],
                    "legacy_oob_frac_positive_pairs": legacy_stats["oob_frac_positive_pairs"],
                    "legacy_oob_ratio_mean": legacy_stats["oob_ratio_mean"],
                    "legacy_oob_ratio_median": legacy_stats["oob_ratio_median"],
                    "legacy_oob_ratio_std": legacy_stats["oob_ratio_std"],
                    "legacy_oob_ratio_min": legacy_stats["oob_ratio_min"],
                    "legacy_oob_ratio_max": legacy_stats["oob_ratio_max"],
                    "legacy_oob_ratio_mean_abs_dev_from_1": legacy_stats["oob_ratio_mean_abs_dev_from_1"],
                    "legacy_oob_ratio_median_abs_dev_from_1": legacy_stats["oob_ratio_median_abs_dev_from_1"],

                    # Mask comparison
                    "oob_mask_same_shape": cmp_stats["oob_mask_same_shape"],
                    "oob_mask_num_diff_entries": cmp_stats["oob_mask_num_diff_entries"],
                    "oob_mask_frac_diff_entries": cmp_stats["oob_mask_frac_diff_entries"],
                    "oob_mask_mean_row_abs_diff": cmp_stats["oob_mask_mean_row_abs_diff"],
                    "oob_mask_max_row_abs_diff": cmp_stats["oob_mask_max_row_abs_diff"],
                    "oob_mask_all_equal": cmp_stats["oob_mask_all_equal"],

                    "status": "ok",
                    "error": "",
                }

                append_and_flush(rows, row)

                log_progress(
                    "    "
                    f"seed={seed} | "
                    f"T={n_estimators} | "
                    f"fk_ratio_mean={fk_stats['oob_ratio_mean']:.3e} | "
                    f"fk_ratio_mad1={fk_stats['oob_ratio_mean_abs_dev_from_1']:.3e} | "
                    f"legacy_ratio_mean={legacy_stats['oob_ratio_mean']:.3e} | "
                    f"legacy_ratio_mad1={legacy_stats['oob_ratio_mean_abs_dev_from_1']:.3e} | "
                    f"mask_frac_diff={cmp_stats['oob_mask_frac_diff_entries']:.3e} | "
                    f"mask_all_equal={cmp_stats['oob_mask_all_equal']}",
                    PROGRESS_LOG,
                )

    flush_results(rows)
    log_progress(f"Saved CSV: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Saved Parquet: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress("Done.", PROGRESS_LOG)


if __name__ == "__main__":
    main()

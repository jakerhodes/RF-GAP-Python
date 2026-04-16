# prox_equivalence_grid.py
#
# Sweep over:
#   - train fractions: 0.1, 0.2, ..., 1.0
#   - number of trees: 100, 200, ..., 1000
#
# For each setting and each proximity pair:
#   ForestKernel method vs legacy RFGAP method
#
# Record:
#   - train/test sizes
#   - shape / nnz / density
#   - max abs difference
#   - relative Frobenius difference
#   - strict allclose
#   - timings
#
# Results are appended dynamically to CSV + Parquet.

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import dataprep
from forestkernel import ForestKernel
from experiments.baselines.rfgap import RFGAP


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

DATASET_NAME = "sign_mnist"
LABEL_COL_IDX = 0
SEEDS = [44, 578, 9, 912, 345]
ROUND_PRINT = 4

TRAIN_FRACTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# N_ESTIMATORS_GRID = list(range(100, 1001, 100))
N_ESTIMATORS_GRID =[60,70,80,90,100,110,120,130,140,150]


# If a dedicated test parquet is not available, use this fraction
HOLDOUT_TEST_SIZE = 0.2

METHOD_PAIRS = [
    # ("original", "original"),
    ("oob", "oob"),
    # ("gap", "rfgap"),
]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"{RUN_ID}_prox_equivalence_grid"
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RUN_DIR / "prox_equivalence_grid_results.csv"
OUT_PARQUET = RUN_DIR / "prox_equivalence_grid_results.parquet"
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
# Helpers
# ---------------------------------------------------------------------
def to_csr(M):
    return M.tocsr() if sparse.issparse(M) else sparse.csr_matrix(M)


def to_dense(M):
    return M.toarray() if sparse.issparse(M) else np.asarray(M)


def max_abs_diff_sparse(A, B) -> float:
    D = (to_csr(A) - to_csr(B)).tocsr()
    if D.nnz == 0:
        return 0.0
    return float(np.max(np.abs(D.data)))


def rel_frob_diff_sparse(A, B) -> float:
    A = to_csr(A)
    B = to_csr(B)
    denom = sparse.linalg.norm(B)
    if denom == 0:
        return np.nan
    return float(sparse.linalg.norm(A - B) / denom)


def percent_nnz(M) -> float:
    M = to_csr(M)
    total = M.shape[0] * M.shape[1]
    if total == 0:
        return np.nan
    return 100.0 * M.nnz / total


def print_dense_matrix(name, M):
    M_dense = to_dense(M)
    if ROUND_PRINT is not None:
        M_dense = np.round(M_dense, ROUND_PRINT)
    print(f"\n{name} dense matrix:")
    print(M_dense)


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
            scale="standardize",
            global_transform=False,
            drop_missing_y=True,
            verbose=False,
        )
        X_test, y_test = dataprep(
            df_test,
            label_col_idx=label_col_idx,
            scale="standardize",
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
            scale="standardize",
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


def build_fk(
    n_estimators: int,
    seed: int,
) -> ForestKernel:
    return ForestKernel(
        prediction_type="classification",
        kernel_method="original",  # overwritten later in build_kernel_cache
        model_type="rf",
        n_estimators=n_estimators,
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )


def build_legacy(
    prox_method: str,
    n_estimators: int,
    seed: int,
) -> RFGAP:
    return RFGAP(
        prediction_type="classification",
        prox_method=prox_method,
        matrix_type="dense",
        triangular=False,
        random_state=seed,
        n_estimators=n_estimators,
        bootstrap=True,
        n_jobs=-1,
    )


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
    log_progress(f"Method pairs: {METHOD_PAIRS}", PROGRESS_LOG)

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

                fk = build_fk(n_estimators=n_estimators, seed=seed)

                t0 = time.perf_counter()
                fk.fit_forest(X_train, y_train)
                forest_fit_time_s = time.perf_counter() - t0

                for fk_method, legacy_method in METHOD_PAIRS:
                    log_progress(
                        f"  Compare ForestKernel='{fk_method}' vs Legacy='{legacy_method}'",
                        PROGRESS_LOG,
                    )

                    # ForestKernel
                    t0 = time.perf_counter()
                    fk.build_kernel_cache(kernel_method=fk_method)
                    fk_build_time_s = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    K_fk = fk.get_kernel(return_dense=True)
                    fk_get_kernel_time_s = time.perf_counter() - t0
                    K_fk_csr = to_csr(K_fk)

                    # Legacy
                    legacy = build_legacy(
                        prox_method=legacy_method,
                        n_estimators=n_estimators,
                        seed=seed,
                    )
                    legacy.set_forest(fk.forest_, y=y_train)

                    t0 = time.perf_counter()
                    legacy.build_proximity_cache(X_train)
                    legacy_build_time_s = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    K_legacy = legacy.get_proximities()
                    legacy_get_kernel_time_s = time.perf_counter() - t0
                    K_legacy_csr = to_csr(K_legacy)

                    # Compare
                    same_shape = K_fk_csr.shape == K_legacy_csr.shape
                    same_nnz = K_fk_csr.nnz == K_legacy_csr.nnz
                    max_diff = max_abs_diff_sparse(K_fk_csr, K_legacy_csr)
                    rel_frob = rel_frob_diff_sparse(K_fk_csr, K_legacy_csr)

                    strict_close = np.allclose(
                        to_dense(K_fk),
                        to_dense(K_legacy),
                        atol=1e-8,
                        rtol=1e-6,
                    )

                    row = {
                        "run_id": RUN_ID,
                        "dataset": DATASET_NAME,
                        "seed": seed,
                        "train_fraction": train_frac,
                        "n_estimators": n_estimators,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "fk_method": fk_method,
                        "legacy_method": legacy_method,
                        "forest_fit_time_s": forest_fit_time_s,
                        "fk_build_time_s": fk_build_time_s,
                        "fk_get_kernel_time_s": fk_get_kernel_time_s,
                        "legacy_build_time_s": legacy_build_time_s,
                        "legacy_get_kernel_time_s": legacy_get_kernel_time_s,
                        "fk_shape_0": K_fk_csr.shape[0],
                        "fk_shape_1": K_fk_csr.shape[1],
                        "legacy_shape_0": K_legacy_csr.shape[0],
                        "legacy_shape_1": K_legacy_csr.shape[1],
                        "same_shape": same_shape,
                        "fk_nnz": K_fk_csr.nnz,
                        "legacy_nnz": K_legacy_csr.nnz,
                        "same_nnz": same_nnz,
                        "fk_percent_nnz": percent_nnz(K_fk_csr),
                        "legacy_percent_nnz": percent_nnz(K_legacy_csr),
                        "max_abs_diff": max_diff,
                        "rel_frob_diff": rel_frob,
                        "strict_close": strict_close,
                        "status": "ok",
                        "error": "",
                    }

                    append_and_flush(rows, row)

                    log_progress(
                        "    "
                        f"seed={seed} | "
                        f"fk_nnz={K_fk_csr.nnz} | "
                        f"legacy_nnz={K_legacy_csr.nnz} | "
                        f"max_diff={max_diff:.3e} | "
                        f"rel_frob={rel_frob:.3e} | "
                        f"strict_close={strict_close}",
                        PROGRESS_LOG,
                    )

    flush_results(rows)
    log_progress(f"Saved CSV: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Saved Parquet: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress("Done.", PROGRESS_LOG)

if __name__ == "__main__":
    main()
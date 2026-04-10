from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forestkernel import ForestKernel
from experiments.runtime_utils import (
    load_dataset_pair_with_raw_labels,
    log_progress,
    resolve_dataset_paths_from_base_names,
    timed_call,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

DATASET_NAMES = [
    "sign_mnist",
    "covertype",
]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"{RUN_ID}_leaf_embedding_experiments"
RUN_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIR = RUN_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RUN_DIR / "embedding_runtime_results.csv"
OUT_PARQUET = RUN_DIR / "embedding_runtime_results.parquet"
PROGRESS_LOG = RUN_DIR / "embedding_progress.log"

SEEDS = [44, 578, 9, 912, 345]

LABEL_COL_IDX = 0
SCALE = "standardize"
GLOBAL_TRANSFORM = False
DROP_MISSING_Y = True
VERBOSE_DATAPREP = False

# Fixed forest setup for leaf-space methods
KERNEL_METHOD = "gap"
MODEL_TYPE = "rf"
FOREST_KWARGS = {
    "bootstrap": True,
    "n_jobs": -1,
}

# Embedding methods
RUN_RAW_PCA = True
RUN_LEAF_PCA = True
RUN_RAW_SVD30_UMAP = True
RUN_LEAF_SVD30_UMAP = True

SVD_N_COMPONENTS = 30
UMAP_N_COMPONENTS = 2
UMAP_KWARGS = {
    "n_components": UMAP_N_COMPONENTS,
    "random_state": None,  # keep same behavior as your notebook
}

KNN_N_NEIGHBORS = 5


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def flush_results(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)


def append_and_flush(rows: list[dict], row: dict) -> None:
    rows.append(row)
    flush_results(rows)


def instantiate_fk(seed: int) -> ForestKernel:
    kwargs = dict(FOREST_KWARGS)
    kwargs["random_state"] = seed
    return ForestKernel(
        prediction_type="classification",
        kernel_method=KERNEL_METHOD,
        model_type=MODEL_TYPE,
        **kwargs,
    )


def make_dataset_seed_dir(dataset_name: str, seed: int) -> Path:
    out_dir = EMB_DIR / dataset_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_embedding(
    out_path: Path,
    y_raw: np.ndarray,
    emb_2d: np.ndarray,
) -> None:
    df = pd.DataFrame({
        "label": np.asarray(y_raw),
        "x1": emb_2d[:, 0],
        "x2": emb_2d[:, 1],
    })
    df.to_csv(out_path, index=False)


def knn_test_accuracy(
    x_train_2d: np.ndarray,
    x_test_2d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = KNN_N_NEIGHBORS,
) -> float:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train_2d, y_train)
    y_pred = clf.predict(x_test_2d)
    return accuracy_score(y_test, y_pred)


def to_dense_if_needed(X):
    if sparse.issparse(X):
        return X.toarray()
    return X


# ---------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------
def run_raw_pca(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    # Train side
    pca = PCA(n_components=2, random_state=seed)

    x_train_2d, fit_time, fit_mem = timed_call(
        pca.fit_transform,
        to_dense_if_needed(X_train),
    )

    # Test side
    x_test_2d, test_time, test_mem = timed_call(
        pca.transform,
        to_dense_if_needed(X_test),
    )

    acc = knn_test_accuracy(x_train_2d, x_test_2d, y_train, y_test)

    save_embedding(out_dir / "raw_pca_train.csv", y_train_raw, x_train_2d)
    save_embedding(out_dir / "raw_pca_test.csv", y_test_raw, x_test_2d)

    return {
        "method_name": "raw_pca",
        "forest_fit_time_s": np.nan,
        "forest_fit_peak_mb": np.nan,
        "cache_build_time_s": np.nan,
        "cache_build_peak_mb": np.nan,
        "reference_map_time_s": np.nan,
        "reference_map_peak_mb": np.nan,
        "query_map_time_s": np.nan,
        "query_map_peak_mb": np.nan,
        "svd_fit_transform_time_s": np.nan,
        "svd_fit_transform_peak_mb": np.nan,
        "svd_transform_time_s": np.nan,
        "svd_transform_peak_mb": np.nan,
        "pca_fit_transform_time_s": fit_time,
        "pca_fit_transform_peak_mb": fit_mem,
        "pca_transform_time_s": test_time,
        "pca_transform_peak_mb": test_mem,
        "umap_fit_transform_time_s": np.nan,
        "umap_fit_transform_peak_mb": np.nan,
        "umap_transform_time_s": np.nan,
        "umap_transform_peak_mb": np.nan,
        "train_total_time_s": fit_time,
        "train_total_peak_mb": fit_mem,
        "test_total_time_s": test_time,
        "test_total_peak_mb": test_mem,
        "knn_test_acc": acc,
        "train_embedding_file": str(out_dir / "raw_pca_train.csv"),
        "test_embedding_file": str(out_dir / "raw_pca_test.csv"),
        "status": "ok",
        "error": "",
    }


def run_leaf_pca(
    fk: ForestKernel,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    # Shared leaf pipeline
    _, forest_fit_time, forest_fit_mem = timed_call(
        fk.fit_forest,
        X_train,
        y_train,
    )

    _, cache_time, cache_mem = timed_call(
        fk.build_kernel_cache,
        kernel_method=KERNEL_METHOD,
    )

    leaf_train, ref_time, ref_mem = timed_call(fk.get_reference_map)
    leaf_test, query_time, query_mem = timed_call(fk.get_query_map, X_test)

    pca = PCA(n_components=2, random_state=seed)

    x_train_2d, pca_fit_time, pca_fit_mem = timed_call(
        pca.fit_transform,
        to_dense_if_needed(leaf_train),
    )

    x_test_2d, pca_test_time, pca_test_mem = timed_call(
        pca.transform,
        to_dense_if_needed(leaf_test),
    )

    acc = knn_test_accuracy(x_train_2d, x_test_2d, y_train, y_test)

    save_embedding(out_dir / "leaf_pca_train.csv", y_train_raw, x_train_2d)
    save_embedding(out_dir / "leaf_pca_test.csv", y_test_raw, x_test_2d)

    train_total_time = forest_fit_time + cache_time + ref_time + pca_fit_time
    train_total_peak = forest_fit_mem + cache_mem + ref_mem + pca_fit_mem
    test_total_time = query_time + pca_test_time
    test_total_peak = query_mem + pca_test_mem

    return {
        "method_name": "leaf_pca",
        "forest_fit_time_s": forest_fit_time,
        "forest_fit_peak_mb": forest_fit_mem,
        "cache_build_time_s": cache_time,
        "cache_build_peak_mb": cache_mem,
        "reference_map_time_s": ref_time,
        "reference_map_peak_mb": ref_mem,
        "query_map_time_s": query_time,
        "query_map_peak_mb": query_mem,
        "svd_fit_transform_time_s": np.nan,
        "svd_fit_transform_peak_mb": np.nan,
        "svd_transform_time_s": np.nan,
        "svd_transform_peak_mb": np.nan,
        "pca_fit_transform_time_s": pca_fit_time,
        "pca_fit_transform_peak_mb": pca_fit_mem,
        "pca_transform_time_s": pca_test_time,
        "pca_transform_peak_mb": pca_test_mem,
        "umap_fit_transform_time_s": np.nan,
        "umap_fit_transform_peak_mb": np.nan,
        "umap_transform_time_s": np.nan,
        "umap_transform_peak_mb": np.nan,
        "train_total_time_s": train_total_time,
        "train_total_peak_mb": train_total_peak,
        "test_total_time_s": test_total_time,
        "test_total_peak_mb": test_total_peak,
        "knn_test_acc": acc,
        "train_embedding_file": str(out_dir / "leaf_pca_train.csv"),
        "test_embedding_file": str(out_dir / "leaf_pca_test.csv"),
        "status": "ok",
        "error": "",
    }


def run_raw_svd30_umap(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=seed)

    x_svd_train, svd_fit_time, svd_fit_mem = timed_call(
        svd.fit_transform,
        X_train,
    )

    x_svd_test, svd_test_time, svd_test_mem = timed_call(
        svd.transform,
        X_test,
    )

    umap = UMAP(**UMAP_KWARGS)

    x_train_2d, umap_fit_time, umap_fit_mem = timed_call(
        umap.fit_transform,
        x_svd_train,
    )

    x_test_2d, umap_test_time, umap_test_mem = timed_call(
        umap.transform,
        x_svd_test,
    )

    acc = knn_test_accuracy(x_train_2d, x_test_2d, y_train, y_test)

    save_embedding(out_dir / "raw_svd30_umap_train.csv", y_train_raw, x_train_2d)
    save_embedding(out_dir / "raw_svd30_umap_test.csv", y_test_raw, x_test_2d)

    train_total_time = svd_fit_time + umap_fit_time
    train_total_peak = svd_fit_mem + umap_fit_mem
    test_total_time = svd_test_time + umap_test_time
    test_total_peak = svd_test_mem + umap_test_mem

    return {
        "method_name": "raw_svd30_umap",
        "forest_fit_time_s": np.nan,
        "forest_fit_peak_mb": np.nan,
        "cache_build_time_s": np.nan,
        "cache_build_peak_mb": np.nan,
        "reference_map_time_s": np.nan,
        "reference_map_peak_mb": np.nan,
        "query_map_time_s": np.nan,
        "query_map_peak_mb": np.nan,
        "svd_fit_transform_time_s": svd_fit_time,
        "svd_fit_transform_peak_mb": svd_fit_mem,
        "svd_transform_time_s": svd_test_time,
        "svd_transform_peak_mb": svd_test_mem,
        "pca_fit_transform_time_s": np.nan,
        "pca_fit_transform_peak_mb": np.nan,
        "pca_transform_time_s": np.nan,
        "pca_transform_peak_mb": np.nan,
        "umap_fit_transform_time_s": umap_fit_time,
        "umap_fit_transform_peak_mb": umap_fit_mem,
        "umap_transform_time_s": umap_test_time,
        "umap_transform_peak_mb": umap_test_mem,
        "train_total_time_s": train_total_time,
        "train_total_peak_mb": train_total_peak,
        "test_total_time_s": test_total_time,
        "test_total_peak_mb": test_total_peak,
        "knn_test_acc": acc,
        "train_embedding_file": str(out_dir / "raw_svd30_umap_train.csv"),
        "test_embedding_file": str(out_dir / "raw_svd30_umap_test.csv"),
        "status": "ok",
        "error": "",
    }


def run_leaf_svd30_umap(
    fk: ForestKernel,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    _, forest_fit_time, forest_fit_mem = timed_call(
        fk.fit_forest,
        X_train,
        y_train,
    )

    _, cache_time, cache_mem = timed_call(
        fk.build_kernel_cache,
        kernel_method=KERNEL_METHOD,
    )

    leaf_train, ref_time, ref_mem = timed_call(fk.get_reference_map)
    leaf_test, query_time, query_mem = timed_call(fk.get_query_map, X_test)

    svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=seed)

    x_svd_train, svd_fit_time, svd_fit_mem = timed_call(
        svd.fit_transform,
        leaf_train,
    )

    x_svd_test, svd_test_time, svd_test_mem = timed_call(
        svd.transform,
        leaf_test,
    )

    umap = UMAP(**UMAP_KWARGS)

    x_train_2d, umap_fit_time, umap_fit_mem = timed_call(
        umap.fit_transform,
        x_svd_train,
    )

    x_test_2d, umap_test_time, umap_test_mem = timed_call(
        umap.transform,
        x_svd_test,
    )

    acc = knn_test_accuracy(x_train_2d, x_test_2d, y_train, y_test)

    save_embedding(out_dir / "leaf_svd30_umap_train.csv", y_train_raw, x_train_2d)
    save_embedding(out_dir / "leaf_svd30_umap_test.csv", y_test_raw, x_test_2d)

    train_total_time = forest_fit_time + cache_time + ref_time + svd_fit_time + umap_fit_time
    train_total_peak = forest_fit_mem + cache_mem + ref_mem + svd_fit_mem + umap_fit_mem
    test_total_time = query_time + svd_test_time + umap_test_time
    test_total_peak = query_mem + svd_test_mem + umap_test_mem

    return {
        "method_name": "leaf_svd30_umap",
        "forest_fit_time_s": forest_fit_time,
        "forest_fit_peak_mb": forest_fit_mem,
        "cache_build_time_s": cache_time,
        "cache_build_peak_mb": cache_mem,
        "reference_map_time_s": ref_time,
        "reference_map_peak_mb": ref_mem,
        "query_map_time_s": query_time,
        "query_map_peak_mb": query_mem,
        "svd_fit_transform_time_s": svd_fit_time,
        "svd_fit_transform_peak_mb": svd_fit_mem,
        "svd_transform_time_s": svd_test_time,
        "svd_transform_peak_mb": svd_test_mem,
        "pca_fit_transform_time_s": np.nan,
        "pca_fit_transform_peak_mb": np.nan,
        "pca_transform_time_s": np.nan,
        "pca_transform_peak_mb": np.nan,
        "umap_fit_transform_time_s": umap_fit_time,
        "umap_fit_transform_peak_mb": umap_fit_mem,
        "umap_transform_time_s": umap_test_time,
        "umap_transform_peak_mb": umap_test_mem,
        "train_total_time_s": train_total_time,
        "train_total_peak_mb": train_total_peak,
        "test_total_time_s": test_total_time,
        "test_total_peak_mb": test_total_peak,
        "knn_test_acc": acc,
        "train_embedding_file": str(out_dir / "leaf_svd30_umap_train.csv"),
        "test_embedding_file": str(out_dir / "leaf_svd30_umap_test.csv"),
        "status": "ok",
        "error": "",
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dataset_groups = resolve_dataset_paths_from_base_names(DATA_DIR, DATASET_NAMES)
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", PROGRESS_LOG)
    log_progress(f"Run directory: {RUN_DIR}", PROGRESS_LOG)
    log_progress(f"Embedding directory: {EMB_DIR}", PROGRESS_LOG)
    log_progress(f"Datasets: {sorted(dataset_groups.keys())}", PROGRESS_LOG)
    log_progress(f"Seeds: {SEEDS}", PROGRESS_LOG)
    log_progress(f"Kernel method: {KERNEL_METHOD}", PROGRESS_LOG)
    log_progress(f"Model type: {MODEL_TYPE}", PROGRESS_LOG)
    log_progress(f"Forest kwargs: {FOREST_KWARGS}", PROGRESS_LOG)
    log_progress(f"SVD components: {SVD_N_COMPONENTS}", PROGRESS_LOG)
    log_progress(f"UMAP kwargs: {UMAP_KWARGS}", PROGRESS_LOG)
    log_progress(f"kNN neighbors: {KNN_N_NEIGHBORS}", PROGRESS_LOG)

    for dataset_name, dataset_paths in dataset_groups.items():
        log_progress(f"=== DATASET: {dataset_name} ===", PROGRESS_LOG)

        for seed in SEEDS:
            log_progress(f">>> SEED: {seed}", PROGRESS_LOG)

            X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, meta = load_dataset_pair_with_raw_labels(
                dataset_name=dataset_name,
                paths=dataset_paths,
                seed=seed,
                label_col_idx=LABEL_COL_IDX,
                scale=SCALE,
                global_transform=GLOBAL_TRANSFORM,
                drop_missing_y=DROP_MISSING_Y,
                verbose_dataprep=VERBOSE_DATAPREP,
            )

            log_progress(
                f"Loaded {dataset_name}: "
                f"train={X_train.shape}, test={X_test.shape}, predefined_split={meta['predefined_split']}",
                PROGRESS_LOG,
            )

            out_dir = make_dataset_seed_dir(dataset_name, seed)

            if RUN_RAW_PCA:
                log_progress("Method: raw_pca", PROGRESS_LOG)
                result = run_raw_pca(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done raw_pca | dataset={dataset_name} | seed={seed} | "
                    f"acc={result['knn_test_acc']:.4f}",
                    PROGRESS_LOG,
                )

            if RUN_LEAF_PCA:
                log_progress("Method: leaf_pca", PROGRESS_LOG)
                fk = instantiate_fk(seed)
                result = run_leaf_pca(
                    fk=fk,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done leaf_pca | dataset={dataset_name} | seed={seed} | "
                    f"acc={result['knn_test_acc']:.4f}",
                    PROGRESS_LOG,
                )

            if RUN_RAW_SVD30_UMAP:
                log_progress("Method: raw_svd30_umap", PROGRESS_LOG)
                result = run_raw_svd30_umap(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done raw_svd30_umap | dataset={dataset_name} | seed={seed} | "
                    f"acc={result['knn_test_acc']:.4f}",
                    PROGRESS_LOG,
                )

            if RUN_LEAF_SVD30_UMAP:
                log_progress("Method: leaf_svd30_umap", PROGRESS_LOG)
                fk = instantiate_fk(seed)
                result = run_leaf_svd30_umap(
                    fk=fk,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done leaf_svd30_umap | dataset={dataset_name} | seed={seed} | "
                    f"acc={result['knn_test_acc']:.4f}",
                    PROGRESS_LOG,
                )

    flush_results(rows)
    log_progress(f"Saved results to: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Saved results to: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress("Done.", PROGRESS_LOG)


if __name__ == "__main__":
    main()
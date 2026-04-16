from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from baselines import PageRankPHATE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree

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
    # "celegans",
    # "pbmc",
    # "sign_mnist",
    # "fashion_mnist",
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
DROP_MISSING_Y = True
VERBOSE_DATAPREP = False

# Image datasets: global normalize
IMAGE_DATASETS = {
    "pathmnist_28",
    "sign_mnist",
    "tissuemnist_28",
    "fashion_mnist",
}
DEFAULT_SCALE = "standardize"
DEFAULT_GLOBAL_TRANSFORM = False
IMAGE_SCALE = "normalize"
IMAGE_GLOBAL_TRANSFORM = True

# Keep only the first 10 SignMNIST letters
# SignMNIST skips J, so this is the first 10 available letters A..K without J
SIGN_MNIST_ALLOWED_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]

# Fixed forest setup for leaf-space methods
KERNEL_METHOD = "kerf"  # do not set to gap which is asymmetric
MODEL_TYPE = "rf"
FOREST_KWARGS = {
    "bootstrap": True,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------
AVAILABLE_METHODS = [
    "raw_pca",
    "leaf_pca",
    "raw_pca_umap",
    "leaf_pca_umap",
    "raw_pca_phate",
    "leaf_pca_phate",
]

METHODS_TO_RUN = [
    "raw_pca",
    "leaf_pca",
    "raw_pca_umap",
    "leaf_pca_umap",
    "raw_pca_phate",
    "leaf_pca_phate",
]

# ---------------------------------------------------------------------
# Embedding configs
# ---------------------------------------------------------------------
PCA_UMAP_N_COMPONENTS = 50
UMAP_N_COMPONENTS = 2
UMAP_KWARGS = {
    "n_components": UMAP_N_COMPONENTS,
    "random_state": None,
    "n_neighbors": 50,
}


PCA_PHATE_N_COMPONENTS = 50
PHATE_N_COMPONENTS = 2
RAW_PHATE_KWARGS = {
    "n_components": PHATE_N_COMPONENTS,
    "random_state": None,
    "knn": 50,
}

LEAF_PHATE_KWARGS_RAW = {
    "n_components": PHATE_N_COMPONENTS,
    "random_state": None,
    "knn": 50,
}

# Fixed k-NN neighborhoods
KNN_K_VALUES = [5, 10, 20]
LOGREG_MAX_ITER = 5000


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


def validate_methods_to_run(methods_to_run: list[str]) -> None:
    unknown = sorted(set(methods_to_run) - set(AVAILABLE_METHODS))
    if unknown:
        raise ValueError(
            f"Unknown methods in METHODS_TO_RUN: {unknown}. "
            f"Available methods are: {AVAILABLE_METHODS}"
        )


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
    row_ids: np.ndarray,
    y_raw: np.ndarray,
    emb_2d: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "row_id": np.asarray(row_ids),
            "label": np.asarray(y_raw),
            "x1": emb_2d[:, 0],
            "x2": emb_2d[:, 1],
        }
    )
    df.to_csv(out_path, index=False)


def get_dataprep_kwargs(dataset_name: str) -> dict[str, object]:
    if dataset_name in IMAGE_DATASETS:
        return {
            "scale": IMAGE_SCALE,
            "global_transform": IMAGE_GLOBAL_TRANSFORM,
        }
    return {
        "scale": DEFAULT_SCALE,
        "global_transform": DEFAULT_GLOBAL_TRANSFORM,
    }


def crop_sign_mnist(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
):
    mask_train = np.isin(np.asarray(y_train_raw), SIGN_MNIST_ALLOWED_LETTERS)
    mask_test = np.isin(np.asarray(y_test_raw), SIGN_MNIST_ALLOWED_LETTERS)

    X_train = X_train[mask_train]
    X_test = X_test[mask_test]
    y_train = np.asarray(y_train)[mask_train]
    y_test = np.asarray(y_test)[mask_test]
    y_train_raw = np.asarray(y_train_raw)[mask_train]
    y_test_raw = np.asarray(y_test_raw)[mask_test]
    id_train_raw = np.asarray(id_train_raw)[mask_train]
    id_test_raw = np.asarray(id_test_raw)[mask_test]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_raw,
        y_test_raw,
        id_train_raw,
        id_test_raw,
    )


def get_knn_neighborhoods(n_train: int) -> list[int]:
    return [k for k in KNN_K_VALUES if k < n_train]


# def knn_test_accuracy_multi(
#     x_train_2d: np.ndarray,
#     x_test_2d: np.ndarray,
#     y_train: np.ndarray,
#     y_test: np.ndarray,
# ) -> tuple[float, list[int], dict[int, float]]:
#     k_values = get_knn_neighborhoods(len(y_train))
#     scores: dict[int, float] = {}

#     for k in k_values:
#         clf = KNeighborsClassifier(n_neighbors=k)
#         clf.fit(x_train_2d, y_train)
#         y_pred = clf.predict(x_test_2d)
#         scores[k] = float(accuracy_score(y_test, y_pred))

#     avg_score = float(np.mean(list(scores.values()))) if scores else np.nan
#     return avg_score, k_values, scores


def knn_test_accuracy_multi(
    x_train_2d: np.ndarray,
    x_test_2d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, list[int], dict[int, float]]:
    k_values = get_knn_neighborhoods(len(y_train))
    if not k_values:
        return np.nan, [], {}

    x_train_2d = np.asarray(x_train_2d, dtype=np.float32, order="C")
    x_test_2d = np.asarray(x_test_2d, dtype=np.float32, order="C")
    y_train = np.asarray(y_train)

    k_max = max(k_values)

    # Build tree once
    tree = KDTree(x_train_2d, leaf_size=64, metric="euclidean")

    # Query once for max K
    nn_idx = tree.query(x_test_2d, k=k_max, return_distance=False)  # (n_test, k_max)

    scores: dict[int, float] = {}
    for k in k_values:
        idx_k = nn_idx[:, :k]                  # (n_test, k)
        neigh_labels = y_train[idx_k]          # (n_test, k)

        # fast majority vote (works for integer labels)
        # If your y_train are not int-coded, convert once outside this function.
        y_pred = np.empty(len(y_test), dtype=y_train.dtype)
        for i in range(len(y_test)):
            vals, counts = np.unique(neigh_labels[i], return_counts=True)
            y_pred[i] = vals[np.argmax(counts)]

        scores[k] = float(accuracy_score(y_test, y_pred))

    avg_score = float(np.mean(list(scores.values()))) if scores else np.nan
    return avg_score, k_values, scores


def logistic_test_accuracy(
    x_train_2d: np.ndarray,
    x_test_2d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> float:
    clf = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(x_train_2d, y_train)
    y_pred = clf.predict(x_test_2d)
    return float(accuracy_score(y_test, y_pred))


def base_result_dict() -> dict:
    return {
        "forest_fit_time_s": np.nan,
        "cache_build_time_s": np.nan,
        "reference_map_time_s": np.nan,
        "query_map_time_s": np.nan,
        "pca_reducer_fit_transform_time_s": np.nan,
        "pca_reducer_transform_time_s": np.nan,
        "pca_fit_transform_time_s": np.nan,
        "pca_transform_time_s": np.nan,
        "umap_fit_transform_time_s": np.nan,
        "umap_transform_time_s": np.nan,
        "phate_fit_transform_time_s": np.nan,
        "phate_transform_time_s": np.nan,
        "train_total_time_s": np.nan,
        "train_total_peak_mb": np.nan,
        "test_total_time_s": np.nan,
        "test_total_peak_mb": np.nan,
        "knn_test_acc_avg": np.nan,
        "knn_k_values": "",
        "knn_test_acc_by_k": "{}",
        "linear_test_acc": np.nan,
        "train_embedding_file": "",
        "test_embedding_file": "",
        "status": "ok",
        "error": "",
    }


def deduplicate_embedding_input(
    X_emb: np.ndarray,
    y,
    y_raw,
    row_ids,
):
    X_emb = np.ascontiguousarray(X_emb)
    _, unique_idx = np.unique(X_emb, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    return (
        X_emb[unique_idx],
        np.asarray(y)[unique_idx],
        np.asarray(y_raw)[unique_idx],
        np.asarray(row_ids)[unique_idx],
        unique_idx,
    )


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
    id_train_raw,
    id_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    def train_pipeline():
        pca = PCA(n_components=2, random_state=seed)
        t0 = time.perf_counter()
        x_train_2d = pca.fit_transform(X_train)
        pca_fit_time = time.perf_counter() - t0
        return pca, x_train_2d, pca_fit_time

    (pca, x_train_2d, pca_fit_time), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        x_test_2d = pca.transform(X_test)
        pca_test_time = time.perf_counter() - t0
        return x_test_2d, pca_test_time

    (x_test_2d, pca_test_time), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(x_train_2d, x_test_2d, y_train, y_test)
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    train_file = out_dir / "raw_pca_train.csv"
    test_file = out_dir / "raw_pca_test.csv"
    save_embedding(train_file, id_train_raw, y_train_raw, x_train_2d)
    save_embedding(test_file, id_test_raw, y_test_raw, x_test_2d)

    result = base_result_dict()
    result.update(
        {
            "method_name": "raw_pca",
            "pca_fit_transform_time_s": pca_fit_time,
            "pca_transform_time_s": pca_test_time,
            "train_total_time_s": train_total_time,
            "train_total_peak_mb": train_total_peak,
            "test_total_time_s": test_total_time,
            "test_total_peak_mb": test_total_peak,
            "knn_test_acc_avg": knn_acc,
            "knn_k_values": str(knn_k_values),
            "knn_test_acc_by_k": str(knn_scores),
            "linear_test_acc": lin_acc,
            "train_embedding_file": str(train_file),
            "test_embedding_file": str(test_file),
        }
    )
    return result


def run_leaf_pca(
    fk: ForestKernel,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    def train_pipeline():
        t0 = time.perf_counter()
        fk.fit_forest(X_train, y_train)
        forest_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        fk.build_kernel_cache(kernel_method=KERNEL_METHOD)
        cache_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        leaf_train = fk.get_reference_map()
        ref_time = time.perf_counter() - t0

        pca = PCA(n_components=2, random_state=seed)
        t0 = time.perf_counter()
        x_train_2d = pca.fit_transform(leaf_train)
        pca_fit_time = time.perf_counter() - t0

        return pca, x_train_2d, forest_fit_time, cache_time, ref_time, pca_fit_time

    (
        pca,
        x_train_2d,
        forest_fit_time,
        cache_time,
        ref_time,
        pca_fit_time,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        leaf_test = fk.get_query_map(X_test)
        query_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test_2d = pca.transform(leaf_test)
        pca_test_time = time.perf_counter() - t0

        return x_test_2d, query_time, pca_test_time

    (x_test_2d, query_time, pca_test_time), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(x_train_2d, x_test_2d, y_train, y_test)
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    train_file = out_dir / "leaf_pca_train.csv"
    test_file = out_dir / "leaf_pca_test.csv"
    save_embedding(train_file, id_train_raw, y_train_raw, x_train_2d)
    save_embedding(test_file, id_test_raw, y_test_raw, x_test_2d)

    result = base_result_dict()
    result.update(
        {
            "method_name": "leaf_pca",
            "forest_fit_time_s": forest_fit_time,
            "cache_build_time_s": cache_time,
            "reference_map_time_s": ref_time,
            "query_map_time_s": query_time,
            "pca_fit_transform_time_s": pca_fit_time,
            "pca_transform_time_s": pca_test_time,
            "train_total_time_s": train_total_time,
            "train_total_peak_mb": train_total_peak,
            "test_total_time_s": test_total_time,
            "test_total_peak_mb": test_total_peak,
            "knn_test_acc_avg": knn_acc,
            "knn_k_values": str(knn_k_values),
            "knn_test_acc_by_k": str(knn_scores),
            "linear_test_acc": lin_acc,
            "train_embedding_file": str(train_file),
            "test_embedding_file": str(test_file),
        }
    )
    return result


def run_raw_pca_umap(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    def train_pipeline():
        pca_reducer = PCA(n_components=PCA_UMAP_N_COMPONENTS, random_state=seed)

        t0 = time.perf_counter()
        x_pca_train = pca_reducer.fit_transform(X_train)
        pca_reducer_fit_time = time.perf_counter() - t0

        umap = UMAP(**UMAP_KWARGS)

        t0 = time.perf_counter()
        x_train_2d = umap.fit_transform(x_pca_train)
        umap_fit_time = time.perf_counter() - t0

        return pca_reducer, umap, x_train_2d, pca_reducer_fit_time, umap_fit_time

    (
        pca_reducer,
        umap,
        x_train_2d,
        pca_reducer_fit_time,
        umap_fit_time,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        x_pca_test = pca_reducer.transform(X_test)
        pca_reducer_test_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test_2d = umap.transform(x_pca_test)
        umap_test_time = time.perf_counter() - t0

        return x_test_2d, pca_reducer_test_time, umap_test_time

    (x_test_2d, pca_reducer_test_time, umap_test_time), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(x_train_2d, x_test_2d, y_train, y_test)
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    train_file = out_dir / f"raw_pca{PCA_UMAP_N_COMPONENTS}_umap_train.csv"
    test_file = out_dir / f"raw_pca{PCA_UMAP_N_COMPONENTS}_umap_test.csv"
    save_embedding(train_file, id_train_raw, y_train_raw, x_train_2d)
    save_embedding(test_file, id_test_raw, y_test_raw, x_test_2d)

    result = base_result_dict()
    result.update(
        {
            "method_name": f"raw_pca{PCA_UMAP_N_COMPONENTS}_umap",
            "pca_reducer_fit_transform_time_s": pca_reducer_fit_time,
            "pca_reducer_transform_time_s": pca_reducer_test_time,
            "umap_fit_transform_time_s": umap_fit_time,
            "umap_transform_time_s": umap_test_time,
            "train_total_time_s": train_total_time,
            "train_total_peak_mb": train_total_peak,
            "test_total_time_s": test_total_time,
            "test_total_peak_mb": test_total_peak,
            "knn_test_acc_avg": knn_acc,
            "knn_k_values": str(knn_k_values),
            "knn_test_acc_by_k": str(knn_scores),
            "linear_test_acc": lin_acc,
            "train_embedding_file": str(train_file),
            "test_embedding_file": str(test_file),
        }
    )
    return result


def run_leaf_pca_umap(
    fk: ForestKernel,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    def train_pipeline():
        t0 = time.perf_counter()
        fk.fit_forest(X_train, y_train)
        forest_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        fk.build_kernel_cache(kernel_method=KERNEL_METHOD)
        cache_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        leaf_train = fk.get_reference_map()
        ref_time = time.perf_counter() - t0

        pca_reducer = PCA(n_components=PCA_UMAP_N_COMPONENTS, random_state=seed)

        t0 = time.perf_counter()
        x_pca_train = pca_reducer.fit_transform(leaf_train)
        pca_reducer_fit_time = time.perf_counter() - t0

        umap = UMAP(**UMAP_KWARGS)

        t0 = time.perf_counter()
        x_train_2d = umap.fit_transform(x_pca_train)
        umap_fit_time = time.perf_counter() - t0

        return (
            pca_reducer,
            umap,
            x_train_2d,
            forest_fit_time,
            cache_time,
            ref_time,
            pca_reducer_fit_time,
            umap_fit_time,
        )

    (
        pca_reducer,
        umap,
        x_train_2d,
        forest_fit_time,
        cache_time,
        ref_time,
        pca_reducer_fit_time,
        umap_fit_time,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        leaf_test = fk.get_query_map(X_test)
        query_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_pca_test = pca_reducer.transform(leaf_test)
        pca_reducer_test_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test_2d = umap.transform(x_pca_test)
        umap_test_time = time.perf_counter() - t0

        return x_test_2d, query_time, pca_reducer_test_time, umap_test_time

    (
        x_test_2d,
        query_time,
        pca_reducer_test_time,
        umap_test_time,
    ), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(x_train_2d, x_test_2d, y_train, y_test)
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    train_file = out_dir / f"leaf_pca{PCA_UMAP_N_COMPONENTS}_umap_train.csv"
    test_file = out_dir / f"leaf_pca{PCA_UMAP_N_COMPONENTS}_umap_test.csv"
    save_embedding(train_file, id_train_raw, y_train_raw, x_train_2d)
    save_embedding(test_file, id_test_raw, y_test_raw, x_test_2d)

    result = base_result_dict()
    result.update(
        {
            "method_name": f"leaf_pca{PCA_UMAP_N_COMPONENTS}_umap",
            "forest_fit_time_s": forest_fit_time,
            "cache_build_time_s": cache_time,
            "reference_map_time_s": ref_time,
            "query_map_time_s": query_time,
            "pca_reducer_fit_transform_time_s": pca_reducer_fit_time,
            "pca_reducer_transform_time_s": pca_reducer_test_time,
            "umap_fit_transform_time_s": umap_fit_time,
            "umap_transform_time_s": umap_test_time,
            "train_total_time_s": train_total_time,
            "train_total_peak_mb": train_total_peak,
            "test_total_time_s": test_total_time,
            "test_total_peak_mb": test_total_peak,
            "knn_test_acc_avg": knn_acc,
            "knn_k_values": str(knn_k_values),
            "knn_test_acc_by_k": str(knn_scores),
            "linear_test_acc": lin_acc,
            "train_embedding_file": str(train_file),
            "test_embedding_file": str(test_file),
        }
    )
    return result


def run_raw_pca_phate(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    def train_pipeline():
        pca_reducer = PCA(n_components=PCA_PHATE_N_COMPONENTS, random_state=seed)

        t0 = time.perf_counter()
        x_pca_train = pca_reducer.fit_transform(X_train)
        pca_reducer_fit_time = time.perf_counter() - t0

        x_pca_train_unique, y_train_unique, y_train_raw_unique, id_train_raw_unique, _ = (
            deduplicate_embedding_input(
                x_pca_train,
                y_train,
                y_train_raw,
                id_train_raw,
            )
        )

        phate_op = PageRankPHATE(**RAW_PHATE_KWARGS)

        t0 = time.perf_counter()
        x_train_2d = phate_op.fit_transform(x_pca_train_unique)
        phate_fit_time = time.perf_counter() - t0

        return (
            pca_reducer,
            phate_op,
            x_train_2d,
            pca_reducer_fit_time,
            phate_fit_time,
            y_train_unique,
            y_train_raw_unique,
            id_train_raw_unique,
        )

    (
        pca_reducer,
        phate_op,
        x_train_2d,
        pca_reducer_fit_time,
        phate_fit_time,
        y_train_unique,
        y_train_raw_unique,
        id_train_raw_unique,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        x_pca_test = pca_reducer.transform(X_test)
        pca_reducer_test_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test_2d = phate_op.transform(x_pca_test)
        phate_test_time = time.perf_counter() - t0

        return x_test_2d, pca_reducer_test_time, phate_test_time

    (x_test_2d, pca_reducer_test_time, phate_test_time), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(
        x_train_2d, x_test_2d, y_train_unique, y_test
    )
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train_unique, y_test, seed)

    train_file = out_dir / f"raw_pca{PCA_PHATE_N_COMPONENTS}_phate_train.csv"
    test_file = out_dir / f"raw_pca{PCA_PHATE_N_COMPONENTS}_phate_test.csv"
    save_embedding(train_file, id_train_raw_unique, y_train_raw_unique, x_train_2d)
    save_embedding(test_file, id_test_raw, y_test_raw, x_test_2d)

    result = base_result_dict()
    result.update(
        {
            "method_name": f"raw_pca{PCA_PHATE_N_COMPONENTS}_phate",
            "pca_reducer_fit_transform_time_s": pca_reducer_fit_time,
            "pca_reducer_transform_time_s": pca_reducer_test_time,
            "phate_fit_transform_time_s": phate_fit_time,
            "phate_transform_time_s": phate_test_time,
            "train_total_time_s": train_total_time,
            "train_total_peak_mb": train_total_peak,
            "test_total_time_s": test_total_time,
            "test_total_peak_mb": test_total_peak,
            "knn_test_acc_avg": knn_acc,
            "knn_k_values": str(knn_k_values),
            "knn_test_acc_by_k": str(knn_scores),
            "linear_test_acc": lin_acc,
            "train_embedding_file": str(train_file),
            "test_embedding_file": str(test_file),
        }
    )
    return result


def run_leaf_pca_phate(
    fk: ForestKernel,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
    seed: int,
    out_dir: Path,
) -> dict:
    def train_pipeline():
        t0 = time.perf_counter()
        fk.fit_forest(X_train, y_train)
        forest_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        fk.build_kernel_cache(kernel_method=KERNEL_METHOD)
        cache_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        leaf_train = fk.get_reference_map()
        ref_time = time.perf_counter() - t0

        pca_reducer = PCA(n_components=PCA_PHATE_N_COMPONENTS, random_state=seed)

        t0 = time.perf_counter()
        x_pca_train = pca_reducer.fit_transform(leaf_train)
        pca_reducer_fit_time = time.perf_counter() - t0

        x_pca_train_unique, y_train_unique, y_train_raw_unique, id_train_raw_unique, _ = (
            deduplicate_embedding_input(
                x_pca_train,
                y_train,
                y_train_raw,
                id_train_raw,
            )
        )

        phate_op = PageRankPHATE(**LEAF_PHATE_KWARGS_RAW)

        t0 = time.perf_counter()
        x_train_2d = phate_op.fit_transform(x_pca_train_unique)
        phate_fit_time = time.perf_counter() - t0

        return (
            pca_reducer,
            phate_op,
            x_train_2d,
            forest_fit_time,
            cache_time,
            ref_time,
            pca_reducer_fit_time,
            phate_fit_time,
            y_train_unique,
            y_train_raw_unique,
            id_train_raw_unique,
        )

    (
        pca_reducer,
        phate_op,
        x_train_2d,
        forest_fit_time,
        cache_time,
        ref_time,
        pca_reducer_fit_time,
        phate_fit_time,
        y_train_unique,
        y_train_raw_unique,
        id_train_raw_unique,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        leaf_test = fk.get_query_map(X_test)
        query_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_pca_test = pca_reducer.transform(leaf_test)
        pca_reducer_test_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test_2d = phate_op.transform(x_pca_test)
        phate_test_time = time.perf_counter() - t0

        return x_test_2d, query_time, pca_reducer_test_time, phate_test_time

    (
        x_test_2d,
        query_time,
        pca_reducer_test_time,
        phate_test_time,
    ), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(
        x_train_2d, x_test_2d, y_train_unique, y_test
    )
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train_unique, y_test, seed)

    train_file = out_dir / f"leaf_pca{PCA_PHATE_N_COMPONENTS}_phate_train.csv"
    test_file = out_dir / f"leaf_pca{PCA_PHATE_N_COMPONENTS}_phate_test.csv"
    save_embedding(train_file, id_train_raw_unique, y_train_raw_unique, x_train_2d)
    save_embedding(test_file, id_test_raw, y_test_raw, x_test_2d)

    result = base_result_dict()
    result.update(
        {
            "method_name": f"leaf_pca{PCA_PHATE_N_COMPONENTS}_phate",
            "forest_fit_time_s": forest_fit_time,
            "cache_build_time_s": cache_time,
            "reference_map_time_s": ref_time,
            "query_map_time_s": query_time,
            "pca_reducer_fit_transform_time_s": pca_reducer_fit_time,
            "pca_reducer_transform_time_s": pca_reducer_test_time,
            "phate_fit_transform_time_s": phate_fit_time,
            "phate_transform_time_s": phate_test_time,
            "train_total_time_s": train_total_time,
            "train_total_peak_mb": train_total_peak,
            "test_total_time_s": test_total_time,
            "test_total_peak_mb": test_total_peak,
            "knn_test_acc_avg": knn_acc,
            "knn_k_values": str(knn_k_values),
            "knn_test_acc_by_k": str(knn_scores),
            "linear_test_acc": lin_acc,
            "train_embedding_file": str(train_file),
            "test_embedding_file": str(test_file),
        }
    )
    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    validate_methods_to_run(METHODS_TO_RUN)

    dataset_groups = resolve_dataset_paths_from_base_names(DATA_DIR, DATASET_NAMES)
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", PROGRESS_LOG)
    log_progress(f"Run directory: {RUN_DIR}", PROGRESS_LOG)
    log_progress(f"Embedding directory: {EMB_DIR}", PROGRESS_LOG)
    log_progress(f"Datasets: {sorted(dataset_groups.keys())}", PROGRESS_LOG)
    log_progress(f"Seeds: {SEEDS}", PROGRESS_LOG)
    log_progress(f"Methods to run: {METHODS_TO_RUN}", PROGRESS_LOG)
    log_progress(f"Kernel method: {KERNEL_METHOD}", PROGRESS_LOG)
    log_progress(f"Model type: {MODEL_TYPE}", PROGRESS_LOG)
    log_progress(f"Forest kwargs: {FOREST_KWARGS}", PROGRESS_LOG)
    log_progress(f"PCA->UMAP components: {PCA_UMAP_N_COMPONENTS}", PROGRESS_LOG)
    log_progress(f"UMAP kwargs: {UMAP_KWARGS}", PROGRESS_LOG)
    log_progress(f"PCA->PHATE components: {PCA_PHATE_N_COMPONENTS}", PROGRESS_LOG)
    log_progress(f"RAW PHATE kwargs: {RAW_PHATE_KWARGS}", PROGRESS_LOG)
    log_progress(f"LEAF PHATE kwargs: {LEAF_PHATE_KWARGS_RAW}", PROGRESS_LOG)
    log_progress(f"kNN k-values: {KNN_K_VALUES}", PROGRESS_LOG)
    log_progress(f"Image datasets: {sorted(IMAGE_DATASETS)}", PROGRESS_LOG)
    log_progress(f"SignMNIST kept letters: {SIGN_MNIST_ALLOWED_LETTERS}", PROGRESS_LOG)

    def save_result_row(
        result: dict,
        dataset_name: str,
        seed: int,
        meta: dict,
        scale,
        global_transform,
        n_train: int,
        n_test: int,
    ) -> None:
        row = {
            "run_id": RUN_ID,
            "dataset": dataset_name,
            "seed": seed,
            "predefined_split": meta["predefined_split"],
            "scale": scale,
            "global_transform": global_transform,
            "n_train": n_train,
            "n_test": n_test,
            "kernel_method": KERNEL_METHOD,
            "model_type": MODEL_TYPE,
            **result,
        }
        append_and_flush(rows, row)

    for dataset_name, dataset_paths in dataset_groups.items():
        log_progress(f"=== DATASET: {dataset_name} ===", PROGRESS_LOG)

        dataprep_kwargs = get_dataprep_kwargs(dataset_name)
        scale = dataprep_kwargs["scale"]
        global_transform = dataprep_kwargs["global_transform"]

        log_progress(
            f"Dataprep scheme | dataset={dataset_name} | "
            f"scale={scale} | global_transform={global_transform}",
            PROGRESS_LOG,
        )

        for seed in SEEDS:
            log_progress(f">>> SEED: {seed}", PROGRESS_LOG)

            (
                X_train,
                X_test,
                y_train,
                y_test,
                y_train_raw,
                y_test_raw,
                id_train_raw,
                id_test_raw,
                meta,
            ) = load_dataset_pair_with_raw_labels(
                dataset_name=dataset_name,
                paths=dataset_paths,
                seed=seed,
                label_col_idx=LABEL_COL_IDX,
                scale=scale,
                global_transform=global_transform,
                drop_missing_y=DROP_MISSING_Y,
                verbose_dataprep=VERBOSE_DATAPREP,
            )

            if dataset_name == "sign_mnist":
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_train_raw,
                    y_test_raw,
                    id_train_raw,
                    id_test_raw,
                ) = crop_sign_mnist(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_train_raw,
                    y_test_raw,
                    id_train_raw,
                    id_test_raw,
                )

            log_progress(
                f"Loaded {dataset_name}: "
                f"train={X_train.shape}, test={X_test.shape}, "
                f"predefined_split={meta['predefined_split']}",
                PROGRESS_LOG,
            )

            out_dir = make_dataset_seed_dir(dataset_name, seed)

            if "raw_pca" in METHODS_TO_RUN:
                log_progress("Method: raw_pca", PROGRESS_LOG)
                result = run_raw_pca(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                save_result_row(result, dataset_name, seed, meta, scale, global_transform, len(y_train), len(y_test))
                log_progress(
                    f"Done raw_pca | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

            if "leaf_pca" in METHODS_TO_RUN:
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
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                save_result_row(result, dataset_name, seed, meta, scale, global_transform, len(y_train), len(y_test))
                log_progress(
                    f"Done leaf_pca | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

            if "raw_pca_umap" in METHODS_TO_RUN:
                method_name = f"raw_pca{PCA_UMAP_N_COMPONENTS}_umap"
                log_progress(f"Method: {method_name}", PROGRESS_LOG)
                result = run_raw_pca_umap(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                save_result_row(result, dataset_name, seed, meta, scale, global_transform, len(y_train), len(y_test))
                log_progress(
                    f"Done {method_name} | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

            if "leaf_pca_umap" in METHODS_TO_RUN:
                method_name = f"leaf_pca{PCA_UMAP_N_COMPONENTS}_umap"
                log_progress(f"Method: {method_name}", PROGRESS_LOG)
                fk = instantiate_fk(seed)
                result = run_leaf_pca_umap(
                    fk=fk,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                save_result_row(result, dataset_name, seed, meta, scale, global_transform, len(y_train), len(y_test))
                log_progress(
                    f"Done {method_name} | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

            if "raw_pca_phate" in METHODS_TO_RUN:
                method_name = f"raw_pca{PCA_PHATE_N_COMPONENTS}_phate"
                log_progress(f"Method: {method_name}", PROGRESS_LOG)
                result = run_raw_pca_phate(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                save_result_row(result, dataset_name, seed, meta, scale, global_transform, len(y_train), len(y_test))
                log_progress(
                    f"Done {method_name} | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

            if "leaf_pca_phate" in METHODS_TO_RUN:
                method_name = f"leaf_pca{PCA_PHATE_N_COMPONENTS}_phate"
                log_progress(f"Method: {method_name}", PROGRESS_LOG)
                fk = instantiate_fk(seed)
                result = run_leaf_pca_phate(
                    fk=fk,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_raw=y_train_raw,
                    y_test_raw=y_test_raw,
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                save_result_row(result, dataset_name, seed, meta, scale, global_transform, len(y_train), len(y_test))
                log_progress(
                    f"Done {method_name} | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

    flush_results(rows)
    log_progress(f"Saved results to: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Saved results to: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress("Done.", PROGRESS_LOG)


if __name__ == "__main__":
    main()
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
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
DROP_MISSING_Y = True
VERBOSE_DATAPREP = False

# Image datasets: global normalize
IMAGE_DATASETS = {
    "pathmnist_28",
    "sign_mnist",
    "tissuemnist_28",
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

# Embedding methods
RUN_RAW_PCA = True
RUN_LEAF_PCA = True
RUN_RAW_SVD_UMAP = True
RUN_LEAF_SVD_UMAP = True

SVD_N_COMPONENTS = 30
UMAP_N_COMPONENTS = 2
UMAP_KWARGS = {
    "n_components": UMAP_N_COMPONENTS,
    "random_state": None,
}

# Scaled k-NN neighborhoods as fractions of train size
KNN_FRACTIONS = [0.001, 0.002, 0.005, 0.01]
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
    k_values = sorted(
        {
            max(3, min(int(round(n_train * frac)), n_train - 1))
            for frac in KNN_FRACTIONS
        }
    )
    return k_values


def knn_test_accuracy_multi(
    x_train_2d: np.ndarray,
    x_test_2d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, list[int], dict[int, float]]:
    k_values = get_knn_neighborhoods(len(y_train))
    scores: dict[int, float] = {}

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train_2d, y_train)
        y_pred = clf.predict(x_test_2d)
        scores[k] = float(accuracy_score(y_test, y_pred))

    avg_score = float(np.mean(list(scores.values())))
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

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(
        x_train_2d, x_test_2d, y_train, y_test
    )
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    save_embedding(out_dir / "raw_pca_train.csv", id_train_raw, y_train_raw, x_train_2d)
    save_embedding(out_dir / "raw_pca_test.csv", id_test_raw, y_test_raw, x_test_2d)

    return {
        "method_name": "raw_pca",
        "forest_fit_time_s": np.nan,
        "cache_build_time_s": np.nan,
        "reference_map_time_s": np.nan,
        "query_map_time_s": np.nan,
        "svd_fit_transform_time_s": np.nan,
        "svd_transform_time_s": np.nan,
        "pca_fit_transform_time_s": pca_fit_time,
        "pca_transform_time_s": pca_test_time,
        "umap_fit_transform_time_s": np.nan,
        "umap_transform_time_s": np.nan,
        "train_total_time_s": train_total_time,
        "train_total_peak_mb": train_total_peak,
        "test_total_time_s": test_total_time,
        "test_total_peak_mb": test_total_peak,
        "knn_test_acc_avg": knn_acc,
        "knn_k_values": str(knn_k_values),
        "knn_test_acc_by_k": str(knn_scores),
        "linear_test_acc": lin_acc,
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

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(
        x_train_2d, x_test_2d, y_train, y_test
    )
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    save_embedding(out_dir / "leaf_pca_train.csv", id_train_raw, y_train_raw, x_train_2d)
    save_embedding(out_dir / "leaf_pca_test.csv", id_test_raw, y_test_raw, x_test_2d)

    return {
        "method_name": "leaf_pca",
        "forest_fit_time_s": forest_fit_time,
        "cache_build_time_s": cache_time,
        "reference_map_time_s": ref_time,
        "query_map_time_s": query_time,
        "svd_fit_transform_time_s": np.nan,
        "svd_transform_time_s": np.nan,
        "pca_fit_transform_time_s": pca_fit_time,
        "pca_transform_time_s": pca_test_time,
        "umap_fit_transform_time_s": np.nan,
        "umap_transform_time_s": np.nan,
        "train_total_time_s": train_total_time,
        "train_total_peak_mb": train_total_peak,
        "test_total_time_s": test_total_time,
        "test_total_peak_mb": test_total_peak,
        "knn_test_acc_avg": knn_acc,
        "knn_k_values": str(knn_k_values),
        "knn_test_acc_by_k": str(knn_scores),
        "linear_test_acc": lin_acc,
        "train_embedding_file": str(out_dir / "leaf_pca_train.csv"),
        "test_embedding_file": str(out_dir / "leaf_pca_test.csv"),
        "status": "ok",
        "error": "",
    }


def run_raw_svd_umap(
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
        svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=seed)

        t0 = time.perf_counter()
        x_svd_train = svd.fit_transform(X_train)
        svd_fit_time = time.perf_counter() - t0

        umap = UMAP(**UMAP_KWARGS)

        t0 = time.perf_counter()
        x_train_2d = umap.fit_transform(x_svd_train)
        umap_fit_time = time.perf_counter() - t0

        return svd, umap, x_train_2d, svd_fit_time, umap_fit_time

    (
        svd,
        umap,
        x_train_2d,
        svd_fit_time,
        umap_fit_time,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        x_svd_test = svd.transform(X_test)
        svd_test_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test_2d = umap.transform(x_svd_test)
        umap_test_time = time.perf_counter() - t0

        return x_test_2d, svd_test_time, umap_test_time

    (x_test_2d, svd_test_time, umap_test_time), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(
        x_train_2d, x_test_2d, y_train, y_test
    )
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    save_embedding(
        out_dir / f"raw_svd{SVD_N_COMPONENTS}_umap_train.csv",
        id_train_raw,
        y_train_raw,
        x_train_2d,
    )
    save_embedding(
        out_dir / f"raw_svd{SVD_N_COMPONENTS}_umap_test.csv",
        id_test_raw,
        y_test_raw,
        x_test_2d,
    )

    return {
        "method_name": f"raw_svd{SVD_N_COMPONENTS}_umap",
        "forest_fit_time_s": np.nan,
        "cache_build_time_s": np.nan,
        "reference_map_time_s": np.nan,
        "query_map_time_s": np.nan,
        "svd_fit_transform_time_s": svd_fit_time,
        "svd_transform_time_s": svd_test_time,
        "pca_fit_transform_time_s": np.nan,
        "pca_transform_time_s": np.nan,
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
        "train_embedding_file": str(out_dir / f"raw_svd{SVD_N_COMPONENTS}_umap_train.csv"),
        "test_embedding_file": str(out_dir / f"raw_svd{SVD_N_COMPONENTS}_umap_test.csv"),
        "status": "ok",
        "error": "",
    }


def run_leaf_svd_umap(
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

        svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=seed)

        t0 = time.perf_counter()
        x_svd_train = svd.fit_transform(leaf_train)
        svd_fit_time = time.perf_counter() - t0

        umap = UMAP(**UMAP_KWARGS)

        t0 = time.perf_counter()
        x_train_2d = umap.fit_transform(x_svd_train)
        umap_fit_time = time.perf_counter() - t0

        return (
            svd,
            umap,
            x_train_2d,
            forest_fit_time,
            cache_time,
            ref_time,
            svd_fit_time,
            umap_fit_time,
        )

    (
        svd,
        umap,
        x_train_2d,
        forest_fit_time,
        cache_time,
        ref_time,
        svd_fit_time,
        umap_fit_time,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    def test_pipeline():
        t0 = time.perf_counter()
        leaf_test = fk.get_query_map(X_test)
        query_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_svd_test = svd.transform(leaf_test)
        svd_test_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_test_2d = umap.transform(x_svd_test)
        umap_test_time = time.perf_counter() - t0

        return x_test_2d, query_time, svd_test_time, umap_test_time

    (
        x_test_2d,
        query_time,
        svd_test_time,
        umap_test_time,
    ), test_total_time, test_total_peak = timed_call(test_pipeline)

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(
        x_train_2d, x_test_2d, y_train, y_test
    )
    lin_acc = logistic_test_accuracy(x_train_2d, x_test_2d, y_train, y_test, seed)

    save_embedding(
        out_dir / f"leaf_svd{SVD_N_COMPONENTS}_umap_train.csv",
        id_train_raw,
        y_train_raw,
        x_train_2d,
    )
    save_embedding(
        out_dir / f"leaf_svd{SVD_N_COMPONENTS}_umap_test.csv",
        id_test_raw,
        y_test_raw,
        x_test_2d,
    )

    return {
        "method_name": f"leaf_svd{SVD_N_COMPONENTS}_umap",
        "forest_fit_time_s": forest_fit_time,
        "cache_build_time_s": cache_time,
        "reference_map_time_s": ref_time,
        "query_map_time_s": query_time,
        "svd_fit_transform_time_s": svd_fit_time,
        "svd_transform_time_s": svd_test_time,
        "pca_fit_transform_time_s": np.nan,
        "pca_transform_time_s": np.nan,
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
        "train_embedding_file": str(out_dir / f"leaf_svd{SVD_N_COMPONENTS}_umap_train.csv"),
        "test_embedding_file": str(out_dir / f"leaf_svd{SVD_N_COMPONENTS}_umap_test.csv"),
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
    log_progress(f"kNN fractions: {KNN_FRACTIONS}", PROGRESS_LOG)
    log_progress(f"Image datasets: {sorted(IMAGE_DATASETS)}", PROGRESS_LOG)
    log_progress(f"SignMNIST kept letters: {SIGN_MNIST_ALLOWED_LETTERS}", PROGRESS_LOG)

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
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "scale": scale,
                    "global_transform": global_transform,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done raw_pca | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
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
                    id_train_raw=id_train_raw,
                    id_test_raw=id_test_raw,
                    seed=seed,
                    out_dir=out_dir,
                )
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "scale": scale,
                    "global_transform": global_transform,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done leaf_pca | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

            if RUN_RAW_SVD_UMAP:
                log_progress(f"Method: raw_svd{SVD_N_COMPONENTS}_umap", PROGRESS_LOG)
                result = run_raw_svd_umap(
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
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "scale": scale,
                    "global_transform": global_transform,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done raw_svd{SVD_N_COMPONENTS}_umap | dataset={dataset_name} | seed={seed} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f} | "
                    f"k={result['knn_k_values']}",
                    PROGRESS_LOG,
                )

            if RUN_LEAF_SVD_UMAP:
                log_progress(f"Method: leaf_svd{SVD_N_COMPONENTS}_umap", PROGRESS_LOG)
                fk = instantiate_fk(seed)
                result = run_leaf_svd_umap(
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
                row = {
                    "run_id": RUN_ID,
                    "dataset": dataset_name,
                    "seed": seed,
                    "predefined_split": meta["predefined_split"],
                    "scale": scale,
                    "global_transform": global_transform,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "kernel_method": KERNEL_METHOD,
                    "model_type": MODEL_TYPE,
                    **result,
                }
                append_and_flush(rows, row)
                log_progress(
                    f"Done leaf_svd{SVD_N_COMPONENTS}_umap | dataset={dataset_name} | seed={seed} | "
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
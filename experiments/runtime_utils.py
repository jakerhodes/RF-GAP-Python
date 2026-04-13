from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from dataset import dataprep


def resolve_dataset_paths_from_base_names(
    data_dir: Path,
    dataset_names: List[str],
) -> Dict[str, Dict[str, Optional[Path]]]:
    groups: Dict[str, Dict[str, Optional[Path]]] = {}

    for base in dataset_names:
        train_path = data_dir / f"{base}_train.parquet"
        test_path = data_dir / f"{base}_test.parquet"
        single_path = data_dir / f"{base}.parquet"

        groups[base] = {"train": None, "test": None, "single": None}

        if train_path.exists() and test_path.exists():
            groups[base]["train"] = train_path
            groups[base]["test"] = test_path
        elif single_path.exists():
            groups[base]["single"] = single_path
        else:
            print(
                f"Skipping dataset '{base}': neither "
                f"({train_path.name}, {test_path.name}) nor {single_path.name} exists."
            )

    return {
        k: v
        for k, v in groups.items()
        if v["train"] is not None or v["single"] is not None
    }


def load_dataset_pair(
    dataset_name: str,
    paths: Dict[str, Optional[Path]],
    seed: int,
    label_col_idx: int = 0,
    scale: str = "standardize",
    global_transform: bool = False,
    drop_missing_y: bool = True,
    verbose_dataprep: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    meta = {
        "dataset": dataset_name,
        "predefined_split": False,
        "train_path": None,
        "test_path": None,
        "single_path": None,
    }

    # ---------------------------------------------------------
    # Case 1: predefined train/test files
    # Apply dataprep ONCE on the concatenated full dataset,
    # then split back to preserve a consistent transform.
    # ---------------------------------------------------------
    if paths["train"] is not None and paths["test"] is not None:
        meta["predefined_split"] = True
        meta["train_path"] = str(paths["train"])
        meta["test_path"] = str(paths["test"])

        df_train = pd.read_parquet(paths["train"])
        df_test = pd.read_parquet(paths["test"])

        if list(df_train.columns) != list(df_test.columns):
            raise ValueError(
                f"Train/test columns differ for dataset '{dataset_name}'."
            )

        n_train_raw = len(df_train)
        n_test_raw = len(df_test)

        df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        out = dataprep(
            df_full,
            label_col_idx=label_col_idx,
            scale=scale,
            global_transform=global_transform,
            drop_missing_y=drop_missing_y,
            verbose=verbose_dataprep,
        )

        if label_col_idx is not None:
            X_full, y_full = out
            X_full = np.asarray(X_full)
            y_full = np.asarray(y_full).reshape(-1)

            if len(y_full) != len(df_full):
                if drop_missing_y:
                    raise ValueError(
                        f"After dataprep, row count changed for predefined split dataset "
                        f"'{dataset_name}'. This makes splitting back ambiguous. "
                        f"Make sure labels are not missing in train/test files."
                    )
                raise ValueError(
                    f"Unexpected row mismatch after dataprep for dataset '{dataset_name}'."
                )

            X_train = X_full[:n_train_raw]
            X_test = X_full[n_train_raw:n_train_raw + n_test_raw]
            y_train = y_full[:n_train_raw]
            y_test = y_full[n_train_raw:n_train_raw + n_test_raw]

            return X_train, X_test, y_train, y_test, meta

        X_full = np.asarray(out)
        X_train = X_full[:n_train_raw]
        X_test = X_full[n_train_raw:n_train_raw + n_test_raw]

        return X_train, X_test, None, None, meta

    # ---------------------------------------------------------
    # Case 2: single file
    # Apply dataprep once, then split
    # ---------------------------------------------------------
    if paths["single"] is not None:
        meta["single_path"] = str(paths["single"])

        df = pd.read_parquet(paths["single"])

        out = dataprep(
            df,
            label_col_idx=label_col_idx,
            scale=scale,
            global_transform=global_transform,
            drop_missing_y=drop_missing_y,
            verbose=verbose_dataprep,
        )

        if label_col_idx is not None:
            X, y = out
            X = np.asarray(X)
            y = np.asarray(y).reshape(-1)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.1,
                random_state=seed,
                stratify=y,
            )
            return X_train, X_test, y_train, y_test, meta

        X = np.asarray(out)
        X_train, X_test = train_test_split(
            X,
            test_size=0.1,
            random_state=seed,
        )
        return X_train, X_test, None, None, meta

    raise ValueError(f"Dataset '{dataset_name}' has no usable parquet file.")


def load_dataset_pair_with_raw_labels(
    dataset_name: str,
    paths: Dict[str, Optional[Path]],
    seed: int,
    label_col_idx: int = 0,
    scale: str = "standardize",
    global_transform: bool = False,
    drop_missing_y: bool = True,
    verbose_dataprep: bool = False,
):
    meta = {
        "dataset": dataset_name,
        "predefined_split": False,
        "train_path": None,
        "test_path": None,
        "single_path": None,
    }

    if paths["train"] is not None and paths["test"] is not None:
        meta["predefined_split"] = True
        meta["train_path"] = str(paths["train"])
        meta["test_path"] = str(paths["test"])

        df_train = pd.read_parquet(paths["train"]).reset_index(drop=True)
        df_test = pd.read_parquet(paths["test"]).reset_index(drop=True)

        y_train_raw = df_train.iloc[:, label_col_idx].to_numpy()
        y_test_raw = df_test.iloc[:, label_col_idx].to_numpy()

        id_train_raw = np.arange(len(df_train))
        id_test_raw = np.arange(len(df_test))

        if list(df_train.columns) != list(df_test.columns):
            raise ValueError(
                f"Train/test columns differ for dataset '{dataset_name}'."
            )

        n_train = len(df_train)
        n_test = len(df_test)

        df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        X_full, y_full = dataprep(
            df_full,
            label_col_idx=label_col_idx,
            scale=scale,
            global_transform=global_transform,
            drop_missing_y=drop_missing_y,
            verbose=verbose_dataprep,
        )

        X_full = np.asarray(X_full)
        y_full = np.asarray(y_full).reshape(-1)

        X_train = X_full[:n_train]
        X_test = X_full[n_train:n_train + n_test]
        y_train = y_full[:n_train]
        y_test = y_full[n_train:n_train + n_test]

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            y_train_raw,
            y_test_raw,
            id_train_raw,
            id_test_raw,
            meta,
        )

    if paths["single"] is not None:
        meta["single_path"] = str(paths["single"])

        df = pd.read_parquet(paths["single"]).reset_index(drop=True)

        y_raw = df.iloc[:, label_col_idx].to_numpy()
        row_ids = np.arange(len(df))

        X, y = dataprep(
            df,
            label_col_idx=label_col_idx,
            scale=scale,
            global_transform=global_transform,
            drop_missing_y=drop_missing_y,
            verbose=verbose_dataprep,
        )

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        idx = np.arange(len(df))
        idx_train, idx_test = train_test_split(
            idx,
            test_size=0.1,
            random_state=seed,
            stratify=y,
        )

        X_train = X[idx_train]
        X_test = X[idx_test]
        y_train = y[idx_train]
        y_test = y[idx_test]

        y_train_raw = y_raw[idx_train]
        y_test_raw = y_raw[idx_test]
        id_train_raw = row_ids[idx_train]
        id_test_raw = row_ids[idx_test]

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            y_train_raw,
            y_test_raw,
            id_train_raw,
            id_test_raw,
            meta,
        )

    raise ValueError(f"Dataset '{dataset_name}' has no usable parquet file.")

def stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if frac >= 1.0:
        return X, y

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=frac,
        random_state=seed,
    )
    idx, _ = next(splitter.split(X, y))
    return X[idx], y[idx]


def stratified_cap_subset(
    X: np.ndarray,
    y: np.ndarray,
    max_train_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if max_train_size is None or n <= max_train_size:
        return X, y

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_train_size,
        random_state=seed,
    )
    idx, _ = next(splitter.split(X, y))
    return X[idx], y[idx]


def kernel_percent_nnz(K) -> float:
    if K is None:
        return np.nan

    if sparse.issparse(K):
        n_rows, n_cols = K.shape
        total = n_rows * n_cols
        if total == 0:
            return np.nan
        return 100.0 * K.nnz / total

    K = np.asarray(K)
    total = K.size
    if total == 0:
        return np.nan
    return 100.0 * np.count_nonzero(K) / total


class MemoryMonitor:
    def __init__(self, poll_seconds: float = 0.01):
        self.poll_seconds = poll_seconds
        self.process = psutil.Process(os.getpid())
        self.start_rss = None
        self.peak_rss = None
        self._stop = None
        self._thread = None

    def __enter__(self):
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self._stop = False

        def _run():
            while not self._stop:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
                time.sleep(self.poll_seconds)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop = True
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self.peak_rss / (1024 ** 2)


def timed_call(fn, *args, poll_seconds: float = 0.01, **kwargs):
    t0 = time.perf_counter()
    with MemoryMonitor(poll_seconds=poll_seconds) as mm:
        out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt, mm.peak_mb


def safe_timed_call(fn, *args, poll_seconds: float = 0.01, **kwargs):
    try:
        out, dt, mem = timed_call(fn, *args, poll_seconds=poll_seconds, **kwargs)
        return out, dt, mem, "ok", ""
    except Exception as e:
        return None, np.nan, np.nan, "failed", str(e)


def log_progress(message: str, log_path: Path):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
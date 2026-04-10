# quick_check_prox_equivalence.py

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit

from dataset import dataprep
from forestkernel import ForestKernel
from experiments.baselines.rfgap import RFGAP


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATA_PATH = Path("/NOBACKUP/aumona/projects/RF-GAP-Python/data/sign_mnist_train.parquet")
LABEL_COL_IDX = 0
SEED = 42
N_SUB = 1000
N_ESTIMATORS = 100
ROUND_PRINT = 4


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def to_csr(M):
    return M.tocsr() if sparse.issparse(M) else sparse.csr_matrix(M)


def to_dense(M):
    return M.toarray() if sparse.issparse(M) else np.asarray(M)


def max_abs_diff_sparse(A, B):
    D = (to_csr(A) - to_csr(B)).tocsr()
    if D.nnz == 0:
        return 0.0
    return float(np.max(np.abs(D.data)))


def rel_frob_diff_sparse(A, B):
    A = to_csr(A)
    B = to_csr(B)
    denom = sparse.linalg.norm(B)
    if denom == 0:
        return np.nan
    return float(sparse.linalg.norm(A - B) / denom)


def print_dense_matrix(name, M):
    M_dense = to_dense(M)
    if ROUND_PRINT is not None:
        M_dense = np.round(M_dense, ROUND_PRINT)
    print(f"\n{name} dense matrix:")
    print(M_dense)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_parquet(DATA_PATH)
X, y = dataprep(
    df,
    label_col_idx=LABEL_COL_IDX,
    scale="standardize",
    global_transform=False,
    drop_missing_y=True,
    verbose=False,
)

X = np.asarray(X)
y = np.asarray(y)

# Small stratified subset
if N_SUB < len(y):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N_SUB, random_state=SEED)
    idx, _ = next(sss.split(X, y))
    X = X[idx]
    y = y[idx]

print(f"Subset shape: {X.shape}")

# ---------------------------------------------------------
# Fit shared ForestKernel forest once
# ---------------------------------------------------------
fk = ForestKernel(
    prediction_type="classification",
    kernel_method="original",
    model_type="rf",
    n_estimators=N_ESTIMATORS,
    bootstrap=True,
    n_jobs=-1,
    random_state=SEED,
)

fk.fit_forest(X, y)

# ---------------------------------------------------------
# Compare all methods
# ---------------------------------------------------------
method_pairs = [
    ("original", "original"),
    ("oob", "oob"),
    ("gap", "rfgap"),
]

for fk_method, legacy_method in method_pairs:
    print("\n" + "=" * 80)
    print(f"Checking ForestKernel='{fk_method}' vs Legacy='{legacy_method}'")

    # ForestKernel
    fk.build_kernel_cache(kernel_method=fk_method)
    K_fk = fk.get_kernel(return_dense=True)
    K_fk_csr = to_csr(K_fk)

    # Legacy
    legacy = RFGAP(
        prediction_type="classification",
        prox_method=legacy_method,
        matrix_type="dense",
        triangular=False,
        random_state=SEED,
        n_estimators=N_ESTIMATORS,
        bootstrap=True,
        n_jobs=-1,
    )
    legacy.set_forest(fk.forest_, y=y)
    legacy.build_proximity_cache(X)
    K_legacy = legacy.get_proximities()
    K_legacy_csr = to_csr(K_legacy)

    # Print dense matrices
    print_dense_matrix(f"ForestKernel ({fk_method})", K_fk)
    print_dense_matrix(f"Legacy ({legacy_method})", K_legacy)

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

    print("\nComparison summary:")
    print("  ForestKernel shape:", K_fk_csr.shape)
    print("  Legacy shape:", K_legacy_csr.shape)
    print("  Same shape:", same_shape)
    print("  ForestKernel nnz:", K_fk_csr.nnz)
    print("  Legacy nnz:", K_legacy_csr.nnz)
    print("  Same nnz:", same_nnz)
    print("  Max absolute difference:", max_diff)
    print("  Relative Frobenius difference:", rel_frob)
    print("  Strict close:", strict_close)
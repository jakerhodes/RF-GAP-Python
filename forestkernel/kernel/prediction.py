import numpy as np
from scipy import sparse
from .sparse_utils import csr_row_scale_inplace


def row_normalize_kernel_block(K, kernel_method):
    """
    Row-normalize a kernel block.

    GAP and KeRF are already normalized by construction and should not
    be renormalized here.
    """
    if kernel_method in {"gap", "kerf"}:
        return K

    if sparse.issparse(K):
        K = K.tocsr(copy=True)
        row_sums = np.asarray(K.sum(axis=1)).ravel()
        row_sums[row_sums == 0.0] = 1.0
        csr_row_scale_inplace(K, 1.0 / row_sums)
        return K

    K = np.asarray(K, dtype=float)
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return K / row_sums


def kernel_predict_regression(K, y_train):
    """
    Proximity-weighted regression prediction.
    """
    y_train = np.asarray(y_train, dtype=float).ravel()

    preds = K.dot(y_train) if sparse.issparse(K) else K @ y_train

    if sparse.issparse(preds):
        preds = preds.toarray()

    return np.asarray(preds, dtype=float).ravel()


def kernel_predict_classification(K, y_train):
    """
    Predict class labels from a kernel block K between query points and
    labeled reference points.
    """
    y_train = np.asarray(y_train).ravel()
    classes, y_idx = np.unique(y_train, return_inverse=True)

    Y = sparse.csr_matrix(
        (
            np.ones(len(y_idx), dtype=np.float32),
            (np.arange(len(y_idx)), y_idx),
        ),
        shape=(len(y_idx), len(classes)),
        dtype=np.float32,
    )

    scores = K.dot(Y) if sparse.issparse(K) else K @ Y.toarray()

    if sparse.issparse(scores):
        scores = scores.toarray()
    else:
        scores = np.asarray(scores)

    # Make sure we really have shape (n_query, n_classes)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    return classes[np.argmax(scores, axis=1)]
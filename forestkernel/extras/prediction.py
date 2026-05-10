import numpy as np
from scipy import sparse


_NORMALIZED_WEIGHT_SCHEMES = {"gap", "kerf"}


def _should_normalize(weight_scheme):
    """
    Return whether prediction weights should be row-normalized.
    """
    return weight_scheme not in _NORMALIZED_WEIGHT_SCHEMES


def _safe_divide_rows(values, row_sums):
    row_sums = np.asarray(row_sums, dtype=float).ravel()
    row_sums[row_sums == 0.0] = 1.0

    if values.ndim == 1:
        return values / row_sums

    return values / row_sums[:, None]


def _factored_row_sums(Q, W):
    """
    Compute row sums of the kernel block P = Q W^T without materializing P.

    This returns

        P 1 = Q (W^T 1),

    where 1 is a vector of ones over the fitted reference samples.

    For sparse Q and W, the cost is

        Time:   O(nnz(W) + nnz(Q))
        Memory: O(L + n_query)

    where L is the number of leaf coordinates. This avoids the potentially
    much larger cost of forming the full n_query x n_reference kernel block.
    """
    ones = np.ones(W.shape[0], dtype=float)
    return Q.dot(W.T.dot(ones))


def _validate_shapes(W, y_train):
    if W.shape[0] != y_train.shape[0]:
        raise ValueError(
            "W and y_train have incompatible shapes: "
            f"W has {W.shape[0]} rows, but y_train has length {y_train.shape[0]}."
        )


def kernel_predict_reg(Q, W, y_train, weight_scheme):
    """
    Kernel-weighted regression using

        P y = Q (W^T y),

    without materializing P = Q W^T.

    Predictions are row-normalized only for weighting schemes that are not
    already normalized by construction.
    """
    y_train = np.asarray(y_train, dtype=float).ravel()
    _validate_shapes(W, y_train)

    preds = Q.dot(W.T.dot(y_train))
    preds = np.asarray(preds, dtype=float).ravel()

    if _should_normalize(weight_scheme):
        row_sums = _factored_row_sums(Q, W)
        preds = _safe_divide_rows(preds, row_sums)

    return preds


def kernel_predict_proba_cls(Q, W, y_train, weight_scheme):
    """
    Kernel-weighted class probabilities using

        P Y = Q (W^T Y),

    without materializing P = Q W^T.

    Class scores are row-normalized only for weighting schemes that are not
    already normalized by construction.
    """
    y_train = np.asarray(y_train).ravel()
    _validate_shapes(W, y_train)

    classes, y_idx = np.unique(y_train, return_inverse=True)

    Y = sparse.csr_matrix(
        (
            np.ones(len(y_idx), dtype=np.float32),
            (np.arange(len(y_idx)), y_idx),
        ),
        shape=(len(y_idx), len(classes)),
        dtype=np.float32,
    )

    proba = Q.dot(W.T.dot(Y))

    if sparse.issparse(proba):
        proba = proba.toarray()
    else:
        proba = np.asarray(proba)

    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)

    if _should_normalize(weight_scheme):
        row_sums = _factored_row_sums(Q, W)
        proba = _safe_divide_rows(proba, row_sums)

    return proba, classes


def kernel_predict_cls(Q, W, y_train, weight_scheme):
    """
    Kernel-weighted classification using

        P Y = Q (W^T Y),

    without materializing P = Q W^T.
    """
    proba, classes = kernel_predict_proba_cls(
        Q,
        W,
        y_train,
        weight_scheme=weight_scheme,
    )

    return classes[np.argmax(proba, axis=1)]
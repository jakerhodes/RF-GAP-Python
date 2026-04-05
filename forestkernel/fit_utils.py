import numpy as np


def prepare_reference_split(X, y, idx_unlabeled=None):
    """
    Prepare the labeled / unlabeled split for the active reference set.

    If idx_unlabeled is provided, transductive mode is activated:
    - the forest is fit on labeled rows only
    - the cache is built on all rows of X

    If idx_unlabeled is None, all rows are treated as labeled.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    idx_unlabeled : array-like of shape (n_unlabeled,), optional
        Row indices in X that should be treated as unlabeled.

    Returns
    -------
    dict
        Dictionary containing:
        - X
        - y
        - X_train
        - y_train
        - idx_labeled
        - idx_unlabeled
        - has_unlabeled
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]

    if y.shape[0] != n:
        raise ValueError("X and y must have the same number of rows.")

    if idx_unlabeled is None:
        idx_unlabeled = np.empty(0, dtype=np.int64)
    else:
        idx_unlabeled = np.asarray(idx_unlabeled, dtype=np.int64)

        if idx_unlabeled.ndim != 1:
            raise ValueError("idx_unlabeled must be a 1D array of row indices.")
        if np.any(idx_unlabeled < 0) or np.any(idx_unlabeled >= n):
            raise ValueError("idx_unlabeled contains out-of-bounds indices.")
        if len(np.unique(idx_unlabeled)) != len(idx_unlabeled):
            raise ValueError("idx_unlabeled contains duplicate indices.")

    has_unlabeled = len(idx_unlabeled) > 0

    mask_labeled = np.ones(n, dtype=bool)
    mask_labeled[idx_unlabeled] = False
    idx_labeled = np.flatnonzero(mask_labeled)

    if len(idx_labeled) == 0:
        raise ValueError("At least one labeled sample is required to fit the forest.")

    X_train = X[idx_labeled]
    y_train = y[idx_labeled]

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "idx_labeled": idx_labeled,
        "idx_unlabeled": idx_unlabeled,
        "has_unlabeled": has_unlabeled,
    }
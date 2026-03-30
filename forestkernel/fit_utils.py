import numpy as np


def prepare_targets(y, prediction_type):
    """
    Convert y to a numpy array and enforce float dtype for regression targets.
    """
    y = np.asarray(y)

    if prediction_type == "regression" and not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32)

    return y


def detect_unlabeled(y, prediction_type):
    """
    Detect unlabeled samples and return their boolean mask.

    Classification:
    - integer labels: -1 is unlabeled
    - floating labels: NaN or -1 is unlabeled

    Regression:
    - NaN is unlabeled
    """
    if prediction_type == "classification":
        if np.issubdtype(y.dtype, np.floating):
            mask_unlabeled = np.isnan(y) | (y == -1)
        else:
            mask_unlabeled = (y == -1)
    else:
        if not np.issubdtype(y.dtype, np.floating):
            y = y.astype(np.float32)
        mask_unlabeled = np.isnan(y)

    return mask_unlabeled


def subset_fit_kwargs(fit_kwargs, idx_labeled, n_total_samples):
    """
    Restrict sample-aligned fit kwargs to the labeled subset.

    Any array-like kwarg whose first dimension matches n_total_samples is
    treated as sample-aligned and subsetted. Other kwargs are passed through.
    """
    subsetted = {}

    for key, value in fit_kwargs.items():
        if value is None:
            subsetted[key] = value
            continue

        try:
            value_array = np.asarray(value)
        except Exception:
            subsetted[key] = value
            continue

        if value_array.ndim >= 1 and value_array.shape[0] == n_total_samples:
            subsetted[key] = value_array[idx_labeled]
        else:
            subsetted[key] = value

    return subsetted


def split_labeled_data(X, y, fit_kwargs, mask_unlabeled):
    """
    Split X, y, and sample-aligned fit kwargs into labeled-only subsets.
    """
    idx_labeled = np.flatnonzero(~mask_unlabeled)
    idx_unlabeled = np.flatnonzero(mask_unlabeled)

    has_unlabeled = bool(np.any(mask_unlabeled))

    if not has_unlabeled:
        X_train = X
        y_train = y
        fit_kwargs_train = dict(fit_kwargs)
    else:
        X_train = X[idx_labeled]
        y_train = y[idx_labeled]
        fit_kwargs_train = subset_fit_kwargs(
            fit_kwargs=fit_kwargs,
            idx_labeled=idx_labeled,
            n_total_samples=X.shape[0],
        )

    return X_train, y_train, fit_kwargs_train, idx_labeled, idx_unlabeled, has_unlabeled


def prepare_fit_inputs(
    X,
    y,
    fit_kwargs,
    prediction_type,
):
    """
    Prepare targets, detect unlabeled samples, and build the labeled fitting subset used by the underlying
    ensemble.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like
        Target values.

    fit_kwargs : dict
        Keyword arguments intended for the underlying estimator fit() call.

    prediction_type : str
        One of {'classification', 'regression'}.

    Returns
    -------
    dict
        Dictionary containing:
        - X
        - y
        - X_train
        - y_train
        - fit_kwargs_train
        - idx_labeled
        - idx_unlabeled
        - has_unlabeled
    """
    y = prepare_targets(y, prediction_type)
    mask_unlabeled = detect_unlabeled(y, prediction_type)
    has_unlabeled = bool(np.any(mask_unlabeled))
    (
        X_train,
        y_train,
        fit_kwargs_train,
        idx_labeled,
        idx_unlabeled,
        has_unlabeled,
    ) = split_labeled_data(
        X=X,
        y=y,
        fit_kwargs=fit_kwargs,
        mask_unlabeled=mask_unlabeled,
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "fit_kwargs_train": fit_kwargs_train,
        "idx_labeled": idx_labeled,
        "idx_unlabeled": idx_unlabeled,
        "has_unlabeled": has_unlabeled,
    }
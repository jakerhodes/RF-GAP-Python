import pandas as pd
import numpy as np

def dataprep(
    data,
    label_col_idx=0,
    scale='standardize',
    drop_missing_y=False,
    verbose=True,
):
    """
    Prepare a dataset for ML models.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    label_col_idx : int or None, default=0
        Index of the label column. Set to None if there is no label column.
    scale : {'normalize', 'standardize', None}, default='standardize'
        Column scaling to apply to features.
    drop_missing_y : bool, default=False
        If True, remove rows whose label is missing before encoding labels.
    verbose : bool, default=True
        If True, print preprocessing information.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,), optional
        Encoded labels, returned only if label_col_idx is not None.
    """
    df = data.copy()

    if verbose:
        print(f"Initial shape: {df.shape}")

    # ---------------------------------------------------------
    # 1. Extract label by index
    # ---------------------------------------------------------
    if label_col_idx is not None:
        label_name = df.columns[label_col_idx]

        if drop_missing_y:
            missing_mask = df[label_name].isna()
            n_missing = int(missing_mask.sum())

            if n_missing > 0:
                df = df.loc[~missing_mask].copy()

            if verbose:
                print(
                    f"Missing labels handling: removed {n_missing} row(s) "
                    f"with missing values in '{label_name}'."
                )
        else:
            n_missing = int(df[label_name].isna().sum())
            if verbose:
                print(
                    f"Missing labels handling: drop_missing_y=False, "
                    f"kept {n_missing} row(s) with missing values in '{label_name}'."
                )

        y_raw = df.pop(label_name)
        y, _ = pd.factorize(y_raw)

        if verbose:
            print(f"Label column: '{label_name}'")
            print(f"Number of classes after factorization: {len(np.unique(y))}")
    else:
        y = None
        if verbose:
            print("No label column used.")

    # ---------------------------------------------------------
    # 2. Handle categorical feature columns
    # ---------------------------------------------------------
    obj_cols = list(df.select_dtypes(include=['object']).columns)
    if len(obj_cols) > 0:
        for col in obj_cols:
            df[col], _ = pd.factorize(df[col])
        if verbose:
            print(f"Factorized {len(obj_cols)} object column(s): {obj_cols}")
    else:
        if verbose:
            print("No object columns to factorize.")

    # ---------------------------------------------------------
    # 3. Scaling
    # ---------------------------------------------------------
    X = df.astype('float32')

    if scale == 'standardize':
        X = (X - X.mean()) / X.std().replace(0, 1)
        if verbose:
            print("Scaling: standardized columns using Z-score.")
    elif scale == 'normalize':
        X = (X - X.min()) / (X.max() - X.min()).replace(0, 1)
        if verbose:
            print("Scaling: normalized columns to [0, 1].")
    elif scale is None:
        if verbose:
            print("Scaling: none.")
    else:
        raise ValueError("scale must be one of {'normalize', 'standardize', None}.")

    if verbose:
        print(f"Final feature shape: {X.shape}")

    # ---------------------------------------------------------
    # 4. Return NumPy arrays
    # ---------------------------------------------------------
    if y is not None:
        return X.to_numpy(), y
    else:
        return X.to_numpy()
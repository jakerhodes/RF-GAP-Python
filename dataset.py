import pandas as pd
import numpy as np

def dataprep(
    data,
    label_col_idx=0,
    scale='standardize',
    global_transform=False,
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
        Scaling to apply to features.
    global_transform : bool, default=False
        If True, apply normalization/standardization globally over all feature
        values. If False, apply it column-wise.
    drop_missing_y : bool, default=False
        If True, remove rows whose label is missing before processing labels.
    verbose : bool, default=True
        If True, print preprocessing information.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,), optional
        Label vector, returned only if label_col_idx is not None.
        Categorical/string labels are factorized into integer codes.
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

        if (
            pd.api.types.is_object_dtype(y_raw)
            or pd.api.types.is_string_dtype(y_raw)
            or pd.api.types.is_categorical_dtype(y_raw)
        ):
            y, uniques = pd.factorize(y_raw.astype(str))
            y = y.astype(np.int64)
            if verbose:
                print(f"Label column: '{label_name}'")
                print("Label handling: factorized categorical/string labels into integer codes.")
                print(f"Number of classes: {len(uniques)}")
        else:
            y = y_raw.to_numpy()
            if verbose:
                print(f"Label column: '{label_name}'")
                print("Label handling: kept numeric labels as numeric.")
                print(f"Number of classes: {len(pd.unique(y))}")
    else:
        y = None
        if verbose:
            print("No label column used.")

    # ---------------------------------------------------------
    # 2. Handle categorical feature columns
    # ---------------------------------------------------------
    obj_cols = list(df.select_dtypes(include=['object', 'string', 'category']).columns)
    if len(obj_cols) > 0:
        for col in obj_cols:
            df[col], _ = pd.factorize(df[col])
        if verbose:
            print(f"Factorized {len(obj_cols)} categorical feature column(s): {obj_cols}")
    else:
        if verbose:
            print("No categorical feature columns to factorize.")

    # ---------------------------------------------------------
    # 3. Scaling
    # ---------------------------------------------------------
    X = df.astype('float32')

    if scale == 'standardize':
        if global_transform:
            x_np = X.to_numpy()
            mean_val = x_np.mean()
            std_val = x_np.std()
            if std_val == 0:
                std_val = 1.0
            X = (X - mean_val) / std_val
            if verbose:
                print("Scaling: globally standardized all feature values using Z-score.")
        else:
            X = (X - X.mean()) / X.std().replace(0, 1)
            if verbose:
                print("Scaling: standardized columns using Z-score.")

    elif scale == 'normalize':
        if global_transform:
            x_np = X.to_numpy()
            min_val = x_np.min()
            max_val = x_np.max()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            X = (X - min_val) / range_val
            if verbose:
                print("Scaling: globally normalized all feature values to [0, 1].")
        else:
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
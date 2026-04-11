import pandas as pd
import numpy as np


def dataprep(
    data,
    label_col_idx=0,
    scale="standardize",
    global_transform=False,
    drop_missing_y=False,
    verbose=True,
):
    """
    Prepare a dataset for ML models.

    Behavior
    --------
    - Extracts the label column if provided.
    - Drops rows with missing labels if requested.
    - Factorizes categorical/string labels into integer codes.
    - One-hot encodes categorical feature columns.
    - Applies scaling only to continuous numeric feature columns.
    - Leaves binary / dummy columns unchanged.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    label_col_idx : int or None, default=0
        Index of the label column. Set to None if there is no label column.
    scale : {'normalize', 'standardize', None}, default='standardize'
        Scaling to apply to continuous numeric features only.
    global_transform : bool, default=False
        If True, apply scaling globally over all continuous feature values.
        If False, apply it column-wise.
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
    # 2. Split feature columns by type
    # ---------------------------------------------------------
    categorical_cols = list(
        df.select_dtypes(include=["object", "string", "category"]).columns
    )

    numeric_cols = list(df.select_dtypes(include=[np.number, "bool"]).columns)

    # Among numeric columns, identify binary columns and continuous columns.
    # Binary columns are left unchanged. Continuous columns may be scaled.
    binary_numeric_cols = []
    continuous_numeric_cols = []

    for col in numeric_cols:
        non_na = df[col].dropna()
        unique_vals = pd.unique(non_na)

        if len(unique_vals) <= 2:
            binary_numeric_cols.append(col)
        else:
            continuous_numeric_cols.append(col)

    if verbose:
        print(f"Categorical feature columns: {categorical_cols if categorical_cols else 'none'}")
        print(f"Continuous numeric feature columns: {continuous_numeric_cols if continuous_numeric_cols else 'none'}")
        print(f"Binary / indicator feature columns: {binary_numeric_cols if binary_numeric_cols else 'none'}")

    # ---------------------------------------------------------
    # 3. One-hot encode categorical features
    # ---------------------------------------------------------
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
        if verbose:
            print(
                f"Feature handling: one-hot encoded {len(categorical_cols)} "
                f"categorical feature column(s)."
            )
    else:
        if verbose:
            print("Feature handling: no categorical feature columns to one-hot encode.")

    # Recompute continuous columns after one-hot encoding
    # Keep only original continuous numeric columns that still exist
    continuous_numeric_cols = [c for c in continuous_numeric_cols if c in df.columns]

    # ---------------------------------------------------------
    # 4. Cast to float32
    # ---------------------------------------------------------
    df = df.astype(np.float32)

    # ---------------------------------------------------------
    # 5. Scale only continuous numeric columns
    # ---------------------------------------------------------
    if scale == "standardize":
        if len(continuous_numeric_cols) > 0:
            if global_transform:
                vals = df[continuous_numeric_cols].to_numpy()
                mean_val = vals.mean()
                std_val = vals.std()
                if std_val == 0:
                    std_val = 1.0
                df.loc[:, continuous_numeric_cols] = (
                    df[continuous_numeric_cols] - mean_val
                ) / std_val
                if verbose:
                    print(
                        "Scaling: globally standardized continuous numeric "
                        "feature values using Z-score."
                    )
            else:
                means = df[continuous_numeric_cols].mean()
                stds = df[continuous_numeric_cols].std().replace(0, 1)
                df.loc[:, continuous_numeric_cols] = (
                    df[continuous_numeric_cols] - means
                ) / stds
                if verbose:
                    print(
                        "Scaling: standardized continuous numeric feature "
                        "columns using Z-score."
                    )
        else:
            if verbose:
                print("Scaling: skipped standardization because no continuous numeric features were found.")

    elif scale == "normalize":
        if len(continuous_numeric_cols) > 0:
            if global_transform:
                vals = df[continuous_numeric_cols].to_numpy()
                min_val = vals.min()
                max_val = vals.max()
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1.0
                df.loc[:, continuous_numeric_cols] = (
                    df[continuous_numeric_cols] - min_val
                ) / range_val
                if verbose:
                    print(
                        "Scaling: globally normalized continuous numeric "
                        "feature values to [0, 1]."
                    )
            else:
                mins = df[continuous_numeric_cols].min()
                ranges = (df[continuous_numeric_cols].max() - mins).replace(0, 1)
                df.loc[:, continuous_numeric_cols] = (
                    df[continuous_numeric_cols] - mins
                ) / ranges
                if verbose:
                    print(
                        "Scaling: normalized continuous numeric feature "
                        "columns to [0, 1]."
                    )
        else:
            if verbose:
                print("Scaling: skipped normalization because no continuous numeric features were found.")

    elif scale is None:
        if verbose:
            print("Scaling: none.")

    else:
        raise ValueError("scale must be one of {'normalize', 'standardize', None}.")

    if verbose:
        print(f"Final feature shape: {df.shape}")

    # ---------------------------------------------------------
    # 6. Return NumPy arrays
    # ---------------------------------------------------------
    X = df.to_numpy(dtype=np.float32)

    if y is not None:
        return X, y
    return X
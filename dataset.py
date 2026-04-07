import pandas as pd
import numpy as np

def dataprep(data, label_col_idx=0, scale='standardize'):
    """
    Prepares dataset for ML models.
    label_col_idx: Index of the label column (default 0). Set to None if no label.
    scale: 'normalize' (0-1), 'standardize' (Z-score), or None.
    """
    df = data.copy()

    # 1. Extract Label by Index
    if label_col_idx is not None:
        # Get the name of the column at that index to pop it
        label_name = df.columns[label_col_idx]
        y_raw = df.pop(label_name)
        
        # Encode labels to integers (A, B, C -> 0, 1, 2)
        # We use pd.Factorize for a clean numerical mapping
        y, _ = pd.factorize(y_raw)
    else:
        y = None

    # 2. Handle Categorical Features (Non-Pixel columns like 'channel')
    # We only encode columns that are 'object' (strings)
    # We AVOID encoding int64/uint8 because those are your pixels!
    for col in df.select_dtypes(include=['object']).columns:
        df[col], _ = pd.factorize(df[col])

    # 3. Scaling (Vectorized Operations)
    X = df.astype('float32') # Convert to float for math operations

    if scale == 'standardize':
        # (X - mean) / std
        X = (X - X.mean()) / X.std().replace(0, 1)

    elif scale == 'normalize':
        # (X - min) / (max - min)
        X = (X - X.min()) / (X.max() - X.min()).replace(0, 1)

    # 4. Return as NumPy arrays
    if y is not None:
        return X.to_numpy(), y
    else:
        return X.to_numpy()
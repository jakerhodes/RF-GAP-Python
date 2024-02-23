import pandas as pd
import numpy as np


# TODO: Rewrite this as a pipeline, either sklearn or pandas
def dataprep(data, label_col_idx = 0, scale = 'normalize'):

    data = data.copy()
    categorical_cols = []
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'int64' or data[col].dtype == 'category':
            categorical_cols.append(col)
            data[col] = pd.Categorical(data[col]).codes


    if label_col_idx is not None:
        label = data.columns[label_col_idx]
        y     = data.pop(label)
        x     = data

    else:
        x = data

    # Need to check dtyps here
    if scale == 'standardize':
        for col in x.columns:
            # if col not in categorical_cols:
            if data[col].std() !=0:
                data[col] = (data[col] - data[col].mean()) / data[col].std()

    elif scale == 'normalize':
        for col in x.columns:
            # if col not in categorical_cols:
            if data[col].max() != data[col].min():
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())


    # Open for return type?
    if label_col_idx is None:
        # return np.array(x)
        return x
    else:
        return x, y
        # return np.array(x), y


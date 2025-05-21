from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from rfgap import RFGAP
from missingmetrics import missing_metrics
import warnings

# TODO: Return RF from best model?
#----------------------------------------------------------------------------------#
#<><><><><><><><><><><><><><><><><> Mean Impute <><><><><><><><><><><><><><><><><>#
#----------------------------------------------------------------------------------#

def stat_impute(df, statistic = 'mean', y = None):

    """
    Perform mean imputation on continuous variables and mode imputation on categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing missing values to be imputed.
    y : pd.Series or None, optional
        A categorical feature to group the data by before imputation. If None, 
        imputation is performed on the entire DataFrame. Default is None.

    Returns
    -------
    pd.DataFrame
        The DataFrame with missing values imputed. Continuous variables are 
        filled with the mean of the respective column, while categorical variables 
        are filled with the mode.
    """

    continuous_features = df.select_dtypes(include='float').columns
    other_features = df.select_dtypes(exclude='float').columns
    df_imp = df.copy()

    if y is None or y.dtype == 'float': # Maybe a better check for continuous variables?

        if len(continuous_features) > 0:
            if statistic == 'mean':
                df_imp[continuous_features] = df_imp[continuous_features].transform(lambda x: x.fillna(x.mean()))

            elif statistic == 'median':
                df_imp[continuous_features] = df_imp[continuous_features].transform(lambda x: x.fillna(x.median()))

        if len(other_features) > 0:
            df_imp[other_features] = df_imp[other_features].transform(lambda x: x.fillna(x.mode()[0]))

    else:
        if len(continuous_features) > 0:
            if statistic == 'mean':
                df_imp[continuous_features] = df_imp.groupby(y)[continuous_features].transform(lambda x: x.fillna(x.mean()))
            elif statistic == 'median':
                df_imp[continuous_features] = df_imp.groupby(y)[continuous_features].transform(lambda x: x.fillna(x.median()))
            
        if len(other_features) > 0:
            df_imp[other_features] = df_imp.groupby(y)[other_features].transform(lambda x: x.fillna(x.mode()[0]))

    return df_imp


#--------------------------------------------------------------------------------#
#<><><><><><><><><><><><><><><><><> KNN Impute <><><><><><><><><><><><><><><><><>#
#--------------------------------------------------------------------------------#


def knn_impute(df, y = None, k = 5):

    """
    Perform K-Nearest Neighbors imputation on a DataFrame to fill missing values.

    Parameters:
    -----------
    df (pd.DataFrame):
        The input DataFrame containing missing values to be imputed.

    y (pd.Series or None, optional):
        A categorical feature to group the data by before imputation. 
        If None, imputation is performed on the entire DataFrame. Default is None.

    k (int, optional):
        The number of nearest neighbors to use for imputation. Default is 5.

    Returns:
    --------
    pd.DataFrame:
        The DataFrame with missing values imputed. The imputed values are filled 
        using KNN, with separate imputation for each group in 'y' if provided.
    """


    continuous_features = df.select_dtypes(include='float').columns
    other_features = df.select_dtypes(exclude='float').columns

    imputer = KNNImputer(n_neighbors = k)

    if y is None or y.dtype == 'float':
        imputed = imputer.fit_transform(df)
        imputed = pd.DataFrame(imputed, index = df.index, columns = df.columns)


    else:
        # Apply imputation separately for each group in y
        groups = df.groupby(y)
        imputed_list = []
        
        for group_name, group_data in groups:
            imputed_group = imputer.fit_transform(group_data)
            imputed_df = pd.DataFrame(imputed_group, index=group_data.index, columns=group_data.columns)
            imputed_list.append(imputed_df)
        
        # Concatenate all imputed groups and preserve the original index
        imputed = pd.concat(imputed_list).reindex(df.index)

    # For categorical variables
    imputed[other_features] = imputed[other_features].round().astype(int)

    # return pd.DataFrame(imputed, columns = df.columns)
    return imputed


#---------------------------------------------------------------------------------#
#<><><><><><><><><><><><><><><><><> RF-GAP Iter <><><><><><><><><><><><><><><><><>#
#---------------------------------------------------------------------------------#

def proximity_impute_iteration(x, missing, proximities, return_nonmissing = False, x_test = None, missing_test = None, proximities_test = None):
    """
    Perform a single iteration of imputation using proximity scores, 
    filtering by the missing mask and re-normalizing proximities.

    Parameters:
    -----------
    x (pd.DataFrame):
        The input DataFrame with missing values that need to be imputed.

    missing (pd.DataFrame or pd.Series):
        A DataFrame or Series indicating missing values (True for missing, False for non-missing).

    proximities (np.ndarray):
        A matrix of proximity scores between rows, used to determine the 'closeness' of data points.

    return_nonmissing (bool, optional):
        Whether to return both the imputed values for missing data and the original values for non-missing data. 
        Default is False, which only returns the imputed DataFrame with missing values filled.

    Returns:
    --------
    pd.DataFrame:
        The DataFrame with imputed values where the original missing values were filled based on proximity.
        If `return_nonmissing` is True, it returns a tuple with two DataFrames:
        - One with the imputed values for missing data only.
        - One with imputed data for both missing and non-missing values (used for internal check for RF-GAP impute)
    """
    
    # Separate continuous and categorical features
    continuous_features = x.select_dtypes(include='float').columns
    other_features = x.select_dtypes(exclude='float').columns

    # Create a copy of the input DataFrame to hold the imputed values
    x_imputed = x.copy()

    if x_test is not None:
        x_test_imputed = x_test.copy()


    # Impute continuous variables
    if len(continuous_features) > 0:
        for feature in continuous_features:
            continuous_values = x[feature].values

            # Apply the missing mask to proximities (set proximities for non-missing values to 0)
            nonmissing_proximities = proximities * (~missing[feature]).astype(int).values
            normalized_proximities = normalize_rows(nonmissing_proximities)

            # Perform imputation using the normalized proximities and continuous values
            x_imputed[feature] = np.dot(normalized_proximities, continuous_values)

            if x_test is not None:
                mask = (~missing[feature]).astype(int).values

                # Apply the mask (broadcasted across rows)
                nonmissing_test_proximities = proximities_test * mask
                normalized_test_proximities = normalize_rows(nonmissing_test_proximities)

                # Compute the imputed values
                x_test_imputed[feature] = np.dot(normalized_test_proximities, continuous_values)




    # Impute categorical variables
    if len(other_features) > 0:
        for feature in other_features:
            # One-hot encode the categorical feature
            other_features_encode = pd.get_dummies(x_imputed[feature])

            # Apply the missing mask to proximities
            nonmissing_proximities = proximities * (~missing[feature]).astype(int).values
            normalized_proximities = normalize_rows(nonmissing_proximities)

            # Perform imputation by choosing the category with the highest proximity-weighted sum
            feature_imputed = np.argmax(np.dot(normalized_proximities, other_features_encode), axis = 1)
            x_imputed[feature] = feature_imputed

            if x_test is not None:
                other_features_encode_test = pd.get_dummies(x_test_imputed[feature])
                nonmissing_test_proximities = proximities_test * (~missing[feature]).astype(int).values
                normalized_test_proximities = normalize_rows(nonmissing_test_proximities)
                feature_imputed_test = np.argmax(np.dot(normalized_test_proximities, other_features_encode), axis = 1)
                x_test_imputed[feature] = feature_imputed_test



    # Return the result
    if return_nonmissing:

        if x_test is not None:
            return x.mask(missing, x_imputed), x.mask(~missing, x_imputed), x_test.mask(missing_test, x_test_imputed) # Does the second return need any mask? Probably not, but doesn't hurt current purpose.

        # Return a tuple with imputed values for missing and non-missing data
        return x.mask(missing, x_imputed), x.mask(~missing, x_imputed)


    
    # Otherwise, return the DataFrame with imputed missing values
    if x_test is not None:
        return x.mask(missing, x_imputed), x_test.mask(missing_test, x_test_imputed)

    return x.mask(missing, x_imputed)



#-----------------------------------------------------------------------------------#
#<><><><><><><><><><><><><><><><><> RF-GAP Impute <><><><><><><><><><><><><><><><><>#
#-----------------------------------------------------------------------------------#

# TODO: Incorporate with RFGAP class
def rfgap_impute(x, y, n_iters=10, initialization='median', global_initialization=False, 
                 numeric_metric='r2', categorical_metric='f1', internal_check=True, 
                 return_scores=False, return_multi_impute=False,  x_test = None, random_state=None, **kwargs):
    """
    Perform RF-GAP imputation on a DataFrame using a Random Forest-based approach over multiple iterations.

    Parameters:
    -----------
    x (pd.DataFrame):
        The input DataFrame containing missing values to be imputed.

    y (pd.Series or pd.DataFrame):
        The target variable (used for supervised learning during imputation).

    n_iters (int, optional):
        The number of iterations for the imputation process. Default is 10.

    initialization (str, optional):
        The method for initializing imputation. Options are 'mean', 'median', and 'knn'.
        Default is 'mean'.

    global_initialization (bool, optional):
        If True, the initialization method is applied globally (to the entire DataFrame). 
        If False, initialization is applied per group defined by `y`. Default is True.

    numeric_metric (str, optional):
        The evaluation metric for continuous features during internal checks. Default is 'r2'.

    categorical_metric (str, optional):
        The evaluation metric for categorical features during internal checks. Default is 'f1'.

    internal_check (bool, optional):
        If True, the function performs internal checks and tracks metrics during each iteration. 
        Default is True.

    return_scores (bool, optional):
        If True, returns the imputations along with the associated scores. Default is False.

    return_multi_impute (bool, optional):
        If True, returns the list of imputations for each iteration. Default is False.

    x_test (pd.DataFrame, optional):
        The test set to impute using the trained RF-GAP model. Default is None.

    **kwargs:
        Additional arguments passed to the Random Forest model (`RFGAP`).

    Returns:
    --------
    pd.DataFrame:
        The imputed DataFrame with missing values filled using the RF-GAP method.
        If `return_scores` is True, also returns a dictionary of the imputation scores (imputed_values, scores).
        If `return_multi_impute` is True, returns a list of imputations across iterations (imputed_values, multiple_imputations, scores).
    """

    if return_multi_impute and not internal_check:
        raise ValueError("return_multi_impute is only supported when internal_check is enabled.")
    
    # Get missing data mask
    missing = get_missing(x)

    if x_test is not None:
        missing_test = get_missing(x_test)
    
    # Initialize Random Forest model for RF-GAP imputation
    rf = RFGAP(y = y, matrix_type = 'dense', oob_score = True, random_state=random_state, **kwargs)
    
    # Dictionary to store metrics during iterations
    scores = {
        'continuous': [],
        'categorical': [],
        'oob_scores': []
    }


    y_dtype = np.asarray(y).dtype

    if not global_initialization and np.issubdtype(y_dtype, np.floating):
        warnings.warn(
            "Regression targets detected are but 'global_initialization' is False. "
            "Cannot run initialization conditional on the response. "
            "Setting 'global_initialization' to True." 
        )

        global_initialization = True




    # Initialize the imputed DataFrame based on the chosen initialization method
    if initialization == 'mean':
        x_imputed = stat_impute(x, statistic = 'mean') if global_initialization else stat_impute(x, y = y, statistic = 'mean')

    elif initialization == 'median':
        x_imputed = stat_impute(x, statistic = 'median') if global_initialization else stat_impute(x, statistic = 'median', y = y)

    elif initialization == 'knn':
        x_imputed = knn_impute(x) if global_initialization else knn_impute(x, y)

    if x_test is not None:
        if initialization == 'mean':
            x_test_imputed = stat_impute(x_test, statistic = 'mean')

        elif initialization == 'median':
            x_test_imputed = stat_impute(x_test, statistic = 'median')

        elif initialization == 'knn':
            x_test_imputed = knn_impute(x_test)


    # Store initial imputation
    imputations = [x_imputed]

    # Perform iterative imputation
    for _ in range(n_iters):
        rf.fit(x_imputed, y)
        scores['oob_scores'].append(rf.oob_score_)
        proximities = rf.get_proximities()

        if x_test is not None:
            proximities_test = rf.prox_extend(x_test_imputed)

        # TODO: Check not working with regression data
        if internal_check:
            # Perform imputation using proximities and evaluate

            if x_test is not None:
                x_imputed, x_imputed_nonmissing, x_test_imputed = proximity_impute_iteration(x_imputed, missing, proximities, return_nonmissing=True, x_test = x_test_imputed, missing_test = missing_test, proximities_test = proximities_test)

            else:
                x_imputed, x_imputed_nonmissing = proximity_impute_iteration(x_imputed, missing, proximities, return_nonmissing=True)

            numeric_score, categorical_score = missing_metrics(
                x, x_imputed_nonmissing, ~missing, numeric_metric=numeric_metric, categorical_metric=categorical_metric
            )

            # Store evaluation scores
            scores['continuous'].append(numeric_score)
            scores['categorical'].append(categorical_score)
            imputations.append(x_imputed)

        else:
            # Just perform imputation without evaluation
            if x_test is not None:
                x_imputed, x_test_imputed = proximity_impute_iteration(x_imputed, missing, proximities, return_nonmissing=False, x_test = x_test_imputed, missing_test = missing_test, proximities_test = proximities_test)

            x_imputed = proximity_impute_iteration(x_imputed, missing, proximities, return_nonmissing=False)

    # Append final out-of-bag score
    scores['oob_scores'].append(rf.oob_score_)


    if internal_check:
        if not scores['continuous'][0]:
            overall_score = np.array(scores['categorical'])
        elif not scores['categorical'][0]:
            overall_score = np.array(scores['continuous'])
        else:
            overall_score = np.mean((scores['continuous'], scores['categorical']), axis=0)

        best_idx = np.argmax(overall_score)

        if return_multi_impute:
            if return_scores:
                return_values = [imputations[best_idx], imputations, scores]
                if x_test is not None:
                    return_values.append(x_test_imputed)
                return tuple(return_values)
            return imputations[best_idx], imputations

        if return_scores:
            return_values = [imputations[best_idx], scores]
            if x_test is not None:
                return_values.append(x_test_imputed)
            return tuple(return_values)

        if x_test is not None:
            return imputations[best_idx], x_test_imputed
        return imputations[best_idx]

    if return_scores:
        return_values = [x_imputed, scores]
        if x_test is not None:
            return_values.append(x_test_imputed)
        return tuple(return_values)

    if x_test is not None:
        return x_imputed, x_test_imputed

    return x_imputed




#-----------------------------------------------------------------------------#
#<><><><><><><><><><><><><><><><><> Helpers <><><><><><><><><><><><><><><><><>#
#-----------------------------------------------------------------------------#

def normalize_rows(matrix):
    """
    Normalize each row of a matrix by dividing each element in the row by the row's sum.

    Parameters
    ----------
    matrix : array-like
        A 2D matrix (e.g., numpy array or pandas DataFrame) where each row will be normalized.

    Returns
    -------
    np.ndarray
        A 2D array with the same shape as the input matrix, but with each row normalized.
        Rows with a sum of zero will be set to zero.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.where(row_sums != 0, matrix / row_sums, 0)


def get_missing(df):
    """
    Identify missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame in which missing values need to be identified.

    Returns
    -------
    pd.DataFrame
        A DataFrame of the same shape as the input `df`, where each cell contains a boolean
        value indicating whether the corresponding entry in the original DataFrame is missing (True)
        or not (False).
    """
    return df.isnull()
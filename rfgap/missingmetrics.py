from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)
from sklearn.metrics.cluster import mutual_info_score


from scipy.stats import entropy, wasserstein_distance, spearmanr, gaussian_kde
import numpy as np
from ot import sliced_wasserstein_distance


#----------------------------------------------------------------------------#
#<><><><><><><><><><><><><> Instance-level Metrics <><><><><><><><><><><><><>#
#----------------------------------------------------------------------------#

def instance_metrics(x, imputed, missing=None):

    """
    Computes various metrics to evaluate the performance of imputation for both 
    numeric and categorical features in a DataFrame.

    Parameters
    ----------
    x : pd.DataFrame
        The original DataFrame with the true values.
    imputed : pd.DataFrame
        The DataFrame with imputed values.
    missing : pd.Series, optional
        A boolean series indicating the missing values in `x`. Default is None, 
        in which case the missing values are assumed to be NaN.

    Returns
    -------
    dict
        A dictionary containing two keys:
        - 'feature_scores' : dict
            A dictionary with individual metrics for each feature.
        - 'global_scores' : dict
            A dictionary with average scores across all features.

    Notes
    -----
    Metrics Computed:
    - For numeric features: R², RMSE, MAE, MAPE, Pearson correlation, Spearman correlation.
    - For categorical features: Accuracy, F1 score, Sensitivity (recall), Mutual information.
    """

    # Default missing values mask if not provided
    if missing is None:
        missing = x.isnull()

    # Identify numeric and categorical features
    numeric_features = x.select_dtypes(include=['float']).columns
    categorical_features = x.select_dtypes(exclude=['float']).columns

    # Initialize dictionaries to store feature-wise and global scores
    feature_scores = {'numeric': {}, 'categorical': {}}
    global_scores = {'numeric': {}, 'categorical': {}}

    # Metrics for numeric features
    for feature in numeric_features:
        true_values = x[feature][missing[feature]]
        imputed_values = imputed[feature][missing[feature]]

        if sum(missing[feature]) == 0:
            # Skip features with no missing values
            continue

        # Ensure non-empty and non-constant inputs for correlation calculations
        if len(true_values) > 1 and len(set(true_values)) > 1 and len(set(imputed_values)) > 1:
            pearson_corr = np.corrcoef(true_values, imputed_values)[0, 1]
            spearman_corr = spearmanr(true_values, imputed_values).correlation
        else:
            pearson_corr = np.nan
            spearman_corr = np.nan

        # Store individual metrics for numeric features
        feature_scores['numeric'][feature] = {
            'R2': r2_score(true_values, imputed_values),
            'RMSE': mean_squared_error(true_values, imputed_values, squared=False),
            'MAE': mean_absolute_error(true_values, imputed_values),
            'MAPE': mean_absolute_percentage_error(true_values, imputed_values),
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
        }

    # Metrics for categorical features
    for feature in categorical_features:
        true_values = x[feature][missing[feature]]
        imputed_values = imputed[feature][missing[feature]]

        if sum(missing[feature]) == 0:
            # Skip features with no missing values
            continue

        # TODO: Check check
        try:
            # Store individual metrics for categorical features
            feature_scores['categorical'][feature] = {
            'Accuracy': accuracy_score(true_values, imputed_values),
            'F1': safe_f1_score(true_values, imputed_values, average='weighted'),
            'Sensitivity': recall_score(true_values, imputed_values, average='weighted'),
            'Mutual Info': mutual_info_score(true_values, imputed_values),
            }
        except Exception as e:
            # Save error details to an external file
            with open("missing_metrics_error_log.txt", "a") as error_file:
                error_file.write(f"Error processing feature: {feature}\n")
                error_file.write(f"Error message: {str(e)}\n")
                error_file.write(f"True values: {true_values.tolist()}\n")
                error_file.write(f"Imputed values: {imputed_values.tolist()}\n")
                error_file.write("\n")

    # Calculate global metrics for numeric features
    global_scores['numeric'] = {
        'Average R2': np.mean([scores['R2'] for scores in feature_scores['numeric'].values()]),
        'Average RMSE': np.mean([scores['RMSE'] for scores in feature_scores['numeric'].values()]),
        'Average MAE': np.mean([scores['MAE'] for scores in feature_scores['numeric'].values()]),
        'Average MAPE': np.mean([scores['MAPE'] for scores in feature_scores['numeric'].values()]),
        'Average Pearson': np.nanmean([scores['Pearson'] for scores in feature_scores['numeric'].values()]),
        'Average Spearman': np.nanmean([scores['Spearman'] for scores in feature_scores['numeric'].values()]),
    }

    # Calculate global metrics for categorical features
    global_scores['categorical'] = {
        'Average Accuracy': np.mean([scores['Accuracy'] for scores in feature_scores['categorical'].values()]),
        'Average F1': np.mean([scores['F1'] for scores in feature_scores['categorical'].values()]),
        'Average Sensitivity': np.mean([scores['Sensitivity'] for scores in feature_scores['categorical'].values()]),
        'Average Mutual Info': np.mean([scores['Mutual Info'] for scores in feature_scores['categorical'].values()]),
    }

    return {'feature_scores': feature_scores, 'global_scores': global_scores}


#---------------------------------------------------------------------------#
#<><><><><><><><><><><><><> Feature-level Metrics <><><><><><><><><><><><><>#
#---------------------------------------------------------------------------#


def feature_metrics(x, imputed, missing=None):

    """
    Computes feature-level metrics for continuous and categorical variables:
    - KL divergence (for categorical and discrete distributions)
    - Wasserstein distance (for continuous distributions)
    
    Also computes global averages for each metric.

    Parameters
    ----------
    x : pd.DataFrame
        Original dataset with missing values.
    imputed : pd.DataFrame
        Dataset with imputed values.
    missing : pd.DataFrame
        Boolean DataFrame indicating missing entries.

    Returns
    -------
    dict
        A dictionary containing feature-level metrics and global averages.
    """

    if missing is None:
        missing = x.isnull()

    numeric_features = x.select_dtypes(include=['float']).columns
    categorical_features = x.select_dtypes(exclude=['float']).columns

    feature_scores = {'numeric': {}, 'categorical': {}}
    global_scores = {'numeric': {}, 'categorical': {}}

    numeric_metrics = []
    categorical_metrics = []

    # Numeric features: Wasserstein distance and KL divergence
    for feature in numeric_features:

        # Currently only working with missing vals.
        # true_values = x[feature][missing[feature]]
        # imputed_values = imputed[feature][missing[feature]]


        true_values = x[feature]
        imputed_values = imputed[feature]

        if sum(missing[feature]) == 0:
            # Skip features with no missing values
            continue

        # Wasserstein distance (continuous)
        wasserstein = wasserstein_distance(true_values, imputed_values)
        feature_scores['numeric'][feature] = {
            'Wasserstein Distance': wasserstein,
        }
        numeric_metrics.append(wasserstein)

        try:

            # KL divergence (continuous): Use Kernel Density Estimation (KDE)
            kde_true = gaussian_kde(true_values)
            kde_imputed = gaussian_kde(imputed_values)

            # Estimate the KL divergence by evaluating the density estimates
            min_val = min(np.min(true_values), np.min(imputed_values))
            max_val = max(np.max(true_values), np.max(imputed_values))

            x_vals = np.linspace(min_val, max_val, 1000)
            true_density = kde_true(x_vals)
            imputed_density = kde_imputed(x_vals)

            # Ensure no zero densities to avoid log(0)
            true_density += 1e-10
            imputed_density += 1e-10

            # KL divergence (continuous)
            kl_div = entropy(true_density, imputed_density)
            feature_scores['numeric'][feature]['KL Divergence'] = kl_div
            numeric_metrics.append(kl_div)

        # TODO: See if this is what we want
        except Exception as e:
            feature_scores['numeric'][feature]['KL Divergence'] = np.nan
            numeric_metrics.append(np.nan)

    # Categorical features: KL divergence
    for feature in categorical_features:

        # Currently only working with missing vals.
        # true_values = x[feature][missing[feature]]
        # imputed_values = imputed[feature][missing[feature]]

        true_values = x[feature]
        imputed_values = imputed[feature]

        if sum(missing[feature]) == 0:
            # Skip features with no missing values
            continue

        # Ensure consistent binning based on true values
        unique_categories = np.unique(true_values)
        bins = np.arange(len(unique_categories) + 1) - 0.5  # Create bins centered on integer categories

        # Compute normalized histograms
        true_hist = np.histogram(true_values, bins=bins, density=True)[0]
        imputed_hist = np.histogram(imputed_values, bins=bins, density=True)[0]

        # Ensure no division by zero for KL divergence
        true_hist += 1e-10
        imputed_hist += 1e-10

        # KL divergence (discrete)
        kl_div = entropy(true_hist, imputed_hist)
        feature_scores['categorical'][feature] = {
            'KL Divergence': kl_div,
        }
        categorical_metrics.append(kl_div)

    # Calculate global metrics
    if numeric_metrics:
        global_scores['numeric'] = {
            'Average Wasserstein Distance': np.mean([feature_scores['numeric'][feature].get('Wasserstein Distance', 0) for feature in numeric_features if sum(missing[feature]) > 0]),
            'Average KL Divergence': np.mean([feature_scores['numeric'][feature].get('KL Divergence', 0) for feature in numeric_features if sum(missing[feature]) > 0]),
        }
    if categorical_metrics:
        global_scores['categorical'] = {
            'Average KL Divergence': np.mean(categorical_metrics),
        }

    return {
        'feature_scores': feature_scores,
        'global_scores': global_scores,
    }




#------------------------------------------------------------------------#
#<><><><><><><><><><><><><> Joint Dist. Metric <><><><><><><><><><><><><>#
#------------------------------------------------------------------------#

# Do we need to separate for variable type?
def joint_metrics(x, imputed, n_projections = 100, random_state = None):

    """
    Compute the sliced Wasserstein distance between the original data (x) 
    and the imputed data.

    Parameters
    ----------
    x : array-like
        Original data.
    imputed : array-like
        Imputed data.
    n_projections : int, optional, default=100
        Number of random projections.

    Returns
    -------
    float
        The sliced Wasserstein distance as a measure of discrepancy.
    """

    if random_state is not None:
        np.random.seed(random_state)

    x = np.asarray(x)
    imputed = np.asarray(imputed)

    # Ensure the inputs have the same shape
    if x.shape != imputed.shape:
        raise ValueError(f"Shape mismatch: x has shape {x.shape}, "
                         f"while imputed has shape {imputed.shape}.")

    # Validate the number of projections
    if not isinstance(n_projections, int) or n_projections <= 0:
        raise ValueError("n_projections must be a positive integer.")

    return sliced_wasserstein_distance(x, imputed, n_projections=n_projections, seed = random_state)



#-----------------------------------------------------------------------------#
#<><><><><><><><><><><><><> Single Instance Metrics <><><><><><><><><><><><><>#
#-----------------------------------------------------------------------------#


def missing_metrics(x, imputed, missing, numeric_metric='mse', categorical_metric='accuracy'):
    
    """
    Computes performance metrics for imputation on categorical and continuous variables.

    This function evaluates the performance of imputation using separate metrics for numeric
    and categorical features, enabling internal checks within the RF-GAP method.

    Parameters
    ----------
    x : pd.DataFrame
        The original DataFrame containing true values.
    imputed : pd.DataFrame
        The DataFrame containing imputed values.
    missing : pd.Series
        A boolean Series indicating missing values in `x`.
    numeric_metric : str, optional
        The metric for evaluating numeric features. Options are:
        - 'mse': Mean Squared Error (default)
        - 'r2': R-squared
        - 'rmse': Root Mean Squared Error
    categorical_metric : str, optional
        The metric for evaluating categorical features. Options are:
        - 'accuracy' (default)
        - 'f1': F1 Score (weighted)

    Returns
    -------
    tuple
        A tuple containing:
        - avg_numeric_metric : float or None
            The average numeric metric across all numeric features, or None if there are no numeric features.
        - avg_categorical_metric : float or None
            The average categorical metric across all categorical features, or None if there are no categorical features.

    Notes
    -----
    - Numeric metrics evaluate imputed values for continuous variables:
    - 'mse': Mean Squared Error
    - 'r2': R-squared
    - 'rmse': Root Mean Squared Error
    - Categorical metrics evaluate imputed values for categorical variables:
    - 'accuracy': Fraction of correct predictions
    - 'f1': Weighted F1 Score
    """

    
    # Identify numeric and non-numeric features
    numeric_features = x.select_dtypes(include=['float']).columns
    other_features = x.select_dtypes(exclude=['float']).columns

    numeric_metrics = {}
    categorical_metrics = {}

    # Compute metrics for numeric features
    for feature in numeric_features:
        true_values = x[feature][missing[feature]]
        imputed_values = imputed[feature][missing[feature]]

        if sum(missing[feature]) == 0:
            # Skip features with no missing values
            continue

        if numeric_metric.lower() == 'mse':
            numeric_metrics[feature] = mean_squared_error(true_values, imputed_values)
        elif numeric_metric.lower() == 'r2':
            numeric_metrics[feature] = r2_score(true_values, imputed_values)
        elif numeric_metric.lower() == 'rmse':
            numeric_metrics[feature] = mean_squared_error(true_values, imputed_values, squared=False)

    # Compute metrics for categorical features
    for feature in other_features:
        true_values = x[feature][missing[feature]]
        imputed_values = imputed[feature][missing[feature]]

        if sum(missing[feature]) == 0:
            # Skip features with no missing values
            continue

        if categorical_metric.lower() == 'accuracy':
            categorical_metrics[feature] = accuracy_score(true_values, imputed_values)
        elif categorical_metric.lower() == 'f1':

            categorical_metrics[feature] = safe_f1_score(true_values, imputed_values, average='weighted')

    # Calculate averages for numeric and categorical metrics
    avg_numeric_metric = np.mean(list(numeric_metrics.values())) if numeric_metrics else None
    avg_categorical_metric = np.mean(list(categorical_metrics.values())) if categorical_metrics else None

    return avg_numeric_metric, avg_categorical_metric



def safe_f1_score(y_true, y_pred, average = 'weighted'):
    y_true = np.asarray(y_true).astype(int).flatten()
    y_pred = np.asarray(y_pred).astype(int).flatten()
    return f1_score(y_true, y_pred, average = average)
import numpy as np
from scipy import sparse
from sklearn import metrics
import warnings


class GAPExtrasMixin:

    
    def prox_predict(self, y):
        
        prox = self.get_kernel()
    
        if self.prediction_type == 'classification':
            y_one_hot = np.zeros((y.size, y.max() + 1))
            y_one_hot[np.arange(y.size), y] = 1
    
            prox_preds = np.argmax(prox @ y_one_hot, axis = 1)
            self.prox_predict_score = metrics.accuracy_score(y, prox_preds)
            return prox_preds
        
        else:
            prox_preds = prox @ y
            self.prox_predict_score = metrics.mean_squared_error(y, prox_preds)
            return prox_preds
        
    
    def get_instance_classification_expectation(self):
        """
        Calculates RF-ICE trust scores based on RF-GAP proximities.
    
        Raises
        ------
        ValueError
            If trust scores are requested for:
            - Non-RF-GAP proximities.
            - Non-classification models.
            - Non-zero diagonal proximities.
    
        Returns
        -------
        numpy.ndarray
            An array of trust scores for each observation.
        """
    
        
        # Validate input conditions
        if self.force_nonzero_diag:
            raise ValueError("Trust scores are only available for RF-GAP proximities with zero diagonal")
        
        if self.prediction_type != 'classification':
            raise ValueError("Classification trust scores are only available for classification models")   
        
        if self.kernel_method != 'gap':
            raise ValueError("Trust scores are only available for RF-GAP proximities")
    
        # Compute out-of-bag probabilities and correctness
        self.oob_proba = self.oob_decision_function_
        self.oob_predictions = np.argmax(self.oob_proba, axis=1)
        self.is_correct_oob = self.oob_predictions == self.y
    
        # Ensure proximities are computed
        if not hasattr(self, "proximities"):
            proximities_result = self.get_kernel()
            self.proximities = proximities_result.toarray() if isinstance(proximities_result, sparse.csr_matrix) else proximities_result
    
        elif isinstance(self.proximities, sparse.csr_matrix):
            self.proximities = self.proximities.toarray()
    
        # Compute trust scores
        self.trust_scores = self.proximities @ self.is_correct_oob
    
        # Compute trust quantiles
        quantile_levels = np.linspace(0, 0.99, 100)
        self.trust_quantiles = np.quantile(self.trust_scores, quantile_levels)
    
        # Compute accuracy rejection curve metrics
        self.trust_auc, self.trust_accuracy_drop, self.trust_n_drop = self.accuracy_rejection_auc(
            self.trust_quantiles, self.trust_scores
        )
    
        return self.trust_scores
    
    
    def get_test_trust(self, x_test):
    
        """
        Calculates RF-ICE trust scores for test data based on RF-GAP proximities.
    
        Parameters
        ----------
        x_test : array-like
            Test data for which trust scores are calculated.
    
        Raises
        ------
        ValueError
            If trust scores are requested for:
            - Non-RF-GAP proximities.
            - Non-classification models.
            - Non-zero diagonal proximities.
    
        Returns
        -------
        numpy.ndarray
            An array of trust scores for each observation in the test data.
        """
    
        
        # Validate input conditions
        if self.force_nonzero_diag:
            raise ValueError("Trust scores are only available for RF-GAP proximities with zero diagonal")
    
        if self.prediction_type != 'classification':
            raise ValueError("Classification trust scores are only available for classification models")   
    
        if self.kernel_method != 'gap':
            raise ValueError("Trust scores are only available for RF-GAP proximities")       
    
        # Compute out-of-bag probabilities and correctness
        self.oob_proba = self.oob_decision_function_
        self.oob_predictions = np.argmax(self.oob_proba, axis=1)
        self.is_correct_oob = self.oob_predictions == self.y 
    
        # Ensure proximities are computed properly
        if not hasattr(self, "kernel_extend"):
            raise AttributeError("The method 'kernel_extend' is not defined in this class.")
    
        self.test_proximities = self.kernel_extend(x_test)
        
        # Convert to dense array if sparse
        if isinstance(self.test_proximities, sparse.csr_matrix):
            self.test_proximities = self.test_proximities.toarray()
    
        # Compute test trust scores
        self.trust_scores_test = self.test_proximities @ self.is_correct_oob
    
        # Compute test trust quantiles
        quantile_levels = np.linspace(0, 0.99, 100)
        self.trust_quantiles_test = np.quantile(self.trust_scores_test, quantile_levels)
    
        return self.trust_scores_test
    
    
    def predict_with_intervals(self, X_test: np.ndarray = None, n_neighbors: int | str = 'auto', 
                        level: float = 0.95, verbose: bool = True) -> tuple:
        """
        Generate point predictions with prediction intervals for the test set using RF-GAP proximities.
    
        Prediction intervals are based on the distribution of OOB residuals conditioned on RF-GAP proximities.
        The model must be fit with `x_test` and `oob_score=True` for this method to work. Since the test data
        is stored in the model object during fitting, `X_test` is optional.
    
        Parameters
        ----------
        X_test : np.ndarray, optional
            Test set for generating predictions and prediction intervals.
            If omitted, the stored test data from model fitting is used.
    
        n_neighbors : int or {'auto', 'all'}, default='auto'
            Number of nearest neighbors to use for estimating residual distribution.
            - If an integer, all test observations use the same number of neighbors.
            - If 'auto', dynamically determines the number of neighbors per test point.
            - If 'all', uses all available training observations.
    
        level : float, default=0.95
            Confidence level for the prediction interval. Must be between 0 and 1.
    
        verbose : bool, default=True
            If True, prints warnings and other messages.
    
        Returns
        -------
        y_pred : np.ndarray of shape (n_test,)
            Point predictions for the test set.
    
        y_pred_lwr : np.ndarray of shape (n_test,)
            Lower bound of the prediction interval.
    
        y_pred_upr : np.ndarray of shape (n_test,)
            Upper bound of the prediction interval.
    
        Raises
        ------
        ValueError
            If RF-GAP proximities are not used, the model is a classification model,
            or `oob_score` was not enabled during training.
        """
        
        # Validate kernel_method applicability
        if self.kernel_method != 'gap':
            raise ValueError("Prediction intervals are only available for RF-GAP proximities.")
        
        if self.prediction_type == 'classification':
            raise ValueError("Prediction intervals are only available for regression models.")
        
        if not hasattr(self, 'oob_score_'):
            raise ValueError("Model must be fit with `oob_score=True`. Returning point predictions only.")
    
        self.interval_level = level
    
        # Retrieve proximities
        self.proximities: np.ndarray = self.get_kernel().toarray()
    
        # Compute test proximities
        test_proximities = self.kernel_extend(X_test)
        if isinstance(test_proximities, sparse.csr_matrix):
            test_proximities = test_proximities.toarray()
    
        self.test_proximities_ = test_proximities
        self.x_test = X_test
    
        # Compute OOB residuals
        oob_residuals = self.y - self.oob_prediction_
    
        # Tile residuals to match test proximity shape
        oob_residuals_tiled = np.tile(oob_residuals, (self.test_proximities_.shape[0], 1))
    
        # Sort OOB residuals by proximity (nearest to farthest)
        nearest_neighbor_indices = np.flip(self.test_proximities_.argsort(axis=1), axis=1)
        nearest_neighbor_residuals = np.take_along_axis(oob_residuals_tiled, nearest_neighbor_indices, axis=1)
        self.nearest_neighbor_residuals_ = nearest_neighbor_residuals
    
        # Validate `n_neighbors`
        match n_neighbors:
            case int() if n_neighbors > 0:
                pass
            case float() if n_neighbors > 0:
                n_neighbors = round(n_neighbors)
                if verbose:
                    warnings.warn(f"n_neighbors must be an integer or 'auto'. Using {n_neighbors} nearest neighbors.", 
                                category=UserWarning)
            case 'auto':
                test_proximities_sorted = np.take_along_axis(self.test_proximities_, nearest_neighbor_indices, axis=1)
                self.test_proximities_sorted_ = test_proximities_sorted
    
                # Remove zero proximities to exclude them from quantile calculation
                nearest_neighbor_residuals[test_proximities_sorted < 1e-10] = np.nan
            case 'all':
                n_neighbors = nearest_neighbor_residuals.shape[1]
            case _:
                raise ValueError("n_neighbors must be a positive integer or 'auto'.")
    
        # Save argument for reference
        self.interval_n_neighbors_: int | str = n_neighbors
    
        # Compute residual quantiles for interval estimation
        if n_neighbors == 'auto':
            resid_lwr = np.nanquantile(nearest_neighbor_residuals, (1 - level) / 2, axis=1)
            resid_upr = np.nanquantile(nearest_neighbor_residuals, 1 - (1 - level) / 2, axis=1)
        else:
            resid_lwr = np.quantile(nearest_neighbor_residuals[:, :n_neighbors], (1 - level) / 2, axis=1)
            resid_upr = np.quantile(nearest_neighbor_residuals[:, :n_neighbors], 1 - (1 - level) / 2, axis=1)
    
        # Compute point predictions and final prediction interval
        y_pred: np.ndarray = self.predict(self.x_test)
    
        y_pred_lwr = y_pred + resid_lwr
        y_pred_upr = y_pred + resid_upr
    
        return y_pred, y_pred_lwr, y_pred_upr
    
        
    def get_nonconformity(self, k: int = 5, x_test: np.ndarray = None, proximity_type = None):
        """
        Calculates nonconformity scores for the training set using RF-GAP proximities.
        Optionally calculates nonconformity scores for a test set.
    
        Parameters
        ----------
        k : int, default=5
            Number of nearest neighbors to consider when computing nonconformity scores.
    
        x_test : np.ndarray, optional
            Test set for which nonconformity scores should be calculated.
    
        Returns
        -------
        None
            Updates the following attributes:
            - `self.nonconformity_scores`
            - `self.conformity_scores`
            - `self.conformity_quantiles`
            - `self.conformity_auc`
            - `self.conformity_accuracy_drop`
            - `self.conformity_n_drop`
            
            If `x_test` is provided, also updates:
            - `self.nonconformity_scores_test`
            - `self.conformity_scores_test`
            - `self.conformity_quantiles_test`
        
        Raises
        ------
        ValueError
            If `k` is not a positive integer.
        """
    
        if not isinstance(k, int) or k <= 0:
            raise ValueError("`k` must be a positive integer.")
    
        try:
            # Store out-of-bag (OOB) probabilities and predictions
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis=1)
    
            # Use RF-GAP proximities to compute nonconformity scores
            original_prox_method = self.kernel_method
    
            if proximity_type is not None:
                self.kernel_method = proximity_type
    
            proximities = self.get_kernel()
    
            # Convert sparse matrix to dense if necessary
            if isinstance(proximities, sparse.csr_matrix):
                proximities = proximities.toarray()
    
            # Normalize proximities
            proximities = proximities / np.max(proximities, axis=1, keepdims=True)
    
            self.nonconformity_scores = np.zeros_like(self.y, dtype=float)
    
            # Calculate nonconformity scores for each class label
            unique_labels = np.unique(self.y)
    
            for label in unique_labels:
                mask = self.y == label
                same_proximities = proximities[:, mask]
                diff_proximities = proximities[:, ~mask]
    
                same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]
    
                diff_mean = np.mean(diff_k, axis=1)[mask]
                same_mean = np.mean(same_k, axis=1)[mask]
    
                # Avoid zero division by replacing zeros with the smallest non-zero value
                min_nonzero = np.min(same_mean[same_mean > 0], initial=1e-10)
                same_mean = np.where(same_mean == 0, min_nonzero, same_mean)
    
                self.nonconformity_scores[mask] = diff_mean / same_mean
    
            # Compute conformity scores and quantiles
            self.conformity_scores = np.max(self.nonconformity_scores) - self.nonconformity_scores
            self.conformity_quantiles = np.quantile(self.conformity_scores, np.linspace(0, 0.99, 100))
    
            # Compute accuracy rejection curve metrics
            self.conformity_auc, self.conformity_accuracy_drop, self.conformity_n_drop = self.accuracy_rejection_auc(
                self.conformity_quantiles, self.conformity_scores
            )
    
            # If test set is provided, calculate nonconformity scores for test predictions
            if x_test is not None:
                # Get predictions for the test set
                self.test_preds = self.predict(x_test)
    
                # Get the proximities between test samples and training samples (shape n_test, n_train)
                proximities_test = self.kernel_extend(x_test)
    
                # Convert sparse matrix to dense if necessary
                if isinstance(proximities_test, sparse.csr_matrix):
                    proximities_test = proximities_test.toarray()
    
                # Initialize array to store test nonconformity scores, shape (n_test,)
                self.nonconformity_scores_test = np.zeros_like(self.test_preds, dtype=float)
    
                # Get the unique predicted labels in the test set
                unique_test_preds = np.unique(self.test_preds)
    
                # Iterate through each unique predicted label found in the test set
                for label in unique_test_preds:
    
                    # Create a boolean mask for test samples predicted as the current label (shape n_test)
                    mask_test = self.test_preds == label
    
                    # Create a boolean mask for training samples with the same label (shape n_train)
                    mask_train_same = self.y == label
    
                    # Create a boolean mask for training samples with different labels (shape n_train)
                    mask_train_diff = self.y != label
    
                    # Select columns corresponding to same-class training samples for ALL test samples
                    # Shape will be (n_test, n_train_same)
                    same_proximities = proximities_test[:, mask_train_same]
    
                    # Select columns corresponding to different-class training samples for ALL test samples
                    # Shape will be (n_test, n_train_diff)
                    diff_proximities = proximities_test[:, mask_train_diff]
    
                    # Partition rows to find the k largest proximities to same-class training samples
                    # Takes the last k columns after partitioning along axis 1 (columns)
                    # Result shape: (n_test, k) - Assumes k <= n_train_same and k <= n_train_diff
                    # If k is too large, np.partition handles it gracefully by taking all available columns
                    same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                    # Result shape: (n_test, k)
                    diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]
    
                    # Calculate the mean of the k largest proximities for each test sample (row)
                    # Result shapes: (n_test,)
                    same_mean_all = np.mean(same_k, axis=1)
                    diff_mean_all = np.mean(diff_k, axis=1)
    
                    # Select the means only for the test samples predicted as the current label
                    # Result shapes: (n_subset,) where n_subset is the count of test samples predicted as 'label'
                    same_mean = same_mean_all[mask_test]
                    diff_mean = diff_mean_all[mask_test]
    
                    # Avoid zero division by replacing zeros with the smallest non-zero value found in same_mean
                    # This exactly mirrors the training logic
                    min_nonzero_test = np.min(same_mean[same_mean > 0], initial=1e-10)
                    same_mean_safe = np.where(same_mean == 0, min_nonzero_test, same_mean)
    
                    # Calculate and assign nonconformity scores for the subset of test samples
                    self.nonconformity_scores_test[mask_test] = diff_mean / same_mean_safe
    
                # Compute conformity scores and quantiles for the test set (using exact same logic as training)
                self.conformity_scores_test = np.max(self.nonconformity_scores_test) - self.nonconformity_scores_test
                self.conformity_quantiles_test = np.quantile(self.conformity_scores_test, np.linspace(0, 0.99, 100))
    
        finally:
            # Restore the original proximity kernel_method
            self.kernel_method = original_prox_method
    
    
    def accuracy_rejection_auc(self, quantiles: np.ndarray, scores: np.ndarray) -> tuple:
        """
        Computes the area under the accuracy-rejection curve (AUC) based on 
        nonconformity scores and rejection quantiles.
    
        The function evaluates model accuracy at different levels of rejection 
        (i.e., removing samples with high nonconformity scores). The result helps 
        assess the trade-off between data rejection and classification accuracy.
    
        Parameters
        ----------
        quantiles : np.ndarray
            Array of quantile thresholds used for rejection.
        
        scores : np.ndarray
            Array of nonconformity scores for each sample, used to determine 
            which samples should be rejected at each quantile level.
    
        Returns
        -------
        auc : float
            The area under the accuracy-rejection curve, computed using the trapezoidal rule.
        
        accuracy_drop : np.ndarray
            An array of accuracy values at different rejection levels.
        
        n_dropped : np.ndarray
            An array representing the proportion of rejected samples at each quantile.
        
        Raises
        ------
        ValueError
            If `quantiles` and `scores` have incompatible dimensions.
        """
    
        if not isinstance(quantiles, np.ndarray) or not isinstance(scores, np.ndarray):
            raise ValueError("Both `quantiles` and `scores` must be NumPy arrays.")
        
        if scores.shape[0] != self.y.shape[0]:
            raise ValueError("Mismatch between `scores` length and the number of target labels in `self.y`.")
    
        # Get out-of-bag (OOB) predictions from the RandomForest model
        oob_preds = np.argmax(self.oob_decision_function_, axis=1)
    
        # Compute the proportion of dropped samples for each quantile threshold
        n_dropped = np.array([np.sum(scores <= q) / len(scores) for q in quantiles])
    
        # Compute classification accuracy among remaining samples for each quantile threshold
        accuracy_drop = np.array([
            np.mean(self.y[scores >= q] == oob_preds[scores >= q]) if np.any(scores >= q) else 1.0 
            for q in quantiles
        ])
    
        # Compute the area under the accuracy-rejection curve (AUC)
        auc = np.trapz(accuracy_drop, n_dropped)
    
        return auc, accuracy_drop, n_dropped
    
    
    def get_outlier_scores(self, y, scaling = 'normalize'):
        """
        Compute class-relative outlier scores based on proximity matrix.
    
        Parameters:
        -----------
        y_train : pandas Series
            Class labels for training samples (e.g., '0', '1', ...)
        prox_matrix : scipy.sparse matrix or np.ndarray
            Square proximity matrix (n_samples x n_samples)
    
        Returns:
        --------
        outlier_scores : np.ndarray
            Standardized outlier scores (higher = more outlier-like)
        """
        try:
            y_arr = y.to_numpy()
        except:
            y_arr = np.asarray(y)
            
        n_samples = len(y_arr)
    
        force_nonzero_diag = self.force_nonzero_diag
    
        is_symmetric = self.force_symmetric
    
        if force_nonzero_diag:
            self.force_nonzero_diag = False
            if is_symmetric:
                self.force_symmetric = False
    
        proximities = self.get_kernel()
        proximities = proximities.toarray() if isinstance(proximities, sparse.csr_matrix) else proximities
    
        self.force_nonzero_diag = force_nonzero_diag
        self.force_symmetric = is_symmetric
    
        # Ensure matrix is dense
        if not isinstance(proximities, np.ndarray):
            prox_dense = proximities.toarray()
        else:
            prox_dense = proximities
    
        # Compute average squared proximities to same-class samples
        avg_prox = np.zeros(n_samples)
        for cls in np.unique(y_arr):
            idx = np.where(y_arr == cls)[0]
            prox_sub = prox_dense[np.ix_(idx, idx)]
            avg_prox[idx] = np.sum(prox_sub ** 2, axis=1)
    
        # Check this out
        if np.any(avg_prox == 0):
            print("Warning: Some samples have zero average proximity to same-class samples. This may affect outlier score calculation.")
    
        avg_prox[avg_prox == 0] = 1e-10
    
        # Compute raw outlier scores
        raw_scores = n_samples / avg_prox
    
        # Standardize within each class
        outlier_scores = np.zeros_like(raw_scores)
        for cls in np.unique(y_arr):
            idx = np.where(y_arr == cls)[0]
            class_scores = raw_scores[idx]
    
            median = np.median(class_scores)
            abs_dev = np.median(np.abs(class_scores - median))
    
            if abs_dev == 0:
                outlier_scores[idx] = 0
            else:
                outlier_scores[idx] = np.abs((class_scores - median)) / abs_dev
    
        if scaling == 'log':
            # Apply log scaling to outlier scores
            outlier_scores = np.log1p(outlier_scores)
    
        elif scaling == 'normalize':
            # Normalize outlier scores to [0, 1] range
            min_score = np.min(outlier_scores)
            max_score = np.max(outlier_scores)
    
            if max_score - min_score > 0:
                outlier_scores = (outlier_scores - min_score) / (max_score - min_score)
            else:
                outlier_scores = np.zeros_like(outlier_scores)
    
        return outlier_scores
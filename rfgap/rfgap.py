# Imports
import numpy as np
from scipy import sparse
import pandas as pd

# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn import metrics

from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble._forest import _generate_sample_indices
else:
    # Before sklearn version 0.24, forest was public, supporting this.
    from sklearn.ensemble.forest import _generate_unsampled_indices # Remove underscore from _forest
    from sklearn.ensemble.forest import _generate_sample_indices # Remove underscore from _forest

from sklearn.utils.validation import check_is_fitted
import warnings

def RFGAP(prediction_type = None, y = None, prox_method = 'rfgap', 
          matrix_type = 'sparse', triangular = True,
          non_zero_diagonal = False, force_symmetric = False, **kwargs):
    """
    A factory method to conditionally create the RFGAP class based on RandomForestClassifier or RandomForestRegressor (depdning on the type of response, y)

    This class takes on a random forest predictors (sklearn) and adds methods to 
    construct proximities from the random forest object. 
        

    Parameters
    ----------

    prediction_type : str
        Options are `regression` or `classification`

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in regression).
        This is an optional way to determine whether RandomForestClassifier or RandomForestRegressor
        should be used

    prox_method : str
        The type of proximity to be constructed.  Options are `original`, `oob`, 
        or `rfgap` (default is `rfgap`)

    matrix_type : str
        Whether the matrix returned proximities whould be sparse or dense 
        (default is sparse)

    triangular : bool
        Should only the upper triangle of the proximity matrix be computed? This speeds up computation
        time. Not available for RF-GAP proximities (default is True)

    non_zero_diagonal : bool
        Only used for RF-GAP proximities. Should the diagonal entries be computed as non-zero? 
        If True, the proximities are also normalized to be between 0 (min) and 1 (max).
        (default is True)

    force_symmetric : bool
        Enforce symmetry of proximities. (default is False)

    **kwargs
        Keyward arguements specific to the RandomForestClassifer or 
        RandomForestRegressor classes

        
    Returns
    -------
    self : object
        The RF object (unfitted)

    """


    if prediction_type is None and y is None:
        prediction_type = 'classification'
    

    if prediction_type is None and y is not None:
        if isinstance(y, pd.Series):
            y_array = y.to_numpy()
        else:
            y_array = np.array(y)

        try:
            if np.issubdtype(y_array.dtype, np.floating):
                prediction_type = 'regression'
            else:
                prediction_type = 'classification'
        except TypeError:
            prediction_type = 'classification'
            y_array = y_array.astype(str)

    if prediction_type == 'classification':
        rf = RandomForestClassifier
    elif prediction_type == 'regression':
        rf = RandomForestRegressor

    class RFGAP(rf):

        def __init__(self, prox_method = prox_method, matrix_type = matrix_type, 
                     triangular = triangular, non_zero_diagonal = non_zero_diagonal,
                     **kwargs):

            super(RFGAP, self).__init__(**kwargs)

            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.triangular  = triangular
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal
            self.force_symmetric = force_symmetric


        def fit(self, X, y, sample_weight = None, x_test = None):

            """Fits the random forest and generates necessary pieces to fit proximities

            Parameters
            ----------

            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in regression).

            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. If None, then samples are equally weighted. Splits that would 
                create child nodes with net zero or negative weight are ignored while searching 
                for a split in each node. In the case of classification, splits are also ignored 
                if they would result in any single class carrying a negative weight in either child node.

            Returns
            -------
            self : object
                Fitted estimator.

            """
            super().fit(X, y, sample_weight)

            # TODO: Check y type; make sure works with the rest of code.
            self.y = y
            self.n = len(y)
            self.leaf_matrix = self.apply(X)

            
            if x_test is not None:
                n_test = np.shape(x_test)[0]
                
                self.leaf_matrix_test = self.apply(x_test)
                self.leaf_matrix = np.concatenate((self.leaf_matrix, self.leaf_matrix_test), axis = 0)
            
                        
            if self.prox_method == 'oob':
                self.oob_indices = self.get_oob_indices(X)
                
                if x_test is not None:
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                
                self.oob_leaves = self.oob_indices * self.leaf_matrix

            if self.prox_method == 'rfgap':

                self.oob_indices = self.get_oob_indices(X)
                self.in_bag_counts = self.get_in_bag_counts(X)

                
                if x_test is not None:
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                    self.in_bag_counts = np.concatenate((self.in_bag_counts, np.zeros((n_test, self.n_estimators))))                
                                
                self.in_bag_indices = 1 - self.oob_indices

                self.in_bag_leaves = self.in_bag_indices * self.leaf_matrix
                self.oob_leaves = self.oob_indices * self.leaf_matrix
            

        
        def _get_oob_samples(self, data):
            
            """This is a helper function for get_oob_indices. 

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)

            """
            n = len(data)
            oob_samples = []
            for tree in self.estimators_:
                # Here at each iteration we obtain out-of-bag samples for every tree.
                oob_indices = _generate_unsampled_indices(tree.random_state, n, n)
                oob_samples.append(oob_indices)

            return oob_samples



        def get_oob_indices(self, data):
            
            """This generates a matrix of out-of-bag samples for each decision tree in the forest

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)


            Returns
            -------
            oob_matrix : array_like (n_samples, n_estimators) 

            """
            n = len(data)
            num_trees = self.n_estimators
            oob_matrix = np.zeros((n, num_trees))
            oob_samples = self._get_oob_samples(data)

            for t in range(num_trees):
                matches = np.unique(oob_samples[t])
                oob_matrix[matches, t] = 1

            return oob_matrix.astype(int)

        def _get_in_bag_samples(self, data):

            """This is a helper function for get_in_bag_indices. 

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)

            """

            n = len(data)
            in_bag_samples = []
            for tree in self.estimators_:
            # Here at each iteration we obtain in-bag samples for every tree.
                in_bag_sample = _generate_sample_indices(tree.random_state, n, n)
                in_bag_samples.append(in_bag_sample)
            return in_bag_samples


        def get_in_bag_counts(self, data):
            
            """This generates a matrix of in-bag samples for each decision tree in the forest

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)


            Returns
            -------
            in_bag_matrix : array_like (n_samples, n_estimators) 

            """
            n = len(data)
            num_trees = self.n_estimators
            in_bag_matrix = np.zeros((n, num_trees))
            in_bag_samples = self._get_in_bag_samples(data)

            for t in range(num_trees):
                matches, n_repeats = np.unique(in_bag_samples[t], return_counts = True)
                in_bag_matrix[matches, t] += n_repeats


            return in_bag_matrix

        def get_proximity_vector(self, ind):

            """This method produces a vector of proximity values for a given observation
            index. This is typically used in conjunction with get_proximities.
            
            Parameters
            ----------
            leaf_matrix : (n_samples, n_estimators) array_like
            oob_indices : (n_samples, n_estimators) array_like
            method      : string: methods may be `original`, `oob`, or `rfgap` (default is `oob`)
            
            Returns
            -------
            prox_vec : (n_samples, 1) array)_like: a vector of proximity values
            """
            n, num_trees = self.leaf_matrix.shape
            
            prox_vec = np.zeros((1, n))
            
            if self.prox_method == 'oob':

                if self.triangular:

                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(self.oob_indices[ind, ind_oob_leaves] == self.oob_indices[ind:, ind_oob_leaves], axis = 1)
                    tree_counts[tree_counts == 0] = 1

                    prox_counts   = np.sum(self.oob_leaves[ind, ind_oob_leaves]  == self.oob_leaves[ind:, ind_oob_leaves], axis = 1)
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols = np.where(prox_vec != 0)[0] + ind
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols - ind]

                else:

                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(self.oob_indices[ind, ind_oob_leaves] == self.oob_indices[:, ind_oob_leaves], axis = 1)
                    tree_counts[tree_counts == 0] = 1

                    prox_counts   = np.sum(self.oob_leaves[ind, ind_oob_leaves]  == self.oob_leaves[:, ind_oob_leaves], axis = 1)
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols = np.nonzero(prox_vec)[0]
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols]

            elif self.prox_method == 'original':

                if self.triangular:

                    tree_inds = self.leaf_matrix[ind, :] # Only indices after selected index
                    prox_vec = np.sum(tree_inds == self.leaf_matrix[ind:, :], axis = 1)

                    cols = np.where(prox_vec != 0)[0] + ind
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols - ind] / num_trees

                else:

                    tree_inds = self.leaf_matrix[ind, :]
                    prox_vec = np.sum(tree_inds == self.leaf_matrix, axis = 1)

                    cols = np.nonzero(prox_vec)[0]
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols] / num_trees


            elif self.prox_method == 'rfgap':

                oob_trees    = np.nonzero(self.oob_indices[ind, :])[0]
                in_bag_trees = np.nonzero(self.in_bag_indices[ind, :])[0]

                terminals = self.leaf_matrix[ind, :]

                matches = terminals == self.in_bag_leaves 

                match_counts = np.where(matches, self.in_bag_counts, 0)

                ks = np.sum(match_counts, axis = 0)
                ks[ks == 0] = 1
                ks_in  = ks[in_bag_trees]
                ks_out = ks[oob_trees]

                S_out = np.count_nonzero(self.oob_indices[ind, :])

                prox_vec = np.sum(np.divide(match_counts[:, oob_trees], ks_out), axis = 1) / S_out

                if self.non_zero_diagonal:
                    S_in  = np.count_nonzero(self.in_bag_indices[ind, :])
                    
                    if S_in > 0:
                        prox_vec[ind] = np.sum(np.divide(match_counts[ind, in_bag_trees], ks_in)) / S_in
                    else: 
                        prox_vec[ind] = np.sum(np.divide(match_counts[ind, in_bag_trees], ks_in))

                    prox_vec = prox_vec / np.max(prox_vec)
                    prox_vec[ind] = 1

                cols = np.nonzero(prox_vec)[0]
                rows = np.ones(len(cols), dtype = int) * ind
                data = prox_vec[cols]

            return data.tolist(), rows.tolist(), cols.tolist()
        
        
        def get_proximities(self):
            
            """This method produces a proximity matrix for the random forest object.
            
            
            Returns
            -------
            array-like
                (if self.matrix_type == `dense`) matrix of pair-wise proximities

            csr_matrix
                (if self.matrix_type == `sparse`) a sparse crs_matrix of pair-wise proximities
            
            """
            check_is_fitted(self)
            n, _ = self.leaf_matrix.shape

            for i in range(n):
                if i == 0:
                        prox_vals, rows, cols = self.get_proximity_vector(i)
                else:
                    if self.verbose:
                        if i % 100 == 0:
                            print('Finished with {} rows'.format(i))

                    prox_val_temp, rows_temp, cols_temp = self.get_proximity_vector(i)
                    prox_vals.extend(prox_val_temp)
                    rows.extend(rows_temp)
                    cols.extend(cols_temp)


            if self.triangular and self.prox_method != 'rfgap':
                prox_sparse = sparse.csr_matrix((np.array(prox_vals + prox_vals), (np.array(rows + cols), np.array(cols + rows))), shape = (n, n)) 
                prox_sparse.setdiag(1)

            else:
                prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(rows), np.array(cols))), shape = (n, n)) 

            if self.force_symmetric:
                prox_sparse = (prox_sparse + prox_sparse.transpose()) / 2

            if self.matrix_type == 'dense':
                return np.array(prox_sparse.toarray())
            
            else:
                return prox_sparse


        def get_test_proximities(self, x_test: np.ndarray) -> np.ndarray:
            """
            Computes proximity values between test samples and the trained model using RF-GAP or OOB proximities.

            This function modifies the internal proximity calculation attributes temporarily to compute
            test proximities and then restores the original values.

            Parameters
            ----------
            x_test : np.ndarray
                Test dataset of shape (n_test, n_features) for which proximities are to be calculated.

            Returns
            -------
            prox_test : np.ndarray
                A proximity matrix of shape (n_test, n_train) representing the similarity between 
                test samples and training samples.

            Raises
            ------
            ValueError
                If `x_test` is not a valid NumPy array or has an incorrect shape.
            """

            if not isinstance(x_test, np.ndarray):
                raise ValueError("`x_test` must be a NumPy array.")

            n_test, _ = x_test.shape

            # Store current attributes to restore later
            leaf_matrix_temp = self.leaf_matrix

            try:
                # Apply test data to obtain leaf assignments
                self.leaf_matrix = self.apply(x_test)

                # Temporarily adjust attributes for different proximity methods
                if self.prox_method in ['oob', 'rfgap']:
                    oob_indices_temp, self.oob_indices = self.oob_indices, np.ones((n_test, self.n_estimators))
                    oob_leaves_temp, self.oob_leaves = self.oob_leaves, self.oob_indices * self.leaf_matrix

                if self.prox_method == 'rfgap':
                    in_bag_indices_temp, self.in_bag_indices = self.in_bag_indices, np.zeros((n_test, self.n_estimators))
                    in_bag_leaves_temp, self.in_bag_leaves = self.in_bag_leaves, self.in_bag_indices * self.leaf_matrix
                    in_bag_counts_temp, self.in_bag_counts = self.in_bag_counts, np.zeros((n_test, self.n_estimators))

                # Calculate proximities using updated test attributes
                prox_test = self.get_proximities()

            finally:
                # Restore original attributes
                self.leaf_matrix = leaf_matrix_temp
                if self.prox_method in ['oob', 'rfgap']:
                    self.oob_indices, self.oob_leaves = oob_indices_temp, oob_leaves_temp
                if self.prox_method == 'rfgap':
                    self.in_bag_indices, self.in_bag_leaves, self.in_bag_counts = (
                        in_bag_indices_temp, in_bag_leaves_temp, in_bag_counts_temp
                    )

            return prox_test



        def prox_extend(self, data):
            """Method to compute proximities between the original training 
            observations and a set of new observations.

            Parameters
            ----------
            data : (n_samples, n_features) array_like (numeric)
            
            Returns
            -------
            array-like
                (if self.matrix_type == `dense`) matrix of pair-wise proximities between
                the training data and the new observations

            csr_matrix
                (if self.matrix_type == `sparse`) a sparse crs_matrix of pair-wise proximities
                between the training data and the new observations
            """
            check_is_fitted(self)
            n, num_trees = self.leaf_matrix.shape
            extended_leaf_matrix = self.apply(data)
            n_ext, _ = extended_leaf_matrix.shape

            prox_vals = []
            rows = []
            cols = []

            if self.prox_method == 'oob':

                for ind in range(n):

                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(self.oob_indices[ind, ind_oob_leaves] == np.ones_like(extended_leaf_matrix[:, ind_oob_leaves]), axis = 1)
                    tree_counts[tree_counts == 0] = 1

                    prox_counts = np.sum(self.oob_leaves[ind, ind_oob_leaves]  == extended_leaf_matrix[:, ind_oob_leaves], axis = 1)
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols_temp = np.nonzero(prox_vec)[0]
                    rows_temp = np.ones(len(cols_temp), dtype = int) * ind
                    prox_temp = prox_vec[cols_temp] 


                    cols.extend(cols_temp)
                    rows.extend(rows_temp)
                    prox_vals.extend(prox_temp)



            elif self.prox_method == 'original':

                for ind in range(n):

                    tree_inds = self.leaf_matrix[ind, :]
                    prox_vec  = np.sum(tree_inds == extended_leaf_matrix, axis = 1)

                    cols_temp = np.nonzero(prox_vec)[0]
                    rows_temp = np.ones(len(cols_temp), dtype = int) * ind
                    prox_temp = prox_vec[cols_temp] / num_trees

                    cols.extend(cols_temp)
                    rows.extend(rows_temp)
                    prox_vals.extend(prox_temp)

            elif self.prox_method == 'rfgap':
  
                for ind in range(n_ext):

                    oob_terminals = extended_leaf_matrix[ind, :] 

                    matches = oob_terminals == self.in_bag_leaves
                    matched_counts = np.where(matches, self.in_bag_counts, 0)

                    ks = np.sum(matched_counts, axis = 0)
                    ks[ks == 0] = 1

                    prox_vec = np.sum(np.divide(matched_counts, ks), axis = 1) / num_trees

                    cols_temp = np.nonzero(prox_vec)[0]
                    rows_temp = np.ones(len(cols_temp), dtype = int) * ind
                    prox_temp = prox_vec[cols_temp]

                    cols.extend(rows_temp)
                    rows.extend(cols_temp)
                    prox_vals.extend(prox_temp)


            prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(cols), np.array(rows))), shape = (n_ext, n))

            if self.matrix_type == 'dense':
                return prox_sparse.toarray() 
            else:
                return prox_sparse
            

        def prox_predict(self, y):
            
            prox = self.get_proximities()

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
            if self.non_zero_diagonal:
                raise ValueError("Trust scores are only available for RF-GAP proximities with zero diagonal")
            
            if self.prediction_type != 'classification':
                raise ValueError("Classification trust scores are only available for classification models")   
            
            if self.prox_method != 'rfgap':
                raise ValueError("Trust scores are only available for RF-GAP proximities")

            # Compute out-of-bag probabilities and correctness
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis=1)
            self.is_correct_oob = self.oob_predictions == self.y

            # Ensure proximities are computed
            if not hasattr(self, "proximities"):
                proximities_result = self.get_proximities()
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
            if self.non_zero_diagonal:
                raise ValueError("Trust scores are only available for RF-GAP proximities with zero diagonal")

            if self.prediction_type != 'classification':
                raise ValueError("Classification trust scores are only available for classification models")   

            if self.prox_method != 'rfgap':
                raise ValueError("Trust scores are only available for RF-GAP proximities")       

            # Compute out-of-bag probabilities and correctness
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis=1)
            self.is_correct_oob = self.oob_predictions == self.y 

            # Ensure proximities are computed properly
            if not hasattr(self, "prox_extend"):
                raise AttributeError("The method 'prox_extend' is not defined in this class.")

            self.test_proximities = self.prox_extend(x_test)
            
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
            
            # Validate method applicability
            if self.prox_method != 'rfgap':
                raise ValueError("Prediction intervals are only available for RF-GAP proximities.")
            
            if self.prediction_type == 'classification':
                raise ValueError("Prediction intervals are only available for regression models.")
            
            if not hasattr(self, 'oob_score_'):
                raise ValueError("Model must be fit with `oob_score=True`. Returning point predictions only.")

            self.interval_level = level

            # Retrieve proximities
            self.proximities: np.ndarray = self.get_proximities().toarray()

            # Compute test proximities
            test_proximities = self.prox_extend(X_test)
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
                original_prox_method = self.prox_method

                if proximity_type is not None:
                    self.prox_method = proximity_type

                proximities = self.get_proximities()

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
                    self.test_preds = self.predict(x_test)
                    proximities_test = self.get_test_proximities(x_test)

                    # Convert sparse matrix to dense if necessary
                    if isinstance(proximities_test, sparse.csr_matrix):
                        proximities_test = proximities_test.toarray()

                    self.nonconformity_scores_test = np.zeros_like(self.test_preds, dtype=float)

                    for label in np.unique(self.test_preds):
                        mask_test = self.test_preds == label
                        same_proximities = proximities_test[:, mask_test]
                        diff_proximities = proximities_test[:, ~mask_test]

                        same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                        diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]

                        # Assign test nonconformity scores
                        diff_mean_test = np.mean(diff_k, axis=1)[mask_test]
                        same_mean_test = np.mean(same_k, axis=1)[mask_test]

                        min_nonzero_test = np.min(same_mean_test[same_mean_test > 0], initial=1e-10)
                        same_mean_test = np.where(same_mean_test == 0, min_nonzero_test, same_mean_test)

                        self.nonconformity_scores_test[mask_test] = diff_mean_test / same_mean_test

                    self.conformity_scores_test = np.max(self.nonconformity_scores_test) - self.nonconformity_scores_test
                    self.conformity_quantiles_test = np.quantile(self.conformity_scores_test, np.linspace(0, 0.99, 100))

            finally:
                # Restore the original proximity method
                self.prox_method = original_prox_method


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
        y_arr = y.to_numpy()
        n_samples = len(y_arr)

        non_zero_diagonal = self.non_zero_diagonal

        if non_zero_diagonal:
            self.non_zero_diagonal = False
            proximities = self.get_proximities()
            proximities = proximities.toarray() if isinstance(proximities, sparse.csr_matrix) else proximities

        self.non_zero_diagonal = non_zero_diagonal

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




    return RFGAP(prox_method = prox_method, matrix_type = matrix_type, triangular = triangular, **kwargs)

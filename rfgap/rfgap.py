# Imports
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

#sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn

from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble import _forest as forest
    from sklearn.ensemble._forest import _generate_sample_indices
else:
    # Before sklearn version 0.24, forest was public, supporting this.
    from sklearn.ensemble.forest import _generate_unsampled_indices # Remove underscore from _forest
    from sklearn.ensemble.forest import _generate_sample_indices # Remove underscore from _forest
    from sklearn.ensemble import forest

from sklearn.utils.validation import check_is_fitted


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
        or `rfgap` (default is `oob`)

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
        if np.dtype(y) == 'float64' or np.dtype(y) == 'float32':
            prediction_type = 'regression'
        else:
            prediction_type = 'classification'


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
                return np.array(prox_sparse.todense())
            
            else:
                return prox_sparse


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
                return prox_sparse.todense() 
            else:
                return prox_sparse
            

        def prox_predict(self, y):
            
            # TODO: need to compute proximities for new points, test points added

            prox = self.get_proximities()

            if self.prediction_type == 'classification':
                y_one_hot = np.zeros((y.size, y.max() + 1))
                y_one_hot[np.arange(y.size), y] = 1

                prox_preds = np.argmax(prox @ y_one_hot, axis = 1)
                self.prox_predict_score = sklearn.metrics.accuracy_score(y, prox_preds)
                return prox_preds
            
            else:
                prox_preds = prox @ y
                self.prox_predict_score = sklearn.metrics.mean_squared_error(y, prox_preds)
                return prox_preds
            


        # TODO: Review and potentially rename. Update trust score verbiage.

        # TODO: UPDATE ALL OF THE BELOW METHODS; NOTHING IS CLEANRED!!

        def get_trust_scores(self):

            """
            Calculates RF-ICE trust scores based on RF-GAP proximities.
            
            Raises:
            - ValueError: If trust scores are requested for non-RF-GAP proximities,
                        or if trust scores are requested for non-classification models,
                        or if non-zero diagonal proximities are used.
            
            Returns:
            - Array of trust scores for each observation.
            """
             
            if self.non_zero_diagonal:
                raise ValueError('Trust scores are only available for RF-GAP proximities with zero diagonal')
            
            if self.prediction_type != 'classification':
                raise ValueError('Classification trust scores are only available for regression models')   
            
            if self.prox_method != 'rfgap':
                raise ValueError('Trust scores are only available for RF-GAP proximities')
        
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis = 1)
            self.is_correct_oob = self.oob_predictions == self.y

            if not 'self.proximities' in locals():
                self.proximities = self.get_proximities().toarray()

            self.trust_scores = self.proximities @ self.is_correct_oob
            self.trust_quantiles = np.quantile(self.trust_scores, np.linspace(0, 0.99, 100))
            self.trust_auc, self.trust_accuarcy_drop, self.trust_n_drop = self.accuracy_rejection_curve_area(self.trust_quantiles, self.trust_scores)

            return self.trust_scores

        
        def get_test_trust(self, x_test):
            """
            Calculates RF-ICE trust scores for test data based on RF-GAP proximities.
            
            Parameters:
            - x_test: Array-like
                Test data to calculate trust scores.
            
            Raises:
            - ValueError: If trust scores are requested for non-RF-GAP proximities,
                        or if trust scores are requested for non-classification models,
                        or if non-zero diagonal proximities are used.
            
            Returns:
            - Array of trust scores for each observation in the test data.
            """

            if self.non_zero_diagonal:
                raise ValueError('Trust scores are only available for RF-GAP proximities with zero diagonal')
            
            if self.prediction_type != 'classification':
                raise ValueError('Classification trust scores are only available for regression models')   
            
            if self.prox_method != 'rfgap':
                raise ValueError('Trust scores are only available for RF-GAP proximities')       

            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis = 1)
            self.is_correct_oob = self.oob_predictions == self.y 

            self.test_proximities = self.prox_extend(x_test).toarray()#[:, :self.n] # Check this last part.

            self.trust_scores_test = self.test_proximities @ self.is_correct_oob
            self.trust_quantiles_test = np.quantile(self.trust_scores_test, np.linspace(0, 0.99, 100))


            return self.trust_scores_test


        def predict_interval(self, X_test:np.array=None, n_neighbors:int|str='auto', level:float=0.95, verbose=True) -> tuple:
            """Generate point predictions with prediction intervals for the test set using RF-GAP proximities. Since
            the prediction intervals are based on the distribution of OOB residuals conditioned on RF-GAP proximities,
            the model must be fit with `x_test` and `oob_score=True`. The test data is stored in the model object when
            the model is fit, so there is no need to pass `x_test` as an argument to this method.

            Parameters
            ----------

            X_test : array-like of shape (n_test, n_features). The test set for which to generate predictions and prediction intervals.
            Does not need to be provided if the model was not fit with `x_test`.

            n_neighbors : int or string, default='auto'. For each test point, how many nearest-neighbor training points should 
            be used to estimate the distribution of possible residuals? If n_neighbors in an interger, all test observations
            will look to the same number of neighbors. If 'auto', the number of neighbors is determined dynamically for each test
            point by using all training neighbors with a non-zero RF-GAP proximity.
            
            level : float, default=0.95. The level of the prediction interval. Must be between 0 and 1.

            verbose : bool, default=True. Whether to print warnings and other messages.

            Returns
            -------
            y_pred : array-like of shape (n_test,) representing the point predictions for the test set
            y_pred_lwr : array-like of shape (n_test,) representing the lower bound of the prediction interval for the test set
            y_pred_upr : array-like of shape (n_test,) representing the upper bound of the prediction interval for the test set

            """

            if self.prox_method != 'rfgap':
                raise ValueError('Prediction intervals are only available for RF-GAP proximities')

            if self.prediction_type == 'classification':
                raise ValueError('Prediction intervals are only available for regression models')

            #if self.oob_score_ is None:
            if not hasattr(self, 'oob_score_'):
                raise ValueError('Model has not been fit with `oob_score = True`. Returning point predictions only.')

            self.interval_level = level

            self.proximities: np.ndarray = self.get_proximities().toarray()

            test_proximities = self.prox_extend(X_test).toarray()
            self.test_proximities_ = test_proximities
            self.x_test = X_test

            ## Calculate OOB residuals and tile residuals to match the shape of proximities
            oob_residuals = self.y - self.oob_prediction_ #self.oob_prediction_ - self.y
            oob_residuals_tiled = np.tile(oob_residuals,(self.test_proximities_.shape[0],1))

            ## Sort OOB residuals row-wise by proximity (nearest to farthest)
            nearest_neighbor_indices = np.flip(self.test_proximities_.argsort(axis=1), axis=1)
            nearest_neighbor_residuals = np.take_along_axis(oob_residuals_tiled, nearest_neighbor_indices, axis=1)
            self.nearest_neighbor_residuals_ = nearest_neighbor_residuals

            ## Perform error checking on n_neighbors
            match n_neighbors:
                case int():
                    pass
                case float():
                    try:
                        n_neighbors: int = round(n_neighbors)
                        if verbose:
                            warnings.warn(f'n_neighbors must be an integer or "auto"; using {n_neighbors} nearest neighbors.', 
                                            category=UserWarning)
                    except Exception as e:
                        raise ValueError('n_neighbors must be an integer or "auto"')
                case 'auto':
                    test_proximities_sorted = np.take_along_axis(self.test_proximities_, nearest_neighbor_indices, axis=1)
                    self.test_proximities_sorted_ = test_proximities_sorted

                    ## Remove zero proximities so they will not be included in quantile
                    nearest_neighbor_residuals[test_proximities_sorted < 1e-10] = np.nan

                case 'all':
                    n_neighbors = nearest_neighbor_residuals.shape[1]
                    
                case _:
                    raise ValueError('n_neighbors must be an integer or "auto"')
                
            ## Save arguments for reference
            self.interval_n_neighbors_: int|str = n_neighbors

            ## Take quantiles of nearest-neighbor OOB residuals to get credible interval on plausible errors
            ## `np.quantile()` will handle the error catching if `level` is out of bounds
            if n_neighbors == 'auto':
                    resid_lwr = np.nanquantile(nearest_neighbor_residuals, (1-level)/2, axis=1)
                    resid_upr = np.nanquantile(nearest_neighbor_residuals, 1-(1-level)/2, axis=1)
            else:
                resid_lwr = np.quantile(nearest_neighbor_residuals[:,:n_neighbors], (1-level)/2, axis=1)
                resid_upr = np.quantile(nearest_neighbor_residuals[:,:n_neighbors], 1-(1-level)/2, axis=1)

            ## Calculate point predictions and prediction interval
            y_pred: np.ndarray = self.predict(self.x_test) #+ np.mean(nearest_neighbor_residuals)

            y_pred_lwr = y_pred + resid_lwr
            y_pred_upr = y_pred + resid_upr

            return y_pred, y_pred_lwr, y_pred_upr
        


        def get_trust_scores(self):

            """
            RF-ICE Trust Scores
            """
            if self.non_zero_diagonal:
                raise ValueError('Trust scores are only available for RF-GAP proximities with zero diagonal')
            
            if self.prediction_type != 'classification':
                raise ValueError('Classification trust scores are only available for regression models')   
            
            if self.prox_method != 'rfgap':
                raise ValueError('Trust scores are only available for RF-GAP proximities')
            
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis = 1)
            self.is_correct_oob = self.oob_predictions == self.y

            if not 'self.proximities' in locals():
                self.proximities = self.get_proximities().toarray()

            # 0 - 1 indication as weighted average of correct predictions
            self.trust_scores = self.proximities @ self.is_correct_oob

            self.trust_quantiles = np.quantile(self.trust_scores, np.linspace(0, 0.99, 100))

            return self.trust_scores
        

        def get_nonconformity(self, k=5, x_test=None):
            """
            Calculates nonconformity scores for the training set and, if provided, for a test set.
            Supports both series and array data formats.
            """
            # Store out-of-bag (OOB) probabilities and predictions
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis=1)
            
            # Use OOB proximities to compute nonconformity scores
            original_prox_method = self.prox_method
            self.prox_method = 'oob'
            oob_proximities = self.get_proximities()
            if matrix_type == 'sparse':
                oob_proximities = oob_proximities.toarray()

            self.nonconformity_scores = np.zeros_like(self.y, dtype=float)

            # Calculate nonconformity scores for each class label
            for label in self.y.unique():
                same_proximities = oob_proximities[:, self.y == label]
                diff_proximities = oob_proximities[:, self.y != label]
                same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]

                # Assign nonconformity scores
                self.nonconformity_scores[self.y == label] = (
                    np.mean(diff_k, axis=1)[self.y == label] /
                    np.mean(same_k, axis=1)[self.y == label]
                )


            self.conformity_scores = np.max(self.nonconformity_scores) - self.nonconformity_scores
            self.conformity_quantiles = np.quantile(self.conformity_scores, np.linspace(0, 0.99, 100))
            self.conformity_auc, self.conformity_accuarcy_drop, self.conformity_n_drop = self.accuracy_rejection_curve_area(self.conformity_quantiles, self.conformity_scores)

            # If test set is provided, calculate nonconformity scores for test predictions
            if x_test is not None:
                self.test_preds = self.predict(x_test)
                oob_proximities_test = self.get_test_proximities(x_test)

                if self.matrix_type == 'sparse':
                    oob_proximities_test = oob_proximities_test.toarray()

                self.nonconformity_scores_test = np.zeros_like(self.test_preds, dtype=float)

                for label in np.unique(self.test_preds):
                    same_proximities = oob_proximities_test[:, self.test_preds == label]
                    diff_proximities = oob_proximities_test[:, self.test_preds != label]
                    same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                    diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]


                    # Assign test nonconformity scores
                    self.nonconformity_scores_test[self.test_preds == label] = (
                        np.mean(diff_k, axis=1)[self.test_preds == label] /
                        np.mean(same_k, axis=1)[self.test_preds == label]
                    )

                self.conformity_scores_test = np.max(self.nonconformity_scores_test) - self.nonconformity_scores_test
                self.conformity_quantiles_test = np.quantile(self.conformity_scores_test, np.linspace(0, 0.99, 100))

            # Restore the original proximity method
            self.prox_method = original_prox_method

        def get_nonconformity_rfgap(self, k=5, x_test=None):
            """
            Calculates nonconformity scores for the training set and, if provided, for a test set.
            Supports both series and array data formats.
            """
            # Store out-of-bag (OOB) probabilities and predictions
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis=1)
            
            # Use OOB proximities to compute nonconformity scores
            original_prox_method = self.prox_method
            self.prox_method = 'rfgap'
            # self.non_zero_diagonal = True # TODO: SHould be TRUE??


            oob_proximities = self.get_proximities()
            if matrix_type == 'sparse':
                oob_proximities = oob_proximities.toarray()


            oob_proximities = oob_proximities / np.max(oob_proximities, axis = 1)[:, np.newaxis]

            self.nonconformity_scores_rfgap = np.zeros_like(self.y, dtype=float)
            # self.conformity_scores_rfgap = np.ones_like(self.y, dtype=float)

            # Calculate nonconformity scores for each class label
            for label in self.y.unique():
                same_proximities = oob_proximities[:, self.y == label]
                diff_proximities = oob_proximities[:, self.y != label]
                same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]


                diff_mean = np.mean(diff_k, axis=1)[self.y == label]
                same_mean = np.mean(same_k, axis=1)[self.y == label]
                # same_mean = np.where(same_mean == 0, 0.00001, same_mean)
                same_min_nonzero = np.min(same_mean[same_mean > 0])  # Find the minimum non-zero value
                same_mean = np.where(same_mean == 0, same_min_nonzero, same_mean)
                # print(same_min_nonzero)


                self.nonconformity_scores_rfgap[self.y == label] = (diff_mean / same_mean)


            # TODO: Make consistent with get_nonconformity
            self.conformity_scores_rfgap = np.max(self.nonconformity_scores_rfgap) - self.nonconformity_scores_rfgap
            self.conformity_quantiles_rfgap = np.quantile(self.conformity_scores_rfgap, np.linspace(0, 0.99, 100))
            self.conformity_rfgap_auc, self.conformity_rfgap_accuarcy_drop, self.conformity_rfgap_n_drop = self.accuracy_rejection_curve_area(self.conformity_quantiles_rfgap, self.conformity_scores_rfgap)

            # If test set is provided, calculate nonconformity scores for test predictions

            # Not currently functional
            if x_test is not None:

                self.test_preds = self.predict(x_test)
                oob_proximities_test = self.get_test_proximities(x_test)

                if self.matrix_type == 'sparse':
                    oob_proximities_test = oob_proximities_test.toarray()

                self.nonconformity_scores_rfgap_test = np.zeros_like(self.test_preds, dtype=float)

                for label in np.unique(self.test_preds):
                    same_proximities = oob_proximities_test[:, self.test_preds == label]
                    diff_proximities = oob_proximities_test[:, self.test_preds != label]
                    same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                    diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]


                    # Assign test nonconformity scores
                    self.nonconformity_scores_rfgap_test[self.test_preds == label] = (
                        np.mean(diff_k, axis=1)[self.test_preds == label] /
                        np.mean(same_k, axis=1)[self.test_preds == label]
                    )

                self.conformity_scores_rfgap_test = np.max(self.nonconformity_scores_rfgap_test) - self.nonconformity_scores_rfgap_test
                self.conformity_quantiles_rfgap_test = np.quantile(self.conformity_scores_rfgap_test, np.linspace(0, 0.99, 100))

            # Restore the original proximity method
            self.prox_method = original_prox_method


        def get_nonconformity_original(self, k=5, x_test=None):
            """
            Calculates nonconformity scores for the training set and, if provided, for a test set.
            Supports both series and array data formats.
            """
            # Store out-of-bag (OOB) probabilities and predictions
            self.oob_proba = self.oob_decision_function_
            self.oob_predictions = np.argmax(self.oob_proba, axis=1)
            
            # Use OOB proximities to compute nonconformity scores
            original_prox_method = self.prox_method
            self.prox_method = 'original'
            # self.non_zero_diagonal = True # TODO: SHould be TRUE??


            proximities = self.get_proximities()
            if matrix_type == 'sparse':
                proximities = proximities.toarray()

            # Already scaled 0 - 1
            # proximities = proximities / np.max(proximities, axis = 1)[:, np.newaxis]

            self.nonconformity_scores_original = np.zeros_like(self.y, dtype=float)
            # self.conformity_scores_rfgap = np.ones_like(self.y, dtype=float)

            # Calculate nonconformity scores for each class label
            for label in self.y.unique():
                same_proximities = proximities[:, self.y == label]
                diff_proximities = proximities[:, self.y != label]
                same_k = np.partition(same_proximities, -k, axis=1)[:, -k:]
                diff_k = np.partition(diff_proximities, -k, axis=1)[:, -k:]


                diff_mean = np.mean(diff_k, axis=1)[self.y == label]
                same_mean = np.mean(same_k, axis=1)[self.y == label]
                # same_mean = np.where(same_mean == 0, 0.00001, same_mean)
                same_min_nonzero = np.min(same_mean[same_mean > 0])  # Find the minimum non-zero value
                same_mean = np.where(same_mean == 0, same_min_nonzero, same_mean)
                # print(same_min_nonzero)


                self.nonconformity_scores_original[self.y == label] = (diff_mean / same_mean)


            # TODO: Make consistent with get_nonconformity
            self.conformity_scores_original = np.max(self.nonconformity_scores_original) - self.nonconformity_scores_original
            self.conformity_quantiles_original = np.quantile(self.conformity_scores_original, np.linspace(0, 0.99, 100))
            self.conformity_original_auc, self.conformity_original_accuarcy_drop, self.conformity_original_n_drop = self.accuracy_rejection_curve_area(self.conformity_quantiles_original, self.conformity_scores_original)

            self.prox_method = original_prox_method


        def accuracy_rejection_curve_area(self, quantiles, scores):
            # Get out-of-bag predictions from the RandomForest model
            oob_preds = np.argmax(self.oob_decision_function_, axis=1)
            
            # Calculate dropped proportion and accuracy for each quantile
            n_dropped = np.array([np.sum(scores <= q) / len(scores) for q in quantiles])
            accuracy_drop = np.array([np.mean(self.y[scores >= q] == oob_preds[scores >= q]) for q in quantiles])
            
            # Calculate area under the accuracy-rejection curve
            return np.trapz(accuracy_drop, n_dropped), accuracy_drop, n_dropped

    return RFGAP(prox_method = prox_method, matrix_type = matrix_type, triangular = triangular, **kwargs)
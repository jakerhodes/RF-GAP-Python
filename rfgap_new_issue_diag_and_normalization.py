# Imports
import numpy as np
from scipy import sparse
import pandas as pd

# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn import metrics

from packaging.version import Version as LooseVersion  # Handles python>3.12
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

def RFGAP_Optimized(prediction_type=None, y=None, prox_method='rfgap', 
          matrix_type='sparse', triangular=True,
          non_zero_diagonal=False, normalize=False, force_symmetric=False, **kwargs):
    """
    Factory function to create an optimized Random Forest Proximity object.
    
    This class takes on a random forest predictors (sklearn) and adds methods to 
    construct proximities from the random forest object.

    This new implementation uses Sparse Matrix Algebra (Inverted Indexing) with Gustavson scipy sparse multiplication
    P = QW^T, where Q and W are query and weight sparse matrices. 
    This achieves  O(N*T) complexity instead of the traditional O(N^2) iterative approach (pairwise comparisons).
    
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
        Whether the proximity matrix is filled triangularly in original and oob proximities. (default is True)
    
    non_zero_diagonal : bool
        Only used for RF-GAP proximities. Should the diagonal entries be computed as non-zero? 
        (default is False, as in original RF-GAP definition)
    
    normalize : bool
        Whether to 0-1 normalize the proximities. (default is False)
    
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

    if prediction_type is None and y is None: prediction_type = 'classification'
    if prediction_type is None and y is not None:
        if isinstance(y, pd.Series): y_array = y.to_numpy()
        else: y_array = np.array(y)
        try:
            if np.issubdtype(y_array.dtype, np.floating): prediction_type = 'regression'
            else: prediction_type = 'classification'
        except TypeError: prediction_type = 'classification'

    if prediction_type == 'classification': rf = RandomForestClassifier
    elif prediction_type == 'regression': rf = RandomForestRegressor

    class RFGAP(rf):
        def __init__(self, prox_method=prox_method, matrix_type=matrix_type, triangular=triangular,
                     non_zero_diagonal=non_zero_diagonal, normalize=normalize, force_symmetric=force_symmetric,
                     **kwargs):
            super(RFGAP, self).__init__(**kwargs)
            
            # BLOCK OOB METHOD
            # The OOB method mathematically requires an N x N denominator matrix 
            # (intersection of OOB status), which breaks sparsity and is O(N^2).
            if prox_method == 'oob':
                raise NotImplementedError(
                    "The 'oob' method is O(N^2) dense and incompatible with this new"
                    "sparse-matrix optimization. Use 'rfgap' (recommended) or 'original'."
                )
            
            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.triangular = triangular
            self.non_zero_diagonal = non_zero_diagonal
            self.normalize = normalize
            self.force_symmetric = force_symmetric
            
            # Internal Cache
            self.W_mat = None   # The "Target" weights matrix (Right side of dot product)
            self._leaf_offsets = None 
            self._total_unique_nodes = 0

        def fit(self, X, y, sample_weight=None):
            """
            Fits the Random Forest and pre-computes the sparse weight matrix W necesssary for proximity calculations.
            Complexity: O(N * T) 

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
            self.y = y
            
            # ---------------------------------------------------------
            # COMPLEXITY STEP 1: Forest Pass -> O(N * T)
            # ---------------------------------------------------------
            # We pass N samples through T trees to get leaf indices.
            # Shape: (N_samples, N_trees)
            self.leaf_matrix = self.apply(X) # LOCAL IDS. Tree 1: Node 0,1,2... Tree 2: Node 0,1,2...
            
            # Calculate offsets to flatten (Tree, Leaf) -> Global Feature ID
            # This allows us to treat every node in the forest as a unique feature column.
            n_leaves_per_tree = [t.tree_.node_count for t in self.estimators_]
            self._leaf_offsets = np.concatenate(([0], np.cumsum(n_leaves_per_tree)[:-1]))  # List of size T showing starting index of each tree's leaves, e.g. [0, 10, 25, 40, ...]
            self._total_unique_nodes = np.sum(n_leaves_per_tree)  # Total unique nodes across all trees (typically NlogN scale)

            # ---------------------------------------------------------
            # COMPLEXITY STEP 2: Statistics Calculation -> O(N * T)
            # ---------------------------------------------------------
            if self.prox_method == 'rfgap':
                # Calculate c_j(t): Multiplicity of sample j in tree t
                self.c_j_t = self.get_in_bag_counts(X)
                
                # Calculate S_i: Set of OOB trees for sample i
                self.oob_indices = self.get_oob_indices(X) 

            # ---------------------------------------------------------
            # COMPLEXITY STEP 3: Build Sparse Weights -> O(N * T)
            # ---------------------------------------------------------
            # We build the W matrix immediately after fit.
            self._build_W_matrix()

            return self

        def get_proximities(self):
            """
            This method produces a proximity matrix for the random forest object.
            Computes the proximity matrix P = Q . W^T using sparse matrix multiplication.
            
            Returns
            -------
            array-like
                (if self.matrix_type == 'dense') matrix of pair-wise proximities
            
            csr_matrix
                (if self.matrix_type == 'sparse') a sparse crs_matrix of pair-wise proximities
            
            """
            check_is_fitted(self)

            # Build Query Matrix Q (Left side of dot product)
            #    Represents the 'i' term in p(i,j). 
            #    Complexity: O(N * T)
            Q = self._build_Q_matrix(is_training=True)
            
            # ---------------------------------------------------------
            # COMPLEXITY STEP 4: Sparse Matrix Multiplication
            # ---------------------------------------------------------
            # Operation: P = Q . W^T
            #
            # In a sparse world, cost is proportional to Number of Non-Zeros (nnz).
            # Since each row has exactly T non-zeros (one leaf per tree), nnz = N * T
            # This effectively makes the operation linear O(k * N) instead of O(N^2).
            prox_matrix = Q.dot(self.W_mat.T)  # this uses Gustavson's algorithm internally, which is very a efficient sparse multiplication

            # Diagonal adjustment for RF-GAP
            if self.non_zero_diagonal and self.prox_method == 'rfgap':
                # The Summation:
                # Since W contains c_i(t)/M_i(t) for in-bag trees (and 0 otherwise),
                # the row sum of W is exactly the summation part of the formula.
                # sparse.sum returns a numpy matrix, so we flatten it to an array.
                numerator = np.array(self.W_mat.sum(axis=1)).flatten()

                # The Denominator (|S_bar_i|):
                # Count of trees where sample i was In-Bag.
                # Since oob_indices is 1 for OOB and 0 for In-Bag:
                n_trees = self.n_estimators
                denominator = n_trees - self.oob_indices.sum(axis=1)

                # Calculate & Set
                # Handle potential division by zero if a sample is never in-bag (unlikely but possible)
                with np.errstate(divide='ignore', invalid='ignore'):
                    diag_values = numerator / denominator
                diag_values[~np.isfinite(diag_values)] = 0.0 # Fallback
                
                prox_matrix.setdiag(diag_values)
                
                # Optional Normalization
                if self.normalize:
                    max_vals = np.array(prox_matrix.max(axis=1).todense()).flatten()
                    max_vals[max_vals == 0] = 1
                    r, _ = prox_matrix.nonzero()
                    prox_matrix.data /= max_vals[r]
                    prox_matrix.setdiag(1.0)

            if self.force_symmetric and self.prox_method == 'rfgap':
                prox_matrix = (prox_matrix + prox_matrix.T) / 2


            return prox_matrix.todense() if self.matrix_type == 'dense' else prox_matrix

        def prox_extend(self, X_new, training_indices=None):
            """
            Calculates proximities between New Data (rows) and Training Data (cols) with the optimized Query-Weight matrix method (sparse)
            
            Parameters
            ----------
            X_new : (n_samples, n_features) array_like (numeric)
                New observations (out-of-sample) for which to compute proximities.
            training_indices : array-like
                Indices of training observations to compute proximities for. Default is None, which uses all training observations.
            
            Returns
            -------
            array-like or csr_matrix
                Pair-wise proximities between the specified training data and new observations.
            """
            check_is_fitted(self)
            
            # Build Query Matrix Q for NEW data (X_new)
            Q_new = self._build_Q_matrix(X_query=X_new, is_training=False)
            
            # Select Target Weights (W)
            if training_indices is not None:
                W_target = self.W_mat[training_indices]
            else:
                W_target = self.W_mat
                
            # Compute Dot Product
            prox_matrix = Q_new.dot(W_target.T)
            
            return prox_matrix.todense() if self.matrix_type == 'dense' else prox_matrix

        # -------------------------------------------------------
        # MATRIX BUILDERS (Mapping to Definitions)
        # -------------------------------------------------------

        def _to_global_leaves(self, leaf_mat):
            """Offset local leaf IDs to global feature IDs."""
            return leaf_mat + self._leaf_offsets

        def _build_W_matrix(self):
            """
            Builds the Weight Matrix 'W' (N_samples x N_total_nodes).
            This matrix handles the 'j' term (target) in the definitions.
            """
            N, T = self.leaf_matrix.shape
            global_leaves = self._to_global_leaves(self.leaf_matrix)
            
            # Flatten indices for sparse construction -> O(N * T)
            flat_rows = np.repeat(np.arange(N), T)
            flat_cols = global_leaves.flatten()
            
            # --- ORIGINAL PROXIMITY ---
            # p(i,j) = (1/T) * Sum[ I(j in v_i(t)) ]
            #
            # Mapping: W handles the (1/T) term.
            if self.prox_method == 'original':
                weights = np.full(N * T, 1.0 / T, dtype=np.float32)
                
            # --- RF-GAP PROXIMITY ---
            # Term inside Sum: c_j(t) / M_i(t)
            #
            # Mapping:
            #   c_j(t) -> self.c_j_t (Numerator)
            #   M_i(t) -> Total weight of the node (Denominator)
            elif self.prox_method == 'rfgap':
                
                # Get c_j(t) [Multiplicity of j in tree t]
                c_j_t = self.c_j_t.flatten()
                
                # Calculate M_i(t) [Total weight of node]
                #    The node mass is the sum of c_j(t) for all samples falling in that node.
                #    bincount is O(N * T)
                M_node_weights = np.bincount(flat_cols, weights=c_j_t, minlength=self._total_unique_nodes)
                
                # Calculate 1 / M_i(t) safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    inverse_M_node = 1.0 / M_node_weights
                inverse_M_node[~np.isfinite(inverse_M_node)] = 0
                
                # Combine: W_val = c_j(t) * (1 / M_node)
                weights = c_j_t * inverse_M_node[flat_cols]
                
            # Construct Sparse Matrix W -> O(N * T)
            self.W_mat = sparse.csr_matrix(
                (weights, (flat_rows, flat_cols)), 
                shape=(N, self._total_unique_nodes)     # shape: N x N_total_leaves (huge but SPARSE)
            )

        def _build_Q_matrix(self, X_query=None, is_training=True):
            """
            Builds the Query Matrix 'Q' (N_query x N_total_nodes).
            This matrix handles the 'i' term and the Summation scope (S_i).
            """
            if is_training:
                leaf_mat = self.leaf_matrix
                # S_i logic: For RFGAP, we only sum over trees where i is OOB.
                oob_mask = self.oob_indices if self.prox_method == 'rfgap' else None
            else:
                leaf_mat = self.apply(X_query)
                # For new data, the sample was not in ANY bag, so it is OOB for all trees.
                oob_mask = np.ones_like(leaf_mat) 

            N, T = leaf_mat.shape
            global_leaves = self._to_global_leaves(leaf_mat)
            
            flat_rows = np.repeat(np.arange(N), T)
            flat_cols = global_leaves.flatten()

            # --- ORIGINAL PROXIMITY ---
            # p(i,j) = Sum[ ... ]  (Sum over all t=1 to T)
            #
            # W already contains (1/T). Q simply indicates existence (1.0).
            if self.prox_method == 'original':
                vals = np.ones(N * T, dtype=np.float32)
                
            # --- RF-GAP PROXIMITY ---
            # p(i,j) = (1 / |S_i|) * Sum_{t in S_i} [ ... ]
            #
            # Q handles:
            # 1. The Summation Scope (t in S_i): Mask out non-OOB trees.
            # 2. The Normalization (1 / |S_i|).
            elif self.prox_method == 'rfgap':
                
                if is_training:
                    # Apply S_i Scope: Keep only OOB trees
                    mask = oob_mask.flatten() == 1
                    flat_rows = flat_rows[mask]
                    flat_cols = flat_cols[mask]
                    
                    # Calculate |S_i|: Count of OOB trees per sample
                    S_i_counts = oob_mask.sum(axis=1)
                    S_i_counts[S_i_counts == 0] = 1 # Avoid div/0
                    
                    # Q_val = 1 / |S_i|
                    vals = (1.0 / S_i_counts[flat_rows]).astype(np.float32)
                    
                else:
                    # For new data, S_i is the set of ALL trees (size T).
                    vals = np.full(N * T, 1.0 / T, dtype=np.float32)
            
            return sparse.csr_matrix(
                (vals, (flat_rows, flat_cols)), 
                shape=(N, self._total_unique_nodes)  # shape: N_query x N_total_leaves (huge but SPARSE)
            )


        def _get_oob_samples(self, data):
            """
            This is a helper function for get_oob_indices. 
            
            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)
            
            """
            n = len(data)
            return [_generate_unsampled_indices(t.random_state, n, n) for t in self.estimators_]

        def get_oob_indices(self, data):

            """
            This generates a matrix of out-of-bag samples for each decision tree in the forest
            
            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)
            
            
            Returns
            -------
            oob_matrix : array_like (n_samples, n_estimators) 
            
            """
            n = len(data)
            oob_matrix = np.zeros((n, self.n_estimators))
            oob_samples = self._get_oob_samples(data)
            for t in range(self.n_estimators): 
                oob_matrix[np.unique(oob_samples[t]), t] = 1
            return oob_matrix.astype(int)

        def _get_in_bag_samples(self, data):
            """
            This is a helper function for get_in_bag_indices. 
            
            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)
            
            """
            n = len(data)
            return [_generate_sample_indices(t.random_state, n, n) for t in self.estimators_]

        def get_in_bag_counts(self, data):
            """
            This generates a matrix of in-bag samples for each decision tree in the forest
            
            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)
            
            
            Returns
            -------
            in_bag_matrix : array_like (n_samples, n_estimators) 
            
            """
            n = len(data)
            in_bag_matrix = np.zeros((n, self.n_estimators))
            in_bag_samples = self._get_in_bag_samples(data)
            for t in range(self.n_estimators):
                matches, n_repeats = np.unique(in_bag_samples[t], return_counts=True)
                in_bag_matrix[matches, t] += n_repeats
            return in_bag_matrix
        
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
                    # Get predictions for the test set
                    self.test_preds = self.predict(x_test)
    
                    # Get the proximities between test samples and training samples (shape n_test, n_train)
                    proximities_test = self.prox_extend(x_test)
    
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
            try:
                y_arr = y.to_numpy()
            except:
                y_arr = np.asarray(y)
                
            n_samples = len(y_arr)
    
            non_zero_diagonal = self.non_zero_diagonal
    
            is_symmetric = self.force_symmetric
    
            if non_zero_diagonal:
                self.non_zero_diagonal = False
                if is_symmetric:
                    self.force_symmetric = False
    
            proximities = self.get_proximities()
            proximities = proximities.toarray() if isinstance(proximities, sparse.csr_matrix) else proximities
    
            self.non_zero_diagonal = non_zero_diagonal
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
    
    return RFGAP(prox_method = prox_method, matrix_type = matrix_type, triangular=triangular, non_zero_diagonal = non_zero_diagonal, normalize=normalize, force_symmetric = force_symmetric, **kwargs)

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

from joblib import Parallel, delayed, effective_n_jobs



def RFGAP(prediction_type = None, y = None, prox_method = 'rfgap', 
          matrix_type = 'sparse', triangular = True,
          non_zero_diagonal = False, normalize=False, force_symmetric = False, batch_size = "auto", **kwargs):
    """
    A factory method to conditionally create the RFGAP class based on RandomForestClassifier or RandomForestRegressor (depending on the type of response, y)

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
        Whether the proximity matrix is filled triangularly in original and oob proximities. (default is True)

    non_zero_diagonal : bool
        Only used for RF-GAP proximities. Should the diagonal entries be computed as non-zero? 
        (default is False, as in original RF-GAP definition)

    normalize : bool
        Whether to 0-1 normalize the proximities. (default is False)

    force_symmetric : bool
        Enforce symmetry of proximities. (default is False)
    
    batch_size : int or 'auto'
        The size of batches to use when computing proximities. If set to 'auto', the batch size
        will be determined heuristically based on the number of samples. (default is 'auto')

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

        def __init__(self, prox_method = prox_method, matrix_type = matrix_type, triangular = triangular,
                     non_zero_diagonal = non_zero_diagonal, normalize = normalize, force_symmetric = force_symmetric,
                     batch_size = batch_size, **kwargs):
            super(RFGAP, self).__init__(**kwargs)

            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.triangular = triangular
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal
            self.normalize = normalize
            self.min_non_zero_diagonal = None
            self.force_symmetric = force_symmetric
            self.batch_size = batch_size


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
            
            x_test : {array-like, sparse matrix} of shape (n_test_samples, n_features), default=None
                Optional test input samples to generate full proximity matrix between both labeled (X) and unlabeled (x_test) data.
                Only available for `original` and `oob` proximity methods. 'rfgap' proximity method only supports proximities to points used to train the underlying RF model.
                Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

            Returns
            -------
            self : object
                Fitted estimator.

            """
            super().fit(X, y, sample_weight)

            self.n_jobs_ = effective_n_jobs(self.n_jobs)  # Get effective n_jobs for parallelization

            # TODO: Check y type; make sure works with the rest of code. Works well for proximities, but nonconformity scores may have issues.
            # Refer to demo notebook on Iris dataset with string labels.
            self.y = y
            self.n = len(y)
            self.leaf_matrix = self.apply(X) # (n_train, T)

            # Store n_test count
            self.n_test_ = 0
            if x_test is not None:
                self.n_test_ = np.shape(x_test)[0]

            # For original and oob, we concatenate matrices if x_test is provided.
            # For rfgap, we *ignore* x_test here and only build train-only matrices.
            if self.prox_method == 'original':
                if x_test is not None:
                    self.leaf_matrix_test = self.apply(x_test)
                    self.leaf_matrix = np.concatenate((self.leaf_matrix, self.leaf_matrix_test), axis = 0)
            
            elif self.prox_method == 'oob':
                self.oob_indices = self.get_oob_indices(X)  # (n_train, T)
                if x_test is not None:
                    self.leaf_matrix_test = self.apply(x_test)
                    self.leaf_matrix = np.concatenate((self.leaf_matrix, self.leaf_matrix_test), axis=0)
                    self.oob_indices = np.concatenate(
                        (self.oob_indices, np.ones((self.n_test_, self.n_estimators), dtype=int)),
                        axis=0)  # Append test set info. Test samples are always OOB (1s)
                self.oob_leaves = self.oob_indices * self.leaf_matrix
                
            elif self.prox_method == 'rfgap':
                # RF-GAP logic *only* builds (n_train, T) matrices.
                # The x_test argument is ignored for this prox_method in fit().
                self.oob_indices = self.get_oob_indices(X)  # Get OOB status (n_samples, n_trees)
                self.in_bag_counts = self.get_in_bag_counts(X)  # Get in-bag counts (M matrix) (n_samples, n_trees)
                self.in_bag_indices = 1 - self.oob_indices  # In-bag status is the inverse of OOB
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
            
            Fully sparse version: returns only (data, rows, cols) for row 'ind'.
        
            All computations follow the exact same math as old code,
            but without allocating dense prox_vec arrays.

            Parameters
            ----------
            ind : int
                Index of the observation for which to compute the proximity vector.

            Returns
            -------
            data : list
                Proximity values for the given observation.
            rows : list
                Row indices corresponding to the proximity values.
            cols : list
                Column indices corresponding to the proximity values.
            """
            check_is_fitted(self)
            _, T = self.leaf_matrix.shape
            method = self.prox_method
        
            # ORIGINAL proximities
            if method == "original":
        
                tree_inds = self.leaf_matrix[ind, :]
        
                if self.triangular:
                    comp = (tree_inds == self.leaf_matrix[ind:, :])
                    sim = comp.sum(axis=1)
                    nz = np.where(sim != 0)[0]
                    cols = nz + ind
                    data = sim[nz] / T
                    rows = np.full(len(cols), ind, dtype=int)
        
                else:
                    comp = (tree_inds == self.leaf_matrix)
                    sim = comp.sum(axis=1)
                    cols = np.nonzero(sim)[0]
                    data = sim[cols] / T
                    rows = np.full(len(cols), ind, dtype=int)
        
                return data.tolist(), rows.tolist(), cols.tolist()
        
            # OOB proximities
            if method == "oob":
        
                ind_oob = np.nonzero(self.oob_leaves[ind])[0]
                if ind_oob.size == 0:
                    return [], [], []
        
                if self.triangular:
                    comp_leaves = self.oob_leaves[ind:, ind_oob]
                    comp_oob    = self.oob_indices[ind:, ind_oob]
                else:
                    comp_leaves = self.oob_leaves[:, ind_oob]
                    comp_oob    = self.oob_indices[:, ind_oob]
        
                tree_counts = (self.oob_indices[ind, ind_oob] == comp_oob).sum(axis=1)
                tree_counts[tree_counts == 0] = 1
        
                prox_counts = (self.oob_leaves[ind, ind_oob] == comp_leaves).sum(axis=1)
                prox_vec = prox_counts / tree_counts
        
                nz = np.nonzero(prox_vec)[0]
        
                if self.triangular:
                    cols = nz + ind
                else:
                    cols = nz
        
                data = prox_vec[nz]
                rows = np.full(len(cols), ind, dtype=int)
                return data.tolist(), rows.tolist(), cols.tolist()
        
            # RFGAP proximities  (full sparse)
            # exact same math as original — but sparse only
            oob_trees    = np.nonzero(self.oob_indices[ind])[0]
            in_bag_trees = np.nonzero(self.in_bag_indices[ind])[0]
        
            terminals = self.leaf_matrix[ind]
            matches = (terminals == self.in_bag_leaves)
        
            match_counts = np.where(matches, self.in_bag_counts, 0)
        
            ks = match_counts.sum(axis=0)
            ks[ks == 0] = 1
        
            ks_out = ks[oob_trees]
            S_out  = len(oob_trees) if len(oob_trees) > 0 else 1
        
            prox_vec = (match_counts[:, oob_trees] / ks_out).sum(axis=1) / S_out
        
            # diagonal adjustments
            if self.non_zero_diagonal:
                ks_in = ks[in_bag_trees]
                S_in  = len(in_bag_trees)
        
                if S_in > 0:
                    prox_vec[ind] = (match_counts[ind, in_bag_trees] / ks_in).sum() / S_in
                else:
                    prox_vec[ind] = 0
                
                if self.normalize:
                    # Store the *minimum* self-similarity across training points
                    if self.min_non_zero_diagonal is None:
                        self.min_non_zero_diagonal = prox_vec[ind]
                    else:
                        self.min_non_zero_diagonal = min(self.min_non_zero_diagonal, prox_vec[ind])
                    
                    # Normalize
                    maxv = prox_vec.max()
                    if maxv > 0:
                        prox_vec = prox_vec / maxv
                    prox_vec[ind] = 1
        
            cols = np.nonzero(prox_vec)[0]
            rows = np.full(len(cols), ind, dtype=int)
            data = prox_vec[cols]
        
            return data.tolist(), rows.tolist(), cols.tolist()
        

        def _run_batched_parallel(self, n_rows, row_fn, out_shape, label="[Batch]"):
            """
            Shared batching logic for get_proximities() and prox_extend().
        
            Parameters
            ----------
            n_rows : int
                Number of rows to compute (n or n_ext).
        
            row_fn : callable
                Function that takes a single row index and returns (data, rows, cols).
        
            out_shape : tuple
                Shape of final sparse matrix.
        
            label : str
                Label for printing progress.
        
            Returns
            -------
            csr_matrix of shape out_shape
            """

            if self.batch_size == "auto":
                # "auto" heuristic to target ~50 batches, but not too small or too large (100<B<2000)
                B = int(max(100, min(2000, n_rows // 50)))
                # Handle n_rows < 100 case
                if n_rows <= 100: 
                    B = n_rows 
                if B == 0: # Handle n_rows == 0
                    B = 1 
            else:
                # Use the user-provided value
                B = int(self.batch_size)
            
            n_batches = (n_rows + B - 1) // B 
        
            if self.verbose:
                print(f"{label}, batch_size={B}, n_rows={n_rows}, n_batches={n_batches}")
        
            blocks = []
        
            if n_batches == 0:
                return sparse.csr_matrix(out_shape) # Handle empty input
        
            for b in range(n_batches):
                start = b * B
                end   = min((b + 1) * B, n_rows)
                idxs  = range(start, end)
        
                if self.verbose:
                    print(f"{label} {b+1}/{n_batches} rows {start}→{end}")
        
                # Parallel per-row computation
                results = Parallel(
                    n_jobs=self.n_jobs,
                    batch_size=1,
                    prefer="threads"
                )(delayed(row_fn)(i) for i in idxs)
        
                # Merge batch into CSR block
                data, rows, cols = [], [], []
                for d, r, c in results:
                    data.extend(d)
                    rows.extend(r)
                    cols.extend(c)
        
                block = sparse.csr_matrix(
                    (np.array(data, dtype=np.float32),
                     (np.array(rows, dtype=np.int32),
                      np.array(cols, dtype=np.int32))),
                    shape=out_shape
                )
        
                blocks.append(block)
        
                del data, rows, cols, results
        
            return sum(blocks)
        

        
        def get_proximities(self):
            """This method produces a proximity matrix for the random forest object.
            
            
            Returns
            -------
            array-like
                (if self.matrix_type == 'dense') matrix of pair-wise proximities
            
            csr_matrix
                (if self.matrix_type == 'sparse') a sparse crs_matrix of pair-wise proximities
            
            """
            check_is_fitted(self)
            n, _ = self.leaf_matrix.shape
        
            # shared batch logic
            prox_sparse = self._run_batched_parallel(
                n_rows=n,
                row_fn=self.get_proximity_vector,
                out_shape=(n, n),
                label="[get_proximities]"
            )
        
            # Post-process symmetry
            if self.triangular and self.prox_method != "rfgap":
                prox_sparse = prox_sparse + prox_sparse.T
                prox_sparse.setdiag(1.0)
        
            if self.prox_method == "rfgap" and self.force_symmetric:
                prox_sparse = (prox_sparse + prox_sparse.T) / 2
        
            return prox_sparse.todense() if self.matrix_type == "dense" else prox_sparse



        def prox_extend(self, X_new, training_indices=None):
            """Compute proximities between specified training indices and new observations.
            
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
        
            leaf_train = self.leaf_matrix
            n_train, T = leaf_train.shape
        
            leaf_new = self.apply(X_new)
            n_ext = leaf_new.shape[0]
        
            if training_indices is None:
                training_indices = np.arange(n_train)
            training_indices = np.asarray(training_indices)
            n_tr = len(training_indices)
        
            # Pre-slice data only once for efficiency
            if self.prox_method == "original":
                leaf_tr = leaf_train[training_indices]
        
            elif self.prox_method == "oob":
                leaf_tr = leaf_train[training_indices]
                oob_tr_bool = self.oob_indices[training_indices].astype(bool)
        
            elif self.prox_method == "rfgap":
                in_bag_leaves_tr = self.in_bag_leaves[training_indices]
                in_bag_counts_tr = self.in_bag_counts[training_indices]
                oob_trees_tr = np.nonzero(self.oob_indices[training_indices].sum(axis=0))[0]
                S_out_tr = max(1, len(oob_trees_tr))
        
            # Per-row function
            def get_proximity_vector_extend(ext_i):
                leaves_i = leaf_new[ext_i]
        
                if self.prox_method == "original":
                    comp = (leaves_i == leaf_tr)
                    sim = comp.sum(axis=1)
                    nz = np.nonzero(sim)[0]
                    data = (sim[nz] / T).astype(np.float32)
                    rows = np.full(len(nz), ext_i, dtype=np.int32)
                    return data, rows, nz
        
                if self.prox_method == "oob":
                    match = (leaves_i == leaf_tr)
                    oob_new = (leaves_i != 0)
                    prox_counts = (match & oob_tr_bool).sum(axis=1)
                    tree_counts = (oob_tr_bool & oob_new).sum(axis=1)
                    tree_counts[tree_counts == 0] = 1
                    prox_vec = prox_counts / tree_counts
                    nz = np.nonzero(prox_vec)[0]
                    data = prox_vec[nz].astype(np.float32)
                    rows = np.full(len(nz), ext_i, dtype=np.int32)
                    return data, rows, nz
        
                # RFGAP
                matches = (leaves_i == in_bag_leaves_tr)
                match_counts = np.where(matches, in_bag_counts_tr, 0)
                ks = match_counts.sum(axis=0)
                ks[ks == 0] = 1
                prox_vec = (match_counts[:, oob_trees_tr] / ks[oob_trees_tr]).sum(axis=1) / S_out_tr
                if self.non_zero_diagonal and self.normalize:
                    # Normalize
                    maxv = max(self.min_non_zero_diagonal, prox_vec.max())
                    if maxv > 0:
                        prox_vec = prox_vec / maxv
                nz = np.nonzero(prox_vec)[0]
                data = prox_vec[nz].astype(np.float32)
                rows = np.full(len(nz), ext_i, dtype=np.int32)
                return data, rows, nz
        
            # Shared batching process
            prox_sparse = self._run_batched_parallel(
                n_rows=n_ext,
                row_fn=get_proximity_vector_extend,
                out_shape=(n_ext, n_tr),
                label="[prox_extend]"
            )
        
            return prox_sparse.todense() if self.matrix_type == "dense" else prox_sparse
        

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




    return RFGAP(prox_method = prox_method, matrix_type = matrix_type, triangular=triangular, non_zero_diagonal = non_zero_diagonal, normalize=normalize, force_symmetric = force_symmetric, batch_size=batch_size, **kwargs)

# Imports
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform, cdist
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

# -----------------------------------------------------------------
# --- START NEW PARALLEL HELPER FUNCTIONS ---
# These must be at the module level (outside the class) for joblib
# -----------------------------------------------------------------


def _get_prox_tree_chunk_oob(t, leaf_matrix, oob_indices):
    """Helper function to compute 'oob' proximities for a single tree."""
    rows_t, cols_t = [], []
    n, T = leaf_matrix.shape
    
    oob_indices_t = np.where(oob_indices[:, t] == 1)[0]
    if oob_indices_t.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), None

    leaves_t = leaf_matrix[oob_indices_t, t]
    
    unique_leaves = np.unique(leaves_t)
    for leaf_val in unique_leaves:
        indices_in_leaf = oob_indices_t[np.where(leaves_t == leaf_val)[0]]
        
        if indices_in_leaf.size == 0:
            continue
            
        i_coords, j_coords = np.meshgrid(indices_in_leaf, indices_in_leaf)
        
        rows_t.append(i_coords.ravel())
        cols_t.append(j_coords.ravel())

    if not rows_t:
        return np.array([], dtype=int), np.array([], dtype=int), None

    return np.concatenate(rows_t), np.concatenate(cols_t), None

def _get_prox_tree_chunk_rfgap(t, leaf_matrix, oob_indices, in_bag_counts, K_matrix):
    """Helper function to compute 'rfgap' proximities for a single tree."""
    data_t, rows_t, cols_t = [], [], []
    n, T = leaf_matrix.shape

    leaves_t = leaf_matrix[:, t]
    K_t = K_matrix[:, t]
    in_bag_counts_t = in_bag_counts[:, t]
    unique_leaves_t = np.unique(leaves_t)
    
    for leaf_val in unique_leaves_t:
        i_mask = (leaves_t == leaf_val) & (oob_indices[:, t] == 1)
        i_in_leaf = np.where(i_mask)[0]
        j_mask = (leaves_t == leaf_val) & (in_bag_counts_t > 0)
        j_in_leaf = np.where(j_mask)[0]
        
        if i_in_leaf.size == 0 or j_in_leaf.size == 0:
            continue
            
        i_coords, j_coords = np.meshgrid(i_in_leaf, j_in_leaf)
        i_flat = i_coords.ravel()
        j_flat = j_coords.ravel()
        
        M_values = in_bag_counts_t[j_flat]
        K_values = K_t[i_flat]
        data_for_this_leaf = M_values / K_values
        
        rows_t.append(i_flat)
        cols_t.append(j_flat)
        data_t.append(data_for_this_leaf)
    
    if not rows_t:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32)

    return np.concatenate(rows_t), np.concatenate(cols_t), np.concatenate(data_t)

def _get_prox_extend_tree_chunk_oob(t, extended_leaf_matrix, train_leaves_subset, train_oob_subset):
    """Helper function to compute 'oob' extended proximities for a single tree."""
    rows_t, cols_t = [], []
    
    train_oob_t = train_oob_subset[:, t]
    train_indices_oob_t = np.where(train_oob_t == 1)[0] # Indices into subset
    
    if train_indices_oob_t.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), None
        
    train_leaves_oob_t = train_leaves_subset[train_indices_oob_t, t]
    test_leaves_t = extended_leaf_matrix[:, t]

    unique_leaves = np.unique(train_leaves_oob_t)
    for leaf_val in unique_leaves:
        i_in_leaf = np.where(test_leaves_t == leaf_val)[0]
        j_in_leaf_subset = np.where(train_leaves_oob_t == leaf_val)[0]
        j_in_leaf = train_indices_oob_t[j_in_leaf_subset]
        
        if i_in_leaf.size == 0 or j_in_leaf.size == 0:
            continue
            
        i_coords, j_coords = np.meshgrid(i_in_leaf, j_in_leaf)
        rows_t.append(i_coords.ravel()) # test index
        cols_t.append(j_coords.ravel()) # train subset index

    if not rows_t:
        return np.array([], dtype=int), np.array([], dtype=int), None
    
    return np.concatenate(rows_t), np.concatenate(cols_t), None

def _get_prox_extend_tree_chunk_rfgap(t, extended_leaf_matrix, train_leaves_subset, K_sub, train_in_bag_counts_subset):
    """Helper function to compute 'rfgap' extended proximities for a single tree."""
    data_t, rows_t, cols_t = [], [], []

    test_leaves_t = extended_leaf_matrix[:, t]
    train_leaves_t = train_leaves_subset[:, t]
    K_t = K_sub[:, t]
    in_bag_counts_t = train_in_bag_counts_subset[:, t]

    unique_leaves_t = np.unique(test_leaves_t)
    for leaf_val in unique_leaves_t:
        i_in_leaf = np.where(test_leaves_t == leaf_val)[0]
        j_mask = (train_leaves_t == leaf_val) & (in_bag_counts_t > 0)
        j_in_leaf = np.where(j_mask)[0]
        
        if i_in_leaf.size == 0 or j_in_leaf.size == 0:
            continue

        i_coords, j_coords = np.meshgrid(i_in_leaf, j_in_leaf)
        i_flat = i_coords.ravel() # test index
        j_flat = j_coords.ravel() # train subset index
        
        M_values = in_bag_counts_t[j_flat]
        K_values = K_t[j_flat]
        
        data_for_this_leaf = M_values / K_values
        
        rows_t.append(i_flat)
        cols_t.append(j_flat)
        data_t.append(data_for_this_leaf)

    if not rows_t:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=np.float32)

    return np.concatenate(rows_t), np.concatenate(cols_t), np.concatenate(data_t)

# -----------------------------------------------------------------
# --- END NEW PARALLEL HELPER FUNCTIONS ---
# -----------------------------------------------------------------




def RFGAP(prediction_type = None, y = None, prox_method = 'rfgap', 
          matrix_type = 'sparse',
          non_zero_diagonal = False, force_symmetric = False, **kwargs):
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

    non_zero_diagonal : bool
        Only used for RF-GAP proximities. Should the diagonal entries be computed as non-zero? 
        (default is False, as in original RF-GAP definition)

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
                     non_zero_diagonal = non_zero_diagonal, force_symmetric = force_symmetric,
                     **kwargs):

            super(RFGAP, self).__init__(**kwargs)

            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal
            self.force_symmetric = force_symmetric


        #TODO: x_test is confusing here. rfgap does not accomodate test-test proximities, so passing x_test
        # results in full zeros for test-test proximities in get_proximities(). This makes sense for other prox_method though.
        # This could be useful for constructing semi-supervised kernel matrix with an adapted RFGAP definition for test-test proximities. To be continued...
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

            self.n_jobs_ = effective_n_jobs(self.n_jobs)  # Get effective n_jobs for parallelization

            # TODO: Check y type; make sure works with the rest of code. Works well for proximities, but nonconformity scores may have issues.
            # Refer to demo notebook on Iris dataset with string labels.
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
                    # Append test set info. Test samples are always OOB (1s)
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                
            
            if self.prox_method == 'rfgap':
            
                # Get OOB status (n_samples, n_trees)
                self.oob_indices = self.get_oob_indices(X)
                # Get in-bag counts (M matrix) (n_samples, n_trees)
                self.in_bag_counts = self.get_in_bag_counts(X)
            
                
                if x_test is not None:
                    # Append test set info. Test samples are always OOB (1s)
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                    # ...and have zero in-bag counts (0s)
                    self.in_bag_counts = np.concatenate((self.in_bag_counts, np.zeros((n_test, self.n_estimators))))                
                                
                # In-bag status is the inverse of OOB
                self.in_bag_indices = 1 - self.oob_indices
                
                if self.verbose:
                    print("Pre-calculating K matrix (in-bag leaf counts)...")
                
                _, T = self.leaf_matrix.shape
                
                # --- START: K-Matrix Calculation (Parallelized) ---
                # This block pre-calculates the K matrix, which is the RF-GAP normalization term.
                #
                # What is K?
                # K_jt: Total in-bag count (sum of M_it) for ALL samples 'i'
                #       that land in the *same leaf* as sample 'j' in tree 't'.
                # Shape: (n_total_samples, n_trees)
                #
                # Why pre-calculate?
                # This is an expensive T-loop. We compute it once during `fit()`
                # so that `get_proximities()` and `prox_extend()` can reuse it.
                # This is critical for the performance of `prox_extend`.
                
                def _get_k_matrix_chunk(t, leaf_matrix, in_bag_counts):
                    """Calculates a single column (K_t) of the K matrix for tree t."""
                    n_total = leaf_matrix.shape[0]
                    leaves_t = leaf_matrix[:, t]  # Leaf IDs for all samples in tree t
                    counts_t = in_bag_counts[:, t]  # In-bag counts (M) for all samples in tree t
                    unique_leaves = np.unique(leaves_t)
                    K_t = np.zeros(n_total, dtype=np.float32) # Initialize column K_t
                    
                    # For each leaf node in the tree
                    for leaf_val in unique_leaves:
                        # Find all samples that landed in this leaf
                        indices = np.where(leaves_t == leaf_val)[0]
                        # Sum the in-bag counts (M) of all those samples
                        total_in_bag_for_this_leaf = np.sum(counts_t[indices])
                        # Assign this total sum to all samples in that leaf
                        K_t[indices] = total_in_bag_for_this_leaf
                    return K_t
            
                # Run the K-matrix column calculation in parallel for each tree
                results = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose)(
                    delayed(_get_k_matrix_chunk)(
                        t, self.leaf_matrix, self.in_bag_counts
                    ) for t in range(T)
                )
                
                # Reconstruct the full (n_total, T) K matrix from the parallel results
                K = np.array(results).T.astype(np.float32)

                # --- END: K-Matrix Calculation (Parallelized) ---
                
                # Avoid division by zero in subsequent proximity calculations
                K[K == 0] = 1.0 

                # Store the final matrix in the class instance
                self.K_matrix = K
            
        
        
        
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
        
        
        def get_proximities(self):
            
            """
            This method produces a proximity matrix for the random forest object.
            
            It uses a hybrid approach:
            - 'original': Uses a highly optimized Hamming distance calculation.
            - 'oob', 'rfgap': Use a fast, parallelized T-loop (looping over trees).
            """
            check_is_fitted(self)
            n, T = self.leaf_matrix.shape
        
        
            # -----------------------------------------------------------------
            # 'original' (Not parallelized, already C-optimized)
            # -----------------------------------------------------------------
            if self.prox_method == 'original':
                if self.verbose:
                    print("Calculating 'original' proximities with Hamming distance...")
                
                hamming_dist_condensed = pdist(self.leaf_matrix, metric='hamming')
                hamming_dist_matrix = squareform(hamming_dist_condensed)
                prox_dense = 1 - hamming_dist_matrix
                np.fill_diagonal(prox_dense, 1)
                prox_sparse = sparse.csr_matrix(prox_dense)
        
            # -----------------------------------------------------------------
            # 'oob' (Parallelized)
            # -----------------------------------------------------------------
            elif self.prox_method == 'oob':
                if self.verbose:
                    print(f"Calculating 'oob' proximities with T-loop (parallelized with n_jobs={self.n_jobs_})...")
                
                # 1. Denominator (Not parallelized, fast matmul)
                if self.verbose:
                    print("Calculating OOB co-occurrence (Denominator)...")
                oob_float = self.oob_indices.astype(np.float32)
                D_dense = oob_float @ oob_float.T
                
                # 2. Numerator (Parallelized T-loop)
                if self.verbose:
                    print("Calculating OOB leaf co-occurrence (Numerator)...")
                
                # --- START: PARALLEL MODIFICATION ---
                results = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose)(
                    delayed(_get_prox_tree_chunk_oob)(
                        t, self.leaf_matrix, self.oob_indices
                    ) for t in range(T)
                )
                
                # Reduce step: concatenate all results
                rows_all_list = [r[0] for r in results if r[0] is not None]
                cols_all_list = [r[1] for r in results if r[1] is not None]
                
                if not rows_all_list:
                    N_sparse = sparse.csr_matrix((n, n), dtype=np.float32)
                else:
                    rows_all = np.concatenate(rows_all_list)
                    cols_all = np.concatenate(cols_all_list)
                    data_all = np.ones(len(rows_all), dtype=np.float32)
                    N_sparse = sparse.csr_matrix((data_all, (rows_all, cols_all)), shape=(n, n))
                # --- END: PARALLEL MODIFICATION ---
        
                # 3. Final Division
                if self.verbose:
                    print("Finalizing proximity matrix (N / D)...")
                
                N_dense = N_sparse.todense()
                prox_dense = np.divide(N_dense, D_dense, 
                                       out=np.zeros_like(N_dense), 
                                       where=D_dense!=0)
                np.fill_diagonal(prox_dense, 1.0)
                prox_sparse = sparse.csr_matrix(prox_dense)
        
        
            # -----------------------------------------------------------------
            # 'rfgap' (Parallelized)
            # -----------------------------------------------------------------
            elif self.prox_method == 'rfgap':

                if self.verbose:
                    print(f"Calculating 'rfgap' proximities with T-loop (parallelized with n_jobs={self.n_jobs_})...")

                K = self.K_matrix
                S_out_i = np.sum(self.oob_indices, axis=1)
                S_out_i_safe = S_out_i.copy()
                S_out_i_safe[S_out_i_safe == 0] = 1

                # --- START: PARALLEL CALCULATION of prox_matrix_sum ---
                if self.verbose:
                    print("Calculating asymmetric proximities (T-loop list extend)...")

                results = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose)(
                    delayed(_get_prox_tree_chunk_rfgap)(
                        t, self.leaf_matrix, self.oob_indices, self.in_bag_counts, self.K_matrix
                    ) for t in range(T)
                )

                rows_list = [r[0] for r in results if r[0] is not None]
                cols_list = [r[1] for r in results if r[1] is not None]
                data_list = [r[2] for r in results if r[2] is not None]

                if not data_list:
                     prox_matrix_sum = sparse.csr_matrix((n, n), dtype=np.float32)
                else:
                    rows = np.concatenate(rows_list)
                    cols = np.concatenate(cols_list)
                    data = np.concatenate(data_list)
                    prox_matrix_sum = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
                # --- END: PARALLEL CALCULATION ---

                # Calculate the raw asymmetric proximities
                inv_S_out_i_scaler = sparse.diags(1.0 / S_out_i_safe, format='csr')
                prox_sparse = inv_S_out_i_scaler @ prox_matrix_sum # Initialize prox_sparse here

                if self.non_zero_diagonal:
                    if self.verbose:
                        print("Calculating 'rfgap' self-similarity (diagonal)...")

                    # Calculate diagonal (self-similarity)
                    S_in_i = np.sum(self.in_bag_indices, axis=1)
                    S_in_i_safe = S_in_i.copy()
                    S_in_i_safe[S_in_i_safe == 0] = 1
                    M_ii_t = self.in_bag_counts
                    K_diag = K[:M_ii_t.shape[0], :]
                    Ratios_ii = M_ii_t / K_diag
                    Terms_ii = Ratios_ii * self.in_bag_indices
                    Diagonal_sum = np.sum(Terms_ii, axis=1)
                    Diagonal_vec = Diagonal_sum / S_in_i_safe
                    diag_matrix = sparse.diags(Diagonal_vec, format='csr')
                    prox_sparse = prox_sparse + diag_matrix # Add diagonal directly

            else:
                 raise ValueError(f"Proximity method '{self.prox_method}' not recognized.")

            if self.force_symmetric:  # RFGAP is asymmetric, others are symmetric by definition
                prox_sparse = (prox_sparse + prox_sparse.transpose()) / 2

            if self.matrix_type == 'dense':
                return np.array(prox_sparse.todense())
            else:
                return prox_sparse



        def prox_extend(self, data, training_indices=None):
            """
            Compute proximities between new test data and specified training data.
            
            This method uses the most efficient strategy for each proximity type:
            - 'original': Fully vectorized Hamming distance calculation (cdist).
            - 'oob', 'rfgap': Fast T-loop (looping over trees) to avoid slow
                n-loops over samples.
            """
            check_is_fitted(self)
            n_train_total, num_trees = self.leaf_matrix.shape
        
            if training_indices is None:
                training_indices = np.arange(n_train_total)
            
            n_sub = len(training_indices)
            
            extended_leaf_matrix = self.apply(data)
            n_ext, _ = extended_leaf_matrix.shape
            
            train_leaves_subset = self.leaf_matrix[training_indices, :]
            
            prox_sparse = None
        
            # -----------------------------------------------------------------
            # 'original' (Not parallelized, already C-optimized)
            # -----------------------------------------------------------------
            if self.prox_method == 'original':
                if self.verbose:
                    print("Calculating 'original' extended proximities with Hamming cdist...")
                
                hamming_dist_matrix = cdist(extended_leaf_matrix, train_leaves_subset, metric='hamming')
                prox_dense = 1 - hamming_dist_matrix
                prox_sparse = sparse.csr_matrix(prox_dense)
        
            # -----------------------------------------------------------------
            # 'oob' (Parallelized)
            # -----------------------------------------------------------------
            elif self.prox_method == 'oob':
                if self.verbose:
                    print(f"Calculating 'oob' extended proximities with T-loop (parallelized with n_jobs={self.n_jobs_})...")
        
                train_oob_subset = self.oob_indices[training_indices, :]
                
                # Denominator (Not parallelized, fast vector op)
                if self.verbose:
                    print("Calculating OOB denominator (D_j)...")
                D_vec = np.sum(train_oob_subset, axis=1, dtype=np.float32)
                D_vec_safe = D_vec.copy()
                D_vec_safe[D_vec_safe == 0] = 1.0
                
                # Numerator (Parallelized T-loop)
                if self.verbose:
                    print("Calculating OOB leaf co-occurrence (Numerator N_ij)...")
        
                # --- START: PARALLEL MODIFICATION ---
                results = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose)(
                    delayed(_get_prox_extend_tree_chunk_oob)(
                        t, extended_leaf_matrix, train_leaves_subset, train_oob_subset
                    ) for t in range(num_trees)
                )
        
                rows_all_list = [r[0] for r in results if r[0] is not None]
                cols_all_list = [r[1] for r in results if r[1] is not None]
                
                if not rows_all_list:
                    N_sparse = sparse.csr_matrix((n_ext, n_sub), dtype=np.float32)
                else:
                    rows_all = np.concatenate(rows_all_list)
                    cols_all = np.concatenate(cols_all_list)
                    data_all = np.ones(len(rows_all), dtype=np.float32)
                    N_sparse = sparse.csr_matrix((data_all, (rows_all, cols_all)), shape=(n_ext, n_sub))
                # --- END: PARALLEL MODIFICATION ---
        
                # Final Division
                if self.verbose:
                    print("Finalizing proximity matrix (N / D)...")
                
                inv_D_vec_scaler = 1.0 / D_vec_safe
                inv_D_scaler_matrix = sparse.diags(inv_D_vec_scaler, format='csr')
                prox_sparse = N_sparse @ inv_D_scaler_matrix
        
            # -----------------------------------------------------------------
            # 'rfgap' (Parallelized)
            # -----------------------------------------------------------------
            elif self.prox_method == 'rfgap':
                if self.verbose:
                    print(f"Calculating 'rfgap' extended proximities with T-loop (parallelized with n_jobs={self.n_jobs_})...")
                
                train_in_bag_counts_subset = self.in_bag_counts[training_indices, :]
                
                if self.verbose:
                    print("Slicing pre-calculated K matrix for subset...")
                K_sub = self.K_matrix[training_indices, :]
            
                # --- START: PARALLEL MODIFICATION ---
                if self.verbose:
                    print("Calculating asymmetric proximities (T-loop list extend)...")
                
                results = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose)(
                    delayed(_get_prox_extend_tree_chunk_rfgap)(
                        t, extended_leaf_matrix, train_leaves_subset, K_sub, train_in_bag_counts_subset
                    ) for t in range(num_trees)
                )
        
                rows_list = [r[0] for r in results if r[0] is not None]
                cols_list = [r[1] for r in results if r[1] is not None]
                data_list = [r[2] for r in results if r[2] is not None]
        
                if not data_list:
                    prox_matrix_sum = sparse.csr_matrix((n_ext, n_sub), dtype=np.float32)
                else:
                    rows = np.concatenate(rows_list)
                    cols = np.concatenate(cols_list)
                    data = np.concatenate(data_list)
                    prox_matrix_sum = sparse.csr_matrix((data, (rows, cols)), shape=(n_ext, n_sub))
                # --- END: PARALLEL MODIFICATION ---
                
                prox_sparse = prox_matrix_sum / num_trees
            
            else:
                 raise ValueError(f"Proximity method '{self.prox_method}' not recognized.")
        
        
            return prox_sparse.todense() if self.matrix_type == 'dense' else prox_sparse
            

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




    return RFGAP(prox_method = prox_method, matrix_type = matrix_type, **kwargs)

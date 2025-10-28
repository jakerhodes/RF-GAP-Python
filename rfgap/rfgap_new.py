# Imports
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform, cdist
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


def RFGAP_NEW_P(prediction_type = None, y = None, prox_method = 'rfgap', matrix_type = 'sparse', triangular = True,
          non_zero_diagonal = True, normalize = True, **kwargs):
    """
    A factory method to conditionally create the RFGAP class based on RandomForestClassifier or RandomForestRegressor

    This class takes on a random forest predictors (sklearn) and adds methods to 
    construct proximities from the random forest object. 
        

    Parameters
    ----------

    prediction_type : str
        Options are 'regression' or 'classification'

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in regression).
        This is an optional way to determine whether RandomForestClassifier or RandomForestRegressor
        should be used

    prox_method : str
        The type of proximity to be constructed.  Options are 'original', 'oob', 
        or 'rfgap' (default is 'rfgap')

    matrix_type : str
        Whether the matrix returned proximities whould be sparse or dense 
        (default is sparse)

    triangular : bool
        Should only the upper triangle of the proximity matrix be computed? This speeds up computation
        time. Not available for RF-GAP proximities (default is True)

    non_zero_diagonal : bool
        Only used for RF-GAP proximities. Should the diagonal entries be computed as non-zero? 
        If True, the proximities are also normalized to be between 0 (min) and 1 (max) by default (see below argument).
        (default is True)
    
    normalize : bool
        Only used for RF-GAP proximities. Should the proximities be normalized to be between 0 (min) and 1 (max)?
        Default is True. Otherwise, the proximities are not normalized but still symmetrized.

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

    class RFGAP_NEW_P(rf):

        def __init__(self, prox_method = prox_method, matrix_type = matrix_type, triangular = triangular,
                     non_zero_diagonal = non_zero_diagonal, normalize = normalize, **kwargs):

            super(RFGAP_NEW_P, self).__init__(**kwargs)

            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.triangular  = triangular
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal
            self.normalize = normalize
            self.min_self_similarity = None


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
            
            # --- MODIFIED: Store n_jobs from base class ---
            super().fit(X, y, sample_weight)
            self.n_jobs_ = effective_n_jobs(self.n_jobs)
            # --- END MODIFIED ---
            
            self.leaf_matrix = self.apply(X)
            
            #---------------------------------------------------------------------------------#
            #                          New Inclusion for Test Set
            #---------------------------------------------------------------------------------#
            
            if x_test is not None:
                n_test = np.shape(x_test)[0]
                
                self.leaf_matrix_test = self.apply(x_test)
                self.leaf_matrix = np.concatenate((self.leaf_matrix, self.leaf_matrix_test), axis = 0)
            
                        
            if self.prox_method == 'oob':
                self.oob_indices = self.get_oob_indices(X)
                
                if x_test is not None:
                    # n_test will be defined from above
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                

            if self.prox_method == 'rfgap':
            
                self.oob_indices = self.get_oob_indices(X)
                self.in_bag_counts = self.get_in_bag_counts(X)

                
                if x_test is not None:
                    n_test = np.shape(x_test)[0] # Get n_test
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                    self.in_bag_counts = np.concatenate((self.in_bag_counts, np.zeros((n_test, self.n_estimators))))                
                                
                self.in_bag_indices = 1 - self.oob_indices
                
                # --- MODIFIED: Added verbose check ---
                if hasattr(self, 'verbose') and self.verbose > 0:
                    print("Pre-calculating K matrix (in-bag leaf counts)...")
                
                n_total, T = self.leaf_matrix.shape
                
                # --- START: K-Matrix Calculation (Parallelized) ---
                # We can also parallelize this K-matrix calculation
                # This block pre-calculates K=total in-bag count of all samples that landed
                # in the same leaf as sample i in tree t, using M (=self.in_bag_counts) and self.leaf_matrix, which are now finalized.
                # This is to avoid re-computing K when calling prox_extend.
                # #NOTE: do not pre-compute K in get_proximities(), because prox_extend() does not need a get_proximities() call, only fit() is required
                def _get_k_matrix_chunk(t, leaf_matrix, in_bag_counts):
                    n_total = leaf_matrix.shape[0]
                    leaves_t = leaf_matrix[:, t]
                    counts_t = in_bag_counts[:, t]
                    unique_leaves = np.unique(leaves_t)
                    K_t = np.zeros(n_total, dtype=np.float32)
                    for leaf_val in unique_leaves:
                        indices = np.where(leaves_t == leaf_val)[0]
                        total_in_bag_for_this_leaf = np.sum(counts_t[indices])
                        K_t[indices] = total_in_bag_for_this_leaf
                    return K_t

                results = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose)(
                    delayed(_get_k_matrix_chunk)(
                        t, self.leaf_matrix, self.in_bag_counts
                    ) for t in range(T)
                )
                
                K = np.array(results).T.astype(np.float32)
                # --- END: K-Matrix Calculation (Parallelized) ---
                
                K[K == 0] = 1.0 # Avoid division by zero
                self.K_matrix = K # <-- Stores K
            

        
        
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
                
                # --- START: PARALLEL MODIFICATION ---
                if self.verbose:
                    print("Calculating asymmetric proximities (T-loop list extend)...")
                
                results = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose)(
                    delayed(_get_prox_tree_chunk_rfgap)(
                        t, self.leaf_matrix, self.oob_indices, self.in_bag_counts, self.K_matrix
                    ) for t in range(T)
                )
                
                # Reduce step: concatenate all results
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
                # --- END: PARALLEL MODIFICATION ---

                inv_S_out_i_scaler = sparse.diags(1.0 / S_out_i_safe, format='csr')
                prox_matrix_asym = inv_S_out_i_scaler @ prox_matrix_sum
                self.proximity_asym = prox_matrix_asym.copy() 
                
                if self.non_zero_diagonal:
                    if self.verbose:
                        print("Calculating 'rfgap' self-similarity (diagonal)...")
                    # ... (diagonal calculation as before, it's fast) ...
                    S_in_i = np.sum(self.in_bag_indices, axis=1)
                    S_in_i_safe = S_in_i.copy()
                    S_in_i_safe[S_in_i_safe == 0] = 1
                    M_ii_t = self.in_bag_counts 
                    Ratios_ii = M_ii_t / K 
                    Terms_ii = Ratios_ii * self.in_bag_indices 
                    Diagonal_sum = np.sum(Terms_ii, axis=1) 
                    Diagonal_vec = Diagonal_sum / S_in_i_safe 
                    self.min_self_similarity = np.min(Diagonal_vec)
                    
                    diag_matrix = None
                    
                    if self.normalize:
                        Diagonal_vec_safe = Diagonal_vec.copy()
                        Diagonal_vec_safe[Diagonal_vec_safe == 0] = 1
                        inv_diag_scaler = sparse.diags(1.0 / Diagonal_vec_safe, format='csr')
                        prox_matrix_asym = inv_diag_scaler @ prox_matrix_asym
                        diag_matrix = sparse.diags(np.ones(n), format='csr')
                    else:
                        diag_matrix = sparse.diags(Diagonal_vec, format='csr')

                    prox_matrix_asym = prox_matrix_asym + diag_matrix

                prox_sparse = (prox_matrix_asym + prox_matrix_asym.transpose()) / 2

            else:
                 raise ValueError(f"Proximity method '{self.prox_method}' not recognized.")

            if self.matrix_type == 'dense':
                return np.array(prox_sparse.todense())
            else:
                return prox_sparse


        def prox_extend(self, data, training_indices=None):
            """
            Compute proximities between new test data and specified training data.
            ... (docstring as before) ...
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
                
                # 1. Denominator (Not parallelized, fast vector op)
                if self.verbose:
                    print("Calculating OOB denominator (D_j)...")
                D_vec = np.sum(train_oob_subset, axis=1, dtype=np.float32)
                D_vec_safe = D_vec.copy()
                D_vec_safe[D_vec_safe == 0] = 1.0
                
                # 2. Numerator (Parallelized T-loop)
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

                # 3. Final Division
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
            
                # 2. Calculate Proximity Sum (Parallelized T-loop)
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
                
                # 3. Apply Normalization
                if self.normalize:
                    if self.verbose:
                        print("Applying 'rfgap' normalization to extended proximities...")
                    
                    if self.min_self_similarity is not None and self.min_self_similarity > 0:
                        prox_sparse.data = prox_sparse.data / self.min_self_similarity
                    
                    # Vectorized normalization
                    if prox_sparse.nnz > 0:
                        max_vals_per_row = prox_sparse.max(axis=1).toarray().ravel()
                        max_vals_per_row[max_vals_per_row <= 1] = 1.0
                        inv_scalers = 1.0 / max_vals_per_row
                        inv_scaler_matrix = sparse.diags(inv_scalers, format='csr')
                        prox_sparse = inv_scaler_matrix @ prox_sparse
            
            else:
                 raise ValueError(f"Proximity method '{self.prox_method}' not recognized.")


            return prox_sparse.todense() if self.matrix_type == 'dense' else prox_sparse


    return RFGAP_NEW_P(prox_method = prox_method, matrix_type = matrix_type, triangular = triangular, non_zero_diagonal = non_zero_diagonal, normalize = normalize, **kwargs)
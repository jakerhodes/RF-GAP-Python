# Imports
import numpy as np
from scipy import sparse
from scipy.sparse import hstack, vstack
import pandas as pd
import gc

# sklearn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
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


def RFGAP(prediction_type=None, y=None, prox_method='rfgap', matrix_type='sparse',
          non_zero_diagonal=False, force_symmetric=False, max_normalize=False,
          model_type='rf', **kwargs):
    """
    Factory function to create an optimized Random Forest or Extra Trees Proximity object.
    
    This class takes on a random forest predictors (sklearn) and adds methods to 
    construct proximities from the random forest object.

    This new implementation uses Sparse Matrix Algebra (Inverted Indexing) with Gustavson scipy sparse multiplication
    P = QW^T, where Q and W are query (i) and weight (j) sparse matrices as per the RF proximity definitions.
    This achieves  O(N*T) complexity instead of the traditional O(N^2) iterative approach (pairwise comparisons), with
    low memory overhead due to sparsity.
    
    Parameters
    ----------
    
    prediction_type : str
        Options are `regression` or `classification`
    
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in regression).
        This is an optional way to determine whether RandomForest/ExtraTrees Classifier or Regressor
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
    
    max_normalize : bool
        Only used for RF-GAP proximities. Whether to max-normalize the proximities 
        after construction (default is False). This might be useful when comparing
        or integrating proximities across multiple models or datasets.

    model_type : str
        'rf' for RandomForest (default) or 'et' for ExtraTrees.
        Note: If 'et' is selected, bootstrap will be set to True by default unless explicitly 
        set to False in kwargs (though True is required for RFGAP proximity calculations).

    **kwargs
        Keyward arguements specific to the RandomForest/ExtraTrees Classifer or Regressor classes
    
        
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

    # Logic to select the correct base class
    if model_type == 'rf':
        if prediction_type == 'classification': base_model = RandomForestClassifier
        elif prediction_type == 'regression': base_model = RandomForestRegressor
    elif model_type == 'et':
        if prediction_type == 'classification': base_model = ExtraTreesClassifier
        elif prediction_type == 'regression': base_model = ExtraTreesRegressor
    else:
        raise ValueError("model_type must be either 'rf' (RandomForest) or 'et' (ExtraTrees)")

    class RFGAP(base_model):
        def __init__(self, prox_method=prox_method, matrix_type=matrix_type,
                     non_zero_diagonal=non_zero_diagonal, force_symmetric=force_symmetric, max_normalize=max_normalize,
                     **kwargs):
            
            # Enforce bootstrapping for ExtraTrees if not specified.
            # Scikit-learn ExtraTrees defaults to bootstrap=False, but RFGAP relies on OOB/bagging mechanics.
            if model_type == 'et':
                if 'bootstrap' not in kwargs:
                    kwargs['bootstrap'] = True
            
            super(RFGAP, self).__init__(**kwargs)
            
            # BLOCK OOB METHOD
            # The OOB method mathematically requires an N x N denominator matrix 
            # (intersection of OOB status), which breaks sparsity and is O(N^2).
            if prox_method == 'oob':
                raise NotImplementedError(
                    "The 'oob' method is O(N^2) dense and incompatible with this new"
                    "sparse-matrix optimization. Use 'rfgap' (highly recommended) or 'original' (traditional RF proximity)."
                )
            
            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal
            self.force_symmetric = force_symmetric
            self.max_normalize = max_normalize
            
            # Internal Cache
            self.W_mat = None   # The "Target" weights matrix (Right side of dot product)
            self._leaf_offsets = None 
            self._total_unique_nodes = None
            
            # Unlabeled Data Cache
            self.leaf_matrix_u = None
            self._n_unlabeled_samples = 0
            self.cached_inverse_M = None  # Caches 1/M for use in P_uu normalization (in-bag leaf size)

            # Partial-label (NaN in y) bookkeeping (preserve original X order)
            self._inv_stack_order = None
            self._n_total_samples = 0

            # Global max normalization cache
            self.global_max = None

        #TODO: do something more elegant than X_train slicing and index reordering. This might cause some memory overhead, but it's fine for now.
        def fit(self, X, y, sample_weight=None):
            """
            Fits the Random Forest and pre-computes the sparse weight matrix W necesssary for proximity calculations.
    
            RUNTIME COMPLEXITY: O(N * T * log(N))
                - Forest Construction: O(N * T * log(N) * #features sampled at each node)
                - Matrix Construction: O(N * T)
                        
            MEMORY COMPLEXITY: O(N * T)
                - Stores leaf indices and sparse weights.
                - Efficiently sparse: Only stores 1 entry per tree per sample.
    
            Parameters
            ----------
            
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.
            
            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in regression).
                If y has missing values (-1/NaNs), those samples are treated as unlabeled and not used for training,
                but are included in proximity calculations after training the RF on the labeled data.
            
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
            # ---------------------------------------------------------
            # STEP 0: Handle Partial Labels (NaNs in y)
            # ---------------------------------------------------------
            
            # 1. Clear previous state
            self._inv_stack_order = None
            self._n_total_samples = X.shape[0]
            
            # 2. Standardize y immediately
            y = np.asarray(y)
            
            # 3. Detect unlabeled samples depending on prediction_type
            if self.prediction_type == "classification":
                if np.issubdtype(y.dtype, np.floating):
                    # Treat BOTH -1 and NaN as unlabeled (robust)
                    mask_unlabeled = np.isnan(y) | (y == -1)
                else:
                    mask_unlabeled = (y == -1)
            else:
                # regression: NaN is unlabeled
                if not np.issubdtype(y.dtype, np.floating):
                    y = y.astype(np.float32)
                mask_unlabeled = np.isnan(y)
            
            if not np.any(mask_unlabeled):
                # Case A: Fully supervised
                X_train = X
                y_train = y
                X_unlabeled = None
            
            else:
                # Case B: Semi-supervised
                idx_labeled = np.flatnonzero(~mask_unlabeled)
                idx_unlabeled = np.flatnonzero(mask_unlabeled)
            
                # Store permutation to restore original order later
                stack_order = np.concatenate([idx_labeled, idx_unlabeled])
                self._inv_stack_order = np.empty(self._n_total_samples, dtype=np.int64)
                self._inv_stack_order[stack_order] = np.arange(self._n_total_samples, dtype=np.int64)
            
                # Slice
                X_train = X[idx_labeled]
                y_train = y[idx_labeled]
                X_unlabeled = X[idx_unlabeled]
            
                # Slice weights if present
                if sample_weight is not None:
                    sample_weight = np.asarray(sample_weight)[idx_labeled]
    
            # 4. Pass valid training data to parent Random Forest
            super().fit(X_train, y_train, sample_weight)
            
            # 5. Store Metadata
            self.y = y
            self._n_train_samples = X_train.shape[0]
            
            # ---------------------------------------------------------
            # COMPLEXITY STEP 1: Forest Pass -> O(N * T * log(N))
            # ---------------------------------------------------------
            # We pass N samples through T trees to get leaf indices.
            # Shape: (N_samples, N_trees)
            self.leaf_matrix = self.apply(X_train) # LOCAL IDS. Tree 1: Node 0,1,2... Tree 2: Node 0,1,2...
            
            # Calculate offsets to flatten (Tree, Leaf) -> Global Feature ID
            # This allows us to treat every node in the forest as a unique feature column.
            n_leaves_per_tree = [t.tree_.node_count for t in self.estimators_]
            self._leaf_offsets = np.concatenate(([0], np.cumsum(n_leaves_per_tree)[:-1]))  # List of size T showing starting index of each tree's leaves
            self._total_unique_nodes = np.sum(n_leaves_per_tree)  # Total unique nodes across all trees (typically NlogN scale)
    
            # Set offset for virtual diagonal nodes (after the last real leaf), needed for non-zero diagonal in RFGAP
            # This avoids the memory-killing setdiag() operation on huge sparse matrices.
            self._diag_offset = self._total_unique_nodes
            
            # ---------------------------------------------------------
            # STEP 1.5: Handle Unlabeled Data (if present)
            # ---------------------------------------------------------
            if X_unlabeled is not None:
                # Pass unlabeled data through the trained forest
                self.leaf_matrix_u = self.apply(X_unlabeled)
                self._n_unlabeled_samples = X_unlabeled.shape[0]
                # Unlabeled proximity matrix (P_uu) naturally has non-zero diagonals and symmetry.
                # We force P_ll to match this behavior for consistency.
                print("Semi-supervised mode. Forcing `non_zero_diagonal`=True and `force_symmetric`=True for consistency.")
                self.non_zero_diagonal = True
                self.force_symmetric = True
            else:
                self.leaf_matrix_u = None
                self._n_unlabeled_samples = 0
    
            # ---------------------------------------------------------
            # COMPLEXITY STEP 2: Statistics Calculation -> O(N * T)
            # ---------------------------------------------------------
            if self.prox_method == 'rfgap':
                # Calculate c_j(t): Multiplicity of sample j in tree t
                self.c_j_t = self.get_in_bag_counts(X_train)
                # Calculate S_i: Set of OOB trees for sample i
                self.oob_indices = self.get_oob_indices(X_train) 

    
            # ---------------------------------------------------------
            # COMPLEXITY STEP 3: Build Sparse Weights -> O(N * T)
            # ---------------------------------------------------------
            # 1. Build Labeled Weights (Stored temporarily in self.W_mat)
            self._build_W_matrix()
    
            # ---------------------------------------------------------
            # STEP 4: Build & Merge Unlabeled Weights -> O(Nu * T)
            # ---------------------------------------------------------
            if self.leaf_matrix_u is not None:
                # 2. Build Unlabeled Weights
                W_u = self._build_Wu_matrix()
                
                # 3. MERGE IMMEDIATELY: Stack Labeled (Top) and Unlabeled (Bottom)
                # This ensures self.W_mat always matches the dimensions of (Labeled + Unlabeled)
                self.W_mat = vstack([self.W_mat, W_u], format='csr', dtype=np.float32)
                
                # Free temp memory
                del W_u
                gc.collect()
            
            # =========================================================
            # Permute W to Original Order HERE
            # =========================================================
            # Instead of permuting the massive N*N output later, we permute the 
            # sparse component W now. This is O(N*Trees), not O(N^2).
            if self._inv_stack_order is not None:
                self.W_mat = self.W_mat[self._inv_stack_order, :]
    
            return self
    
    
        def get_proximities(self):
            """
            This method produces a proximity matrix for the random forest object.
            Computes the proximity matrix P = Q . W^T using sparse matrix multiplication.
    
            RUNTIME COMPLEXITY: O(N * T * k_bar)
                - Where k_bar is the average number of samples per leaf (i.e. average leaf size)
                - Asymmetric: 1x Sparse Matmul.
                - Symmetric (RFGAP only):  2x Sparse Matmul (via Block method).
                        
            MEMORY COMPLEXITY: the output P of the dot product is O(NNZ_Prox) approx O(N^2 * Density), where Density = % of non-zeros in proximity matrix = % points sharing at least 1 leaf.
                - Block Symmetrization keeps peak memory close to the asymmetric output size
            
            Returns
            -------
            array-like
                (if self.matrix_type == 'dense') matrix of pair-wise proximities
            
            csr_matrix
                (if self.matrix_type == 'sparse') a sparse crs_matrix of pair-wise proximities
            
            """
            check_is_fitted(self)
    
            # 1. Retrieve Labeled Components
            Q_l = self._build_Q_matrix(leaves=self.leaf_matrix, is_training=True)
    
            # 2. Retrieve Unlabeled Components (if any) and Stack Q
            if self.leaf_matrix_u is not None:
                # Build Q for unlabeled (is_training=False ensures correct column width/padding)
                Q_u = self._build_Q_matrix(leaves=self.leaf_matrix_u, is_training=False)
                
                # STACKING Q: Labeled on top, Unlabeled on bottom
                Q_total = vstack([Q_l, Q_u], format='csr', dtype=np.float32)
                del Q_l, Q_u
                gc.collect()
            else:
                Q_total = Q_l
    
            # 3. Safety Check: Ensure W_mat was correctly merged in fit()
            if Q_total.shape[0] != self.W_mat.shape[0]:
                raise RuntimeError(
                    f"Dimension Mismatch: Q has {Q_total.shape[0]} rows (Labeled+Unlabeled) but W has {self.W_mat.shape[0]}. "
                    "This usually implies fit() was not re-run after updating the code. Please re-run fit()."
                )
            
            # =========================================================
            # Permute Q to Original Order HERE
            # =========================================================
            # We match the reordering done to W_mat in fit()
            if self._inv_stack_order is not None:
                Q_total = Q_total[self._inv_stack_order, :]

            # 4. Monolithic Dot Product
            # -------------------------
            prox_matrix = None
    
            if self.force_symmetric and self.prox_method == 'rfgap':
                # P = 0.5 * (Q W^T + W Q^T)
                
                # Left side: [Q, W] -> (N_total, 2*F)
                left_side = hstack([Q_total, self.W_mat], format='csr', dtype=np.float32)
                
                # Right side: [W.T, Q.T] -> (2*F, N_total) -> CSC for fast multiplication
                right_side_T = vstack([self.W_mat.T, Q_total.T], format='csc', dtype=np.float32)
                
                # Single Pass Multiplication
                prox_matrix = left_side.dot(right_side_T)
                prox_matrix *= 0.5
                
                del left_side, right_side_T
                gc.collect()
                
            else:
                # Asymmetric: P = Q W^T
                prox_matrix = Q_total.dot(self.W_mat.T)

            # Normalize proximities to [0,1] by global max if requested
            if self.max_normalize and self.prox_method == 'rfgap':
                self.global_max = prox_matrix.max()
                prox_matrix.data /= self.global_max
            
            # Cleanup
            del Q_total
            
            return prox_matrix.todense() if self.matrix_type == 'dense' else prox_matrix
    
        #TODO: Check memory usage of prox_extend in semi-supervised mode (X_unlabeled present)
        def prox_extend(self, X_new):
            """
            Calculates proximities between New Data (rows) and the existing data (cols) passed during fit().
            

            RUNTIME: O(N_test * T * log(N_train) + N_test * T * k_bar)
            MEMORY: O(N_test * T) (Query Q overhead) + Output Matrix

            Parameters
            ----------
            X_new : (n_samples, n_features) array_like
                New observations.

            Returns
            -------
            array-like or csr_matrix
                Proximities between X_new and the fitted data.
            """
            # 1. Pass X_new through the forest to get leaf indices
            leaves_new = self.apply(X_new)
            
            # 2. Build Query Matrix Q for NEW data
            # Q_new shape: (N_new, Total_Unique_Nodes)
            Q_new = self._build_Q_matrix(leaves=leaves_new, is_training=False)
            
            # 3. Compute Dot Product efficiently
            # P = Q . W^T
            prox_matrix = Q_new.dot(self.W_mat.T)
            
            # Cleanup
            del Q_new

            # Max-Normalize if requested using training global max to ensure consistent scaling
            if self.max_normalize and self.prox_method == 'rfgap':
                prox_matrix.data /= self.global_max
            
            return prox_matrix.todense() if self.matrix_type == 'dense' else prox_matrix
    
    
        # MATRIX BUILDERS (Mapping to Definitions)
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
            
            # ORIGINAL PROXIMITY
            # p(i,j) = (1/T) * Sum[ I(j in v_i(t)) ]
            #
            # Mapping: W handles the (1/T) term.
            if self.prox_method == 'original':
                weights = np.full(N * T, 1.0 / T, dtype=np.float32)
                
            # RF-GAP PROXIMITY
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
                
                # Cache this for use in P_uu (Unlabeled-Unlabeled) proximity
                # This ensures unlabeled data is normalized by the same density map as labeled data.
                self.cached_inverse_M = inverse_M_node 
                
                # Combine: W_val = c_j(t) * (1 / M_node)
                weights = c_j_t * inverse_M_node[flat_cols]
                
            # Non-zero diagonal trick via Virtual Diagonal Injection
            # This avoids the memory-killing setdiag() operation on huge sparse matrices.
            total_cols = self._total_unique_nodes
            if self.non_zero_diagonal and self.prox_method == 'rfgap':
                # 1. Calculate what the diagonal values SHOULD be
                row_sums = np.bincount(flat_rows, weights=weights, minlength=N)
                n_trees = self.n_estimators
                denominators = n_trees - self.oob_indices.sum(axis=1)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    diag_vals = row_sums / denominators
                diag_vals[~np.isfinite(diag_vals)] = 0.0
                
                # 2. Append them as new "features" at the end of the matrix
                diag_rows = np.arange(N)
                diag_cols = np.arange(N) + self._diag_offset
                
                flat_rows = np.concatenate([flat_rows, diag_rows])
                flat_cols = np.concatenate([flat_cols, diag_cols])
                weights = np.concatenate([weights, diag_vals])
                
                total_cols += N 
    
            # Filter zeros and build Sparse Matrix W -> O(N * T)
            mask = weights > 0
            self.W_mat = sparse.csr_matrix(
                (weights[mask], (flat_rows[mask], flat_cols[mask])), 
                shape=(N, total_cols),  # shape: N x N_total_leaves (huge but SPARSE)
                dtype=np.float32
            )

        def _build_Wu_matrix(self):
            """
            [NEW] Builds the Weight Matrix 'W_u' (N_unlabeled x N_total_nodes).
            This logic was previously inside get_proximities(). Moving it here allows
            caching and efficient re-use.
            """
            N_u = self._n_unlabeled_samples
            T = self.n_estimators
            global_leaves_u = self._to_global_leaves(self.leaf_matrix_u)
            
            flat_rows = np.repeat(np.arange(N_u), T)
            flat_cols = global_leaves_u.flatten()
            
            # Ensure we use density-weighted normalization (1/M)
            if self.prox_method == 'original':
                # Original Method: just 1/T
                w_vals = np.full(N_u * T, 1.0/T, dtype=np.float32)
            else:
                # RFGAP Method: Use cached density weights (1/M)
                if self.cached_inverse_M is None:
                    # Should not happen if fit() logic is correct
                    raise ValueError("RFGAP weights not found. Model must be fitted with prox_method='rfgap'.")
                
                # Map global leaf IDs to their cached inverse weights
                w_vals = self.cached_inverse_M[flat_cols]

            # Determine shape
            # We must ensure W_u has the same number of columns as W_mat 
            # (which is usually _total_unique_nodes, or +N_train if diagonals involved)
            # This ensures dimensions match when doing dot products or combining matrices.
            total_cols = self.W_mat.shape[1]

            return sparse.csr_matrix(
                    (w_vals, (flat_rows, flat_cols)),
                    shape=(N_u, total_cols), 
                    dtype=np.float32
                )
    
    
        def _build_Q_matrix(self, leaves=None, is_training=True):
            """
            Builds the Query Matrix 'Q' (N_query x N_total_nodes).
            This matrix handles the 'i' term and the Summation scope (S_i).
            
            Now accepts 'leaves' directly to allow pre-computed leaf matrices.
            """
            if leaves is None:
                leaves = self.leaf_matrix
    
            N, T = leaves.shape
            global_leaves = self._to_global_leaves(leaves)
            
            flat_rows = np.repeat(np.arange(N), T)
            flat_cols = global_leaves.flatten()
    
            # Determine OOB mask logic
            if is_training:
                # S_i logic: For RFGAP, we only sum over trees where i is OOB.
                oob_mask = self.oob_indices if self.prox_method == 'rfgap' else None
            else:
                # For new data, the sample was not in ANY bag, so it is OOB for all trees.
                oob_mask = None # Treated as all ones later
    
            # ORIGINAL PROXIMITY
            # p(i,j) = Sum[ ... ]  (Sum over all t=1 to T)
            #
            # W already contains (1/T). Q simply indicates existence (1.0).
            if self.prox_method == 'original':
                vals = np.ones(N * T, dtype=np.float32)
                
            # RF-GAP PROXIMITY
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
                    # vals = 1/T for normalization
                    vals = np.full(N * T, 1.0 / T, dtype=np.float32)
        
            # Non-zero diagonal trick via Virtual Diagonal Injection
            # This avoids the memory-killing setdiag() operation on huge sparse matrices.
            total_cols = self._total_unique_nodes
            if self.non_zero_diagonal and self.prox_method == 'rfgap':
                # FIX: Use the Training Size, not the current Query Size (N)
                total_cols += self._n_train_samples
                
                # Only add Identity (1.0) if we are training (samples match themselves)
                if is_training:
                    diag_rows = np.arange(N)
                    diag_cols = np.arange(N) + self._diag_offset
                    
                    flat_rows = np.concatenate([flat_rows, diag_rows])
                    flat_cols = np.concatenate([flat_cols, diag_cols])
                    vals = np.concatenate([vals, np.ones(N)])
                
                # If is_training=False (prox_extend), we DO NOT add values, 
                # but we DO keep the total_cols expanded so dimensions match W.
    
            return sparse.csr_matrix(
                (vals, (flat_rows, flat_cols)), 
                shape=(N, total_cols), 
                dtype=np.float32
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
    
    return RFGAP(prox_method = prox_method, matrix_type = matrix_type,
                 non_zero_diagonal = non_zero_diagonal, force_symmetric = force_symmetric, max_normalize = max_normalize, **kwargs)

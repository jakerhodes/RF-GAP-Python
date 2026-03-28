# Imports
import numpy as np
from scipy import sparse
from scipy.sparse import hstack, vstack
import pandas as pd

# sklearn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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

# Extras
from .extras import GAPExtrasMixin
from forestkernel.wrappers.bagged_rotation_forest import BaggedRotationForest

#TODO: add support for sklearn Quantile RandomForests and other tree-based models in sklearn and beyond (e.g. LightGBM, XGBoost, CatBoost)
def ForestKernel(prediction_type=None, y=None, prox_method='gap', matrix_type='sparse',
          force_nonzero_diag=False, force_symmetric=None, max_normalize=False,
          model_type='rf', **kwargs):
    """
    Factory function to create an optimized Random Forest, Extra Trees, or Gradient Boosting Proximity object.
    
    This class takes on a tree ensemble predictor (sklearn) and adds methods to 
    construct proximities from the fitted ensemble object.

    This new implementation uses Sparse Matrix Algebra (Inverted Indexing) with Gustavson scipy sparse multiplication
    P = QW^T, where Q and W are query (i) and weight (j) sparse matrices as per the proximity definitions.
    This achieves  O(N*T) complexity instead of the traditional O(N^2) iterative approach (pairwise comparisons), with
    low memory overhead due to sparsity.
    
    Parameters
    ----------
    
    prediction_type : str
        Options are `regression` or `classification`
    
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in regression).
        This is an optional way to determine whether RandomForest/ExtraTrees/GradientBoosting Classifier or Regressor
        should be used
    
    prox_method : str
        The type of proximity to be constructed. Options are `original`, `oob`,
        `gap`, `kerf`, or `gbt` (default is `gap`)
    
    matrix_type : str
        Whether the matrix returned proximities whould be sparse or dense 
        (default is sparse)
    
    force_nonzero_diag : bool
        Only used for RF-GAP proximities. Should the diagonal entries be computed as non-zero? 
        (default is False, as in original RF-GAP definition)
    
    force_symmetric : bool
        Whether to force the output proximities to be symmetric via the Block Symmetrization Trick.
    
    max_normalize : bool
        Only used for RF-GAP proximities. Whether to row-wise max-normalize the proximities 
        after construction (default is False). This might be useful when comparing
        or integrating proximities across multiple models or datasets.

    model_type : str
        'rf' for RandomForest (default), 'et' for ExtraTrees, 'gbt' for GradientBoosting,
        or 'rotf' for Bagged Rotation Forest.

    **kwargs
        Keyward arguements specific to the RandomForest/ExtraTrees/GradientBoosting Classifer or Regressor classes
    
        
    Returns
    -------
    self : object
        The RF/ET/GBT/ROTF object (unfitted)
    
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
    if prox_method == 'gbt' and model_type != 'gbt':
        raise ValueError("prox_method='gbt' requires model_type='gbt'")
    
    if model_type == 'gbt' and prox_method != 'gbt':
        raise ValueError("When model_type='gbt', prox_method must be 'gbt'")

    if model_type == 'rotf' and prediction_type != 'classification':
        raise ValueError("model_type='rotf' currently supports classification only.")
    
    if model_type == 'rf':
        if prediction_type == 'classification': base_model = RandomForestClassifier
        elif prediction_type == 'regression': base_model = RandomForestRegressor
    elif model_type == 'et':
        if prediction_type == 'classification': base_model = ExtraTreesClassifier
        elif prediction_type == 'regression': base_model = ExtraTreesRegressor
    elif model_type == 'gbt':
        if prediction_type == 'classification': base_model = GradientBoostingClassifier
        elif prediction_type == 'regression': base_model = GradientBoostingRegressor
    elif model_type == 'rotf':
        base_model = BaggedRotationForest
    else:
        raise ValueError("model_type must be either 'rf' (RandomForest), 'et' (ExtraTrees), 'gbt' (GradientBoosting), or 'rotf' (BaggedRotationForest)")

    class ForestKernel(GAPExtrasMixin, base_model):
        def __init__(self, prox_method=prox_method, matrix_type=matrix_type,
                     force_nonzero_diag=force_nonzero_diag, force_symmetric=force_symmetric, max_normalize=max_normalize,
                     **kwargs):
            
            # OOB- and RF-GAP-based proximities require bootstrap sampling.
            # We enforce the standard full-bootstrap setting used by this implementation.
            if prox_method in ['oob', 'gap'] and model_type in ['rf', 'et']:
                kwargs['bootstrap'] = True
                kwargs['max_samples'] = None
            
            super(ForestKernel, self).__init__(**kwargs)
            
            
            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.prediction_type = prediction_type
            self.force_nonzero_diag = force_nonzero_diag
            self.force_symmetric = force_symmetric
            self.max_normalize = max_normalize
            self.model_type = model_type
            
            # Internal Cache
            self.W_mat = None   # The "Target/Reference" weights matrix (Right side of dot product)
            self._leaf_offsets = None 
            self._total_unique_nodes = None
            
            self.idx_labeled_ = None
            self.idx_unlabeled_ = None
            
            self.leaf_matrix_all = None     # (N_total, T) in ORIGINAL X order
            self.oob_mask_all = None        # (N_total, T) int8/bool in ORIGINAL order
            self.c_all = None               # (N_total, T) float32 in ORIGINAL order

            self._flat_rows_all = None          # flattened row ids for all sample-tree incidences
            self._flat_cols_all = None          # flattened global leaf ids for all sample-tree incidences
            
            self._leaf_mass_labeled_unit = None
            self._leaf_mass_labeled_mult = None
            
            self._inv_sqrt_leaf_mass_labeled_unit = None
            self._inv_leaf_mass_labeled_mult = None

            # GBT-specific cache
            self._tree_list = None
            self._gbt_tree_weights = None

            # Partial-label (NaN in y) bookkeeping (preserve original X order)
            self._n_total_samples = 0

        def fit(self, X, y, sample_weight=None):
            """
            Fits the tree ensemble and pre-computes the sparse weight matrix W necesssary for proximity calculations.
        
            RUNTIME COMPLEXITY: O(N * T * log(N))
                - Ensemble Construction: O(N * T * log(N) * #features sampled at each node)
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
                but are included in proximity calculations after training the ensemble on the labeled data.
            
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
            # STEP 0: Prepare Data
            # ---------------------------------------------------------
            
            # 1. Standardize y and clear previous state
            y = np.asarray(y)
            if self.prediction_type == "regression" and not np.issubdtype(y.dtype, np.floating):
                y = y.astype(np.float32)
        
            self._n_samples = X.shape[0]
        
            # 2. Pass training data to parent tree ensemble
            try:
                super().fit(X, y, sample_weight=sample_weight)
            except TypeError:
                if sample_weight is not None:
                    warnings.warn(
                        "sample_weight was provided but is ignored because the selected base model does not support it."
                    )
                super().fit(X, y)
            
            # 3. Store Metadata
            self.y = y
            
            # ---------------------------------------------------------
            # COMPLEXITY STEP 1: Forest Pass -> O(N * T * log(N))
            # ---------------------------------------------------------
            # We pass N samples through T trees to get leaf indices.
            # Shape: (N_samples, N_trees)
            if self.model_type == 'gbt':
                self._tree_list = self._get_tree_list()

            self.leaf_matrix_all = self._get_leaf_matrix(X)  # LOCAL IDS across all trees
                
            # Calculate offsets to flatten (Tree, Leaf) -> Global Feature ID
            # This allows us to treat every node in the forest as a unique feature column.
            n_leaves_per_tree = self._get_n_nodes_per_tree()
            self._leaf_offsets = np.concatenate(([0], np.cumsum(n_leaves_per_tree)[:-1]))
            self._total_unique_nodes = np.sum(n_leaves_per_tree)
        
            # Set offset for virtual diagonal nodes (after the last real leaf), needed for non-zero diagonal in GAP
            # This avoids the memory-killing setdiag() operation on huge sparse matrices.
            self._diag_offset = self._total_unique_nodes
            
            # ---------------------------------------------------------
            # STEP 1.5: Cache global flattened leaf structure
            # ---------------------------------------------------------
            T = self.leaf_matrix_all.shape[1]
            global_leaves_all = self._to_global_leaves(self.leaf_matrix_all)
            
            self._flat_rows_all = np.repeat(np.arange(self._n_samples), T)
            self._flat_cols_all = global_leaves_all.flatten()
        
            # ---------------------------------------------------------
            # COMPLEXITY STEP 2: Statistics Calculation -> O(N * T)
            # ---------------------------------------------------------
            if self.prox_method in ['oob', 'gap']:
                # Calculate S_i: Set of OOB trees for sample i
                self.oob_mask_all = self.get_oob_indices(X).astype(np.int8)
        
                if self.prox_method == 'gap':
                    # Calculate c_j(t): Multiplicity of sample j in tree t
                    self.c_all = self.get_in_bag_counts(X).astype(np.float32)
            
            # ---------------------------------------------------------
            # STEP 2.5: Precompute leaf statistics used by proximity methods
            # ---------------------------------------------------------
            
            # ---------------------------------------------------------
            # Unit leaf mass
            #   counts sample-tree incidences in each global leaf
            #   used by KeRF
            # ---------------------------------------------------------
            self._leaf_mass_unit = np.bincount(
                self._flat_cols_all,
                minlength=self._total_unique_nodes
            ).astype(np.float32)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                self._inv_sqrt_leaf_mass_unit = 1.0 / np.sqrt(self._leaf_mass_unit)
            self._inv_sqrt_leaf_mass_unit[~np.isfinite(self._inv_sqrt_leaf_mass_unit)] = 0.0
            
            # ---------------------------------------------------------
            # Multiplicity leaf mass
            #   sums in-bag multiplicities in each global leaf
            #   used by RF-GAP
            # ---------------------------------------------------------
            if self.prox_method == 'gap':
                c_flat = self.c_all.flatten()
            
                self._leaf_mass_mult = np.bincount(
                    self._flat_cols_all,
                    weights=c_flat,
                    minlength=self._total_unique_nodes
                ).astype(np.float32)
            
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._inv_leaf_mass_mult = 1.0 / self._leaf_mass_mult
                self._inv_leaf_mass_mult[~np.isfinite(self._inv_leaf_mass_mult)] = 0.0

            
            # Tree-wise weights for GBT proximity
            if self.prox_method == 'gbt':
                self._gbt_tree_weights = self._compute_gbt_tree_weights(X)
        
            # ---------------------------------------------------------
            # COMPLEXITY STEP 3: Build Sparse Weights -> O(N * T)
            # ---------------------------------------------------------
            # 1. Build Weights (Stored temporarily in self.W_mat)
            self._build_W_matrix()
        
            return self
        
        
        # TODO: check memory/time of normalization step
        def get_proximities(self):
            """
            This method produces a proximity matrix for the random forest object.
            Computes the proximity matrix P = Q . W^T using sparse matrix multiplication.
        
            RUNTIME COMPLEXITY: O(N * T * k_bar)
                - Where k_bar is the average number of samples per leaf (i.e. average leaf size)
                - Asymmetric: 1x Sparse Matmul. Already symmetric if 'original' or 'oob' method since Q=W.
                - Symmetric (GAP only):  2x Sparse Matmul (via Block method).
                        
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
        
            # 1. Retrieve Query Leaf Components
            Q_total = self._build_Q_matrix(leaves=self.leaf_matrix_all, is_training=True)
            
            # =========================================================
            # Fast Row-Max Normalization (Hadamard Trick)
            # =========================================================
            # We perform normalization BEFORE symmetrization to keep the Block Trick efficient.
            # Calculate the diagonal (self-similarity) using fast element-wise mult.
            #    (For proximities, the diagonal is generally the Row Max).
            if self.max_normalize and ((self.prox_method == 'gap' and self.force_nonzero_diag) or self.prox_method == 'kerf'):
                # diagonal = row max = sum(Q ⊙ W) row-wise (fast + sparse)
                # Hadamard product O(NNZ) is much faster than Dot Product O(N^2 * density)
                diagonal = Q_total.multiply(self.W_mat).sum(axis=1).A.ravel()
                
                # Safety for Division
                diagonal[diagonal == 0] = 1.0
                
                # In-Place Row Scaling
                # Q <- D^{-1} Q. 
                # This ensures that (Q . W^T) will have 1s on the diagonal.
                self._csr_row_scale_inplace(Q_total, 1.0 / diagonal)
        
            # 2. Monolithic Dot Product
            # -------------------------
            prox_matrix = None
        
            if (self.force_symmetric and (self.prox_method == 'gap' or (self.prox_method == 'kerf' and self.max_normalize))):
                prox_matrix = self._block_symmetrize(Q_total, self.W_mat)
            else:
                # Asymmetric: P = Q W^T 
                # (For naturally symmetric methods, this is simply W W^T)
                prox_matrix = Q_total.dot(self.W_mat.T)
            
            # # Diagonal clipping for OOB  (mimics diagonal correction to make diagonal=1 for OOB)
            # if self.prox_method == 'oob':
            #     prox_matrix.data = np.minimum(prox_matrix.data, 1.0)
            
            return prox_matrix.toarray() if self.matrix_type == 'dense' else prox_matrix
        
        # TODO: Add max-normalization option to prox_extend, consistent with get_proximities
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
            leaves_new = self._get_leaf_matrix(X_new)
            
            # 2. Build Query Matrix Q for NEW data
            # Q_new shape: (N_new, Total_Unique_Nodes)
            Q_new = self._build_Q_matrix(leaves=leaves_new, is_training=False)
            
            # 3. Compute Dot Product efficiently
            # P = Q . W^T
            prox_new = Q_new.dot(self.W_mat.T)
        
            # Cleanup
            del Q_new
                
            return prox_new.toarray() if self.matrix_type == 'dense' else prox_new
        
        # Helpers
        @staticmethod
        def _csr_row_scale_inplace(A, scale):
            """
            In-place row scaling of a CSR matrix.
            A[i, :] *= scale[i]
            """
            if not sparse.isspmatrix_csr(A):
                raise ValueError("Matrix must be CSR for in-place scaling.")
                
            scale = np.asarray(scale, dtype=A.data.dtype)
            
            # Repeat scale factor for every non-zero element in the row
            # indptr diff gives the count of non-zeros per row
            nnz_per_row = np.diff(A.indptr)
            
            # In-place multiplication of the data array
            A.data *= np.repeat(scale, nnz_per_row)
        
        @staticmethod
        def _block_symmetrize(Q, W):
            """
            Computes symmetric proximity P using optimized sparse strategies.
            P = 0.5 * (Q W^T + W Q^T) = 0.5 [Q W] [W.T, Q.T]--> Uses block matrix trick for 1x memory overhead.
            """
            # Efficiency: We stack [Q, W] to compute QW^T + WQ^T in one pass
            left_block = sparse.hstack([Q, W], format='csr', dtype=np.float32)
            right_block_T = sparse.vstack([W.T, Q.T], format='csc', dtype=np.float32)
            P = 0.5 * left_block.dot(right_block_T)
            del left_block, right_block_T
        
            return P
        
        # MATRIX BUILDERS (Mapping to Definitions)
        def _to_global_leaves(self, leaf_mat):
            """Offset local leaf IDs to global feature IDs."""
            return leaf_mat + self._leaf_offsets

        def _extract_tree(self, estimator):
            """
            Helper to safely retrieve the raw decision tree object.
            - RF / ET: estimator is already a sklearn tree
            - GBT: tree list is already flattened raw trees
            - ROTF: estimator may be a PCA/tree pipeline
            """
            if self.model_type == 'rotf':
                if hasattr(estimator, 'steps'):
                    return estimator.steps[-1][1]
                return estimator
            return estimator

        def _get_base_estimators(self):
            """
            Returns the fitted estimator objects used by the ensemble.
            """
            if self.model_type == 'gbt':
                return self._tree_list
            return self.estimators_

        def _get_leaf_matrix(self, X):
            """
            Return matrix of leaf ids of shape (N, T).
            """
            if self.model_type == 'gbt':
                return self._apply_all_trees(X)
            return self.apply(X).astype(np.int32)

        def _get_n_nodes_per_tree(self):
            """
            Number of nodes per tree, used to offset local node ids into global ids.
            """
            estimators = self._get_base_estimators()
            return [self._extract_tree(est).tree_.node_count for est in estimators]
        
        
        def _build_W_matrix(self):
            """
            Builds the Weight Matrix 'W' (N_samples x N_total_nodes).
            This matrix handles the 'j' term (target) in the definitions.
            """
            leaves = self.leaf_matrix_all
            N, T = leaves.shape
            global_leaves = self._to_global_leaves(leaves)  # shape (N, T) with global node IDs
            
            # Flatten indices for sparse construction -> O(N * T)
            flat_rows = np.repeat(np.arange(N), T)
            flat_cols = global_leaves.flatten()
            total_cols = self._total_unique_nodes  # Base number of columns for sparse W building (before virtual diagonal)
            
            # ORIGINAL PROXIMITY
            # p(i,j) = (1/T) * Sum[ I(j in v_i(t)) ]
            #
            # Mapping: W handles the sqrt(1/T) (uniform weights).
            if self.prox_method == 'original':
                scale_factor = np.float32(1.0 / np.sqrt(T))
                weights = np.full(N * T, scale_factor, dtype=np.float32)
        
            # GBT PROXIMITY
            # p(i,j) = Sum_t w_t * I( leaf_t(i) = leaf_t(j) )
            #
            # Mapping: use a symmetric factorization with sqrt(w_t) on both sides
            elif self.prox_method == 'gbt':
                sqrt_w = np.sqrt(self._gbt_tree_weights).astype(np.float32)
                weights = np.tile(sqrt_w, N)
            
            elif self.prox_method == 'kerf':
                # KeRF-style symmetric kernel:
                # (1/T) * sum_t I(leaf_i(t)=leaf_j(t)) / M_leaf(t)
                weights = (1.0 / np.sqrt(T)) * self._inv_sqrt_leaf_mass_unit[flat_cols]
            
            # OOB proximity (approximated to make it separable).
            elif self.prox_method == 'oob':
                mask = self.oob_mask_all.flatten() == 1
                flat_rows = flat_rows[mask]
                flat_cols = flat_cols[mask]
            
                # M_j = number of OOB trees for sample j (row-sum of OOB mask)
                M = self.oob_mask_all.sum(axis=1).astype(np.float32)
                M[M == 0] = 1.0  # safety
            
                # weights for row=j: sqrt(T) / M_j
                weights = (np.sqrt(T) / M[flat_rows]).astype(np.float32)
        
                # Exact diagonal replacement trick for OOB.
                # The base OOB factorization gives raw self-similarity T / M_j, which is often > 1.
                # We append one private coordinate per training sample so that
                #    QW^T  ->  QW^T + diag(1 - raw_diag),
                # thereby discarding the raw diagonal and replacing it with 1.
                raw_diag = (T / M).astype(np.float32)
                diag_vals = (1.0 - raw_diag).astype(np.float32)
        
                diag_rows = np.arange(N)
                diag_cols = np.arange(N) + self._diag_offset
        
                flat_rows = np.concatenate([flat_rows, diag_rows])
                flat_cols = np.concatenate([flat_cols, diag_cols])
                weights   = np.concatenate([weights, diag_vals])
                total_cols += N
                
            # RF-GAP PROXIMITY
            # Term inside Sum: c_j(t) / M_i(t)
            elif self.prox_method == 'gap':
                c_j_t = self.c_all.flatten()
                weights = c_j_t * self._inv_leaf_mass_mult[flat_cols]
                
                # Optional virtual diagonal injection for RF-GAP.
                # This avoids sparse setdiag() and restores nonzero self-similarity.
                #
                # For row i, we inject
                #    d_i = (sum_t c_i(t) / M_{leaf(i,t)}) / #{t : c_i(t) > 0}.
                if self.force_nonzero_diag:
                    row_sums = np.bincount(flat_rows, weights=weights, minlength=N).astype(np.float32)
                    denom = (self.c_all > 0).sum(axis=1).astype(np.float32)
                    denom[denom == 0] = 1.0
                    diag_vals = row_sums / denom
                
                    diag_rows = np.arange(N)
                    diag_cols = np.arange(N) + self._diag_offset
                    flat_rows = np.concatenate([flat_rows, diag_rows])
                    flat_cols = np.concatenate([flat_cols, diag_cols])
                    weights   = np.concatenate([weights, diag_vals])
                    total_cols += N
        
            # Filter zeros and build Sparse Matrix W -> O(N * T)
            mask = weights != 0
            self.W_mat = sparse.csr_matrix(
                (weights[mask], (flat_rows[mask], flat_cols[mask])), 
                shape=(N, total_cols),  # shape: N x N_total_leaves (huge but SPARSE)
                dtype=np.float32
            )
        
        
        def _build_Q_matrix(self, leaves=None, is_training=True):
            """
            Builds the Query Matrix 'Q' (N_query x N_total_nodes).
            This matrix handles the 'i' term and the Summation scope (S_i).
        
            Arguments:
            - leaves: Optional pre-computed leaf matrix for the query data. If None, uses self.leaf_matrix_all.
            - is_training: Applies to GAP and OOB. Whether the query data is the original training data.
            """
            if leaves is None:
                leaves = self.leaf_matrix_all
        
            N, T = leaves.shape
            global_leaves = self._to_global_leaves(leaves)
            
            flat_rows = np.repeat(np.arange(N), T)
            flat_cols = global_leaves.flatten()
            total_cols = self._total_unique_nodes  # Base number of columns for sparse Q building (before virtual diagonal)
        
            # ORIGINAL PROXIMITY
            # p(i,j) = Sum[ ... ]  (Sum over all t=1 to T)
            if self.prox_method == 'original':
                scale_factor = np.float32(1.0 / np.sqrt(T))
                vals = np.full(N * T, scale_factor, dtype=np.float32)
        
            # GBT PROXIMITY
            # p(i,j) = Sum_t w_t * I( leaf_t(i) = leaf_t(j) )
            elif self.prox_method == 'gbt':
                sqrt_w = np.sqrt(self._gbt_tree_weights).astype(np.float32)
                vals = np.tile(sqrt_w, N)
            
            elif self.prox_method == 'kerf':
                vals = (1.0 / np.sqrt(T)) * self._inv_sqrt_leaf_mass_unit[flat_cols]
        
            # approximated OOB proximity.
            # If is_training=True, build exactly as in W, except that the private diagonal
            # coordinates are set to 1 so that QW^T adds diag(1 - raw_diag).
            # If is_training=False, we treat each query as OOB for all trees.
            elif self.prox_method == 'oob':
                
                if is_training:
                    # Apply OOB scope: Keep only OOB trees
                    mask = self.oob_mask_all.flatten() == 1
                    flat_rows = flat_rows[mask]
                    flat_cols = flat_cols[mask]
                    
                    # Calculate |S_i|: Count of OOB trees per sample
                    S_i_counts = self.oob_mask_all.sum(axis=1).astype(np.float32)
                    S_i_counts[S_i_counts == 0] = 1.0  # Avoid div/0
                    
                    # Q_val = sqrt(T) / |S_i|
                    vals = (np.sqrt(T) / S_i_counts[flat_rows]).astype(np.float32)
        
                    # Matching private diagonal coordinates for exact diagonal replacement
                    total_cols += self._n_samples
                    diag_rows = np.arange(N)
                    diag_cols = np.arange(N) + self._diag_offset
                    diag_vals = np.ones(N, dtype=np.float32)
        
                    flat_rows = np.concatenate([flat_rows, diag_rows])
                    flat_cols = np.concatenate([flat_cols, diag_cols])
                    vals      = np.concatenate([vals, diag_vals])
                            
                else:
                    # For new data, all trees are considered OOB by convention (size T).
                    vals = np.full(N * T, np.sqrt(T) / T, dtype=np.float32)
        
                    # The reference-side W includes private diagonal coordinates for the training set.
                    # New queries should have zero mass on these coordinates, but the matrix width must match.
                    total_cols += self._n_samples
                
            # RF-GAP PROXIMITY
            # p(i,j) = (1 / |S_i|) * Sum_{t in S_i} [ ... ]
            elif self.prox_method == 'gap':
                
                if is_training:
                    # Apply S_i Scope: Keep only OOB trees
                    mask = self.oob_mask_all.flatten() == 1
                    flat_rows = flat_rows[mask]
                    flat_cols = flat_cols[mask]
                    
                    # Calculate |S_i|: Count of OOB trees per sample
                    S_i_counts = self.oob_mask_all.sum(axis=1).astype(np.float32)
                    S_i_counts[S_i_counts == 0] = 1 # Avoid div/0
                    
                    # Q_val = 1 / |S_i|
                    vals = (1.0 / S_i_counts[flat_rows]).astype(np.float32)
                    
                else:
                    # For new data, S_i is the set of ALL trees (size T).
                    vals = np.full(N * T, 1.0 / T, dtype=np.float32)
        
                # Non-zero diagonal trick via Virtual Diagonal Injection
                if self.force_nonzero_diag:
                    total_cols += self._n_samples
                
                    if is_training:
                        diag_rows = np.arange(N)
                        diag_cols = np.arange(N) + self._diag_offset
                
                        # Default diagonal injection as 1.0
                        diag_vals = np.ones(N, dtype=np.float32)
                
                        flat_rows = np.concatenate([flat_rows, diag_rows])
                        flat_cols = np.concatenate([flat_cols, diag_cols])
                        vals      = np.concatenate([vals, diag_vals])
            
            mask = vals != 0
            return sparse.csr_matrix(
                (vals, (flat_rows[mask], flat_cols[mask])), 
                shape=(N, total_cols), 
                dtype=np.float32
            )

        # Forest/OOB Helpers
        def get_oob_indices(self, X_train=None):
            """
            Returns OOB mask matrix of shape (N_train, T), where entry (i,t)=1 if sample i is OOB for tree t.
            """
            if self.model_type == 'gbt':
                raise ValueError("OOB indices are not defined for GradientBoosting.")

            if self.model_type == 'rotf':
                c = self.get_in_bag_counts(X_train)
                return (c == 0).astype(np.int8)
            
            n_samples = X_train.shape[0]
            n_trees = len(self.estimators_)
            oob_mask = np.zeros((n_samples, n_trees), dtype=np.int8)
            
            for t, tree in enumerate(self.estimators_):
                unsampled = _generate_unsampled_indices(
                    tree.random_state,
                    n_samples,
                    n_samples
                )
                oob_mask[unsampled, t] = 1
            
            return oob_mask

        def get_in_bag_counts(self, X_train=None):
            """
            Returns in-bag multiplicity matrix of shape (N_train, T), where entry (i,t)
            is the number of times sample i was drawn for tree t.
            """
            if self.model_type == 'gbt':
                raise ValueError("In-bag counts are not defined for GradientBoosting.")

            if self.model_type == 'rotf':
                n_trees = len(self.estimators_)
                first_tree = self._extract_tree(self.estimators_[0])
                n_samples = len(first_tree.in_bag_counts_)
                counts = np.zeros((n_samples, n_trees), dtype=np.float32)

                for t, est in enumerate(self.estimators_):
                    tree = self._extract_tree(est)
                    counts[:, t] = tree.in_bag_counts_

                return counts
            
            n_samples = X_train.shape[0]
            n_trees = len(self.estimators_)
            counts = np.zeros((n_samples, n_trees), dtype=np.int32)
            
            for t, tree in enumerate(self.estimators_):
                sampled = _generate_sample_indices(
                    tree.random_state,
                    n_samples,
                    n_samples
                )
                binc = np.bincount(sampled, minlength=n_samples)
                counts[:, t] = binc
            
            return counts.astype(np.float32)
        
        # GBT Helpers
        def _get_tree_list(self):
            """
            Flatten sklearn GBT estimators_ into a single list of trees.
        
            - Regression: estimators_.shape = (T, 1)
            - Binary classification: estimators_.shape = (T, 1)
            - Multiclass classification: estimators_.shape = (T, K)
            """
            return [tree for tree in self.estimators_.ravel()]
        
        def _apply_all_trees(self, X):
            """
            Apply every tree in the flattened GBT ensemble and return
            a leaf matrix of shape (N, T_total).
            """
            return np.column_stack([tree.apply(X) for tree in self._tree_list]).astype(np.int32)
        
        def _compute_gbt_tree_weights(self, X_ref):
            """
            Computes tree-specific weights for Gradient Boosted Tree (GBT) proximities.
        
            Following Tan et al. (2016), we weight each tree by its 'importance'—the 
            magnitude of its contribution to the final boosted predictor. Unlike Random 
            Forests where trees are i.i.d., GBT trees are learned iteratively and 
            contribute unequally.
        
            The weight w_t is proportional to the squared L2-norm of the tree's 
            shrunken predictions:
                w_t = || learning_rate * h_t(X_ref) ||_2^2
            
            This is mathematically equivalent to the variance-based weighting 
            scheme (gamma_i^2 * Var(tree_output)) mentioned in the literature.
        
            Returns
            -------
            weights : ndarray of shape (n_trees,)
                Normalized weights (sum to 1.0) for each tree in the ensemble.
                Note: Downstream matrix construction should use sqrt(weights) 
                to maintain the separable p(i,j) = sum(w_t * I) property.
            """
            # gamma_i in paper: global shrinkage factor applied to all trees
            lr = np.float32(self.learning_rate) 
            weights = []
        
            for tree in self._tree_list:
                # c_i^{Tree}(s): raw residual prediction of the i-th tree
                # We use X_ref (usually the training set) to estimate the tree's 'energy'
                contrib = lr * tree.predict(X_ref)
                
                # We square the norm to match the L2-size/Variance definition.
                # This gives high-impact trees significantly more weight than 
                # trees that only perform minor residual corrections.
                wt = np.linalg.norm(contrib, ord=2)**2
                weights.append(wt)
        
            weights = np.asarray(weights, dtype=np.float32)
        
            if weights.size == 0:
                raise RuntimeError("No trees found in fitted GradientBoosting model.")
        
            # Standardize weights to prevent numerical overflow and ensure 
            # the proximity matrix values are bounded and interpretable.
            total_weight = weights.sum()
            if total_weight <= 0:
                # Fallback to uniform weights if trees have zero variance (unlikely)
                weights[:] = 1.0 / len(weights)
            else:
                weights /= total_weight
        
            return weights.astype(np.float32)
        
    
    return ForestKernel(prox_method = prox_method, matrix_type = matrix_type,
                 force_nonzero_diag = force_nonzero_diag, force_symmetric = force_symmetric, max_normalize = max_normalize, **kwargs)
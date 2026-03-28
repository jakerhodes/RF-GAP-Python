import numpy as np
import warnings

from sklearn.utils.validation import check_is_fitted

from .extras import GAPExtrasMixin
from .config import (
    infer_prediction_type,
    validate_model_configuration,
    get_base_model,
    sanitize_model_kwargs,
    validate_model_kwargs,
)
from .adapters import make_adapter
from .proximity import (
    initialize_cache,
    attach_oob_structure,
    attach_gbt_weights,
    compute_unit_leaf_mass,
    compute_multiplicity_leaf_mass,
    build_W_matrix,
    build_Q_matrix,
    csr_row_scale_inplace,
    block_symmetrize,
)


# TODO: add support for sklearn Quantile RandomForests and other tree-based models
# in sklearn and beyond (e.g. LightGBM, XGBoost, CatBoost)
def ForestKernel(
    prediction_type=None,
    y=None,
    prox_method="gap",
    matrix_type="sparse",
    force_nonzero_diag=False,
    force_symmetric=None,
    normalize_diagonal=False,
    model_type="rf",
    allow_semi_supervised=False,
    **kwargs,
):
    """
    Factory function to create an optimized Random Forest, Extra Trees,
    Gradient Boosting, or Rotation Forest proximity object.

    This class takes on a tree ensemble predictor and adds methods to
    construct proximities from the fitted ensemble object.

    This implementation uses Sparse Matrix Algebra (Inverted Indexing)
    with Gustavson scipy sparse multiplication:
        P = Q W^T
    where Q and W are query (i) and weight (j) sparse matrices as per the
    proximity definitions.

    Parameters
    ----------
    prediction_type : str
        Options are 'regression' or 'classification'.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Optional target values used to infer prediction_type.

    prox_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'gbt'}.

    matrix_type : str
        'sparse' or 'dense'.

    force_nonzero_diag : bool
        Only used for RF-GAP proximities. Whether to inject non-zero diagonal.

    force_symmetric : bool or None
        Whether to force symmetry via block symmetrization.

    normalize_diagonal : bool
        Whether to row-normalize by self-similarity so that the diagonal becomes 1 when applicable

    model_type : str
        One of {'rf', 'et', 'gbt', 'rotf'}.

    allow_semi_supervised : bool
        Whether to allow the transductive semi-supervised heuristic when y
        contains unlabeled entries (-1 and/or NaN depending on task type).
        If False and unlabeled targets are detected, fit() raises an error.

    **kwargs
        Estimator-specific keyword arguments.

    Returns
    -------
    self : object
        Unfitted estimator enhanced with proximity methods.
    """
    prediction_type = infer_prediction_type(prediction_type=prediction_type, y=y)
    validate_model_configuration(
        model_type=model_type,
        prox_method=prox_method,
        prediction_type=prediction_type,
    )
    base_model = get_base_model(model_type=model_type, prediction_type=prediction_type)
    kwargs = sanitize_model_kwargs(model_type=model_type, prox_method=prox_method, kwargs=kwargs)
    validate_model_kwargs(base_model, kwargs)

    class ForestKernel(GAPExtrasMixin, base_model):
        def __init__(
            self,
            prox_method=prox_method,
            matrix_type=matrix_type,
            force_nonzero_diag=force_nonzero_diag,
            force_symmetric=force_symmetric,
            normalize_diagonal=normalize_diagonal,
            allow_semi_supervised=allow_semi_supervised,
            **kwargs,
        ):
            super(ForestKernel, self).__init__(**kwargs)

            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.prediction_type = prediction_type
            self.force_nonzero_diag = force_nonzero_diag
            self.force_symmetric = force_symmetric
            self.normalize_diagonal = normalize_diagonal
            self.model_type = model_type
            self.allow_semi_supervised = allow_semi_supervised

            # Proximity internals
            self.cache = None
            self._adapter = None

            # Partial-label bookkeeping placeholder
            self.idx_labeled_ = None
            self.idx_unlabeled_ = None
            self._n_total_samples = 0
            self._n_train_samples = 0

        def fit(self, X, y, sample_weight=None):
            """
            Fits the tree ensemble and pre-computes the sparse weight matrix W
            necessary for proximity calculations.

            In semi-supervised mode, unlabeled points are not used to fit the
            forest, but are still inserted into the reference set used to build
            proximities. This is a transductive heuristic and is enabled only
            when allow_semi_supervised=True.

            RUNTIME COMPLEXITY: O(N * T * log(N))
                - Ensemble Construction: O(N * T * log(N) * #features sampled at each node)
                - Matrix Construction: O(N * T)

            MEMORY COMPLEXITY: O(N * T)
                - Stores leaf indices and sparse weights.
                - Efficiently sparse: only stores 1 entry per tree per sample.
            """
            # ---------------------------------------------------------
            # STEP 0: Handle partial labels (semi-supervised heuristic)
            # ---------------------------------------------------------
            self._n_total_samples = X.shape[0]

            y = np.asarray(y)
            if self.prediction_type == "regression" and not np.issubdtype(y.dtype, np.floating):
                y = y.astype(np.float32)

            # Detect unlabeled samples depending on task type
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

            has_unlabeled = bool(np.any(mask_unlabeled))

            if has_unlabeled and not self.allow_semi_supervised:
                raise ValueError(
                    "Unlabeled targets were detected in y, but semi-supervised mode is disabled. "
                    "This transductive proximity heuristic is only enabled when "
                    "allow_semi_supervised=True."
                )

            # Preserve original X order through labeled/unlabeled indices
            self.idx_labeled_ = np.flatnonzero(~mask_unlabeled)
            self.idx_unlabeled_ = np.flatnonzero(mask_unlabeled)

            if not has_unlabeled:
                # Case A: Fully supervised
                X_train = X
                y_train = y
                sample_weight_train = sample_weight
            else:
                # Case B: Semi-supervised (transductive heuristic)
                X_train = X[self.idx_labeled_]
                y_train = y[self.idx_labeled_]

                if sample_weight is not None:
                    sample_weight_train = np.asarray(sample_weight)[self.idx_labeled_]
                else:
                    sample_weight_train = None

            self._n_train_samples = X_train.shape[0]
            self.y = y

            # ---------------------------------------------------------
            # STEP 1: Fit underlying ensemble on labeled data only
            # ---------------------------------------------------------
            try:
                super().fit(X_train, y_train, sample_weight=sample_weight_train)
            except TypeError:
                if sample_weight_train is not None:
                    warnings.warn(
                        "sample_weight was provided but is ignored because the selected "
                        "base model does not support it."
                    )
                super().fit(X_train, y_train)

            # Build ensemble adapter
            self._adapter = make_adapter(self, self.model_type)

            # ---------------------------------------------------------
            # STEP 2: Leaf structure for ALL points (original order)
            # ---------------------------------------------------------
            # NOTE: we compute leaves for ALL points (labeled+unlabeled) in ORIGINAL order.
            leaf_matrix_all = self._adapter.get_leaf_matrix(X)
            n_nodes_per_tree = self._adapter.get_n_nodes_per_tree()

            self.cache = initialize_cache(
                leaf_matrix_all=leaf_matrix_all,
                n_nodes_per_tree=n_nodes_per_tree,
                n_samples=X.shape[0],
                idx_labeled=self.idx_labeled_,
                idx_unlabeled=self.idx_unlabeled_,
                n_train_samples=X_train.shape[0],
            )

            # ---------------------------------------------------------
            # STEP 3: OOB / multiplicity structure
            # ---------------------------------------------------------
            if self.prox_method in ["oob", "gap"]:
                # NOTE: OOB / in-bag statistics are computed on TRAINING (labeled) samples only
                oob_labeled = self._adapter.get_oob_mask(X_train)

                if has_unlabeled:
                    T = self.cache.n_trees

                    # Unlabeled rows are treated as OOB everywhere in the transductive heuristic
                    oob_mask_all = np.ones((self.cache.n_samples, T), dtype=np.int8)
                    oob_mask_all[self.idx_labeled_, :] = oob_labeled.astype(np.int8)

                    if self.prox_method == "gap":
                        # Unlabeled rows follow the phantom-target convention: c_j(t)=1 for all trees
                        c_labeled = self._adapter.get_in_bag_counts(X_train).astype(np.float32)
                        c_all = np.ones((self.cache.n_samples, T), dtype=np.float32)
                        c_all[self.idx_labeled_, :] = c_labeled
                    else:
                        c_all = None
                else:
                    oob_mask_all = oob_labeled.astype(np.int8)
                    c_all = (
                        self._adapter.get_in_bag_counts(X_train).astype(np.float32)
                        if self.prox_method == "gap"
                        else None
                    )

                attach_oob_structure(self.cache, oob_mask_all=oob_mask_all, c_all=c_all)

            # ---------------------------------------------------------
            # STEP 4: GBT tree weights
            # ---------------------------------------------------------
            if self.prox_method == "gbt":
                gbt_tree_weights = self._adapter.get_tree_weights(X_train)
                attach_gbt_weights(self.cache, gbt_tree_weights)

            # ---------------------------------------------------------
            # STEP 5: Leaf statistics used by specific proximities
            # ---------------------------------------------------------
            # In semi-supervised mode, these are computed from LABELED incidences only,
            # according to the heuristic implemented in builders.py.
            if self.prox_method == "kerf":
                compute_unit_leaf_mass(self.cache)

            if self.prox_method == "gap":
                compute_multiplicity_leaf_mass(self.cache)

            # ---------------------------------------------------------
            # STEP 6: Build sparse right factor W
            # ---------------------------------------------------------
            # NOTE: W is built for ALL points directly in original order.
            self.cache.W_mat = build_W_matrix(
                self.cache,
                prox_method=self.prox_method,
                force_nonzero_diag=self.force_nonzero_diag,
            )

            return self

        def get_proximities(self):
            """
            Computes the proximity matrix P = Q W^T using sparse matrix multiplication.

            Returns
            -------
            array-like or csr_matrix
                Dense or sparse proximity matrix depending on matrix_type.
            """
            check_is_fitted(self)

            Q_total = build_Q_matrix(
                self.cache,
                prox_method=self.prox_method,
                leaves=self.cache.leaf_matrix_all,
                is_training=True,
                force_nonzero_diag=self.force_nonzero_diag,
            )

            # Fast diagonal normalization (Hadamard trick)
            if self.normalize_diagonal and (
                (self.prox_method == "gap" and self.force_nonzero_diag)
                or self.prox_method == "kerf"
            ):
                diagonal = Q_total.multiply(self.cache.W_mat).sum(axis=1).A.ravel()
                diagonal[diagonal == 0] = 1.0
                csr_row_scale_inplace(Q_total, 1.0 / diagonal)

            # Final sparse multiplication
            if self.force_symmetric and (
                self.prox_method == "gap"
                or (self.prox_method == "kerf" and self.normalize_diagonal)
            ):
                prox_matrix = block_symmetrize(Q_total, self.cache.W_mat)
            else:
                prox_matrix = Q_total.dot(self.cache.W_mat.T)

            return prox_matrix.toarray() if self.matrix_type == "dense" else prox_matrix

        def prox_extend(self, X_new):
            """
            Calculates proximities between new data (rows) and the fitted
            reference data (cols).

            Parameters
            ----------
            X_new : array-like of shape (n_samples_new, n_features)

            Returns
            -------
            array-like or csr_matrix
                Proximities between X_new and the fitted reference data.
            """
            check_is_fitted(self)

            leaves_new = self._adapter.get_leaf_matrix(X_new)

            Q_new = build_Q_matrix(
                self.cache,
                prox_method=self.prox_method,
                leaves=leaves_new,
                is_training=False,
                force_nonzero_diag=self.force_nonzero_diag,
            )

            prox_new = Q_new.dot(self.cache.W_mat.T)
            return prox_new.toarray() if self.matrix_type == "dense" else prox_new

    return ForestKernel(
        prox_method=prox_method,
        matrix_type=matrix_type,
        force_nonzero_diag=force_nonzero_diag,
        force_symmetric=force_symmetric,
        normalize_diagonal=normalize_diagonal,
        allow_semi_supervised=allow_semi_supervised,
        **kwargs,
    )
import numpy as np

from .extras import GAPExtrasMixin
from .config import (
    infer_prediction_type,
    validate_model_configuration,
    get_base_model,
    validate_model_kwargs,
)
from .adapters import make_adapter
from .fit_utils import prepare_reference_split
from .kernel import (
    initialize_cache,
    attach_bootstrap_stats,
    attach_boosted_weights,
    attach_inv_sqrt_leaf_mass,
    attach_inv_inbag_leaf_mass,
    attach_unlabeled_multiplicity_surrogates,
    build_W_matrix,
    build_Q_matrix,
    csr_row_scale_inplace,
    block_symmetrize,
    row_normalize_kernel_block,
    kernel_predict_regression,
    kernel_predict_classification,
)


def ForestKernel(
    prediction_type=None,
    y=None,
    kernel_method="gap",
    matrix_type="sparse",
    force_nonzero_diag=False,
    force_symmetric=None,
    normalize_diagonal=False,
    model_type="rf",
    **kwargs,
):
    """
    Factory function creating a tree ensemble equipped with leaf-space
    kernel maps and kernel construction methods.

    The fitted kernel is represented in factored form as

        K = Q W^T

    where:
    - Q is the query-side leaf feature map
    - W is the reference-side leaf feature map

    In symmetric cases, this reduces to an ordinary dot-product kernel.
    In asymmetric cases such as GAP, this is a bilinear kernel between
    two distinct feature maps.
    """
    prediction_type = infer_prediction_type(prediction_type=prediction_type, y=y)
    validate_model_configuration(
        model_type=model_type,
        kernel_method=kernel_method,
        prediction_type=prediction_type,
        kwargs=kwargs,
    )
    base_model = get_base_model(
        model_type=model_type,
        prediction_type=prediction_type,
    )
    validate_model_kwargs(base_model, kwargs)

    class ForestKernel(GAPExtrasMixin):
        def __init__(
            self,
            kernel_method=kernel_method,
            matrix_type=matrix_type,
            force_nonzero_diag=force_nonzero_diag,
            force_symmetric=force_symmetric,
            normalize_diagonal=normalize_diagonal,
            **model_kwargs,
        ):
            self.kernel_method = kernel_method
            self.matrix_type = matrix_type
            self.prediction_type = prediction_type
            self.force_nonzero_diag = force_nonzero_diag
            self.force_symmetric = force_symmetric
            self.normalize_diagonal = normalize_diagonal
            self.model_type = model_type

            # Underlying model kwargs
            self.model_kwargs = dict(model_kwargs)

            # Kernel internals
            self.cache = None
            self._adapter = None
            self.y = None

            # Real underlying fitted forest / tree ensemble object
            self.forest_ = base_model(**self.model_kwargs)

        def _check_fitted(self):
            if self._adapter is None or self.cache is None:
                raise AttributeError(
                    "This ForestKernel instance is not fitted yet. Call 'fit' first."
                )

        def _check_forest_fitted(self):
            if self._adapter is None:
                raise AttributeError(
                    "The underlying forest is not fitted yet. Call 'fit' first."
                )

        def fit_forest(self, X, y, idx_unlabeled=None, **fit_kwargs):
            """
            Fit only the underlying ensemble, without building kernel metadata.

            Useful for benchmarking forest fitting separately from kernel
            preprocessing.
            """
            fit_data = prepare_reference_split(
                X=X,
                y=y,
                idx_unlabeled=idx_unlabeled,
            )

            if fit_data["has_unlabeled"] and self.kernel_method == "gap" and not self.force_nonzero_diag:
                raise ValueError(
                    "Transductive GAP currently requires force_nonzero_diag=True. "
                    "The zero-diagonal transductive GAP variant is not supported "
                    "because it relies on signed private correction coordinates."
                )

            self.y = fit_data["y"]

            self.forest_.fit(
                fit_data["X_train"],
                fit_data["y_train"],
                **fit_kwargs,
            )
            self._adapter = make_adapter(self.forest_, self.model_type)

            return {
                "X": fit_data["X"],
                "X_train": fit_data["X_train"],
                "idx_labeled": fit_data["idx_labeled"],
                "idx_unlabeled": fit_data["idx_unlabeled"],
                "has_unlabeled": fit_data["has_unlabeled"],
            }

        def build_kernel_cache(
            self,
            X,
            X_train,
            idx_labeled,
            idx_unlabeled,
            has_unlabeled,
        ):
            """
            Build all post-fit kernel metadata and the reference-side leaf map W.

            Useful for benchmarking kernel preprocessing separately from
            forest fitting.
            """
            self._check_forest_fitted()

            # ---------------------------------------------------------
            # STEP 1: initialize cache from leaf structure on ALL points
            # ---------------------------------------------------------
            leaf_matrix = self._adapter.get_leaf_matrix(X)
            n_nodes_per_tree = self._adapter.get_n_nodes_per_tree()

            self.cache = initialize_cache(
                leaf_matrix=leaf_matrix,
                n_nodes_per_tree=n_nodes_per_tree,
                n_samples=X.shape[0],
                idx_labeled=idx_labeled,
                idx_unlabeled=idx_unlabeled,
                n_train_samples=X_train.shape[0],
            )

            # ---------------------------------------------------------
            # STEP 2: attach OOB / multiplicity structure to cache when needed
            # ---------------------------------------------------------
            if self.kernel_method in ["oob", "gap"]:
                oob_labeled = self._adapter.get_oob_mask(X_train)

                if has_unlabeled:
                    T = self.cache.n_trees

                    # Unlabeled rows are treated as OOB in every tree, so the OOB
                    # mask is initialized with ones and then labeled rows are filled in.
                    oob_mask = np.ones((self.cache.n_samples, T), dtype=np.int8)
                    oob_mask[idx_labeled, :] = oob_labeled.astype(np.int8)

                    if self.kernel_method == "gap":
                        # Unlabeled rows are initialized with placeholder zeros.
                        # The GAP builders later replace these target-side values using
                        # the cached treewise surrogates.
                        inbag_counts_labeled = self._adapter.get_in_bag_counts(X_train).astype(np.float32)
                        inbag_counts = np.zeros((self.cache.n_samples, T), dtype=np.float32)
                        inbag_counts[idx_labeled, :] = inbag_counts_labeled
                    else:
                        inbag_counts = None
                else:
                    oob_mask = oob_labeled.astype(np.int8)
                    inbag_counts = (
                        self._adapter.get_in_bag_counts(X_train).astype(np.float32)
                        if self.kernel_method == "gap"
                        else None
                    )

                attach_bootstrap_stats(
                    self.cache,
                    oob_mask=oob_mask,
                    inbag_counts=inbag_counts,
                )

            # ---------------------------------------------------------
            # STEP 3: attach tree weights when needed
            # ---------------------------------------------------------
            if self.kernel_method == "boosted":
                boosted_tree_weights = self._adapter.get_tree_weights(X_train)
                attach_boosted_weights(self.cache, boosted_tree_weights)

            # ---------------------------------------------------------
            # STEP 4: attach kernel-specific cached statistics
            # ---------------------------------------------------------
            if self.kernel_method == "kerf":
                attach_inv_sqrt_leaf_mass(self.cache)

            if self.kernel_method == "gap":
                attach_inv_inbag_leaf_mass(self.cache)

                if has_unlabeled:
                    attach_unlabeled_multiplicity_surrogates(self.cache)

            # ---------------------------------------------------------
            # STEP 5: build the reference-side feature map W
            # ---------------------------------------------------------
            self.cache.W_mat = build_W_matrix(
                self.cache,
                kernel_method=self.kernel_method,
                force_nonzero_diag=self.force_nonzero_diag,
            )

            return self

        def fit(self, X, y, idx_unlabeled=None, **fit_kwargs):
            """
            Fit the ensemble and precompute the reference-side leaf map W.

            In transductive mode, unlabeled samples are excluded from the
            ensemble fit, but retained in the reference set used for kernel
            construction through the transductive heuristic implemented in
            builders.py.

            Additional keyword arguments are forwarded to the underlying base
            estimator fit() method.
            """
            fit_context = self.fit_forest(
                X,
                y,
                idx_unlabeled=idx_unlabeled,
                **fit_kwargs,
            )
            self.build_kernel_cache(**fit_context)
            return self

        def get_reference_map(self):
            """
            Return the fitted reference-side leaf feature map W.
            """
            self._check_fitted()
            return self.cache.W_mat

        def get_train_query_map(self):
            """
            Return the query-side leaf feature map Q on the fitted reference set.
            """
            self._check_fitted()

            Q_train = build_Q_matrix(
                self.cache,
                kernel_method=self.kernel_method,
                leaves=self.cache.leaf_matrix,
                is_training=True,
                force_nonzero_diag=self.force_nonzero_diag,
            )

            if self.normalize_diagonal and (
                (self.kernel_method == "gap" and self.force_nonzero_diag)
                or self.kernel_method == "kerf"
            ):
                diagonal = Q_train.multiply(self.cache.W_mat).sum(axis=1).A.ravel()
                diagonal[diagonal == 0] = 1.0
                csr_row_scale_inplace(Q_train, 1.0 / diagonal)

            return Q_train

        def get_query_map(self, X_new):
            """
            Return the out-of-sample query-side leaf feature map Q(X_new).
            """
            self._check_fitted()

            leaves_new = self._adapter.get_leaf_matrix(X_new)

            return build_Q_matrix(
                self.cache,
                kernel_method=self.kernel_method,
                leaves=leaves_new,
                is_training=False,
                force_nonzero_diag=self.force_nonzero_diag,
            )

        def get_kernel_from_query_map(self, Q):
            """
            Form a kernel block K = Q W^T from a query-side map Q.
            """
            self._check_fitted()
            K = Q.dot(self.cache.W_mat.T)
            return K.toarray() if self.matrix_type == "dense" else K

        def get_kernel(self):
            """
            Return the fitted kernel matrix on the reference set.
            """
            self._check_fitted()

            Q_train = self.get_train_query_map()

            if self.force_symmetric and (
                self.kernel_method == "gap"
                or (self.kernel_method == "kerf" and self.normalize_diagonal)
            ):
                K = block_symmetrize(Q_train, self.cache.W_mat)
            else:
                K = Q_train.dot(self.cache.W_mat.T)

            return K.toarray() if self.matrix_type == "dense" else K

        def kernel_extend(self, X_new):
            """
            Return the kernel block between X_new and the fitted reference set.
            """
            self._check_fitted()
            Q_new = self.get_query_map(X_new)
            return self.get_kernel_from_query_map(Q_new)

        def kernel_predict(self, X_new):
            """
            Proximity-weighted prediction on new samples using the fitted
            labeled reference subset only.

            In transductive mode, unlabeled reference points still influence
            the geometry through the cache construction, but do not directly
            vote in the prediction step.
            """
            self._check_fitted()

            Q_new = self.get_query_map(X_new)
            K_ext = self.get_kernel_from_query_map(Q_new)

            idx_labeled = self.cache.idx_labeled
            if idx_labeled is None:
                idx_labeled = np.arange(K_ext.shape[1], dtype=np.int64)

            K_ext_labeled = K_ext[:, idx_labeled]
            y_labeled = np.asarray(self.y[idx_labeled]).ravel()

            K_ext_labeled = row_normalize_kernel_block(K_ext_labeled, self.kernel_method)

            if self.prediction_type == "regression":
                return kernel_predict_regression(K_ext_labeled, y_labeled)

            return kernel_predict_classification(K_ext_labeled, y_labeled)

        # ---------------------------------------------------------
        # Convenience accessors to the underlying fitted ensemble
        # ---------------------------------------------------------
        def predict_forest(self, X):
            self._check_forest_fitted()
            return self.forest_.predict(np.asarray(X))

        def predict_proba_forest(self, X):
            self._check_forest_fitted()
            if not hasattr(self.forest_, "predict_proba"):
                raise AttributeError("Underlying forest does not implement predict_proba().")
            return self.forest_.predict_proba(np.asarray(X))

        def apply(self, X):
            self._check_forest_fitted()
            if not hasattr(self.forest_, "apply"):
                raise AttributeError("Underlying forest does not implement apply().")
            return self.forest_.apply(np.asarray(X))

        def get_params(self):
            """
            Return wrapper and model parameters.
            """
            return {
                "kernel_method": self.kernel_method,
                "matrix_type": self.matrix_type,
                "force_nonzero_diag": self.force_nonzero_diag,
                "force_symmetric": self.force_symmetric,
                "normalize_diagonal": self.normalize_diagonal,
                "model_type": self.model_type,
                **self.model_kwargs,
            }

    return ForestKernel(
        kernel_method=kernel_method,
        matrix_type=matrix_type,
        force_nonzero_diag=force_nonzero_diag,
        force_symmetric=force_symmetric,
        normalize_diagonal=normalize_diagonal,
        **kwargs,
    )
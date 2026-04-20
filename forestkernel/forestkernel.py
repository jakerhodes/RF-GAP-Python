import numpy as np

from .extras import GAPExtrasMixin
from .config import (
    infer_prediction_type,
    validate_model_configuration,
    get_base_model,
    validate_model_kwargs,
)
from .adapters import make_adapter
from .kernel import (
    initialize_cache,
    attach_bootstrap_stats,
    attach_boosted_weights,
    attach_inv_sqrt_leaf_mass,
    attach_inv_inbag_leaf_mass,
    build_W_matrix,
    build_Q_matrix,
    csr_row_scale_inplace,
    block_symmetrize,
    format_output_matrix,
    row_normalize_kernel_block,
    kernel_predict_regression,
    kernel_predict_classification,
)


def ForestKernel(
    prediction_type=None,
    y=None,
    kernel_method="gap",
    force_nonzero_diag=False,
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
            force_nonzero_diag=force_nonzero_diag,
            **model_kwargs,
        ):
            self.kernel_method = kernel_method
            self.prediction_type = prediction_type
            self.force_nonzero_diag = force_nonzero_diag
            self.model_type = model_type

            # Underlying model kwargs
            self.model_kwargs = dict(model_kwargs)

            # Kernel internals
            self.cache = None
            self.fit_forest_context_ = None
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

        def set_kernel_config(
            self,
            kernel_method=None,
            force_nonzero_diag=None,
        ):
            """
            Update the active kernel construction configuration without refitting
            the forest.
            """
            next_kernel_method = self.kernel_method if kernel_method is None else kernel_method
            next_force_nonzero_diag = (
                self.force_nonzero_diag if force_nonzero_diag is None else force_nonzero_diag
            )

            validate_model_configuration(
                model_type=self.model_type,
                kernel_method=next_kernel_method,
                prediction_type=self.prediction_type,
                kwargs=self.model_kwargs,
            )

            self.kernel_method = next_kernel_method
            self.force_nonzero_diag = next_force_nonzero_diag

            return self

        def fit_forest(self, X, y, **fit_kwargs):
            """
            Fit only the underlying ensemble, without building kernel metadata.

            Useful for benchmarking forest fitting separately from kernel
            preprocessing.
            """
            X = np.asarray(X)
            y = np.asarray(y)

            self.y = y
            self.forest_.fit(X, y, **fit_kwargs)
            self._adapter = make_adapter(self.forest_, self.model_type)

            self.fit_forest_context_ = {
                "X": X,
            }

            # Any old cache becomes invalid after refitting the forest.
            self.cache = None

            return {
                "X": X,
            }

        def build_kernel_cache(
            self,
            kernel_method=None,
            force_nonzero_diag=None,
        ):
            """
            Build all post-fit kernel metadata and the reference-side leaf map W.

            Useful for benchmarking kernel preprocessing separately from
            forest fitting.
            """
            self._check_forest_fitted()

            self.set_kernel_config(
                kernel_method=kernel_method,
                force_nonzero_diag=force_nonzero_diag,
            )

            if self.fit_forest_context_ is None:
                raise ValueError(
                    "No stored fit context available. "
                    "Call fit(...) or fit_forest(...) first."
                )

            X = self.fit_forest_context_["X"]

            # ---------------------------------------------------------
            # STEP 1: initialize cache from leaf structure on all points
            # ---------------------------------------------------------
            leaf_matrix = self._adapter.get_leaf_matrix(X)
            n_nodes_per_tree = self._adapter.get_n_nodes_per_tree()

            self.cache = initialize_cache(
                leaf_matrix=leaf_matrix,
                n_nodes_per_tree=n_nodes_per_tree,
                n_samples=X.shape[0],
            )

            # ---------------------------------------------------------
            # STEP 2: attach OOB / multiplicity structure to cache when needed
            # ---------------------------------------------------------
            if self.kernel_method in ["oob", "gap"]:
                oob_mask = self._adapter.get_oob_mask(X).astype(np.int8)
                inbag_counts = (
                    self._adapter.get_in_bag_counts(X).astype(np.float32)
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
                boosted_tree_weights = self._adapter.get_tree_weights(X)
                attach_boosted_weights(self.cache, boosted_tree_weights)

            # ---------------------------------------------------------
            # STEP 4: attach kernel-specific cached statistics
            # ---------------------------------------------------------
            if self.kernel_method == "kerf":
                attach_inv_sqrt_leaf_mass(self.cache)

            if self.kernel_method == "gap":
                attach_inv_inbag_leaf_mass(self.cache)

            # ---------------------------------------------------------
            # STEP 5: build the reference-side feature map W
            # ---------------------------------------------------------
            self.cache.W_mat = build_W_matrix(
                self.cache,
                kernel_method=self.kernel_method,
                force_nonzero_diag=self.force_nonzero_diag,
            )

            return self

        def fit(self, X, y, **fit_kwargs):
            """
            Fit the ensemble and precompute the reference-side leaf map W.

            Additional keyword arguments are forwarded to the underlying base
            estimator fit() method.
            """
            self.fit_forest(
                X,
                y,
                **fit_kwargs,
            )
            self.build_kernel_cache()
            return self

        def get_reference_map(self, return_dense=False):
            """
            Return the fitted reference-side leaf feature map W.
            """
            self._check_fitted()
            return format_output_matrix(self.cache.W_mat, return_dense=return_dense)

        def get_train_query_map(self, normalize_diagonal=False, return_dense=False):
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

            if normalize_diagonal and (
                (self.kernel_method == "gap" and self.force_nonzero_diag)
                or self.kernel_method == "kerf"
            ):
                # Hadamard trick to get the diagonal of Q W^T without forming the full kernel matrix.
                diagonal = Q_train.multiply(self.cache.W_mat).sum(axis=1).A.ravel()
                diagonal[diagonal == 0] = 1.0
                csr_row_scale_inplace(Q_train, 1.0 / diagonal)

            return format_output_matrix(Q_train, return_dense=return_dense)

        def get_query_map(self, X_new, return_dense=False):
            """
            Return the out-of-sample query-side leaf feature map Q(X_new).
            """
            self._check_fitted()

            leaves_new = self._adapter.get_leaf_matrix(X_new)

            Q_new = build_Q_matrix(
                self.cache,
                kernel_method=self.kernel_method,
                leaves=leaves_new,
                is_training=False,
                force_nonzero_diag=self.force_nonzero_diag,
            )

            return format_output_matrix(Q_new, return_dense=return_dense)

        def get_kernel_from_query_map(self, Q, return_dense=False):
            """
            Form a kernel block K = Q W^T from a query-side map Q.
            """
            self._check_fitted()
            K = Q.dot(self.cache.W_mat.T)
            return format_output_matrix(K, return_dense=return_dense)

        def get_kernel(self, force_symmetric=False, normalize_diagonal=False, return_dense=False):
            """
            Return the fitted kernel matrix on the reference set.
            """
            self._check_fitted()

            Q_train = self.get_train_query_map(
                normalize_diagonal=normalize_diagonal
            )

            if force_symmetric and (
                self.kernel_method == "gap"
                or (self.kernel_method == "kerf" and normalize_diagonal)
            ):
                K = block_symmetrize(Q_train, self.cache.W_mat)
            else:
                K = Q_train.dot(self.cache.W_mat.T)

            return format_output_matrix(K, return_dense=return_dense)

        def kernel_extend(self, X_new, return_dense=False):
            """
            Return the kernel block between X_new and the fitted reference set.
            """
            self._check_fitted()
            Q_new = self.get_query_map(X_new)
            return self.get_kernel_from_query_map(Q_new, return_dense=return_dense)

        def kernel_predict(self, X_new):
            """
            Proximity-weighted prediction on new samples using the fitted
            reference set.
            """
            self._check_fitted()

            Q_new = self.get_query_map(X_new)
            K_ext = self.get_kernel_from_query_map(Q_new)

            y_ref = np.asarray(self.y).ravel()
            K_ext = row_normalize_kernel_block(K_ext, self.kernel_method)

            if self.prediction_type == "regression":
                return kernel_predict_regression(K_ext, y_ref)

            return kernel_predict_classification(K_ext, y_ref)

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
                "force_nonzero_diag": self.force_nonzero_diag,
                "model_type": self.model_type,
                **self.model_kwargs,
            }

    return ForestKernel(
        kernel_method=kernel_method,
        force_nonzero_diag=force_nonzero_diag,
        **kwargs,
    )
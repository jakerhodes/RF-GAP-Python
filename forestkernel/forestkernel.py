import numpy as np

from sklearn.utils.validation import check_is_fitted

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
    attach_gbt_weights,
    attach_inv_sqrt_leaf_mass,
    attach_inv_inbag_leaf_mass,
    attach_unlabeled_multiplicity_surrogates,
    build_W_matrix,
    build_Q_matrix,
    csr_row_scale_inplace,
    block_symmetrize,
)
from .fit_utils import prepare_fit_inputs


def ForestKernel(
    prediction_type=None,
    y=None,
    kernel_method="gap",
    matrix_type="sparse",
    force_nonzero_diag=False,
    force_symmetric=None,
    normalize_diagonal=False,
    model_type="rf",
    allow_semi_supervised=False,
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

    class ForestKernel(GAPExtrasMixin, base_model):
        def __init__(
            self,
            kernel_method=kernel_method,
            matrix_type=matrix_type,
            force_nonzero_diag=force_nonzero_diag,
            force_symmetric=force_symmetric,
            normalize_diagonal=normalize_diagonal,
            allow_semi_supervised=allow_semi_supervised,
            **kwargs,
        ):
            super(ForestKernel, self).__init__(**kwargs)

            self.kernel_method = kernel_method
            self.matrix_type = matrix_type
            self.prediction_type = prediction_type
            self.force_nonzero_diag = force_nonzero_diag
            self.force_symmetric = force_symmetric
            self.normalize_diagonal = normalize_diagonal
            self.model_type = model_type
            self.allow_semi_supervised = allow_semi_supervised

            # Kernel internals
            self.cache = None
            self._adapter = None
            self.y = None


        def fit_forest(self, X, y, **fit_kwargs):
            """
            Fit only the underlying ensemble, without building kernel metadata.

            Useful for benchmarking forest fitting separately from kernel
            preprocessing.
            """
            fit_data = prepare_fit_inputs(
                X=X,
                y=y,
                fit_kwargs=fit_kwargs,
                prediction_type=self.prediction_type,
            )
            if fit_data["has_unlabeled"]:
                if not self.allow_semi_supervised:
                    raise ValueError(
                        "Unlabeled targets were detected in y, but semi-supervised mode is disabled. "
                        "Set allow_semi_supervised=True to enable the transductive kernel heuristic."
                    )
                if self.kernel_method == "gap" and not self.force_nonzero_diag:
                    raise ValueError(
                        "Semi-supervised GAP currently requires force_nonzero_diag=True. "
                        "The zero-diagonal semi-supervised GAP variant is not supported "
                        "because it relies on signed private correction coordinates."
                    )
                
            self.y = fit_data["y"]

            super().fit(
                fit_data["X_train"],
                fit_data["y_train"],
                **fit_data["fit_kwargs_train"],
            )
            self._adapter = make_adapter(self, self.model_type)

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
            check_is_fitted(self)

            # ---------------------------------------------------------
            # STEP 1: initialize cache from leaf structure on ALL points
            # ---------------------------------------------------------
            leaf_matrix_all = self._adapter.get_leaf_matrix(X)
            n_nodes_per_tree = self._adapter.get_n_nodes_per_tree()

            max_ids = leaf_matrix_all.max(axis=0)
            bad = np.where(max_ids >= np.asarray(n_nodes_per_tree))[0]
            
            print("max raw node ids per tree:", max_ids)
            print("n_nodes_per_tree:", n_nodes_per_tree)
            print("bad trees:", bad)

            self.cache = initialize_cache(
                leaf_matrix_all=leaf_matrix_all,
                n_nodes_per_tree=n_nodes_per_tree,
                n_samples=X.shape[0],
                idx_labeled=idx_labeled,
                idx_unlabeled=idx_unlabeled,
                n_train_samples=X_train.shape[0],
            )

            max_ids = leaf_matrix_all.max(axis=0)
            bad = np.where(max_ids >= np.asarray(n_nodes_per_tree))[0]
            
            print("max raw node ids per tree:", max_ids)
            print("n_nodes_per_tree:", n_nodes_per_tree)
            print("bad trees:", bad)

            # ---------------------------------------------------------
            # STEP 2: attach OOB / multiplicity structure to cache when needed
            # ---------------------------------------------------------
            if self.kernel_method in ["oob", "gap"]:
                oob_labeled = self._adapter.get_oob_mask(X_train)

                if has_unlabeled:
                    T = self.cache.n_trees

                    # Unlabeled rows are treated as OOB in every tree, so the OOB mask is initialized with ones and then labeled rows are filled in.
                    oob_mask_all = np.ones((self.cache.n_samples, T), dtype=np.int8)
                    oob_mask_all[idx_labeled, :] = oob_labeled.astype(np.int8)

                    if self.kernel_method == "gap":
                        # Unlabeled rows are initialized with placeholder zeros.
                        # The GAP builders later replace these target-side values using
                        # the cached treewise surrogates.
                        c_labeled = self._adapter.get_in_bag_counts(X_train).astype(np.float32)
                        c_all = np.zeros((self.cache.n_samples, T), dtype=np.float32)
                        c_all[idx_labeled, :] = c_labeled
                    else:
                        c_all = None
                else:
                    oob_mask_all = oob_labeled.astype(np.int8)
                    c_all = (
                        self._adapter.get_in_bag_counts(X_train).astype(np.float32)
                        if self.kernel_method == "gap"
                        else None
                    )

                attach_bootstrap_stats(
                    self.cache,
                    oob_mask_all=oob_mask_all,
                    c_all=c_all,
                )

            # ---------------------------------------------------------
            # STEP 3: attach tree weights when needed
            # ---------------------------------------------------------
            if self.kernel_method == "gbt":
                gbt_tree_weights = self._adapter.get_tree_weights(X_train)
                attach_gbt_weights(self.cache, gbt_tree_weights)

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

        def fit(self, X, y, **fit_kwargs):
            """
            Fit the ensemble and precompute the reference-side leaf map W.

            In semi-supervised mode, unlabeled samples are excluded from the
            ensemble fit, but retained in the reference set used for kernel
            construction through the transductive heuristic implemented in
            builders.py.

            Additional keyword arguments are forwarded to the underlying base
            estimator fit() method. In semi-supervised mode, sample-aligned
            fit kwargs are restricted to labeled rows.
            """
            fit_context = self.fit_forest(X, y, **fit_kwargs)
            self.build_kernel_cache(**fit_context)
            return self

        def get_reference_map(self):
            """
            Return the fitted reference-side leaf feature map W.
            """
            check_is_fitted(self)
            return self.cache.W_mat

        def get_train_query_map(self):
            """
            Return the query-side leaf feature map Q on the fitted reference set.
            """
            check_is_fitted(self)

            Q_train = build_Q_matrix(
                self.cache,
                kernel_method=self.kernel_method,
                leaves=self.cache.leaf_matrix_all,
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
            check_is_fitted(self)

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
            check_is_fitted(self)
            K = Q.dot(self.cache.W_mat.T)
            return K.toarray() if self.matrix_type == "dense" else K

        def get_kernel(self):
            """
            Return the fitted kernel matrix on the reference set.
            """
            check_is_fitted(self)

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
            check_is_fitted(self)
            Q_new = self.get_query_map(X_new)
            return self.get_kernel_from_query_map(Q_new)

    return ForestKernel(
        kernel_method=kernel_method,
        matrix_type=matrix_type,
        force_nonzero_diag=force_nonzero_diag,
        force_symmetric=force_symmetric,
        normalize_diagonal=normalize_diagonal,
        allow_semi_supervised=allow_semi_supervised,
        **kwargs,
    )
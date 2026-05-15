# forestkernel/encoder.py

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from .adapters import make_adapter
from .maps import (
    initialize_cache,
    attach_bootstrap_stats,
    attach_boosted_weights,
    attach_inv_sqrt_leaf_mass,
    attach_inv_inbag_leaf_mass,
    build_W_matrix,
    build_Q_matrix,
    augment_leaf_maps,
    block_symmetrize,
    format_output_matrix,
)


class LeafEncoder(TransformerMixin, BaseEstimator):
    """
    Sparse forest leaf encoder.

    The fitted encoder represents forest kernels in factored form

        P = Q W^T,

    where Q is the query-side leaf map and W is the reference-side leaf map.
    Both Q and W are highly sparse (at most T nonzeros per row), so the
    factorization enables efficient storage and computation without
    materializing the full kernel matrix P.

    The encoder provides direct access to the leaf maps Q and W for custom downstream tasks
    (e.g., kernel methods, manifold learning, dimensionality reduction, visualization...),
    as well as higher-level utilities such as `kernel()` and `kernel_extend()` for explicitly
    computing the kernel/proximity matrix P (dot product in leaf space) in dense or sparse form.

    While forming P is more expensive than working with the sparse maps directly,
    the sparse factorization keeps construction much more scalable than explicit,
    dense pairwise comparisons.
    """

    def __init__(self, forest=None, weight_scheme="uniform"):
        """
        Initialize the leaf encoder.

        Parameters
        ----------
        forest : BaseEstimator, default=None
            The underlying tree ensemble (e.g., RandomForestRegressor).
            This model will be cloned and fitted during `fit()`.

        weight_scheme : str, default="uniform"
            Leaf-weighting scheme used to build the query and reference leaf maps.
            Supported: {"uniform", "oob", "gap", "kerf", "boosted"}.
        """
        self.forest = forest
        self.weight_scheme = weight_scheme

    def _check_forest_fitted(self):
        check_is_fitted(self, attributes=["forest_"])
    
    def _check_fitted(self):
        check_is_fitted(self, attributes=["forest_", "cache_"])
    
        if self.cache_ is None:
            raise NotFittedError(
                "This LeafEncoder instance is not fitted yet. "
                "Call `fit(...)` first."
            )

    def _format(self, matrix, return_dense=False):
        return format_output_matrix(matrix, return_dense=return_dense)

    def _fit_forest(self, X, y, **fit_kwargs):
        """
        Fit only the underlying ensemble.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
    
        if self.forest is None:
            raise ValueError("`forest` must be provided.")
    
        adapter = make_adapter(
            self.forest,
            weight_scheme=self.weight_scheme,
        )
    
        adapter.fit(X, y, **fit_kwargs)
    
        self.forest_ = adapter
        self.X_fit_ = X
        self.y_ = y
        self.classes_ = (
            np.unique(y)
            if callable(getattr(adapter.estimator, "predict_proba", None))
            else None
        )
        self.cache_ = None
    
        return self

    def _build_cache(self):
        """
        Build leaf-map metadata and the fitted query map Q and reference map W.
        """
        self._check_forest_fitted()

        X = self.X_fit_

        leaf_matrix = self.forest_.get_leaf_matrix(X)
        n_nodes_per_tree = self.forest_.get_n_nodes_per_tree()

        self.cache_ = initialize_cache(
            leaf_matrix=leaf_matrix,
            n_nodes_per_tree=n_nodes_per_tree,
            n_samples=X.shape[0],
        )

        if self.weight_scheme in ("oob", "gap"):
            oob_mask = self.forest_.get_oob_mask(X).astype(np.int8)

            inbag_counts = (
                self.forest_.get_in_bag_counts(X).astype(np.float32)
                if self.weight_scheme == "gap"
                else None
            )

            attach_bootstrap_stats(
                self.cache_,
                oob_mask=oob_mask,
                inbag_counts=inbag_counts,
            )

        if self.weight_scheme == "boosted":
            boosted_tree_weights = self.forest_.get_tree_weights(X)
            attach_boosted_weights(self.cache_, boosted_tree_weights)

        if self.weight_scheme == "kerf":
            attach_inv_sqrt_leaf_mass(self.cache_)

        if self.weight_scheme == "gap":
            attach_inv_inbag_leaf_mass(self.cache_)

        self.cache_.Q_mat = build_Q_matrix(
            self.cache_,
            weight_scheme=self.weight_scheme,
            leaves=self.cache_.leaf_matrix,
            is_training=True,
        )

        self.cache_.W_mat = build_W_matrix(
            self.cache_,
            weight_scheme=self.weight_scheme,
        )

        return self

    def set_weight_scheme(self, weight_scheme):
        """
        Update the active leaf-weighting scheme without refitting the forest.
        """
        self._check_forest_fitted()
    
        old_scheme = self.weight_scheme
        old_cache = getattr(self, "cache_", None)
    
        try:
            self.forest_.validate_weight_scheme(weight_scheme)
            self.weight_scheme = weight_scheme
    
            if weight_scheme != old_scheme or old_cache is None:
                self._build_cache()
    
        except Exception:
            self.weight_scheme = old_scheme
            self.cache_ = old_cache
            raise
    
        return self

    def fit(self, X, y, **fit_kwargs):
        """
        Fit the ensemble and build fitted leaf-map metadata.
        """
        self._fit_forest(X, y, **fit_kwargs)
        self._build_cache()
        return self

    def fit_transform(self, X, y, return_dense=False, **fit_kwargs):
        """
        Fit the encoder and return the fitted training query map Q.
        """
        self.fit(X, y, **fit_kwargs)
        return self.training_query_map(return_dense=return_dense)

    def training_query_map(self, return_dense=False):
        """
        Return the fitted training query-side leaf map Q.

        This may differ from transform(X_fit_) for weighting schemes with
        training-specific behavior (e.g., OOB, GAP).
        """
        self._check_fitted()
        return self._format(self.cache_.Q_mat, return_dense=return_dense)

    def reference_map(self, return_dense=False):
        """
        Return the fitted reference-side leaf map W.

        For symmetric kernels, this is the same as `training_query_map()`.
        For asymmetric kernels such as GAP, W may differ from Q.
        """
        self._check_fitted()
        return self._format(self.cache_.W_mat, return_dense=return_dense)

    def transform(self, X, return_dense=False):
        """
        Return the inductive query-side leaf map Q(X).
        """
        self._check_fitted()

        leaves = self.forest_.get_leaf_matrix(np.asarray(X))

        Q = build_Q_matrix(
            self.cache_,
            weight_scheme=self.weight_scheme,
            leaves=leaves,
            is_training=False,
        )

        return self._format(Q, return_dense=return_dense)

    def kernel(self, force_symmetric=False, adjust_diagonal=False, return_dense=False):
        """
        Return the fitted train-train forest kernel matrix P = Q W^T.
        """
        self._check_fitted()

        Q = self.cache_.Q_mat
        W = self.cache_.W_mat

        Q, W = augment_leaf_maps(
            self.cache_,
            self.weight_scheme,
            Q,
            W,
            adjust_diagonal=adjust_diagonal,
            is_training=True,
        )

        if force_symmetric and self.weight_scheme in {"gap"}:
            P = block_symmetrize(Q, W)
        else:
            P = Q.dot(W.T)

        return self._format(P, return_dense=return_dense)

    def kernel_extend(self, X, return_dense=False):
        """
        Extend the fitted forest kernel to new samples.

        Returns the out-of-sample kernel block

            P_new = Q(X) W_train^T,

        between new query samples and the fitted reference set.
        """
        self._check_fitted()

        Q = self.transform(X, return_dense=False)
        P = Q.dot(self.cache_.W_mat.T)

        return self._format(P, return_dense=return_dense)
# Imports
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from packaging.version import Version as LooseVersion

if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble._forest import _generate_sample_indices
else:
    # Before sklearn version 0.24, forest was public.
    from sklearn.ensemble.forest import _generate_unsampled_indices
    from sklearn.ensemble.forest import _generate_sample_indices


def RFGAP(
    prediction_type=None,
    y=None,
    prox_method="rfgap",
    matrix_type="sparse",
    triangular=False,
    non_zero_diagonal=False,
    force_symmetric=False,
    **kwargs,
):
    """
    Factory method creating a benchmark-oriented legacy random-forest proximity
    class based on RandomForestClassifier or RandomForestRegressor.

    This version is simplified to support clean benchmarking against ForestKernel
    with three stages:
        1) fit_forest(...)
        2) build_proximity_cache(...)
        3) get_proximities()

    Parameters
    ----------
    prediction_type : str or None
        One of {'classification', 'regression'}. If None, infer from y.

    y : array-like or None
        Optional target values used to infer classification vs regression.

    prox_method : str
        One of {'original', 'oob', 'rfgap'}.

    matrix_type : str
        One of {'sparse', 'dense'}.

    triangular : bool
        If True, only the upper triangle is computed for 'original' and 'oob'.

    non_zero_diagonal : bool
        Only used for 'rfgap'. If True, RF-GAP diagonal is filled using the
        original legacy rule.

    force_symmetric : bool
        Whether to symmetrize the final sparse proximity matrix.

    **kwargs
        Keyword arguments passed to the underlying sklearn random forest.
    """
    if prediction_type is None and y is None:
        prediction_type = "classification"

    if prediction_type is None and y is not None:
        if isinstance(y, pd.Series):
            y_array = y.to_numpy()
        else:
            y_array = np.asarray(y)

        try:
            if np.issubdtype(y_array.dtype, np.floating):
                prediction_type = "regression"
            else:
                prediction_type = "classification"
        except TypeError:
            prediction_type = "classification"

    if prediction_type == "classification":
        rf = RandomForestClassifier
    elif prediction_type == "regression":
        rf = RandomForestRegressor
    else:
        raise ValueError("prediction_type must be 'classification' or 'regression'.")

    class RFGAP(rf):
        def __init__(
            self,
            prox_method=prox_method,
            matrix_type=matrix_type,
            triangular=triangular,
            non_zero_diagonal=non_zero_diagonal,
            force_symmetric=force_symmetric,
            **kwargs,
        ):
            super(RFGAP, self).__init__(**kwargs)

            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.triangular = triangular
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal
            self.force_symmetric = force_symmetric

            # Benchmark-oriented internals
            self.y = None
            self.n = None
            self._adapter_ready = False

            self.leaf_matrix = None
            self.leaf_matrix_test = None

            self.oob_indices = None
            self.in_bag_counts = None
            self.in_bag_indices = None
            self.in_bag_leaves = None
            self.oob_leaves = None

        def _get_oob_samples(self, X):
            """
            Helper for get_oob_indices.
            """
            n = len(X)
            oob_samples = []
            for tree in self.estimators_:
                oob_idx = _generate_unsampled_indices(tree.random_state, n, n)
                oob_samples.append(oob_idx)
            return oob_samples

        def get_oob_indices(self, X):
            """
            Return OOB indicator matrix of shape (n_samples, n_estimators).
            """
            n = len(X)
            num_trees = self.n_estimators
            oob_matrix = np.zeros((n, num_trees), dtype=np.int8)

            oob_samples = self._get_oob_samples(X)
            for t in range(num_trees):
                matches = np.unique(oob_samples[t])
                oob_matrix[matches, t] = 1

            return oob_matrix

        def _get_in_bag_samples(self, X):
            """
            Helper for get_in_bag_counts.
            """
            n = len(X)
            in_bag_samples = []
            for tree in self.estimators_:
                in_bag_sample = _generate_sample_indices(tree.random_state, n, n)
                in_bag_samples.append(in_bag_sample)
            return in_bag_samples

        def get_in_bag_counts(self, X):
            """
            Return in-bag multiplicity matrix of shape (n_samples, n_estimators).
            """
            n = len(X)
            num_trees = self.n_estimators
            in_bag_matrix = np.zeros((n, num_trees), dtype=np.float32)

            in_bag_samples = self._get_in_bag_samples(X)
            for t in range(num_trees):
                matches, n_repeats = np.unique(in_bag_samples[t], return_counts=True)
                in_bag_matrix[matches, t] = n_repeats

            return in_bag_matrix

        def fit_forest(self, X, y, **fit_kwargs):
            """
            Fit only the underlying sklearn random forest.
        
            This is the part that should count as forest training time when
            benchmarking against ForestKernel.
            """
            if self._adapter_ready:
                # Forest already injected through set_forest(...)
                if y is not None:
                    self.y = np.asarray(y)
                    self.n = len(y)
                return {"X": X}
        
            super().fit(X, y, **fit_kwargs)
        
            self.y = y
            self.n = len(y)
            self._adapter_ready = True
        
            return {"X": X}
        
        def set_forest(self, fitted_forest, y=None):
            """
            Reuse an already fitted sklearn random forest instead of fitting again.
        
            Parameters
            ----------
            fitted_forest : fitted RandomForestClassifier or RandomForestRegressor
                Forest to reuse.
        
            y : array-like or None, default=None
                Optional labels corresponding to the training samples. Stored only
                for consistency with the legacy API.
            """
            check_is_fitted(fitted_forest)
        
            # Preserve wrapper-specific configuration
            prox_method = self.prox_method
            matrix_type = self.matrix_type
            triangular = self.triangular
            prediction_type = self.prediction_type
            non_zero_diagonal = self.non_zero_diagonal
            force_symmetric = self.force_symmetric
        
            # Preserve legacy cache fields
            leaf_matrix = self.leaf_matrix
            leaf_matrix_test = self.leaf_matrix_test
            oob_indices = self.oob_indices
            in_bag_counts = self.in_bag_counts
            in_bag_indices = self.in_bag_indices
            in_bag_leaves = self.in_bag_leaves
            oob_leaves = self.oob_leaves
        
            # Copy fitted sklearn forest state into self
            self.__dict__.update(fitted_forest.__dict__)
        
            # Restore wrapper-specific configuration
            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.triangular = triangular
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal
            self.force_symmetric = force_symmetric
        
            # Restore legacy cache fields
            self.leaf_matrix = leaf_matrix
            self.leaf_matrix_test = leaf_matrix_test
            self.oob_indices = oob_indices
            self.in_bag_counts = in_bag_counts
            self.in_bag_indices = in_bag_indices
            self.in_bag_leaves = in_bag_leaves
            self.oob_leaves = oob_leaves
        
            self.y = None if y is None else np.asarray(y)
            self.n = None if y is None else len(y)
            self._adapter_ready = True
        
            return self

        def build_proximity_cache(self, X, x_test=None):
            """
            Build all legacy proximity metadata after forest fitting.

            This is the part that should count as proximity preprocessing time.
            """
            check_is_fitted(self)
            if not self._adapter_ready:
                raise ValueError("Forest must be fitted before building proximity cache.")

            self.leaf_matrix = self.apply(X)

            if x_test is not None:
                n_test = np.shape(x_test)[0]
                self.leaf_matrix_test = self.apply(x_test)
                self.leaf_matrix = np.concatenate(
                    (self.leaf_matrix, self.leaf_matrix_test),
                    axis=0,
                )
            else:
                n_test = 0

            if self.prox_method == "oob":
                self.oob_indices = self.get_oob_indices(X)

                if x_test is not None:
                    self.oob_indices = np.concatenate(
                        (self.oob_indices, np.ones((n_test, self.n_estimators), dtype=np.int8)),
                        axis=0,
                    )

                self.oob_leaves = self.oob_indices * self.leaf_matrix

            elif self.prox_method == "rfgap":
                self.oob_indices = self.get_oob_indices(X)
                self.in_bag_counts = self.get_in_bag_counts(X)

                if x_test is not None:
                    self.oob_indices = np.concatenate(
                        (self.oob_indices, np.ones((n_test, self.n_estimators), dtype=np.int8)),
                        axis=0,
                    )
                    self.in_bag_counts = np.concatenate(
                        (self.in_bag_counts, np.zeros((n_test, self.n_estimators), dtype=np.float32)),
                        axis=0,
                    )

                self.in_bag_indices = 1 - self.oob_indices
                self.in_bag_leaves = self.in_bag_indices * self.leaf_matrix
                self.oob_leaves = self.oob_indices * self.leaf_matrix

            return self

        def fit(self, X, y, x_test=None, **fit_kwargs):
            """
            Full legacy workflow:
                1) fit forest
                2) build proximity metadata
            """
            self.fit_forest(X, y, **fit_kwargs)
            self.build_proximity_cache(X, x_test=x_test)
            return self

        def get_proximity_vector(self, ind):
            """
            Produce one row of the legacy proximity matrix.
            Returns only (data, cols), which is enough for CSR row-wise construction.
            """
            n, num_trees = self.leaf_matrix.shape

            if self.prox_method == "oob":
                if self.triangular:
                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(
                        self.oob_indices[ind, ind_oob_leaves]
                        == self.oob_indices[ind:, ind_oob_leaves],
                        axis=1,
                    )
                    tree_counts[tree_counts == 0] = 1

                    prox_counts = np.sum(
                        self.oob_leaves[ind, ind_oob_leaves]
                        == self.oob_leaves[ind:, ind_oob_leaves],
                        axis=1,
                    )
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols = np.where(prox_vec != 0)[0] + ind
                    data = prox_vec[cols - ind]
                else:
                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(
                        self.oob_indices[ind, ind_oob_leaves]
                        == self.oob_indices[:, ind_oob_leaves],
                        axis=1,
                    )
                    tree_counts[tree_counts == 0] = 1

                    prox_counts = np.sum(
                        self.oob_leaves[ind, ind_oob_leaves]
                        == self.oob_leaves[:, ind_oob_leaves],
                        axis=1,
                    )
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols = np.nonzero(prox_vec)[0]
                    data = prox_vec[cols]

            elif self.prox_method == "original":
                if self.triangular:
                    tree_inds = self.leaf_matrix[ind, :]
                    prox_vec = np.sum(tree_inds == self.leaf_matrix[ind:, :], axis=1)

                    cols = np.where(prox_vec != 0)[0] + ind
                    data = prox_vec[cols - ind] / num_trees
                else:
                    tree_inds = self.leaf_matrix[ind, :]
                    prox_vec = np.sum(tree_inds == self.leaf_matrix, axis=1)

                    cols = np.nonzero(prox_vec)[0]
                    data = prox_vec[cols] / num_trees

            elif self.prox_method == "rfgap":
                oob_trees = np.nonzero(self.oob_indices[ind, :])[0]
                in_bag_trees = np.nonzero(self.in_bag_indices[ind, :])[0]

                terminals = self.leaf_matrix[ind, :]
                matches = terminals == self.in_bag_leaves
                match_counts = np.where(matches, self.in_bag_counts, 0)

                ks = np.sum(match_counts, axis=0)
                ks[ks == 0] = 1
                ks_in = ks[in_bag_trees]
                ks_out = ks[oob_trees]

                S_out = np.count_nonzero(self.oob_indices[ind, :])
                if S_out == 0:
                    S_out = 1

                prox_vec = np.sum(
                    np.divide(match_counts[:, oob_trees], ks_out),
                    axis=1,
                ) / S_out

                if self.non_zero_diagonal:
                    S_in = np.count_nonzero(self.in_bag_indices[ind, :])

                    if S_in > 0:
                        prox_vec[ind] = np.sum(
                            np.divide(match_counts[ind, in_bag_trees], ks_in)
                        ) / S_in
                    else:
                        prox_vec[ind] = np.sum(
                            np.divide(match_counts[ind, in_bag_trees], ks_in)
                        )

                    prox_vec = prox_vec / np.max(prox_vec)
                    prox_vec[ind] = 1

                cols = np.nonzero(prox_vec)[0]
                data = prox_vec[cols]

            else:
                raise ValueError(f"Unknown prox_method='{self.prox_method}'")

            return np.asarray(data), np.asarray(cols, dtype=np.int64)

        def get_proximities(self):
            """
            Materialize the legacy proximity matrix.

            This version avoids giant Python triplet lists and instead builds
            the CSR structure row by row.
            """
            check_is_fitted(self)

            if self.leaf_matrix is None:
                raise ValueError(
                    "Proximity cache has not been built yet. "
                    "Call build_proximity_cache(...) first."
                )

            n, _ = self.leaf_matrix.shape

            data_parts = []
            indices_parts = []
            indptr = np.zeros(n + 1, dtype=np.int64)

            for i in range(n):
                data_i, cols_i = self.get_proximity_vector(i)
                data_parts.append(data_i)
                indices_parts.append(cols_i)
                indptr[i + 1] = indptr[i] + cols_i.size

            if len(data_parts) > 0:
                data = np.concatenate(data_parts)
                indices = np.concatenate(indices_parts)
            else:
                data = np.array([], dtype=np.float32)
                indices = np.array([], dtype=np.int64)

            prox_sparse = sparse.csr_matrix(
                (data, indices, indptr),
                shape=(n, n),
            )

            if self.triangular and self.prox_method != "rfgap":
                prox_sparse = prox_sparse + prox_sparse.transpose()
                prox_sparse.setdiag(1)

            if self.force_symmetric:
                prox_sparse = (prox_sparse + prox_sparse.transpose()) / 2

            if self.matrix_type == "dense":
                return prox_sparse.toarray()
            return prox_sparse

        def prox_extend(self, data):
            """
            Compute proximities between training observations and new data.
            """
            check_is_fitted(self)

            if self.leaf_matrix is None:
                raise ValueError(
                    "Proximity cache has not been built yet. "
                    "Call build_proximity_cache(...) first."
                )

            n, num_trees = self.leaf_matrix.shape
            extended_leaf_matrix = self.apply(data)
            n_ext, _ = extended_leaf_matrix.shape

            data_parts = []
            indices_parts = []
            indptr = np.zeros(n_ext + 1, dtype=np.int64)

            if self.prox_method == "oob":
                for ind in range(n_ext):
                    ind_oob_leaves = np.nonzero(self.oob_leaves[:, :])[1]  # unused placeholder to keep structure similar

                    # Compute proximities from test row ind to all train rows
                    tree_counts = np.sum(
                        self.oob_indices[:, :] == np.ones_like(self.oob_indices[:, :]),
                        axis=1,
                    )
                    tree_counts[tree_counts == 0] = 1

                    prox_counts = np.sum(
                        self.oob_leaves == extended_leaf_matrix[ind, :],
                        axis=1,
                    )
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols_i = np.nonzero(prox_vec)[0]
                    data_i = prox_vec[cols_i]

                    data_parts.append(np.asarray(data_i))
                    indices_parts.append(np.asarray(cols_i, dtype=np.int64))
                    indptr[ind + 1] = indptr[ind] + len(cols_i)

            elif self.prox_method == "original":
                for ind in range(n_ext):
                    tree_inds = extended_leaf_matrix[ind, :]
                    prox_vec = np.sum(tree_inds == self.leaf_matrix, axis=1)

                    cols_i = np.nonzero(prox_vec)[0]
                    data_i = prox_vec[cols_i] / num_trees

                    data_parts.append(np.asarray(data_i))
                    indices_parts.append(np.asarray(cols_i, dtype=np.int64))
                    indptr[ind + 1] = indptr[ind] + len(cols_i)

            elif self.prox_method == "rfgap":
                for ind in range(n_ext):
                    oob_terminals = extended_leaf_matrix[ind, :]
                    matches = oob_terminals == self.in_bag_leaves
                    matched_counts = np.where(matches, self.in_bag_counts, 0)

                    ks = np.sum(matched_counts, axis=0)
                    ks[ks == 0] = 1

                    prox_vec = np.sum(np.divide(matched_counts, ks), axis=1) / num_trees

                    cols_i = np.nonzero(prox_vec)[0]
                    data_i = prox_vec[cols_i]

                    data_parts.append(np.asarray(data_i))
                    indices_parts.append(np.asarray(cols_i, dtype=np.int64))
                    indptr[ind + 1] = indptr[ind] + len(cols_i)

            else:
                raise ValueError(f"Unknown prox_method='{self.prox_method}'")

            if len(data_parts) > 0:
                data_arr = np.concatenate(data_parts)
                indices_arr = np.concatenate(indices_parts)
            else:
                data_arr = np.array([], dtype=np.float32)
                indices_arr = np.array([], dtype=np.int64)

            prox_sparse = sparse.csr_matrix(
                (data_arr, indices_arr, indptr),
                shape=(n_ext, n),
            )

            if self.matrix_type == "dense":
                return prox_sparse.toarray()
            return prox_sparse

    return RFGAP(
        prox_method=prox_method,
        matrix_type=matrix_type,
        triangular=triangular,
        non_zero_diagonal=non_zero_diagonal,
        force_symmetric=force_symmetric,
        **kwargs,
    )
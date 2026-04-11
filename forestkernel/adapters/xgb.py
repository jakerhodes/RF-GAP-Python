import numpy as np

from .base import EnsembleAdapter


class XGBoostAdapter(EnsembleAdapter):
    """
    Adapter for XGBoost sklearn-style estimators such as
    xgboost.XGBClassifier and xgboost.XGBRegressor.

    This adapter supports only leaf-based kernels such as the original
    proximity. It does not support OOB-, in-bag-, or tree-weighted quantities.
    """

    def _get_booster(self):
        if not hasattr(self.estimator, "get_booster"):
            raise ValueError("XGBoost estimator must be fitted before using XGBoostAdapter.")
        return self.estimator.get_booster()

    def get_leaf_matrix(self, X):
        """
        Return matrix of leaf ids of shape (N, T).
        """
        leaf_matrix = self.estimator.apply(X)
        leaf_matrix = np.asarray(leaf_matrix, dtype=np.int32)

        if leaf_matrix.ndim == 1:
            leaf_matrix = leaf_matrix.reshape(-1, 1)
        elif leaf_matrix.ndim > 2:
            leaf_matrix = leaf_matrix.reshape(leaf_matrix.shape[0], -1)

        return leaf_matrix

    def get_n_nodes_per_tree(self):
        """
        Return number of nodes per tree using the dumped booster structure.
        Cached after first call.
        """
        if hasattr(self, "_n_nodes_per_tree_cache"):
            return self._n_nodes_per_tree_cache

        booster = self._get_booster()
        tree_df = booster.trees_to_dataframe()

        self._n_nodes_per_tree_cache = (
            tree_df.groupby("Tree")
            .size()
            .astype(int)
            .tolist()
        )
        return self._n_nodes_per_tree_cache

    def get_oob_mask(self, X_train=None):
        raise ValueError("OOB indices are not defined for XGBoost.")

    def get_in_bag_counts(self, X_train=None):
        raise ValueError("In-bag counts are not defined for XGBoost.")

    def get_tree_weights(self, X_ref):
        raise ValueError("Tree weights are not defined for XGBoost.")

    def supports_oob(self):
        return False

    def supports_in_bag_counts(self):
        return False

    def supports_tree_weights(self):
        return False
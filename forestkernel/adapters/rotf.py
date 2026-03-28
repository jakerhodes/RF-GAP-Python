import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import EnsembleAdapter


class ROTFAdapter(EnsembleAdapter):
    """
    Adapter for Bagged Rotation Forest.

    Assumes the fitted estimator follows the aeon RotationForestClassifier
    internals used by your wrapper:
    - estimators_ : tuple of fitted tree estimators
    - _groups     : tuple of per-tree feature groups
    - _pcas       : tuple of per-tree PCA objects
    - _min, _ptp  : feature-wise normalization statistics
    """

    def _extract_tree(self, estimator):
        """
        In aeon Rotation Forest, estimators_ already stores the fitted trees.
        """
        return estimator

    def _transform_for_tree(self, X, t_idx):
        """
        Reproduce aeon's per-tree Rotation Forest feature transform.

        aeon stores:
        - self.estimator._groups[t_idx]: tuple/list of feature-index groups for tree t
        - self.estimator._pcas[t_idx]:   fitted PCA objects, one per group

        During fit/predict, aeon:
        1. normalizes X with (X - self.estimator._min) / self.estimator._ptp
        2. applies each PCA to its corresponding feature group
        3. concatenates the transformed blocks
        4. applies nan_to_num
        """
        X = np.asarray(X)

        ptp = np.where(self.estimator._ptp == 0, 1.0, self.estimator._ptp)
        X_norm = (X - self.estimator._min) / ptp

        pcas_t = self.estimator._pcas[t_idx]
        groups_t = self.estimator._groups[t_idx]

        X_t = np.concatenate(
            [pcas_t[i].transform(X_norm[:, group]) for i, group in enumerate(groups_t)],
            axis=1
        )

        X_t = X_t.astype(np.float32)
        X_t = np.nan_to_num(
            X_t,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min
        )
        return X_t

    def get_leaf_matrix(self, X):
        """
        Passes data through each tree's specific Rotation Forest transform,
        then fetches the terminal leaf indices.
        """
        check_is_fitted(self.estimator)
        X = np.atleast_2d(X)
        N = X.shape[0]
        T = len(self.estimator.estimators_)
        leaf_matrix = np.zeros((N, T), dtype=np.int32)

        for t_idx, tree in enumerate(self.estimator.estimators_):
            X_t = self._transform_for_tree(X, t_idx)
            leaf_matrix[:, t_idx] = tree.apply(X_t)

        return leaf_matrix

    def get_n_nodes_per_tree(self):
        """
        Return number of nodes per tree.
        """
        return [self._extract_tree(est).tree_.node_count for est in self.estimator.estimators_]

    def get_oob_mask(self, X_train=None):
        """
        Returns OOB mask matrix of shape (N_train, T), where entry (i,t)=1 if
        sample i is OOB for tree t.
        """
        c = self.get_in_bag_counts(X_train)
        return (c == 0).astype(np.int8)

    def get_in_bag_counts(self, X_train=None):
        """
        Retrieves the exact bootstrap sample counts (multiplicities) per tree.
        Output shape: (N_train_samples, N_trees)
        """
        check_is_fitted(self.estimator)
        T = len(self.estimator.estimators_)

        first_tree = self._extract_tree(self.estimator.estimators_[0])
        N = len(first_tree.in_bag_counts_)

        c_matrix = np.zeros((N, T), dtype=np.float32)

        for t_idx, tree in enumerate(self.estimator.estimators_):
            tree = self._extract_tree(tree)
            c_matrix[:, t_idx] = tree.in_bag_counts_

        return c_matrix

    def get_tree_weights(self, X_ref):
        raise ValueError("Tree weights are not defined for Rotation Forest.")

    def supports_oob(self):
        return True

    def supports_in_bag_counts(self):
        return True
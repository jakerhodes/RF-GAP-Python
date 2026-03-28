import numpy as np

from .base import EnsembleAdapter


class GBTAdapter(EnsembleAdapter):
    """
    Adapter for sklearn GradientBoosting ensembles.
    """

    def _get_tree_list(self):
        """
        Flatten sklearn GBT estimators_ into a single list of trees.

        - Regression: estimators_.shape = (T, 1)
        - Binary classification: estimators_.shape = (T, 1)
        - Multiclass classification: estimators_.shape = (T, K)
        """
        return [tree for tree in self.estimator.estimators_.ravel()]

    def get_leaf_matrix(self, X):
        """
        Apply every tree in the flattened GBT ensemble and return a leaf matrix
        of shape (N, T_total).
        """
        tree_list = self._get_tree_list()
        return np.column_stack([tree.apply(X) for tree in tree_list]).astype(np.int32)

    def get_n_nodes_per_tree(self):
        """
        Return number of nodes per tree.
        """
        tree_list = self._get_tree_list()
        return [tree.tree_.node_count for tree in tree_list]

    def get_oob_mask(self, X_train=None):
        raise ValueError("OOB indices are not defined for GradientBoosting.")

    def get_in_bag_counts(self, X_train=None):
        raise ValueError("In-bag counts are not defined for GradientBoosting.")

    def get_tree_weights(self, X_ref):
        """
        Computes tree-specific weights for Gradient Boosted Tree (GBT) proximities.

        Following Tan et al. (2016), we weight each tree by its 'importance'—the
        magnitude of its contribution to the final boosted predictor. Unlike Random
        Forests where trees are i.i.d., GBT trees are learned iteratively and
        contribute unequally.

        The weight w_t is proportional to the squared L2-norm of the tree's
        shrunken predictions:
            w_t = || learning_rate * h_t(X_ref) ||_2^2
        """
        lr = np.float32(self.estimator.learning_rate)
        tree_list = self._get_tree_list()
        weights = []

        for tree in tree_list:
            contrib = lr * tree.predict(X_ref)
            wt = np.linalg.norm(contrib, ord=2) ** 2
            weights.append(wt)

        weights = np.asarray(weights, dtype=np.float32)

        if weights.size == 0:
            raise RuntimeError("No trees found in fitted GradientBoosting model.")

        total_weight = weights.sum()
        if total_weight <= 0:
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total_weight

        return weights.astype(np.float32)

    def supports_tree_weights(self):
        return True
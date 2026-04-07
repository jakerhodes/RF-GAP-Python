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
        Compute tree-specific weights for boosted-tree proximities.
    
        Following the boosted-tree proximity definition of Tan et al. (2020),
        each tree is weighted by the variance of its contribution over the
        reference set. If h_t(s) denotes the shrunken output of tree t for
        sample s, then the weight is taken proportional to
    
            w_t ∝ Var({h_t(s) : s in X_ref}).
    
        For sklearn GradientBoosting, the per-tree shrunken contribution is
    
            learning_rate * tree.predict(X_ref),
    
        so we compute the empirical variance of these values over the
        reference samples.
    
        Notes
        -----
        - This differs from using the squared L2 norm of the tree outputs.
          The two coincide only when the tree outputs are centered.
        - For multiclass GradientBoosting, the flattened tree list contains
          class-specific trees from successive boosting stages, so these
          weights should be interpreted as per-flattened-tree variance
          weights.
        """
        lr = np.float32(self.estimator.learning_rate)
        tree_list = self._get_tree_list()
        weights = []
    
        for tree in tree_list:
            contrib = lr * tree.predict(X_ref).astype(np.float32, copy=False)
            centered = contrib - contrib.mean()
            wt = np.mean(centered ** 2, dtype=np.float32)
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
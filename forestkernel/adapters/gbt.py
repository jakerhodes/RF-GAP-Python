import numpy as np
from .base import EnsembleAdapter


class GBTAdapter(EnsembleAdapter):
    """
    Adapter for sklearn GradientBoosting estimators.
    """

    supported_weight_schemes = {"uniform", "kerf", "boosted"}

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

    def _predict_tree_outputs(self, X_ref):
        """
        Helper to get shrunken tree outputs (learning_rate * tree_output).
        Used to calculate variance for weights.
        """
        leaf_matrix = self.get_leaf_matrix(X_ref)
        tree_list = self._get_tree_list()
        lr = np.float32(self.estimator.learning_rate)
        
        n_samples, n_trees = leaf_matrix.shape
        outputs = np.zeros((n_samples, n_trees), dtype=np.float32)

        for t, tree in enumerate(tree_list):
            # tree.tree_.value has shape (n_nodes, 1, 1)
            # We map the leaf indices to their corresponding values
            leaf_values = tree.tree_.value.reshape(-1).astype(np.float32)
            outputs[:, t] = lr * leaf_values[leaf_matrix[:, t]]
            
        return outputs

    def get_n_nodes_per_tree(self):
        """
        Return number of nodes per tree.
        """
        tree_list = self._get_tree_list()
        return [tree.tree_.node_count for tree in tree_list]

    def get_tree_weights(self, X_ref):
        """
        Compute normalized tree-specific weights for boosted-tree proximities.
    
        Returns
        -------
        weights : np.ndarray of shape (T_total,), dtype np.float32
            One nonnegative weight per tree in the flattened GradientBoosting
            ensemble.
    
            The entries are normalized to sum to one. Therefore, ``weights[t]``
            gives the relative contribution of flattened tree ``t`` to the
            boosted-tree proximity.
    
            Here, ``T_total`` is the number of trees after flattening
            ``estimator.estimators_``:
    
            - regression: ``T_total = n_estimators``
            - binary classification: ``T_total = n_estimators``
            - multiclass classification:
              ``T_total = n_estimators * n_classes``
    
        Notes
        -----
        Gradient Boosting models are additive ensembles of the form
    
            F(x) = sum_t h_t(x),
    
        where ``h_t(x)`` denotes the prediction contribution of tree ``t`` for
        sample ``x``. For sklearn GradientBoosting estimators,
    
            h_t(x) = learning_rate * f_t(x),
    
        where ``f_t(x)`` is the raw prediction of tree ``t``.
    
        In this implementation, the tree contributions are obtained directly from
        the fitted leaf values and the leaf assignments of ``X_ref``.
    
        Following the boosted-tree proximity definition of Tan et al. (2020),
        each tree is weighted according to the empirical variance of its
        contribution over the reference set:
    
            w_t ∝ Var({h_t(s) : s in X_ref}).
    
        The intuition is that trees whose outputs vary more across the dataset
        carry more geometric and predictive information, and should therefore
        contribute more strongly to the proximity. Conversely, trees whose
        predictions are nearly constant over ``X_ref`` contribute little to
        distinguishing samples and receive smaller weights.
    
        This differs from Random Forests, where trees are approximately
        exchangeable and are therefore typically weighted uniformly. In boosting,
        trees are built sequentially and can have very different importance,
        making variance-based weighting more natural.
    
        If all empirical variances are numerically zero, the method falls back to
        uniform weights.
    
        For multiclass GradientBoosting, sklearn stores one tree per class at each
        boosting stage. The returned weights are therefore per flattened
        class-specific tree, not per boosting stage.
        """
        # contribs shape: (n_samples, n_trees)
        contribs = self._predict_tree_outputs(X_ref)
        
        if contribs.shape[1] == 0:
            raise RuntimeError("No trees found in fitted GradientBoosting model.")

        # Uniform calculation using np.var (Empirical variance)
        weights = np.var(contribs, axis=0).astype(np.float32)
    
        total_weight = weights.sum()
        if total_weight <= 1e-12:
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total_weight
    
        return weights.astype(np.float32)
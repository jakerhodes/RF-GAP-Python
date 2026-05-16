import numpy as np
from .base import EnsembleAdapter

class XGBoostAdapter(EnsembleAdapter):
    """
    Adapter for XGBoost sklearn-style estimators.

    Notes
    -----
    XGBoost typically grows trees depth-wise by default. This can produce
    more balanced leaves than LightGBM, which may increase shared-leaf
    collisions and yield denser forest kernels.
    """

    supported_weight_schemes = {"uniform", "kerf", "boosted"}

    def _get_booster(self):
        if not hasattr(self.estimator, "get_booster"):
            raise ValueError("XGBoost estimator must be fitted before using XGBoostAdapter.")
        return self.estimator.get_booster()

    def get_leaf_matrix(self, X):
        """
        Return matrix of leaf ids of shape (N, T).
        """
        # XGBoost apply returns the leaf node indices
        leaf_matrix = self.estimator.apply(X)
        leaf_matrix = np.asarray(leaf_matrix, dtype=np.int32)

        if leaf_matrix.ndim == 1:
            leaf_matrix = leaf_matrix.reshape(-1, 1)
        elif leaf_matrix.ndim > 2:
            # Handle multiclass case where apply might return (N, rounds, classes)
            leaf_matrix = leaf_matrix.reshape(leaf_matrix.shape[0], -1)

        return leaf_matrix

    def _predict_tree_outputs(self, X_ref):
        """
        Return per-tree fitted contributions of shape
        (n_samples, n_trees_total).
    
        For each sample in ``X_ref`` and each XGBoost tree, this method returns
        the leaf value reached by that sample. These values define the per-tree
        contribution ``h_t(x)`` used by the fitted boosted model.
    
        In XGBoost, the leaf values stored in the dumped booster representation
        are already scaled by the learning rate. Therefore, no additional
        multiplication by ``learning_rate`` is applied here.
    
        Notes
        -----
        The values are read from ``booster.trees_to_dataframe()``. For leaf nodes,
        XGBoost stores the fitted leaf output in the ``Gain`` column.
        """
        leaf_matrix = self.get_leaf_matrix(X_ref)
        n_samples, n_trees = leaf_matrix.shape
        
        booster = self._get_booster()
        df = booster.trees_to_dataframe()
        
        # Only interested in leaves
        leaves_df = df[df['Feature'] == 'Leaf']
        
        outputs = np.zeros((n_samples, n_trees), dtype=np.float32)

        for t in range(n_trees):
            # Map for specific tree: NodeID -> Leaf Value
            tree_leaves = leaves_df[leaves_df['Tree'] == t]
            leaf_map = dict(zip(tree_leaves['ID'].str.split('-').str[-1].astype(int), 
                                tree_leaves['Gain'].astype(np.float32)))
            
            # Reconstruct contributions
            # Note: XGBoost leaf values are already shrunken by the learning rate
            outputs[:, t] = np.array([leaf_map[idx] for idx in leaf_matrix[:, t]], dtype=np.float32)
            
        return outputs

    def get_n_nodes_per_tree(self):
        """
        Return number of nodes per tree using the dumped booster structure.
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
        """
        Compute normalized tree-specific weights for boosted-tree proximities.
    
        Returns
        -------
        weights : np.ndarray of shape (n_trees_total,), dtype np.float32
            One nonnegative weight per XGBoost tree.
    
            The entries are normalized to sum to one. Therefore, ``weights[t]``
            gives the relative contribution of tree ``t`` to the boosted-tree
            proximity.
    
            Here, ``n_trees_total`` is the number of trees returned by the fitted
            XGBoost booster. For multiclass models, XGBoost stores one tree per
            class at each boosting round, so
    
                ``n_trees_total = n_estimators * n_classes``.
    
        Notes
        -----
        Boosted trees are additive ensembles of the form
    
            F(x) = sum_t h_t(x),
    
        where ``h_t(x)`` denotes the contribution of tree ``t`` to the model score
        for sample ``x``.
    
        In this implementation, ``h_t(x)`` is obtained from the fitted XGBoost
        leaf values reached by ``x``. These leaf values are already scaled by the
        learning rate in the dumped booster representation, so no additional
        shrinkage is applied.
    
        Following the boosted-tree proximity definition of Tan et al. (2020),
        each tree is weighted according to the empirical variance of its
        contribution over the reference set:
    
            w_t ∝ Var({h_t(s) : s in X_ref}).
    
        The intuition is that trees whose outputs vary more across the dataset
        carry more geometric and predictive information, and should therefore
        contribute more strongly to the proximity. Conversely, trees whose
        predictions are nearly constant over ``X_ref`` contribute little to
        distinguishing samples and receive smaller weights.
    
        If all empirical variances are numerically zero, the method falls back to
        uniform weights.
    
        For multiclass XGBoost models, the returned weights are per flattened
        class-specific tree, not per boosting round.
        """
        contribs = self._predict_tree_outputs(X_ref)
        
        if contribs.shape[1] == 0:
            raise RuntimeError("No trees found in fitted XGBoost model.")

        # Uniform calculation using np.var (Empirical variance)
        weights = np.var(contribs, axis=0).astype(np.float32)
    
        total_weight = weights.sum()
        if total_weight <= 1e-12:
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total_weight
    
        return weights.astype(np.float32)
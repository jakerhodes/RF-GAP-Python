import numpy as np
from .base import EnsembleAdapter

class XGBoostAdapter(EnsembleAdapter):
    """
    Adapter for XGBoost sklearn-style estimators such as
    xgboost.XGBClassifier and xgboost.XGBRegressor.
    """

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
        Helper to get shrunken tree outputs. 
        In XGBoost, the 'Gain' column for leaf nodes in trees_to_dataframe() 
        contains the raw leaf output.
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
        Compute tree-specific weights for boosted-tree proximities.
    
        Following the boosted-tree proximity definition of Tan et al. (2020),
        each tree is weighted by the variance of its contribution over the
        reference set. If h_t(s) denotes the shrunken output of tree t for
        sample s, then the weight is taken proportional to
    
            w_t ∝ Var({h_t(s) : s in X_ref}).
    
        Since XGBoost leaf values in the booster dump are already shrunken by 
        the learning rate, this amounts to computing the empirical variance 
        of these values over the reference samples.
    
        Notes
        -----
        - This differs from using the squared L2 norm of the tree outputs.
          The two coincide only when the tree outputs are centered.
        - For multiclass XGBoost, the flattened tree list contains
          class-specific trees from successive boosting rounds, so these
          weights should be interpreted as per-flattened-tree variance
          weights.
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

    def supports_tree_weights(self):
        return True
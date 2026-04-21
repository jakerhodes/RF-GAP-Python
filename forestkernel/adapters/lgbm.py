import numpy as np
from .base import EnsembleAdapter

class LightGBMAdapter(EnsembleAdapter):
    """
    Adapter for LightGBM sklearn-style estimators such as
    lightgbm.LGBMClassifier and lightgbm.LGBMRegressor.
    """

    def _get_booster(self):
        if not hasattr(self.estimator, "booster_"):
            raise ValueError(
                "LightGBM estimator must be fitted before using LightGBMAdapter."
            )
        return self.estimator.booster_

    def get_leaf_matrix(self, X):
        """
        Apply every LightGBM tree and return a leaf matrix of shape
        (n_samples, n_trees_total).
        """
        booster = self._get_booster()
        leaf_matrix = booster.predict(X, pred_leaf=True)

        leaf_matrix = np.asarray(leaf_matrix, dtype=np.int32)
        if leaf_matrix.ndim == 1:
            leaf_matrix = leaf_matrix.reshape(-1, 1)

        return leaf_matrix

    def _get_leaf_value_maps(self):
        """
        Return one dictionary per tree mapping leaf_index to leaf_value.
        Defensively handles cases where LightGBM produces empty trees or 
        missing columns in the dataframe dump.
        """
        booster = self._get_booster()
        df = booster.trees_to_dataframe()
        
        # If the dataframe is empty, return an empty list
        if df.empty:
            return []

        n_trees = int(df['tree_index'].max() + 1)
        maps = [{} for _ in range(n_trees)]

        # If 'leaf_index' is missing, it means no trees have splits.
        # We return the list of empty dicts, which results in 0 variance.
        if 'leaf_index' not in df.columns:
            return maps
            
        # Filter for rows that represent actual leaves
        leaves_df = df[df['leaf_index'].notnull()].copy()
        
        for t in range(n_trees):
            tree_leaves = leaves_df[leaves_df['tree_index'] == t]
            if not tree_leaves.empty:
                # Map leaf_index to pre-shrunken 'value'
                maps[t] = dict(zip(tree_leaves['leaf_index'].astype(int), 
                                    tree_leaves['value'].astype(np.float32)))
            
        return maps

    def _predict_tree_outputs(self, X_ref):
        """
        Return per-tree shrunken contributions of shape (n_samples, n_trees_total).
        """
        leaf_matrix = self.get_leaf_matrix(X_ref)
        leaf_value_maps = self._get_leaf_value_maps()

        n_samples, n_trees = leaf_matrix.shape
        outputs = np.zeros((n_samples, n_trees), dtype=np.float32)

        for t in range(n_trees):
            leaf_map = leaf_value_maps[t]
            # Use .get(idx, 0.0) to safely handle any empty/stub trees
            outputs[:, t] = np.array(
                [leaf_map.get(int(leaf_idx), 0.0) for leaf_idx in leaf_matrix[:, t]],
                dtype=np.float32,
            )

        return outputs

    def get_n_nodes_per_tree(self):
        """
        Return total node counts for each LightGBM tree using the dataframe.
        """
        booster = self._get_booster()
        df = booster.trees_to_dataframe()
        return df.groupby("tree_index").size().astype(int).tolist()

    def get_oob_mask(self, X_train=None):
        raise ValueError("OOB indices are not defined for LightGBM.")

    def get_in_bag_counts(self, X_train=None):
        raise ValueError("In-bag counts are not defined for LightGBM.")

    def get_tree_weights(self, X_ref):
        """
        Compute tree-specific weights for boosted-tree proximities.
        Following Tan et al. (2020) variance-based weighting.
        """
        contribs = self._predict_tree_outputs(X_ref)
        
        if contribs.shape[1] == 0:
            raise RuntimeError("No trees found in fitted LightGBM model.")

        weights = np.var(contribs, axis=0).astype(np.float32)
    
        total_weight = weights.sum()
        if total_weight <= 1e-12:
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total_weight
    
        return weights.astype(np.float32)

    def supports_oob(self):
        return False

    def supports_in_bag_counts(self):
        return False

    def supports_tree_weights(self):
        return True
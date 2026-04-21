import numpy as np

from .base import EnsembleAdapter


class LightGBMAdapter(EnsembleAdapter):
    """
    Adapter for LightGBM sklearn-style estimators such as
    lightgbm.LGBMClassifier and lightgbm.LGBMRegressor.

    This adapter supports leaf-based kernels and tree-weighted boosted-tree
    proximities, but does not support OOB- or in-bag-based quantities.
    """

    def _get_booster(self):
        if not hasattr(self.estimator, "booster_"):
            raise ValueError(
                "LightGBM estimator must be fitted before using LightGBMAdapter."
            )
        return self.estimator.booster_

    def _get_tree_info_list(self):
        """
        Return the flattened list of LightGBM trees from the dumped model.
        """
        booster = self._get_booster()
        model_dump = booster.dump_model()
        return model_dump["tree_info"]

    def _count_nodes(self, node):
        """
        Recursively count all nodes in a LightGBM tree.
        """
        if "left_child" not in node and "right_child" not in node:
            return 1
        return (
            1
            + self._count_nodes(node["left_child"])
            + self._count_nodes(node["right_child"])
        )

    def _collect_leaf_values(self, node, out):
        """
        Recursively collect a mapping {leaf_index -> leaf_value} for one tree.
        """
        if "left_child" not in node and "right_child" not in node:
            out[int(node["leaf_index"])] = np.float32(node["leaf_value"])
            return

        self._collect_leaf_values(node["left_child"], out)
        self._collect_leaf_values(node["right_child"], out)

    def _get_leaf_value_maps(self):
        """
        Return one dictionary per tree mapping LightGBM leaf_index to leaf_value.
        """
        tree_info_list = self._get_tree_info_list()
        maps = []

        for tree_info in tree_info_list:
            leaf_map = {}
            self._collect_leaf_values(tree_info["tree_structure"], leaf_map)
            maps.append(leaf_map)

        return maps

    def _predict_tree_outputs(self, X_ref):
        """
        Return per-tree shrunken contributions of shape (n_samples, n_trees_total).

        Following Tan et al. (2020), we reconstruct the contribution h_t(s)
        for each tree. In LightGBM's dump_model(), the 'leaf_value' already
        accounts for the learning rate (shrinkage), so we map leaf indices 
        directly to these values.
        """
        # leaf_matrix shape: (n_samples, n_trees)
        leaf_matrix = self.get_leaf_matrix(X_ref)
        # leaf_value_maps is a list of dicts: [{leaf_idx: value}, ...]
        leaf_value_maps = self._get_leaf_value_maps()

        n_samples, n_trees = leaf_matrix.shape
        outputs = np.zeros((n_samples, n_trees), dtype=np.float32)

        for t, leaf_map in enumerate(leaf_value_maps):
            # Map the leaf indices to their pre-shrunken values.
            # We use a list comprehension for the mapping; for massive X_ref,
            # consider np.vectorize or a lookup array for extra speed.
            outputs[:, t] = np.array(
                [leaf_map[int(leaf_idx)] for leaf_idx in leaf_matrix[:, t]],
                dtype=np.float32,
            )

        return outputs

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

    def get_n_nodes_per_tree(self):
        """
        Return total node counts for each LightGBM tree.
        """
        tree_info_list = self._get_tree_info_list()
        return [
            self._count_nodes(tree_info["tree_structure"])
            for tree_info in tree_info_list
        ]

    def get_oob_mask(self, X_train=None):
        raise ValueError("OOB indices are not defined for LightGBM.")

    def get_in_bag_counts(self, X_train=None):
        raise ValueError("In-bag counts are not defined for LightGBM.")

    def get_tree_weights(self, X_ref):
        """
        Compute tree-specific weights for boosted-tree proximities.
    
        Following the boosted-tree proximity definition of Tan et al. (2020),
        each tree is weighted by the variance of its contribution over the
        reference set. If h_t(s) denotes the shrunken output of tree t for
        sample s, then the weight is taken proportional to
    
            w_t ∝ Var({h_t(s) : s in X_ref}).
    
        Since `_predict_tree_outputs(X_ref)` already returns the per-tree
        shrunken contributions, this amounts to computing the empirical
        variance of each tree's output across the reference samples.
    
        Notes
        -----
        - This differs from using the squared L2 norm of the tree outputs.
            The two coincide only when the tree outputs are centered.
        - For multiclass LightGBM, the flattened tree list contains
            class-specific trees from successive boosting rounds, so these
            weights should be interpreted as per-flattened-tree variance
            weights.
        """
        # contribs shape: (n_samples, n_trees)
        contribs = self._predict_tree_outputs(X_ref)
    
        if contribs.shape[1] == 0:
            raise RuntimeError("No trees found in fitted LightGBM model.")

        # Compute empirical variance of each tree contribution
        weights = np.var(contribs, axis=0).astype(np.float32)
    
        # Handle edge case where trees might have zero variance (e.g., single-leaf trees)
        total_weight = weights.sum()
        if total_weight <= 1e-12:
            # Fallback to uniform weighting if no variance is found
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total_weight
    
        return weights.astype(np.float32)

    def supports_tree_weights(self):
        return True
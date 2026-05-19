import numpy as np
from .base import EnsembleAdapter

class LightGBMAdapter(EnsembleAdapter):
    """
    Adapter for LightGBM sklearn-style estimators.

    Notes
    -----
    LightGBM uses leaf-wise tree growth by default. This can produce more
    specialized leaves than depth-wise/tree-level methods, which may reduce
    the number of shared-leaf collisions and yield sparser forest kernels.
    """

    supported_weight_schemes = {"uniform", "kerf", "boosted"}

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
        booster = self._get_booster()
        tree_info = booster.dump_model()["tree_info"]
    
        maps = []
    
        def collect_leaves(node, out):
            if "leaf_index" in node:
                out[int(node["leaf_index"])] = float(node["leaf_value"])
                return
        
            if "left_child" in node:
                collect_leaves(node["left_child"], out)
        
            if "right_child" in node:
                collect_leaves(node["right_child"], out)
    
        for tree in tree_info:
            out = {}
            collect_leaves(tree["tree_structure"], out)
            maps.append(out)
    
        return maps

    def _predict_tree_outputs(self, X_ref):
        """
        Return per-tree fitted contributions of shape
        (n_samples, n_trees_total).
    
        LightGBM leaf values from ``booster.dump_model()`` already include the
        learning-rate shrinkage used by the fitted booster, so no additional
        multiplication by ``learning_rate`` is applied here.
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

    def get_tree_weights(self, X_ref):
        """
        Compute normalized tree-specific weights for boosted-tree proximities.
    
        Returns
        -------
        weights : np.ndarray of shape (n_trees_total,), dtype np.float32
            One nonnegative weight per LightGBM tree.
    
            The entries are normalized to sum to one. Therefore, ``weights[t]``
            gives the relative contribution of tree ``t`` to the boosted-tree
            proximity.
    
            Here, ``n_trees_total`` is the number of trees returned by the fitted
            LightGBM booster. For multiclass models, LightGBM stores one tree per
            class at each boosting iteration, so
    
                ``n_trees_total = n_estimators * n_classes``.
    
        Notes
        -----
        Boosted trees are additive ensembles of the form
    
            F(x) = sum_t h_t(x),
    
        where ``h_t(x)`` denotes the contribution of tree ``t`` to the model score
        for sample ``x``.
    
        In this implementation, ``h_t(x)`` is obtained from the fitted LightGBM
        leaf values reached by ``x``. These leaf values are read directly from the
        dumped booster model and correspond to the per-tree contribution used by
        the fitted LightGBM ensemble.
    
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
    
        For multiclass LightGBM models, the returned weights are per flattened
        class-specific tree, not per boosting stage.
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
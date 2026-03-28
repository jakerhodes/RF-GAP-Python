import numpy as np
import sklearn
from packaging.version import Version as LooseVersion  # Handles python>3.12

if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble._forest import _generate_sample_indices
else:
    # Before sklearn version 0.24, forest was public, supporting this.
    from sklearn.ensemble.forest import _generate_unsampled_indices
    from sklearn.ensemble.forest import _generate_sample_indices

from .base import EnsembleAdapter


class RFETAdapter(EnsembleAdapter):
    """
    Adapter for sklearn RandomForest / ExtraTrees ensembles.
    """

    def _extract_tree(self, estimator):
        return estimator

    def get_leaf_matrix(self, X):
        """
        Return matrix of leaf ids of shape (N, T).
        """
        return self.estimator.apply(X).astype(np.int32)

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
        n_samples = X_train.shape[0]
        n_trees = len(self.estimator.estimators_)
        oob_mask = np.zeros((n_samples, n_trees), dtype=np.int8)

        for t, tree in enumerate(self.estimator.estimators_):
            unsampled = _generate_unsampled_indices(
                tree.random_state,
                n_samples,
                n_samples
            )
            oob_mask[unsampled, t] = 1

        return oob_mask

    def get_in_bag_counts(self, X_train=None):
        """
        Returns in-bag multiplicity matrix of shape (N_train, T), where entry
        (i,t) is the number of times sample i was drawn for tree t.
        """
        n_samples = X_train.shape[0]
        n_trees = len(self.estimator.estimators_)
        counts = np.zeros((n_samples, n_trees), dtype=np.int32)

        for t, tree in enumerate(self.estimator.estimators_):
            sampled = _generate_sample_indices(
                tree.random_state,
                n_samples,
                n_samples
            )
            binc = np.bincount(sampled, minlength=n_samples)
            counts[:, t] = binc

        return counts.astype(np.float32)

    def get_tree_weights(self, X_ref):
        raise ValueError("Tree weights are not defined for RandomForest / ExtraTrees.")

    def supports_oob(self):
        return True

    def supports_in_bag_counts(self):
        return True
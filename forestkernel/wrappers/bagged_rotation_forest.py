import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

# Graceful import for modern sktime/aeon implementations
try:
    from aeon.classification.sklearn import RotationForestClassifier
except ImportError: "RotationForestClassifier not found in aeon.classification.sklearn. Please ensure you have a compatible version of aeon installed."


class BootstrappedTreeWrapper(DecisionTreeClassifier):
    """
    A transparent wrapper around sklearn's DecisionTreeClassifier that forces
    bootstrap sampling during fit().
    
    By using this as the base estimator in Rotation Forest, bagging occurs 
    AFTER the PCA rotation phase, generating true Out-Of-Bag (OOB) samples 
    for calculating RF-GAP proximities.
    """
    def fit(self, X, y, sample_weight=None, check_input=True):
        rng = check_random_state(self.random_state)
        N = X.shape[0]
        
        # 1. Tree-wise Bagging: Draw bootstrap indices
        indices = rng.randint(0, N, N)
        
        # Save exact multiplicities for this specific tree
        self.in_bag_counts_ = np.bincount(indices, minlength=N).astype(np.float32)
        
        # Subset the rotated data
        X_boot = X[indices]
        y_boot = y[indices]
        
        if sample_weight is not None:
            sample_weight_boot = sample_weight[indices]
        else:
            sample_weight_boot = None
            
        # Fit the underlying decision tree on the bootstrapped, rotated subset
        return super().fit(X_boot, y_boot, sample_weight=sample_weight_boot, check_input=check_input)


class BaggedRotationForest(RotationForestClassifier):
    """
    An extended Rotation Forest that uses bootstrapped base trees and exposes
    the internal structure needed for ForestKernel proximities.
    """
    def __init__(self, n_estimators=200, base_estimator=None, **kwargs):
        
        # Inject our custom bootstrapped tree if no estimator is provided
        if base_estimator is None:
            # Note: aeon Rotation Forest accepts base_estimator
            base_estimator = BootstrappedTreeWrapper(criterion="entropy")
            
        super().__init__(n_estimators=n_estimators, base_estimator=base_estimator, **kwargs)

    def _extract_tree(self, estimator_obj):
        """In aeon, estimators_ already stores the fitted trees."""
        return estimator_obj

    def _transform_for_tree(self, X, t_idx):
        """
        Reproduce aeon's per-tree Rotation Forest feature transform.

        aeon stores:
        - self._groups[t_idx]: tuple/list of feature-index groups for tree t
        - self._pcas[t_idx]:   fitted PCA objects, one per group

        During fit/predict, aeon:
        1. normalizes X with (X - self._min) / self._ptp
        2. applies each PCA to its corresponding feature group
        3. concatenates the transformed blocks
        4. applies nan_to_num
        """
        X = np.asarray(X)

        # Match aeon's normalization step
        X_norm = (X - self._min) / self._ptp

        pcas_t = self._pcas[t_idx]
        groups_t = self._groups[t_idx]

        X_t = np.concatenate(
            [pcas_t[i].transform(X_norm[:, group]) for i, group in enumerate(groups_t)],
            axis=1
        )

        X_t = X_t.astype(np.float32)
        X_t = np.nan_to_num(
            X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
        )
        return X_t

    def get_underlying_estimators(self):
        """
        Returns the fitted tree estimators stored by Rotation Forest.
        """
        check_is_fitted(self)
        return self.estimators_

    # 2. The Apply Method
    def apply(self, X):
        """
        Passes data through each tree's specific Rotation Forest transform,
        then fetches the terminal leaf indices.
        """
        check_is_fitted(self)
        X = np.atleast_2d(X)
        N = X.shape[0]
        T = len(self.estimators_)
        leaf_matrix = np.zeros((N, T), dtype=np.int32)

        for t_idx, tree in enumerate(self.estimators_):
            X_t = self._transform_for_tree(X, t_idx)
            leaf_matrix[:, t_idx] = tree.apply(X_t)

        return leaf_matrix

    # 3. The In-Bag and OOB Mechanisms
    def get_in_bag_counts(self, X=None):
        """
        Retrieves the exact bootstrap sample counts (multiplicities) per tree.
        Output shape: (N_train_samples, N_trees)
        """
        check_is_fitted(self)
        T = len(self.estimators_)

        first_tree = self._extract_tree(self.estimators_[0])
        N = len(first_tree.in_bag_counts_)

        c_matrix = np.zeros((N, T), dtype=np.float32)

        for t_idx, tree in enumerate(self.estimators_):
            tree = self._extract_tree(tree)
            c_matrix[:, t_idx] = tree.in_bag_counts_

        return c_matrix

    def get_oob_indices(self, X=None):
        """
        Returns a binary mask of OOB samples.
        Output shape: (N_train_samples, N_trees)
        """
        c_matrix = self.get_in_bag_counts(X)
        return (c_matrix == 0).astype(np.int8)
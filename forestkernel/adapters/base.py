class EnsembleAdapter:
    """
    Base adapter interface used to abstract away ensemble-specific internals.

    Each adapter wraps a fitted ForestKernel estimator instance and exposes a
    unified interface for:
    - retrieving leaf indices
    - retrieving per-tree node counts
    - retrieving OOB masks / in-bag counts when available
    - retrieving tree-specific weights when relevant (e.g. GBT)

    Notes
    -----
    The adapter does not own the estimator. It simply delegates to it.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def get_leaf_matrix(self, X):
        """
        Return matrix of leaf ids of shape (N, T).
        """
        raise NotImplementedError

    def get_n_nodes_per_tree(self):
        """
        Return number of nodes per tree, used to offset local node ids into
        global ids.
        """
        raise NotImplementedError

    def get_oob_mask(self, X_train=None):
        """
        Return OOB mask matrix of shape (N_train, T), where entry (i,t)=1 if
        sample i is OOB for tree t.
        """
        raise NotImplementedError

    def get_in_bag_counts(self, X_train=None):
        """
        Return in-bag multiplicity matrix of shape (N_train, T), where entry
        (i,t) is the number of times sample i was drawn for tree t.
        """
        raise NotImplementedError

    def get_tree_weights(self, X_ref):
        """
        Return per-tree weights when the proximity requires them.
        Only relevant for some ensembles such as Gradient Boosting.
        """
        raise NotImplementedError

    def supports_oob(self):
        return False

    def supports_in_bag_counts(self):
        return False

    def supports_tree_weights(self):
        return False
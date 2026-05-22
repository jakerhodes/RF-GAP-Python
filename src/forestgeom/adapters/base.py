from sklearn.base import clone

class EnsembleAdapter:
    """
    Base adapter interface used to abstract away ensemble-specific internals.

    Each adapter wraps a fitted estimator instance and exposes a
    unified interface for:
    - retrieving leaf indices
    - retrieving per-tree node counts
    - retrieving OOB masks / in-bag counts when available
    - retrieving tree-specific weights when relevant (e.g. GBT)

    Each adapter wraps an estimator instance and exposes a unified interface
    for leaf-based forest kernel construction.
    

    Notes
    -----
    The adapter does not own the estimator. It simply delegates to it.
    """



    supported_weight_schemes = {"uniform", "kerf"}

    def __init__(self, estimator, weight_scheme=None):
        self.estimator = estimator

        if weight_scheme is not None:
            self.validate_weight_scheme(weight_scheme)

    def validate_weight_scheme(self, weight_scheme):
        """
        Validate whether this adapter supports the requested weight scheme.
        """
        if weight_scheme not in self.supported_weight_schemes:
            raise ValueError(
                f"{type(self).__name__} does not support "
                f"weight_scheme='{weight_scheme}'. "
                f"Supported schemes are {sorted(self.supported_weight_schemes)}."
            )

        return self

    def fit(self, X, y, **fit_kwargs):
        """
        Clone and fit the wrapped estimator.

        Additional keyword arguments are passed to estimator.fit(...),
        e.g. sample_weight.
        """
        self.estimator = clone(self.estimator)
        self.estimator.fit(X, y, **fit_kwargs)
        return self
    
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
    
    def __getattr__(self, name):
        return getattr(self.estimator, name)
import numpy as np
from scipy import sparse

from .cache import KernelCache


def to_global_leaves(leaf_mat, leaf_offsets):
    """
    Offset local leaf/node ids to global feature ids.

    Parameters
    ----------
    leaf_mat : ndarray of shape (N, T)
        Local node ids returned by apply().
    leaf_offsets : ndarray of shape (T,)
        Cumulative per-tree offsets.

    Returns
    -------
    ndarray of shape (N, T)
        Global node ids.
    """
    return leaf_mat + leaf_offsets


def initialize_cache(
    leaf_matrix,
    n_nodes_per_tree,
    n_samples,
):
    """
    Initialize the reusable structural part of the kernel cache from
    a leaf matrix.

    This includes:
    - global leaf indexing
    - flattened sample-tree incidences
    - flattened tree ids used by tree-specific quantities
    """
    cache = KernelCache()
    cache.leaf_matrix = leaf_matrix.astype(np.int32, copy=False)
    cache.n_samples = int(n_samples)
    cache.n_trees = int(leaf_matrix.shape[1])

    cache.leaf_offsets = np.concatenate(([0], np.cumsum(n_nodes_per_tree)[:-1])).astype(np.int64)
    cache.total_unique_nodes = int(np.sum(n_nodes_per_tree))
    cache.diag_offset = cache.total_unique_nodes

    global_leaves = to_global_leaves(cache.leaf_matrix, cache.leaf_offsets)

    cache.flat_rows = np.repeat(np.arange(cache.n_samples), cache.n_trees)
    cache.flat_cols = global_leaves.flatten()

    # Cached tree ids for flattened sample-tree arrays
    cache.flat_tree_ids = np.tile(np.arange(cache.n_trees, dtype=np.int64), cache.n_samples)

    return cache


def attach_bootstrap_stats(cache, oob_mask, inbag_counts=None):
    """
    Attach OOB and optional multiplicity information to an existing cache.
    """
    cache.oob_mask = oob_mask.astype(np.int8, copy=False)
    if inbag_counts is not None:
        cache.inbag_counts = inbag_counts.astype(np.float32, copy=False)
    return cache


def attach_boosted_weights(cache, boosted_tree_weights):
    """
    Attach boosted-tree weights to an existing cache.
    """
    cache.boosted_tree_weights = np.asarray(boosted_tree_weights, dtype=np.float32)
    return cache


def attach_inv_sqrt_leaf_mass(cache):
    """
    Attach inverse square-root unit leaf-mass statistics used by KeRF.
    """
    leaf_mass = np.bincount(
        cache.flat_cols,
        minlength=cache.total_unique_nodes,
    ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        cache.inv_sqrt_leaf_mass = 1.0 / np.sqrt(leaf_mass)
    cache.inv_sqrt_leaf_mass[~np.isfinite(cache.inv_sqrt_leaf_mass)] = 0.0

    return cache


def attach_inv_inbag_leaf_mass(cache):
    """
    Attach inverse multiplicity leaf-mass statistics used by GAP.
    """
    if cache.inbag_counts is None:
        raise ValueError("cache.inbag_counts is required to compute multiplicity leaf mass.")

    inbag_counts = cache.inbag_counts.astype(np.float32, copy=False)
    c_flat = inbag_counts.flatten()

    inbag_leaf_mass = np.bincount(
        cache.flat_cols,
        weights=c_flat,
        minlength=cache.total_unique_nodes,
    ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        cache.inv_inbag_leaf_mass = 1.0 / inbag_leaf_mass
    cache.inv_inbag_leaf_mass[~np.isfinite(cache.inv_inbag_leaf_mass)] = 0.0

    return cache


def build_W_matrix(cache, kernel_method):
    """
    Builds the raw Weight Matrix W (N_ref x N_total_nodes).

    This matrix handles the 'j' term (target/reference) in the kernel
    definitions, restricted to the true leaf coordinates only.

    Parameters
    ----------
    cache : KernelCache
    kernel_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'boosted'}

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    N = cache.n_samples
    T = cache.n_trees

    # Reuse the cached flattened structure of the reference/training set.
    # This avoids recomputing global leaves and flattened indices.
    flat_rows = cache.flat_rows
    flat_cols = cache.flat_cols

    total_cols = cache.total_unique_nodes

    # ---------------------------------------------------------
    # ORIGINAL PROXIMITY
    # p(i,j) = (1/T) * Sum_t [ I( j in v_i(t) ) ]
    #
    # Mapping:
    #   Use a symmetric factorization with sqrt(1/T) on both sides.
    #   W handles the target/reference side.
    # ---------------------------------------------------------
    if kernel_method == "original":
        scale_factor = np.float32(1.0 / np.sqrt(T))
        weights = np.full(N * T, scale_factor, dtype=np.float32)

    # ---------------------------------------------------------
    # TREE-WEIGHTED BOOSTED PROXIMITY
    # p(i,j) = Sum_t w_t * I( leaf_t(i) = leaf_t(j) )
    #
    # Mapping:
    #   Use a symmetric factorization with sqrt(w_t) on both sides.
    # ---------------------------------------------------------
    elif kernel_method == "boosted":
        if cache.boosted_tree_weights is None:
            raise ValueError("cache.boosted_tree_weights is required for kernel_method='boosted'.")
        sqrt_w = np.sqrt(cache.boosted_tree_weights).astype(np.float32)
        weights = np.tile(sqrt_w, N)

    # ---------------------------------------------------------
    # KeRF PROXIMITY
    # p(i,j) = (1/T) * Sum_t [ I(leaf_t(i)=leaf_t(j)) / M_leaf(t) ]
    #
    # Mapping:
    #   Again use a symmetric factorization:
    #       1/sqrt(T) * 1/sqrt(M_leaf)
    #   on both Q and W.
    # ---------------------------------------------------------
    elif kernel_method == "kerf":
        if cache.inv_sqrt_leaf_mass is None:
            raise ValueError("cache.inv_sqrt_leaf_mass is required for kernel_method='kerf'.")
        weights = (1.0 / np.sqrt(T)) * cache.inv_sqrt_leaf_mass[flat_cols]

    # ---------------------------------------------------------
    # OOB PROXIMITY (separable approximation)
    #
    # Reference-side weighting:
    #   Keep only the trees where the reference sample j is OOB.
    #
    # Let M_j = number of OOB trees for sample j.
    # Then W carries sqrt(T) / M_j on the retained sample-tree incidences.
    # ---------------------------------------------------------
    elif kernel_method == "oob":
        if cache.oob_mask is None:
            raise ValueError("cache.oob_mask is required for kernel_method='oob'.")

        # Apply OOB scope on the reference side: keep only OOB trees for each j.
        mask = cache.oob_mask.flatten() == 1
        flat_rows = flat_rows[mask]
        flat_cols = flat_cols[mask]

        # M_j = number of OOB trees for sample j
        M = cache.oob_mask.sum(axis=1).astype(np.float32)
        M[M == 0] = 1.0  # safety

        # Reference-side weights: sqrt(T) / M_j
        weights = (np.sqrt(T) / M[flat_rows]).astype(np.float32)

    # ---------------------------------------------------------
    # GAP PROXIMITY
    #
    # Ordinary target-side term:
    #   W stores the target/reference-side factor
    #
    #       c_j(t) / M_leaf
    #
    #   on each sample-tree incidence.
    # ---------------------------------------------------------
    elif kernel_method == "gap":
        if cache.inbag_counts is None:
            raise ValueError("cache.inbag_counts is required for kernel_method='gap'.")
        if cache.inv_inbag_leaf_mass is None:
            raise ValueError("cache.inv_inbag_leaf_mass is required for kernel_method='gap'.")

        c_j_t = cache.inbag_counts.flatten().astype(np.float32, copy=False)
        weights = c_j_t * cache.inv_inbag_leaf_mass[flat_cols]

    else:
        raise ValueError(f"Unknown kernel_method='{kernel_method}'.")

    mask = weights != 0
    W_mat = sparse.csr_matrix(
        (weights[mask], (flat_rows[mask], flat_cols[mask])),
        shape=(N, total_cols),
        dtype=np.float32
    )
    return W_mat


def build_Q_matrix(
    cache,
    kernel_method,
    leaves=None,
    is_training=True,
):
    """
    Builds the raw Query Matrix Q (N_query x N_total_nodes).

    This matrix handles the 'i' term and the summation scope S_i,
    restricted to the true leaf coordinates only.

    Parameters
    ----------
    cache : KernelCache
    kernel_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'boosted'}
    leaves : ndarray of shape (N_query, T), optional
        Query leaf matrix. If None, uses cache.leaf_matrix.
    is_training : bool, default=True
        Whether the query points are the same as the fitted reference points.
        Relevant for OOB and GAP.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    if leaves is None:
        leaves = cache.leaf_matrix

    N, T = leaves.shape
    global_leaves = to_global_leaves(leaves, cache.leaf_offsets)

    flat_rows = np.repeat(np.arange(N), T)
    flat_cols = global_leaves.flatten()

    total_cols = cache.total_unique_nodes

    # ---------------------------------------------------------
    # ORIGINAL PROXIMITY
    # p(i,j) = (1/T) * Sum_t [ I( j in v_i(t) ) ]
    #
    # Mapping:
    #   Use the same symmetric factorization as W:
    #       sqrt(1/T)
    # ---------------------------------------------------------
    if kernel_method == "original":
        scale_factor = np.float32(1.0 / np.sqrt(T))
        vals = np.full(N * T, scale_factor, dtype=np.float32)

    # ---------------------------------------------------------
    # TREE-WEIGHTED BOOSTED PROXIMITY
    # p(i,j) = Sum_t w_t * I( leaf_t(i) = leaf_t(j) )
    #
    # Mapping:
    #   Use sqrt(w_t) on the query side too.
    # ---------------------------------------------------------
    elif kernel_method == "boosted":
        if cache.boosted_tree_weights is None:
            raise ValueError("cache.boosted_tree_weights is required for kernel_method='boosted'.")
        sqrt_w = np.sqrt(cache.boosted_tree_weights).astype(np.float32)
        vals = np.tile(sqrt_w, N)

    # ---------------------------------------------------------
    # KeRF PROXIMITY
    # p(i,j) = (1/T) * Sum_t [ I(leaf_t(i)=leaf_t(j)) / M_leaf(t) ]
    #
    # Mapping:
    #   Same symmetric factorization as W:
    #       1/sqrt(T) * 1/sqrt(M_leaf)
    # ---------------------------------------------------------
    elif kernel_method == "kerf":
        if cache.inv_sqrt_leaf_mass is None:
            raise ValueError("cache.inv_sqrt_leaf_mass is required for kernel_method='kerf'.")
        vals = (1.0 / np.sqrt(T)) * cache.inv_sqrt_leaf_mass[flat_cols]

    # ---------------------------------------------------------
    # OOB PROXIMITY (separable approximation)
    #
    # If is_training=True:
    #   Restrict to the OOB trees for each query sample i.
    #   Let |S_i| be the number of such trees.
    #   Then Q carries sqrt(T) / |S_i|.
    #
    # If is_training=False:
    #   By convention, new points are treated as OOB for all trees, so |S_i| = T.
    # ---------------------------------------------------------
    elif kernel_method == "oob":
        if is_training:
            if cache.oob_mask is None:
                raise ValueError("cache.oob_mask is required for training-time kernel_method='oob'.")

            # Apply OOB scope on the query side: keep only OOB trees for each i.
            mask = cache.oob_mask.flatten() == 1
            flat_rows = flat_rows[mask]
            flat_cols = flat_cols[mask]

            # |S_i| = number of OOB trees for sample i
            S_i_counts = cache.oob_mask.sum(axis=1).astype(np.float32)
            S_i_counts[S_i_counts == 0] = 1.0

            # Query-side weights: sqrt(T) / |S_i|
            vals = (np.sqrt(T) / S_i_counts[flat_rows]).astype(np.float32)

        else:
            # For new data, all trees are considered OOB by convention (size T).
            vals = np.full(N * T, np.sqrt(T) / T, dtype=np.float32)

    # ---------------------------------------------------------
    # GAP PROXIMITY
    #
    # Ordinary query term:
    #   Q stores only the query-side normalization
    #
    #       1 / |S_i|
    #
    #   where S_i is:
    #   - the OOB set of sample i during training
    #   - all trees for extension points
    # ---------------------------------------------------------
    elif kernel_method == "gap":
        if is_training:
            if cache.oob_mask is None:
                raise ValueError("cache.oob_mask is required for training-time kernel_method='gap'.")

            mask = cache.oob_mask.flatten() == 1
            flat_rows = flat_rows[mask]
            flat_cols = flat_cols[mask]

            S_i_counts = cache.oob_mask.sum(axis=1).astype(np.float32)
            S_i_counts[S_i_counts == 0] = 1.0
            vals = (1.0 / S_i_counts[flat_rows]).astype(np.float32)

        else:
            # OOS: average over all trees
            vals = np.full(N * T, 1.0 / T, dtype=np.float32)

    else:
        raise ValueError(f"Unknown kernel_method='{kernel_method}'.")

    mask = vals != 0
    Q = sparse.csr_matrix(
        (vals[mask], (flat_rows[mask], flat_cols[mask])),
        shape=(N, total_cols),
        dtype=np.float32
    )
    return Q


def augment_kernel_maps(cache, kernel_method, Q, W, adjust_diagonal=False, is_training=True):
    """
    Optionally augment raw query/reference maps with private diagonal
    correction coordinates for exact kernel construction.

    This helper is shared by OOB and GAP. It assumes that build_Q_matrix()
    and build_W_matrix() return only the raw leaf maps, and appends private
    diagonal-correction coordinates only when requested.

    Parameters
    ----------
    cache : KernelCache
    kernel_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'boosted'}
    Q : scipy.sparse.csr_matrix
        Raw query-side map.
    W : scipy.sparse.csr_matrix
        Raw reference-side map.
    adjust_diagonal : bool, default=False
        Whether to append private diagonal-correction coordinates.
    is_training : bool, default=True
        Whether Q corresponds to the fitted reference set.

    Returns
    -------
    Q_aug : scipy.sparse.csr_matrix
    W_aug : scipy.sparse.csr_matrix
    """
    if not adjust_diagonal:
        return Q, W

    if kernel_method not in {"oob", "gap"}:
        return Q, W

    n_ref = cache.n_samples

    # ---------------------------------------------------------
    # Query-side augmentation
    # ---------------------------------------------------------
    if not is_training:
        # OOS queries do not get active private diagonal coordinates.
        # We only need to widen Q so it matches the augmented W width.
        n_query = Q.shape[0]
        zero_block = sparse.csr_matrix((n_query, n_ref), dtype=np.float32)
        Q_aug = sparse.hstack([Q, zero_block], format="csr")
    else:
        diag_rows = np.arange(n_ref, dtype=np.int64)
        diag_cols = np.arange(n_ref, dtype=np.int64)
        diag_vals_q = np.ones(n_ref, dtype=np.float32)
        Q_diag = sparse.csr_matrix(
            (diag_vals_q, (diag_rows, diag_cols)),
            shape=(n_ref, n_ref),
            dtype=np.float32,
        )
        Q_aug = sparse.hstack([Q, Q_diag], format="csr")

    # ---------------------------------------------------------
    # Reference-side augmentation
    # ---------------------------------------------------------
    if kernel_method == "oob":
        if cache.oob_mask is None:
            raise ValueError("cache.oob_mask is required for kernel_method='oob'.")

        T = cache.n_trees
        M = cache.oob_mask.sum(axis=1).astype(np.float32)
        M[M == 0] = 1.0
        raw_diag = (T / M).astype(np.float32)
        diag_vals_w = (1.0 - raw_diag).astype(np.float32)

    elif kernel_method == "gap":
        if cache.inbag_counts is None:
            raise ValueError("cache.inbag_counts is required for kernel_method='gap'.")

        row_sums = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
        inbag_counts_per_row = (cache.inbag_counts > 0).sum(axis=1).astype(np.float32)
        inbag_counts_per_row[inbag_counts_per_row == 0] = 1.0
        diag_vals_w = (row_sums / inbag_counts_per_row).astype(np.float32)

    else:
        diag_vals_w = None

    diag_rows = np.arange(n_ref, dtype=np.int64)
    diag_cols = np.arange(n_ref, dtype=np.int64)
    W_diag = sparse.csr_matrix(
        (diag_vals_w, (diag_rows, diag_cols)),
        shape=(n_ref, n_ref),
        dtype=np.float32,
    )
    W_aug = sparse.hstack([W, W_diag], format="csr")

    return Q_aug, W_aug
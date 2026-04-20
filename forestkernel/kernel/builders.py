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


def build_W_matrix(cache, kernel_method, force_nonzero_diag=False):
    """
    Builds the Weight Matrix W (N_ref x N_total_nodes_plus_optional_diag).

    This matrix handles the 'j' term (target/reference) in the kernel
    definitions.

    Parameters
    ----------
    cache : KernelCache
    kernel_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'boosted'}
    force_nonzero_diag : bool, default=False
        Only relevant for GAP. Whether to inject virtual diagonal
        coordinates to restore non-zero self-similarities.

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

    # Base number of columns for sparse W building (before optional virtual diagonal).
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
    #
    # Diagonal trick:
    #   The raw separable OOB factorization yields self-similarity T / M_j,
    #   which is generally > 1. To replace the diagonal exactly by 1 without
    #   calling sparse setdiag(), we append one private coordinate per sample:
    #
    #       QW^T  ->  QW^T + diag(1 - raw_diag)
    #
    #   This is done by adding N virtual columns after the real leaf columns.
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

        # Exact diagonal replacement trick for OOB.
        raw_diag = (T / M).astype(np.float32)
        diag_vals = (1.0 - raw_diag).astype(np.float32)

        diag_rows = np.arange(N, dtype=np.int64)
        diag_cols = diag_rows + cache.diag_offset

        flat_rows = np.concatenate([flat_rows, diag_rows])
        flat_cols = np.concatenate([flat_cols, diag_cols])
        weights = np.concatenate([weights, diag_vals])
        total_cols += N

    # ---------------------------------------------------------
    # GAP PROXIMITY
    #
    # Ordinary target-side term:
    #   W stores the target/reference-side factor
    #
    #       c_j(t) / M_leaf
    #
    #   on each sample-tree incidence.
    #
    # Private diagonal coordinates:
    #   When force_nonzero_diag=True, one private coordinate per training
    #   sample is appended after the leaf coordinates. These coordinates
    #   are used only to correct training self-similarities.
    #
    #   These private coordinates are training-only correction features:
    #   out-of-sample queries keep the corresponding columns in Q, but with
    #   zero values.
    # ---------------------------------------------------------
    elif kernel_method == "gap":
        if cache.inbag_counts is None:
            raise ValueError("cache.inbag_counts is required for kernel_method='gap'.")
        if cache.inv_inbag_leaf_mass is None:
            raise ValueError("cache.inv_inbag_leaf_mass is required for kernel_method='gap'.")

        # ----- Ordinary target-side term -----
        c_j_t = cache.inbag_counts.flatten().astype(np.float32, copy=False)
        weights = c_j_t * cache.inv_inbag_leaf_mass[flat_cols]

        # ----- Private diagonal correction -----
        if force_nonzero_diag:
            total_cols += N

            row_sums = np.bincount(flat_rows, weights=weights, minlength=N).astype(np.float32)
            inbag_counts_per_row = (cache.inbag_counts > 0).sum(axis=1).astype(np.float32)
            inbag_counts_per_row[inbag_counts_per_row == 0] = 1.0
            diag_vals = (row_sums / inbag_counts_per_row).astype(np.float32)

            diag_rows = np.arange(N, dtype=np.int64)
            diag_cols = diag_rows + cache.diag_offset

            flat_rows = np.concatenate([flat_rows, diag_rows])
            flat_cols = np.concatenate([flat_cols, diag_cols])
            weights = np.concatenate([weights, diag_vals])

    else:
        raise ValueError(f"Unknown kernel_method='{kernel_method}'.")

    # Filter zeros and build sparse W
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
    force_nonzero_diag=False,
):
    """
    Builds the Query Matrix Q (N_query x N_total_nodes_plus_optional_diag).

    This matrix handles the 'i' term and the summation scope S_i.

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
    force_nonzero_diag : bool, default=False
        Only relevant for GAP.

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

    # Base number of columns for sparse Q building (before optional virtual diagonal).
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
    #   To match the exact diagonal replacement in W, append the same private
    #   virtual coordinates with value 1 on the query side.
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

            # Matching private diagonal coordinates for exact diagonal replacement
            total_cols += cache.n_samples
            diag_rows = np.arange(N, dtype=np.int64)
            diag_cols = diag_rows + cache.diag_offset
            diag_vals = np.ones(N, dtype=np.float32)

            flat_rows = np.concatenate([flat_rows, diag_rows])
            flat_cols = np.concatenate([flat_cols, diag_cols])
            vals = np.concatenate([vals, diag_vals])

        else:
            # For new data, all trees are considered OOB by convention (size T).
            vals = np.full(N * T, np.sqrt(T) / T, dtype=np.float32)

            # The reference-side W includes private diagonal coordinates for the training set.
            # New queries should have zero mass on these coordinates, but the matrix width must match.
            total_cols += cache.n_samples

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
    #
    # Private diagonal term:
    #   These private coordinates must match the extra columns created in W.
    #
    #   - If force_nonzero_diag=True, Q places value 1 on all private
    #     coordinates for training points, and W determines the final
    #     diagonal magnitude.
    #
    #   - For OOS points, the ambient feature dimension must still match W,
    #     so we keep the extra private-diagonal columns in the shape, but
    #     OOS queries do not activate them.
    # ---------------------------------------------------------
    elif kernel_method == "gap":
        # ----- Ordinary query-side term -----
        if is_training:
            if cache.oob_mask is None:
                raise ValueError("cache.oob_mask is required for training-time kernel_method='gap'.")

            mask = cache.oob_mask.flatten() == 1
            flat_rows = flat_rows[mask]
            flat_cols = flat_cols[mask]

            S_i_counts = cache.oob_mask.sum(axis=1).astype(np.float32)
            S_i_counts[S_i_counts == 0] = 1.0
            vals = (1.0 / S_i_counts[flat_rows]).astype(np.float32)

            # ----- Final assembly -----
            if force_nonzero_diag:
                total_cols += cache.n_samples

                diag_rows = np.arange(N, dtype=np.int64)
                diag_cols = diag_rows + cache.diag_offset
                diag_vals = np.ones(N, dtype=np.float32)

                flat_rows = np.concatenate([flat_rows, diag_rows])
                flat_cols = np.concatenate([flat_cols, diag_cols])
                vals = np.concatenate([vals, diag_vals])

        else:
            # OOS: average over all trees
            vals = np.full(N * T, 1.0 / T, dtype=np.float32)

            # IMPORTANT:
            # keep the same ambient feature dimension as W when
            # force_nonzero_diag=True, but do NOT activate any private
            # diagonal coordinates for OOS points.
            if force_nonzero_diag:
                total_cols += cache.n_samples

    else:
        raise ValueError(f"Unknown kernel_method='{kernel_method}'.")

    # Filter zeros and build sparse Q
    mask = vals != 0
    Q = sparse.csr_matrix(
        (vals[mask], (flat_rows[mask], flat_cols[mask])),
        shape=(N, total_cols),
        dtype=np.float32
    )
    return Q
import numpy as np
from scipy import sparse

from .cache import ProximityCache


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
    leaf_matrix_all,
    n_nodes_per_tree,
    n_samples,
):
    """
    Initialize the structural part of the proximity cache from a leaf matrix.

    Parameters
    ----------
    leaf_matrix_all : ndarray of shape (N, T)
        Local node ids for all reference samples.
    n_nodes_per_tree : sequence[int]
        Number of nodes in each tree.
    n_samples : int
        Number of reference samples.

    Returns
    -------
    ProximityCache
    """
    cache = ProximityCache()
    cache.leaf_matrix_all = leaf_matrix_all.astype(np.int32, copy=False)
    cache.n_samples = int(n_samples)
    cache.n_trees = int(leaf_matrix_all.shape[1])

    cache.leaf_offsets = np.concatenate(([0], np.cumsum(n_nodes_per_tree)[:-1])).astype(np.int64)
    cache.total_unique_nodes = int(np.sum(n_nodes_per_tree))
    cache.diag_offset = cache.total_unique_nodes

    global_leaves_all = to_global_leaves(cache.leaf_matrix_all, cache.leaf_offsets)

    cache.flat_rows_all = np.repeat(np.arange(cache.n_samples), cache.n_trees)
    cache.flat_cols_all = global_leaves_all.flatten()

    return cache


def attach_oob_structure(cache, oob_mask_all, c_all=None):
    """
    Attach OOB and optional multiplicity information to an existing cache.
    """
    cache.oob_mask_all = oob_mask_all.astype(np.int8, copy=False)
    if c_all is not None:
        cache.c_all = c_all.astype(np.float32, copy=False)
    return cache


def attach_gbt_weights(cache, gbt_tree_weights):
    """
    Attach GBT tree weights to an existing cache.
    """
    cache.gbt_tree_weights = np.asarray(gbt_tree_weights, dtype=np.float32)
    return cache


def compute_unit_leaf_mass(cache):
    """
    Precompute unit leaf mass statistics used by KeRF:
        M_leaf = number of sample-tree incidences in each global leaf
        inv_sqrt_leaf_mass_unit = 1 / sqrt(M_leaf)
    """
    cache.leaf_mass_unit = np.bincount(
        cache.flat_cols_all,
        minlength=cache.total_unique_nodes
    ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        cache.inv_sqrt_leaf_mass_unit = 1.0 / np.sqrt(cache.leaf_mass_unit)
    cache.inv_sqrt_leaf_mass_unit[~np.isfinite(cache.inv_sqrt_leaf_mass_unit)] = 0.0

    return cache


def compute_multiplicity_leaf_mass(cache):
    """
    Precompute multiplicity leaf mass statistics used by RF-GAP:
        M_leaf = sum_j c_j(t) over all sample-tree incidences in each global leaf
        inv_leaf_mass_mult = 1 / M_leaf
    """
    if cache.c_all is None:
        raise ValueError("cache.c_all is required to compute multiplicity leaf mass.")

    c_flat = cache.c_all.flatten()

    cache.leaf_mass_mult = np.bincount(
        cache.flat_cols_all,
        weights=c_flat,
        minlength=cache.total_unique_nodes
    ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        cache.inv_leaf_mass_mult = 1.0 / cache.leaf_mass_mult
    cache.inv_leaf_mass_mult[~np.isfinite(cache.inv_leaf_mass_mult)] = 0.0

    return cache


def build_W_matrix(cache, prox_method, force_nonzero_diag=False):
    """
    Builds the Weight Matrix W (N_ref x N_total_nodes_plus_optional_diag).

    This matrix handles the 'j' term (target/reference) in the proximity
    definitions.

    Parameters
    ----------
    cache : ProximityCache
    prox_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'gbt'}
    force_nonzero_diag : bool, default=False
        Only relevant for RF-GAP. Whether to inject virtual diagonal
        coordinates to restore non-zero self-similarities.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    N = cache.n_samples
    T = cache.n_trees
    flat_rows = cache.flat_rows_all
    flat_cols = cache.flat_cols_all
    total_cols = cache.total_unique_nodes

    if prox_method == "original":
        scale_factor = np.float32(1.0 / np.sqrt(T))
        weights = np.full(N * T, scale_factor, dtype=np.float32)

    elif prox_method == "gbt":
        if cache.gbt_tree_weights is None:
            raise ValueError("cache.gbt_tree_weights is required for prox_method='gbt'.")
        sqrt_w = np.sqrt(cache.gbt_tree_weights).astype(np.float32)
        weights = np.tile(sqrt_w, N)

    elif prox_method == "kerf":
        if cache.inv_sqrt_leaf_mass_unit is None:
            raise ValueError("cache.inv_sqrt_leaf_mass_unit is required for prox_method='kerf'.")
        weights = (1.0 / np.sqrt(T)) * cache.inv_sqrt_leaf_mass_unit[flat_cols]

    elif prox_method == "oob":
        if cache.oob_mask_all is None:
            raise ValueError("cache.oob_mask_all is required for prox_method='oob'.")

        mask = cache.oob_mask_all.flatten() == 1
        flat_rows = flat_rows[mask]
        flat_cols = flat_cols[mask]

        M = cache.oob_mask_all.sum(axis=1).astype(np.float32)
        M[M == 0] = 1.0

        weights = (np.sqrt(T) / M[flat_rows]).astype(np.float32)

        raw_diag = (T / M).astype(np.float32)
        diag_vals = (1.0 - raw_diag).astype(np.float32)

        diag_rows = np.arange(N)
        diag_cols = np.arange(N) + cache.diag_offset

        flat_rows = np.concatenate([flat_rows, diag_rows])
        flat_cols = np.concatenate([flat_cols, diag_cols])
        weights = np.concatenate([weights, diag_vals])
        total_cols += N

    elif prox_method == "gap":
        if cache.c_all is None:
            raise ValueError("cache.c_all is required for prox_method='gap'.")
        if cache.inv_leaf_mass_mult is None:
            raise ValueError("cache.inv_leaf_mass_mult is required for prox_method='gap'.")

        c_j_t = cache.c_all.flatten()
        weights = c_j_t * cache.inv_leaf_mass_mult[flat_cols]

        if force_nonzero_diag:
            row_sums = np.bincount(flat_rows, weights=weights, minlength=N).astype(np.float32)
            denom = (cache.c_all > 0).sum(axis=1).astype(np.float32)
            denom[denom == 0] = 1.0
            diag_vals = row_sums / denom

            diag_rows = np.arange(N)
            diag_cols = np.arange(N) + cache.diag_offset

            flat_rows = np.concatenate([flat_rows, diag_rows])
            flat_cols = np.concatenate([flat_cols, diag_cols])
            weights = np.concatenate([weights, diag_vals])
            total_cols += N

    else:
        raise ValueError(f"Unknown prox_method='{prox_method}'.")

    mask = weights != 0
    W_mat = sparse.csr_matrix(
        (weights[mask], (flat_rows[mask], flat_cols[mask])),
        shape=(N, total_cols),
        dtype=np.float32
    )
    return W_mat


def build_Q_matrix(
    cache,
    prox_method,
    leaves=None,
    is_training=True,
    force_nonzero_diag=False,
):
    """
    Builds the Query Matrix Q (N_query x N_total_nodes_plus_optional_diag).

    This matrix handles the 'i' term and the summation scope S_i.

    Parameters
    ----------
    cache : ProximityCache
    prox_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'gbt'}
    leaves : ndarray of shape (N_query, T), optional
        Query leaf matrix. If None, uses cache.leaf_matrix_all.
    is_training : bool, default=True
        Whether the query points are the same as the fitted reference points.
        Relevant for OOB and GAP.
    force_nonzero_diag : bool, default=False
        Only relevant for RF-GAP.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    if leaves is None:
        leaves = cache.leaf_matrix_all

    N, T = leaves.shape
    global_leaves = to_global_leaves(leaves, cache.leaf_offsets)

    flat_rows = np.repeat(np.arange(N), T)
    flat_cols = global_leaves.flatten()
    total_cols = cache.total_unique_nodes

    if prox_method == "original":
        scale_factor = np.float32(1.0 / np.sqrt(T))
        vals = np.full(N * T, scale_factor, dtype=np.float32)

    elif prox_method == "gbt":
        if cache.gbt_tree_weights is None:
            raise ValueError("cache.gbt_tree_weights is required for prox_method='gbt'.")
        sqrt_w = np.sqrt(cache.gbt_tree_weights).astype(np.float32)
        vals = np.tile(sqrt_w, N)

    elif prox_method == "kerf":
        if cache.inv_sqrt_leaf_mass_unit is None:
            raise ValueError("cache.inv_sqrt_leaf_mass_unit is required for prox_method='kerf'.")
        vals = (1.0 / np.sqrt(T)) * cache.inv_sqrt_leaf_mass_unit[flat_cols]

    elif prox_method == "oob":
        if is_training:
            if cache.oob_mask_all is None:
                raise ValueError("cache.oob_mask_all is required for training-time prox_method='oob'.")

            mask = cache.oob_mask_all.flatten() == 1
            flat_rows = flat_rows[mask]
            flat_cols = flat_cols[mask]

            S_i_counts = cache.oob_mask_all.sum(axis=1).astype(np.float32)
            S_i_counts[S_i_counts == 0] = 1.0

            vals = (np.sqrt(T) / S_i_counts[flat_rows]).astype(np.float32)

            total_cols += cache.n_samples
            diag_rows = np.arange(N)
            diag_cols = np.arange(N) + cache.diag_offset
            diag_vals = np.ones(N, dtype=np.float32)

            flat_rows = np.concatenate([flat_rows, diag_rows])
            flat_cols = np.concatenate([flat_cols, diag_cols])
            vals = np.concatenate([vals, diag_vals])

        else:
            vals = np.full(N * T, np.sqrt(T) / T, dtype=np.float32)
            total_cols += cache.n_samples

    elif prox_method == "gap":
        if is_training:
            if cache.oob_mask_all is None:
                raise ValueError("cache.oob_mask_all is required for training-time prox_method='gap'.")

            mask = cache.oob_mask_all.flatten() == 1
            flat_rows = flat_rows[mask]
            flat_cols = flat_cols[mask]

            S_i_counts = cache.oob_mask_all.sum(axis=1).astype(np.float32)
            S_i_counts[S_i_counts == 0] = 1.0

            vals = (1.0 / S_i_counts[flat_rows]).astype(np.float32)

        else:
            vals = np.full(N * T, 1.0 / T, dtype=np.float32)

        if force_nonzero_diag:
            total_cols += cache.n_samples

            if is_training:
                diag_rows = np.arange(N)
                diag_cols = np.arange(N) + cache.diag_offset
                diag_vals = np.ones(N, dtype=np.float32)

                flat_rows = np.concatenate([flat_rows, diag_rows])
                flat_cols = np.concatenate([flat_cols, diag_cols])
                vals = np.concatenate([vals, diag_vals])

    else:
        raise ValueError(f"Unknown prox_method='{prox_method}'.")

    mask = vals != 0
    Q = sparse.csr_matrix(
        (vals[mask], (flat_rows[mask], flat_cols[mask])),
        shape=(N, total_cols),
        dtype=np.float32
    )
    return Q
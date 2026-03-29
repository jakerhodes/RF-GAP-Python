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
    leaf_matrix_all,
    n_nodes_per_tree,
    n_samples,
    idx_labeled=None,
    idx_unlabeled=None,
    n_train_samples=None,
):
    """
    Initialize the reusable structural part of the kernel cache from
    a leaf matrix.
    
    This includes:
    - global leaf indexing
    - flattened sample-tree incidences
    - semi-supervised row/flat masks
    - flattened tree ids used by tree-specific GAP surrogates
    """
    cache = KernelCache()
    cache.leaf_matrix_all = leaf_matrix_all.astype(np.int32, copy=False)
    cache.n_samples = int(n_samples)
    cache.n_trees = int(leaf_matrix_all.shape[1])

    cache.idx_labeled = None if idx_labeled is None else np.asarray(idx_labeled, dtype=np.int64)
    cache.idx_unlabeled = None if idx_unlabeled is None else np.asarray(idx_unlabeled, dtype=np.int64)
    cache.n_train_samples = int(n_train_samples) if n_train_samples is not None else int(n_samples)
    cache.is_semi_supervised = (
        cache.idx_unlabeled is not None and len(cache.idx_unlabeled) > 0
    )

    cache.leaf_offsets = np.concatenate(([0], np.cumsum(n_nodes_per_tree)[:-1])).astype(np.int64)
    cache.total_unique_nodes = int(np.sum(n_nodes_per_tree))
    cache.diag_offset = cache.total_unique_nodes

    global_leaves_all = to_global_leaves(cache.leaf_matrix_all, cache.leaf_offsets)

    cache.flat_rows_all = np.repeat(np.arange(cache.n_samples), cache.n_trees)
    cache.flat_cols_all = global_leaves_all.flatten()

    # Cached row-level masks for semi-supervised builders
    cache.row_is_labeled = np.zeros(cache.n_samples, dtype=bool)
    if cache.idx_labeled is not None:
        cache.row_is_labeled[cache.idx_labeled] = True

    cache.row_is_unlabeled = np.zeros(cache.n_samples, dtype=bool)
    if cache.idx_unlabeled is not None:
        cache.row_is_unlabeled[cache.idx_unlabeled] = True

    cache.flat_is_labeled = np.repeat(cache.row_is_labeled, cache.n_trees)
    cache.flat_is_unlabeled = np.repeat(cache.row_is_unlabeled, cache.n_trees)

    # Cached tree ids for flattened sample-tree arrays
    cache.flat_tree_ids = np.tile(np.arange(cache.n_trees, dtype=np.int64), cache.n_samples)

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
    Precompute unit leaf mass statistics used by KeRF.

    In semi-supervised mode, only labeled sample-tree incidences contribute
    to the leaf mass. This matches the old transductive heuristic.
    """
    if cache.idx_labeled is None:
        flat_cols = cache.flat_cols_all
    else:
        labeled_incidence_mask = cache.flat_is_labeled
        flat_cols = cache.flat_cols_all[labeled_incidence_mask]

    cache.leaf_mass_unit = np.bincount(
        flat_cols,
        minlength=cache.total_unique_nodes
    ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        cache.inv_sqrt_leaf_mass_unit = 1.0 / np.sqrt(cache.leaf_mass_unit)
    cache.inv_sqrt_leaf_mass_unit[~np.isfinite(cache.inv_sqrt_leaf_mass_unit)] = 0.0

    return cache


def compute_multiplicity_leaf_mass(cache):
    """
    Precompute multiplicity leaf-mass statistics used by RF-GAP.

    In semi-supervised mode, only labeled sample-tree incidences contribute
    to the denominator leaf mass, even though c_all is defined for all points.

    For unlabeled phantom targets, we use the following surrogates:

    - empirical_mult_all_by_tree[t]
      average multiplicity among all labeled samples in tree t,
      used for the ordinary unlabeled off-diagonal target contribution

    - empirical_mult_inbag_by_tree[t]
      average multiplicity among in-bag labeled samples in tree t,
      used for the unlabeled diagonal target
    """
    if cache.c_all is None:
        raise ValueError("cache.c_all is required to compute multiplicity leaf mass.")

    c_all = cache.c_all.astype(np.float32, copy=False)
    c_flat = c_all.flatten()

    if cache.idx_labeled is None:
        weights = c_flat
    else:
        weights = c_flat.copy()
        weights[~cache.flat_is_labeled] = 0.0

    cache.leaf_mass_mult = np.bincount(
        cache.flat_cols_all,
        weights=weights,
        minlength=cache.total_unique_nodes
    ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        cache.inv_leaf_mass_mult = 1.0 / cache.leaf_mass_mult
    cache.inv_leaf_mass_mult[~np.isfinite(cache.inv_leaf_mass_mult)] = 0.0

    T = cache.n_trees

    if cache.idx_labeled is not None and len(cache.idx_labeled) > 0:
        c_labeled = c_all[cache.idx_labeled]

        # Per-tree average multiplicity among all labeled samples,
        # used for the ordinary unlabeled off-diagonal target contribution
        cache.empirical_mult_all_by_tree = c_labeled.mean(axis=0).astype(np.float32)

        # Per-tree average multiplicity among in-bag labeled samples,
        # used only for the unlabeled diagonal target
        inbag_mask = c_labeled > 0
        inbag_counts = inbag_mask.sum(axis=0).astype(np.float32)
        inbag_sums = c_labeled.sum(axis=0).astype(np.float32)

        empirical_mult_inbag_by_tree = np.ones(T, dtype=np.float32)
        positive = inbag_counts > 0
        empirical_mult_inbag_by_tree[positive] = (
            inbag_sums[positive] / inbag_counts[positive]
        )
        cache.empirical_mult_inbag_by_tree = empirical_mult_inbag_by_tree
    else:
        cache.empirical_mult_all_by_tree = np.ones(T, dtype=np.float32)
        cache.empirical_mult_inbag_by_tree = np.ones(T, dtype=np.float32)

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

    # Reuse the cached flattened structure of the reference/training set.
    # This avoids recomputing global leaves and flattened indices.
    flat_rows = cache.flat_rows_all
    flat_cols = cache.flat_cols_all

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
    # GBT PROXIMITY
    # p(i,j) = Sum_t w_t * I( leaf_t(i) = leaf_t(j) )
    #
    # Mapping:
    #   Use a symmetric factorization with sqrt(w_t) on both sides.
    # ---------------------------------------------------------
    elif kernel_method == "gbt":
        if cache.gbt_tree_weights is None:
            raise ValueError("cache.gbt_tree_weights is required for kernel_method='gbt'.")
        sqrt_w = np.sqrt(cache.gbt_tree_weights).astype(np.float32)
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
        if cache.inv_sqrt_leaf_mass_unit is None:
            raise ValueError("cache.inv_sqrt_leaf_mass_unit is required for kernel_method='kerf'.")
        weights = (1.0 / np.sqrt(T)) * cache.inv_sqrt_leaf_mass_unit[flat_cols]

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
        if cache.oob_mask_all is None:
            raise ValueError("cache.oob_mask_all is required for kernel_method='oob'.")

        # Apply OOB scope on the reference side: keep only OOB trees for each j.
        mask = cache.oob_mask_all.flatten() == 1
        flat_rows = flat_rows[mask]
        flat_cols = flat_cols[mask]

        # M_j = number of OOB trees for sample j
        M = cache.oob_mask_all.sum(axis=1).astype(np.float32)
        M[M == 0] = 1.0  # safety

        # Reference-side weights: sqrt(T) / M_j
        weights = (np.sqrt(T) / M[flat_rows]).astype(np.float32)

        # Exact diagonal replacement trick for OOB.
        raw_diag = (T / M).astype(np.float32)
        diag_vals = (1.0 - raw_diag).astype(np.float32)

        diag_rows = np.arange(N)
        diag_cols = np.arange(N) + cache.diag_offset

        flat_rows = np.concatenate([flat_rows, diag_rows])
        flat_cols = np.concatenate([flat_cols, diag_cols])
        weights = np.concatenate([weights, diag_vals])
        total_cols += N

    # ---------------------------------------------------------
    # RF-GAP PROXIMITY
    #
    # Ordinary leaf term:
    #   W stores the target-side factor
    #
    #       c_j(t) / M_leaf
    #
    #   on each sample-tree incidence.
    #
    #   - labeled targets use their observed multiplicity c_j(t)
    #   - unlabeled phantom targets use the empirical unconditional surrogate
    #     empirical_mult_all_by_tree[t]
    #
    # Private diagonal term:
    #   One optional private coordinate per training sample is appended
    #   after the leaf coordinates.
    #
    #   - If force_nonzero_diag=False and semi-supervised mode is active,
    #     unlabeled private coordinates are used so that Q can cancel the
    #     ordinary unlabeled diagonal exactly.
    #
    #   - If force_nonzero_diag=True, the private coordinates restore the
    #     desired training diagonal:
    #
    #       labeled rows:
    #           (sum_t c_i(t) / M_leaf(i,t)) / #{t : c_i(t) > 0}
    #
    #       unlabeled rows:
    #           (1/T) * sum_t empirical_mult_inbag_by_tree[t] / M_i(t)
    #
    #     where empirical_mult_inbag_by_tree[t] is the average multiplicity
    #     among in-bag labeled samples in tree t.
    #
    #     Since the ordinary leaf part already contributes a nonzero
    #     unlabeled diagonal, the private coordinate stores only the
    #     missing correction.
    # ---------------------------------------------------------
    elif kernel_method == "gap":
        if cache.c_all is None:
            raise ValueError("cache.c_all is required for kernel_method='gap'.")
        if cache.inv_leaf_mass_mult is None:
            raise ValueError("cache.inv_leaf_mass_mult is required for kernel_method='gap'.")

        has_unlabeled = cache.is_semi_supervised
        needs_private_cols = force_nonzero_diag or has_unlabeled

        # ----- Ordinary target-side term -----
        c_j_t = cache.c_all.flatten().astype(np.float32, copy=True)
        if has_unlabeled:
            c_j_t[cache.flat_is_unlabeled] = cache.empirical_mult_all_by_tree[
                cache.flat_tree_ids[cache.flat_is_unlabeled]
            ]

        weights = c_j_t * cache.inv_leaf_mass_mult[flat_cols]

        # ----- Private diagonal correction -----
        diag_rows = np.empty(0, dtype=np.int64)
        diag_cols = np.empty(0, dtype=np.int64)
        diag_vals = np.empty(0, dtype=np.float32)

        if force_nonzero_diag:
            # Labeled target diagonal
            row_sums = np.bincount(flat_rows, weights=weights, minlength=N).astype(np.float32)
            inbag_counts = (cache.c_all > 0).sum(axis=1).astype(np.float32)
            inbag_counts[inbag_counts == 0] = 1.0
            labeled_target_diag = row_sums / inbag_counts

            if has_unlabeled:
                unl = cache.idx_unlabeled.astype(np.int64, copy=False)

                # Desired unlabeled diagonal:
                #   (1/T) * sum_t empirical_mult_inbag_by_tree[t] / M_i(t)
                desired_unl_diag_all = np.bincount(
                    flat_rows,
                    weights=cache.empirical_mult_inbag_by_tree[cache.flat_tree_ids] * cache.inv_leaf_mass_mult[flat_cols],
                    minlength=N
                ).astype(np.float32) / np.float32(T)
                desired_unl_diag = desired_unl_diag_all[unl]

                # Ordinary unlabeled diagonal contributed by the non-private leaf part:
                #   (1 / |S_i|) * sum_{t in S_i} empirical_mult_all_by_tree[t] / M_i(t)
                if cache.oob_mask_all is None:
                    raise ValueError("cache.oob_mask_all is required for training-time kernel_method='gap'.")

                q_mask = cache.oob_mask_all.flatten() == 1
                q_rows = cache.flat_rows_all[q_mask]
                q_cols = cache.flat_cols_all[q_mask]
                q_tree_ids = cache.flat_tree_ids[q_mask]

                S_i_counts = cache.oob_mask_all.sum(axis=1).astype(np.float32)
                S_i_counts[S_i_counts == 0] = 1.0
                q_vals = (1.0 / S_i_counts[q_rows]).astype(np.float32)

                ordinary_unl_diag = np.bincount(
                    q_rows,
                    weights=q_vals * cache.empirical_mult_all_by_tree[q_tree_ids] * cache.inv_leaf_mass_mult[q_cols],
                    minlength=N
                ).astype(np.float32)[unl]

                # Labeled rows get their full target diagonal
                lab_mask = ~cache.row_is_unlabeled
                lab = np.arange(N, dtype=np.int64)[lab_mask]

                lab_diag_rows = lab
                lab_diag_cols = lab + cache.diag_offset
                lab_diag_vals = labeled_target_diag[lab]

                # Unlabeled rows get only the missing correction
                unl_diag_rows = unl
                unl_diag_cols = unl + cache.diag_offset
                unl_diag_vals = desired_unl_diag - ordinary_unl_diag

                diag_rows = np.concatenate([lab_diag_rows, unl_diag_rows])
                diag_cols = np.concatenate([lab_diag_cols, unl_diag_cols])
                diag_vals = np.concatenate([lab_diag_vals, unl_diag_vals]).astype(np.float32)

            else:
                diag_rows = np.arange(N, dtype=np.int64)
                diag_cols = diag_rows + cache.diag_offset
                diag_vals = labeled_target_diag.astype(np.float32)

        elif has_unlabeled:
            # In the zero-diagonal semi-supervised case, W places value 1 on
            # unlabeled private coordinates so that Q can cancel the ordinary
            # unlabeled diagonal exactly.
            diag_rows = cache.idx_unlabeled.astype(np.int64, copy=False)
            diag_cols = diag_rows + cache.diag_offset
            diag_vals = np.ones(len(diag_rows), dtype=np.float32)

        # ----- Final assembly -----
        if needs_private_cols:
            total_cols += N

        if diag_vals.size > 0:
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
    cache : ProximityCache
    kernel_method : str
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
    # GBT PROXIMITY
    # p(i,j) = Sum_t w_t * I( leaf_t(i) = leaf_t(j) )
    #
    # Mapping:
    #   Use sqrt(w_t) on the query side too.
    # ---------------------------------------------------------
    elif kernel_method == "gbt":
        if cache.gbt_tree_weights is None:
            raise ValueError("cache.gbt_tree_weights is required for kernel_method='gbt'.")
        sqrt_w = np.sqrt(cache.gbt_tree_weights).astype(np.float32)
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
        if cache.inv_sqrt_leaf_mass_unit is None:
            raise ValueError("cache.inv_sqrt_leaf_mass_unit is required for kernel_method='kerf'.")
        vals = (1.0 / np.sqrt(T)) * cache.inv_sqrt_leaf_mass_unit[flat_cols]

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
            if cache.oob_mask_all is None:
                raise ValueError("cache.oob_mask_all is required for training-time kernel_method='oob'.")

            # Apply OOB scope on the query side: keep only OOB trees for each i.
            mask = cache.oob_mask_all.flatten() == 1
            flat_rows = flat_rows[mask]
            flat_cols = flat_cols[mask]

            # |S_i| = number of OOB trees for sample i
            S_i_counts = cache.oob_mask_all.sum(axis=1).astype(np.float32)
            S_i_counts[S_i_counts == 0] = 1.0

            # Query-side weights: sqrt(T) / |S_i|
            vals = (np.sqrt(T) / S_i_counts[flat_rows]).astype(np.float32)

            # Matching private diagonal coordinates for exact diagonal replacement
            total_cols += cache.n_samples
            diag_rows = np.arange(N)
            diag_cols = np.arange(N) + cache.diag_offset
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
    # RF-GAP PROXIMITY
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
    #     coordinates, and W determines the final diagonal magnitude.
    #
    #   - If force_nonzero_diag=False in semi-supervised mode, Q places a
    #     negative value on unlabeled private coordinates to cancel the
    #     ordinary unlabeled diagonal induced by the leaf part.
    #
    #     In that ordinary term, unlabeled phantom targets use the
    #     empirical unconditional surrogate empirical_mult_all_by_tree[t]
    #     on the target side.
    # ---------------------------------------------------------
    elif kernel_method == "gap":
        has_unlabeled = cache.is_semi_supervised
        needs_private_cols = force_nonzero_diag or has_unlabeled

        diag_rows = np.empty(0, dtype=np.int64)
        diag_cols = np.empty(0, dtype=np.int64)
        diag_vals = np.empty(0, dtype=np.float32)

        # ----- Ordinary query-side term -----
        if is_training:
            if cache.oob_mask_all is None:
                raise ValueError("cache.oob_mask_all is required for training-time kernel_method='gap'.")

            mask = cache.oob_mask_all.flatten() == 1
            flat_rows = flat_rows[mask]
            flat_cols = flat_cols[mask]
            flat_tree_ids = cache.flat_tree_ids[mask]

            S_i_counts = cache.oob_mask_all.sum(axis=1).astype(np.float32)
            S_i_counts[S_i_counts == 0] = 1.0
            vals = (1.0 / S_i_counts[flat_rows]).astype(np.float32)

            if force_nonzero_diag:
                # Q carries value 1 on all private coordinates.
                # W determines the final diagonal target.
                diag_rows = np.arange(N, dtype=np.int64)
                diag_cols = diag_rows + cache.diag_offset
                diag_vals = np.ones(N, dtype=np.float32)

            elif has_unlabeled:
                # Ordinary unlabeled diagonal induced by the leaf coordinates:
                # unlabeled phantom targets use empirical_mult_all_by_tree[t]
                ordinary_unl_diag = np.bincount(
                    flat_rows,
                    weights=vals * cache.empirical_mult_all_by_tree[flat_tree_ids] * cache.inv_leaf_mass_mult[flat_cols],
                    minlength=N
                ).astype(np.float32)

                diag_rows = cache.idx_unlabeled.astype(np.int64, copy=False)
                diag_cols = diag_rows + cache.diag_offset
                diag_vals = -ordinary_unl_diag[diag_rows]

        else:
            # Extension points average over all trees
            vals = np.full(N * T, 1.0 / T, dtype=np.float32)

        # ----- Final assembly -----
        if needs_private_cols:
            total_cols += cache.n_samples

        if diag_vals.size > 0:
            flat_rows = np.concatenate([flat_rows, diag_rows])
            flat_cols = np.concatenate([flat_cols, diag_cols])
            vals = np.concatenate([vals, diag_vals])

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
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse


@dataclass
class ProximityCache:
    """
    Container for all cached quantities used to build proximity matrices.

    This separates:
    - fitted ensemble state
    - proximity algebra state

    so that the main estimator class remains lighter and easier to maintain.
    """

    # Core leaf structure
    leaf_matrix_all: Optional[np.ndarray] = None          # shape: (N, T), local node ids
    leaf_offsets: Optional[np.ndarray] = None             # shape: (T,)
    total_unique_nodes: Optional[int] = None             # total number of global node ids
    diag_offset: Optional[int] = None                    # offset for virtual diagonal coordinates

    # Optional OOB / multiplicity structure
    oob_mask_all: Optional[np.ndarray] = None            # shape: (N, T), int8/bool
    c_all: Optional[np.ndarray] = None                   # shape: (N, T), float32

    # Flattened global leaf incidences
    flat_rows_all: Optional[np.ndarray] = None           # shape: (N*T,)
    flat_cols_all: Optional[np.ndarray] = None           # shape: (N*T,)

    # Leaf masses
    leaf_mass_unit: Optional[np.ndarray] = None          # counts sample-tree incidences per global leaf
    inv_sqrt_leaf_mass_unit: Optional[np.ndarray] = None # used by KeRF

    leaf_mass_mult: Optional[np.ndarray] = None          # sums multiplicities per global leaf
    inv_leaf_mass_mult: Optional[np.ndarray] = None      # used by RF-GAP

    # GBT-specific
    gbt_tree_weights: Optional[np.ndarray] = None        # shape: (T_total,)

    # Sparse right factor
    W_mat: Optional[sparse.csr_matrix] = None            # shape: (N_ref, total_cols)

    # Metadata
    n_samples: Optional[int] = None                      # number of reference samples
    n_trees: Optional[int] = None                        # number of trees
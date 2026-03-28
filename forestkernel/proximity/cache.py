from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse


@dataclass
class ProximityCache:
    """
    Container for all cached quantities used to build proximity matrices.
    """

    # Core leaf structure
    leaf_matrix_all: Optional[np.ndarray] = None
    leaf_offsets: Optional[np.ndarray] = None
    total_unique_nodes: Optional[int] = None
    diag_offset: Optional[int] = None

    # Optional OOB / multiplicity structure
    oob_mask_all: Optional[np.ndarray] = None
    c_all: Optional[np.ndarray] = None

    # Flattened global leaf incidences
    flat_rows_all: Optional[np.ndarray] = None
    flat_cols_all: Optional[np.ndarray] = None

    # Leaf masses
    leaf_mass_unit: Optional[np.ndarray] = None
    inv_sqrt_leaf_mass_unit: Optional[np.ndarray] = None

    leaf_mass_mult: Optional[np.ndarray] = None
    inv_leaf_mass_mult: Optional[np.ndarray] = None

    # GBT-specific
    gbt_tree_weights: Optional[np.ndarray] = None

    # Sparse right factor
    W_mat: Optional[sparse.csr_matrix] = None

    # Metadata
    n_samples: Optional[int] = None          # total reference samples (labeled + unlabeled if transductive)
    n_trees: Optional[int] = None
    n_train_samples: Optional[int] = None    # number of labeled samples actually used to fit the forest

    # Semi-supervised bookkeeping (original X order)
    idx_labeled: Optional[np.ndarray] = None
    idx_unlabeled: Optional[np.ndarray] = None
    is_semi_supervised: bool = False
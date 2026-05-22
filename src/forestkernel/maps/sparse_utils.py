import numpy as np
from scipy import sparse


def block_symmetrize(Q, W):
    """
    Computes symmetric 'kernel' P using optimized sparse strategies.

    P = 0.5 * (Q W^T + W Q^T)
      = 0.5 * [Q, W] [W^T; Q^T]

    Uses the block matrix trick to avoid explicitly materializing both
    asymmetric products separately.
    """
    left_block = sparse.hstack([Q, W], format="csr", dtype=np.float32)
    right_block_T = sparse.vstack([W.T, Q.T], format="csc", dtype=np.float32)
    P = 0.5 * left_block.dot(right_block_T)
    del left_block, right_block_T
    return P


def format_output_matrix(M, return_dense=False):
    """
    Format matrix-like outputs as sparse by default, or dense on demand.
    """
    if return_dense and hasattr(M, "toarray"):
        return M.toarray()
    return M

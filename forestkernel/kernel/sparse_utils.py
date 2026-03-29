import numpy as np
from scipy import sparse


def csr_row_scale_inplace(A, scale):
    """
    In-place row scaling of a CSR matrix.
    A[i, :] *= scale[i]
    """
    if not sparse.isspmatrix_csr(A):
        raise ValueError("Matrix must be CSR for in-place scaling.")

    scale = np.asarray(scale, dtype=A.data.dtype)
    nnz_per_row = np.diff(A.indptr)
    A.data *= np.repeat(scale, nnz_per_row)


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
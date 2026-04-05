import numpy as np


def original_proximity_dense_from_leaves(leaf_matrix):
    """
    Compute the naive dense original proximity matrix from a leaf matrix.

    The original proximity between samples i and j is

        p(i, j) = (1 / T) * sum_t 1{ leaf_t(i) == leaf_t(j) }

    Parameters
    ----------
    leaf_matrix : ndarray of shape (n_samples, n_trees)
        Leaf indices returned by apply().

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Dense original proximity matrix.
    """
    leaf_matrix = np.asarray(leaf_matrix)
    n_samples, n_trees = leaf_matrix.shape

    prox = np.zeros((n_samples, n_samples), dtype=np.float32)

    for t in range(n_trees):
        prox += (leaf_matrix[:, t][:, None] == leaf_matrix[:, t][None, :])

    prox /= np.float32(n_trees)
    return prox


def original_proximity_block_dense_from_leaves(leaf_matrix_ref, leaf_matrix_query):
    """
    Compute the naive dense original proximity block between query samples
    and reference samples from two leaf matrices.

    Parameters
    ----------
    leaf_matrix_ref : ndarray of shape (n_ref, n_trees)
        Leaf indices for the reference samples.

    leaf_matrix_query : ndarray of shape (n_query, n_trees)
        Leaf indices for the query samples.

    Returns
    -------
    ndarray of shape (n_query, n_ref)
        Dense original proximity block.
    """
    leaf_matrix_ref = np.asarray(leaf_matrix_ref)
    leaf_matrix_query = np.asarray(leaf_matrix_query)

    n_query, n_trees = leaf_matrix_query.shape
    n_ref = leaf_matrix_ref.shape[0]

    if leaf_matrix_ref.shape[1] != n_trees:
        raise ValueError("Reference and query leaf matrices must have the same number of trees.")

    prox = np.zeros((n_query, n_ref), dtype=np.float32)

    for t in range(n_trees):
        prox += (leaf_matrix_query[:, t][:, None] == leaf_matrix_ref[None, :, t])

    prox /= np.float32(n_trees)
    return prox


def original_proximity_dense_from_forest(forest, X):
    """
    Compute the naive dense original proximity matrix directly from a fitted forest
    using its apply() method.

    Parameters
    ----------
    forest : fitted tree ensemble
        Any fitted object exposing an apply(X) method returning a leaf matrix.

    X : array-like of shape (n_samples, n_features)
        Input samples.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Dense original proximity matrix.
    """
    leaf_matrix = forest.apply(X)
    return original_proximity_dense_from_leaves(leaf_matrix)


def original_proximity_block_dense_from_forest(forest, X_ref, X_query):
    """
    Compute the naive dense original proximity block between query samples
    and reference samples directly from a fitted forest using apply().

    Parameters
    ----------
    forest : fitted tree ensemble
        Any fitted object exposing an apply(X) method returning a leaf matrix.

    X_ref : array-like of shape (n_ref, n_features)
        Reference samples.

    X_query : array-like of shape (n_query, n_features)
        Query samples.

    Returns
    -------
    ndarray of shape (n_query, n_ref)
        Dense original proximity block.
    """
    leaf_matrix_ref = forest.apply(X_ref)
    leaf_matrix_query = forest.apply(X_query)
    return original_proximity_block_dense_from_leaves(
        leaf_matrix_ref=leaf_matrix_ref,
        leaf_matrix_query=leaf_matrix_query,
    )
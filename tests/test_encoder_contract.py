import numpy as np
import pytest

from forestkernel import LeafEncoder


RF_ET_FORESTS_AND_DATA = [
    ("rf_classifier", "classification_data"),
    ("et_classifier", "classification_data"),
    ("rf_regressor", "regression_data"),
    ("et_regressor", "regression_data"),
]

BOOSTED_FORESTS_AND_DATA = [
    ("gbt_classifier", "classification_data"),
    ("lgbm_classifier", "classification_data"),
    ("xgb_classifier", "classification_data"),
]

ALL_SUPPORTED_CASES = [
    *[(f, d, s) for f, d in RF_ET_FORESTS_AND_DATA for s in ["uniform", "kerf", "oob", "gap"]],
    *[(f, d, s) for f, d in BOOSTED_FORESTS_AND_DATA for s in ["uniform", "kerf", "boosted"]],
]


def _max_nnz_per_row(A):
    return np.diff(A.tocsr().indptr).max()


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", ALL_SUPPORTED_CASES)
def test_leaf_maps_have_consistent_shapes(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    Q_train = enc.training_query_map(return_dense=False)
    W = enc.reference_map(return_dense=False)
    Q_test = enc.transform(X_test, return_dense=False)

    assert Q_train.shape[0] == X_train.shape[0]
    assert Q_test.shape[0] == X_test.shape[0]

    assert Q_train.shape[1] == W.shape[1]
    assert Q_test.shape[1] == W.shape[1]

    assert W.shape[0] == X_train.shape[0]


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", ALL_SUPPORTED_CASES)
def test_leaf_maps_have_at_most_one_nonzero_per_tree_per_row(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    Q_train = enc.training_query_map(return_dense=False)
    W = enc.reference_map(return_dense=False)
    Q_test = enc.transform(X_test, return_dense=False)

    n_trees = enc.cache_.leaf_matrix.shape[1]

    assert _max_nnz_per_row(Q_train) <= n_trees
    assert _max_nnz_per_row(W) <= n_trees
    assert _max_nnz_per_row(Q_test) <= n_trees


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", ALL_SUPPORTED_CASES)
def test_kernel_matches_leaf_factorization(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    K = enc.kernel(return_dense=True)
    Q = enc.training_query_map(return_dense=False)
    W = enc.reference_map(return_dense=False)

    assert np.allclose(K, (Q @ W.T).toarray())


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", ALL_SUPPORTED_CASES)
def test_kernel_extend_matches_leaf_factorization(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    K_test = enc.kernel_extend(X_test, return_dense=True)
    Q_test = enc.transform(X_test, return_dense=False)
    W = enc.reference_map(return_dense=False)

    assert np.allclose(K_test, (Q_test @ W.T).toarray())


@pytest.mark.parametrize("forest_fixture,data_fixture", RF_ET_FORESTS_AND_DATA)
def test_kerf_train_kernel_is_doubly_stochastic_and_test_rows_sum_to_one(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme="kerf").fit(X_train, y_train)

    K_train = enc.kernel(return_dense=False)
    K_test = enc.kernel_extend(X_test, return_dense=False)

    train_row_sums = np.asarray(K_train.sum(axis=1)).ravel()
    train_col_sums = np.asarray(K_train.sum(axis=0)).ravel()
    test_row_sums = np.asarray(K_test.sum(axis=1)).ravel()

    assert np.allclose(train_row_sums, 1.0, rtol=1e-5, atol=1e-6)
    assert np.allclose(train_col_sums, 1.0, rtol=1e-5, atol=1e-6)
    assert np.allclose(test_row_sums, 1.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("forest_fixture,data_fixture", BOOSTED_FORESTS_AND_DATA)
def test_boosted_kernel_matches_explicit_weighted_leaf_collisions(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme="boosted").fit(X_train, y_train)

    K_fast = enc.kernel(return_dense=True)
    leaves = enc.forest_.get_leaf_matrix(X_train)
    weights = enc.forest_.get_tree_weights(X_train)

    n_samples, n_trees = leaves.shape
    K_slow = np.zeros((n_samples, n_samples), dtype=np.float32)

    for t in range(n_trees):
        same_leaf = leaves[:, [t]] == leaves[:, [t]].T
        K_slow += weights[t] * same_leaf.astype(np.float32)

    assert np.allclose(K_fast, K_slow, rtol=1e-5, atol=1e-6)
import numpy as np
import pytest

from forestkernel import LeafEncoder

from tests.fixtures.constants import RF_ET_FORESTS_AND_DATA, RF_ET_WEIGHT_SCHEMES


@pytest.mark.parametrize("forest_fixture,data_fixture", RF_ET_FORESTS_AND_DATA)
def test_gap_train_and_test_rows_sum_to_one_exactly(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme="gap").fit(X_train, y_train)

    K_train = enc.kernel(return_dense=False)
    K_test = enc.kernel_extend(X_test, return_dense=False)

    train_row_sums = np.asarray(K_train.sum(axis=1)).ravel()
    test_row_sums = np.asarray(K_test.sum(axis=1)).ravel()

    assert np.allclose(train_row_sums, 1.0, rtol=1e-5, atol=1e-6)
    assert np.allclose(test_row_sums, 1.0, rtol=1e-5, atol=1e-6)


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


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", [
    (f, d, s)
    for f, d in RF_ET_FORESTS_AND_DATA
    for s in RF_ET_WEIGHT_SCHEMES
    if s != "gap"
])
def test_training_kernel_is_symmetric(request, forest_fixture, data_fixture, weight_scheme):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    K = enc.kernel(return_dense=True)

    assert K.shape[0] == K.shape[1]
    assert np.allclose(K, K.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("forest_fixture,data_fixture", RF_ET_FORESTS_AND_DATA)
def test_gap_force_symmetric_kernel_is_symmetric(request, forest_fixture, data_fixture):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme="gap").fit(X_train, y_train)

    K = enc.kernel(force_symmetric=True, return_dense=True)

    assert K.shape[0] == K.shape[1]
    assert np.allclose(K, K.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("forest_fixture,data_fixture", RF_ET_FORESTS_AND_DATA)
@pytest.mark.parametrize("weight_scheme", ["gap", "oob"])
def test_adjust_diagonal_matches_expected_gap_and_oob_training_diagonal(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = LeafEncoder(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    K = enc.kernel(adjust_diagonal=True, return_dense=True)
    diag = np.diag(K)

    if weight_scheme == "oob":
        assert np.allclose(diag, 1.0, rtol=1e-6, atol=1e-8)
        return

    ref_mass = np.asarray(enc.reference_map(return_dense=False).sum(axis=1)).ravel()
    inbag_tree_counts = (enc.cache_.inbag_counts > 0).sum(axis=1).astype(np.float32)
    inbag_tree_counts[inbag_tree_counts == 0] = 1.0
    expected_diag = ref_mass / inbag_tree_counts

    assert np.allclose(diag, expected_diag, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "est_class,data_fixture",
    [
        ("RandomForestClassifier", "classification_data"),
        ("ExtraTreesClassifier", "classification_data"),
        ("RandomForestRegressor", "regression_data"),
        ("ExtraTreesRegressor", "regression_data"),
    ],
)
@pytest.mark.parametrize("weight_scheme", ["gap", "oob"])
def test_gap_oob_require_bootstrap_true(request, est_class, data_fixture, weight_scheme):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)

    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        RandomForestRegressor,
        ExtraTreesRegressor,
    )

    cls_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
    }

    est = cls_map[est_class](n_estimators=10, bootstrap=False, random_state=0, n_jobs=1)
    enc = LeafEncoder(forest=est, weight_scheme=weight_scheme)

    with pytest.raises(ValueError):
        enc.fit(X_train, y_train)

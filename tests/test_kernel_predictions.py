import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error
import pytest

from forestkernel import LeafEncoder

from tests.test_constants import RF_ET_WEIGHT_SCHEMES


@pytest.mark.parametrize("forest_fixture", ["rf_classifier", "et_classifier"])
@pytest.mark.parametrize("data_fixture", ["classification_data"])
def test_gap_matches_base_classifier_predictions(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)

    kernel_model = LeafEncoder(forest=clone(forest), weight_scheme="gap")
    kernel_preds = kernel_model.fit(X_train, y_train).predict(X_test)

    np.testing.assert_array_equal(kernel_preds, base_forest.predict(X_test))

    proba = kernel_model.predict_proba(X_test)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(proba.shape[0]), atol=1e-6)
    np.testing.assert_array_equal(kernel_model.classes_, base_forest.classes_)


@pytest.mark.parametrize("forest_fixture", ["rf_regressor", "et_regressor"])
@pytest.mark.parametrize("data_fixture", ["regression_data"])
def test_gap_matches_base_regressor_predictions(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)

    kernel_model = LeafEncoder(forest=clone(forest), weight_scheme="gap")
    kernel_preds = kernel_model.fit(X_train, y_train).predict(X_test)

    np.testing.assert_allclose(kernel_preds, base_forest.predict(X_test))

    with pytest.raises(AttributeError):
        kernel_model.predict_proba(X_test)


@pytest.mark.parametrize("forest_fixture", ["rf_classifier", "et_classifier"])
@pytest.mark.parametrize("data_fixture", ["classification_data_10k"])
def test_gap_is_closest_to_base_classifier_error(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)
    base_error = 1.0 - accuracy_score(y_test, base_forest.predict(X_test))

    scheme_errors = {}
    for scheme in RF_ET_WEIGHT_SCHEMES:
        kernel_model = LeafEncoder(forest=clone(forest), weight_scheme=scheme)
        preds = kernel_model.fit(X_train, y_train).predict(X_test)
        scheme_errors[scheme] = abs((1.0 - accuracy_score(y_test, preds)) - base_error)

    assert scheme_errors["gap"] <= min(v for k, v in scheme_errors.items() if k != "gap") + 1e-8


@pytest.mark.parametrize("forest_fixture", ["rf_regressor", "et_regressor"])
@pytest.mark.parametrize("data_fixture", ["regression_data_10k"])
def test_gap_is_closest_to_base_regressor_mse(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)
    base_mse = mean_squared_error(y_test, base_forest.predict(X_test))

    scheme_mses = {}
    for scheme in RF_ET_WEIGHT_SCHEMES:
        kernel_model = LeafEncoder(forest=clone(forest), weight_scheme=scheme)
        preds = kernel_model.fit(X_train, y_train).predict(X_test)
        scheme_mses[scheme] = abs(mean_squared_error(y_test, preds) - base_mse)

    assert scheme_mses["gap"] <= min(v for k, v in scheme_mses.items() if k != "gap") + 1e-8

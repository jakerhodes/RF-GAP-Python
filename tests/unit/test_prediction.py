import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error
import pytest

from forestgeom import LeafEncoder

from tests.fixtures.constants import RF_ET_WEIGHT_SCHEMES


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


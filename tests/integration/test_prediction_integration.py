import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error

from forestgeom import LeafEncoder

from tests.fixtures.constants import RF_ET_WEIGHT_SCHEMES


@pytest.mark.integration
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
        proximity_model = LeafEncoder(forest=clone(forest), weight_scheme=scheme)
        preds = proximity_model.fit(X_train, y_train).proximity_predict(X_test)
        scheme_errors[scheme] = abs((1.0 - accuracy_score(y_test, preds)) - base_error)

    assert scheme_errors["gap"] <= min(v for k, v in scheme_errors.items() if k != "gap") + 1e-8


@pytest.mark.integration
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
        proximity_model = LeafEncoder(forest=clone(forest), weight_scheme=scheme)
        preds = proximity_model.fit(X_train, y_train).proximity_predict(X_test)
        scheme_mses[scheme] = abs(mean_squared_error(y_test, preds) - base_mse)

    assert scheme_mses["gap"] <= min(v for k, v in scheme_mses.items() if k != "gap") + 1e-8

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=150,
        n_classes=3,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=150,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def classification_data_10k():
    X, y = make_classification(
        n_samples=10_000,
        n_classes=3,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def regression_data_10k():
    X, y = make_regression(
        n_samples=10_000,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=0,
    )
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture
def rf_classifier():
    return RandomForestClassifier(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def et_classifier():
    return ExtraTreesClassifier(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def rf_regressor():
    return RandomForestRegressor(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def et_regressor():
    return ExtraTreesRegressor(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def gbt_classifier():
    return GradientBoostingClassifier(
        n_estimators=50,
        random_state=0,
    )


@pytest.fixture
def lgbm_classifier():
    if LGBMClassifier is None:
        pytest.importorskip("lightgbm")

    return LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=0,
        verbose=-1,
        n_jobs=-1,
    )


@pytest.fixture
def xgb_classifier():
    if XGBClassifier is None:
        pytest.importorskip("xgboost")

    return XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=0,
        verbosity=0,
        n_jobs=-1,
        eval_metric="logloss",
    )
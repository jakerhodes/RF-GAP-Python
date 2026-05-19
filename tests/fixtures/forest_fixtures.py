import pytest
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


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

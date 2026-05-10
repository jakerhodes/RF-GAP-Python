from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from .rf_et import RFETAdapter
from .gbt import GBTAdapter


try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from .lgbm import LightGBMAdapter

    _LGBM_CLASSES = (LGBMClassifier, LGBMRegressor)
except ImportError:
    LightGBMAdapter = None
    _LGBM_CLASSES = ()


try:
    from xgboost import XGBClassifier, XGBRegressor
    from .xgb import XGBoostAdapter

    _XGB_CLASSES = (XGBClassifier, XGBRegressor)
except ImportError:
    XGBoostAdapter = None
    _XGB_CLASSES = ()


_RF_ET_CLASSES = (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

_GBT_CLASSES = (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


def make_adapter(estimator, weight_scheme=None):
    """
    Return the correct ensemble adapter for a supported estimator.

    If weight_scheme is provided, validate that the selected adapter supports
    this forest / weight_scheme combination.
    """
    if isinstance(estimator, _RF_ET_CLASSES):
        adapter = RFETAdapter(estimator)

    elif isinstance(estimator, _GBT_CLASSES):
        adapter = GBTAdapter(estimator)

    elif _LGBM_CLASSES and isinstance(estimator, _LGBM_CLASSES):
        adapter = LightGBMAdapter(estimator)

    elif _XGB_CLASSES and isinstance(estimator, _XGB_CLASSES):
        adapter = XGBoostAdapter(estimator)

    else:
        supported = [
            "RandomForestClassifier/Regressor",
            "ExtraTreesClassifier/Regressor",
            "GradientBoostingClassifier/Regressor",
        ]

        if _LGBM_CLASSES:
            supported.append("LGBMClassifier/Regressor")

        if _XGB_CLASSES:
            supported.append("XGBClassifier/Regressor")

        raise TypeError(
            "Unsupported forest estimator. Expected one of: "
            + ", ".join(supported)
            + "."
        )

    if weight_scheme is not None:
        adapter.validate_weight_scheme(weight_scheme)

    return adapter
from .rf_et import RFETAdapter
from .gbt import GBTAdapter
from .lgbm import LightGBMAdapter
from .xgb import XGBoostAdapter



def make_adapter(estimator, forest_type):
    """
    Factory returning the correct ensemble adapter for the fitted estimator.
    """
    if forest_type in ("rf", "et"):
        return RFETAdapter(estimator)
    if forest_type == "gbt":
        return GBTAdapter(estimator)
    if forest_type == "lgbm":
        return LightGBMAdapter(estimator)
    if forest_type == "xgb":
        return XGBoostAdapter(estimator)
    
    raise ValueError(f"Unsupported forest_type: {forest_type}")
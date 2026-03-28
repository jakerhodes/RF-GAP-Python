from .rf_et import RFETAdapter
from .gbt import GBTAdapter
from .rotf import ROTFAdapter


def make_adapter(estimator, model_type):
    """
    Factory returning the correct ensemble adapter for the fitted estimator.
    """
    if model_type in ("rf", "et"):
        return RFETAdapter(estimator)
    if model_type == "gbt":
        return GBTAdapter(estimator)
    if model_type == "rotf":
        return ROTFAdapter(estimator)

    raise ValueError(f"Unsupported model_type='{model_type}'")
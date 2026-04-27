import inspect
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


def infer_prediction_type(prediction_type=None, y=None):
    """
    Infer prediction type from user input.

    Parameters
    ----------
    prediction_type : str or None
        Explicit prediction type. If provided, it is returned unchanged.
    y : array-like or None
        Optional target values used to infer regression vs classification.

    Returns
    -------
    str
        'classification' or 'regression'
    """
    if prediction_type is not None:
        return prediction_type

    if y is None:
        return "classification"

    if isinstance(y, pd.Series):
        y_array = y.to_numpy()
    else:
        y_array = np.asarray(y)

    try:
        if np.issubdtype(y_array.dtype, np.floating):
            return "regression"
        return "classification"
    except TypeError:
        return "classification"


def validate_model_configuration(forest_type, weight_scheme, kwargs=None):
    """
    Validate the high-level compatibility between forest_type, weight_scheme, and a few kernel-specific estimator requirements.

    Parameters
    ----------
    forest_type : str
        One of {'rf', 'et', 'gbt', 'lgbm', 'xgb'}.
    weight_scheme : str
        One of {'original', 'oob', 'gap', 'kerf', 'boosted'}.
    kwargs : dict or None
        Estimator kwargs supplied by the user. These are not modified here,
        only checked when needed for kernel-specific requirements.
    """
    if forest_type not in {"rf", "et", "gbt", "lgbm", "xgb"}:
        raise ValueError(
            "forest_type must be one of {'rf', 'et', 'gbt', 'lgbm', 'xgb'}"
        )

    if weight_scheme not in {"original", "oob", "gap", "kerf", "boosted"}:
        raise ValueError(
            "weight_scheme must be one of {'original', 'oob', 'gap', 'kerf', 'boosted'}"
        )

    kwargs = {} if kwargs is None else dict(kwargs)

    # ---------------------------------------------------------
    # RF / ET:
    # all forest-style kernels are allowed, but OOB / GAP need bootstrap
    # ---------------------------------------------------------
    if forest_type in {"rf", "et"}:
        if weight_scheme in {"oob", "gap"}:
            if "bootstrap" in kwargs and kwargs["bootstrap"] is not True:
                raise ValueError(
                    f"weight_scheme='{weight_scheme}' with forest_type='{forest_type}' "
                    "requires bootstrap=True."
                )
        return

    # ---------------------------------------------------------
    # Boosted-tree models:
    # allow leaf-based kernels that do not require OOB / in-bag structure
    # ---------------------------------------------------------
    if forest_type in {"gbt", "lgbm", "xgb"}:
        if weight_scheme in {"oob", "gap"}:
            raise ValueError(
                f"weight_scheme='{weight_scheme}' is not supported for "
                f"forest_type='{forest_type}' because boosted trees do not provide "
                "the RF-style OOB / in-bag structure required by these kernels."
            )
        return


def get_base_model(forest_type, prediction_type):
    """
    Return the base estimator class corresponding to forest_type / prediction_type.
    """
    if forest_type == "rf":
        return (
            RandomForestClassifier
            if prediction_type == "classification"
            else RandomForestRegressor
        )

    if forest_type == "et":
        return (
            ExtraTreesClassifier
            if prediction_type == "classification"
            else ExtraTreesRegressor
        )

    if forest_type == "gbt":
        return (
            GradientBoostingClassifier
            if prediction_type == "classification"
            else GradientBoostingRegressor
        )

    if forest_type == "lgbm":
        return (
            LGBMClassifier
            if prediction_type == "classification"
            else LGBMRegressor
        )

    if forest_type == "xgb":
        return (
            XGBClassifier
            if prediction_type == "classification"
            else XGBRegressor
        )

    raise ValueError(f"Unsupported forest_type='{forest_type}'")


def validate_forest_kwargs(base_model, kwargs, extra_allowed=None):
    """
    Validate kwargs against the constructor signature of the selected base model.

    Parameters
    ----------
    base_model : class
        Estimator class that will be instantiated.
    kwargs : dict
        User-provided kwargs to forward to the estimator constructor.
    extra_allowed : iterable[str] or None
        Optional additional kwargs to tolerate.

    Raises
    ------
    TypeError
        If unsupported kwargs are present.
    """
    extra_allowed = set() if extra_allowed is None else set(extra_allowed)

    sig = inspect.signature(base_model.__init__)
    params = sig.parameters

    # If constructor accepts **kwargs, strict validation is not possible here.
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )
    if has_var_kw:
        return

    allowed = {
        name
        for name, p in params.items()
        if name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    allowed |= extra_allowed

    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(
            f"Unsupported keyword argument(s) for {base_model.__name__}: {unknown}. "
            f"Allowed parameters are: {sorted(allowed)}"
        )
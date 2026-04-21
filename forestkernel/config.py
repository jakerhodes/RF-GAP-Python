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


def validate_model_configuration(model_type, kernel_method, kwargs=None):
    """
    Validate the high-level compatibility between model_type, kernel_method, and a few kernel-specific estimator requirements.

    Parameters
    ----------
    model_type : str
        One of {'rf', 'et', 'gbt', 'lgbm', 'xgb'}.
    kernel_method : str
        One of {'original', 'oob', 'gap', 'kerf', 'boosted'}.
    kwargs : dict or None
        Estimator kwargs supplied by the user. These are not modified here,
        only checked when needed for kernel-specific requirements.
    """
    if model_type not in {"rf", "et", "gbt", "lgbm", "xgb"}:
        raise ValueError(
            "model_type must be one of {'rf', 'et', 'gbt', 'lgbm', 'xgb'}"
        )

    if kernel_method not in {"original", "oob", "gap", "kerf", "boosted"}:
        raise ValueError(
            "kernel_method must be one of {'original', 'oob', 'gap', 'kerf', 'boosted'}"
        )

    kwargs = {} if kwargs is None else dict(kwargs)

    # ---------------------------------------------------------
    # RF / ET:
    # all forest-style kernels are allowed, but OOB / GAP need bootstrap
    # ---------------------------------------------------------
    if model_type in {"rf", "et"}:
        if kernel_method in {"oob", "gap"}:
            if "bootstrap" in kwargs and kwargs["bootstrap"] is not True:
                raise ValueError(
                    f"kernel_method='{kernel_method}' with model_type='{model_type}' "
                    "requires bootstrap=True."
                )
        return

    # ---------------------------------------------------------
    # Boosted-tree models:
    # allow leaf-based kernels that do not require OOB / in-bag structure
    # ---------------------------------------------------------
    if model_type in {"gbt", "lgbm", "xgb"}:
        if kernel_method in {"oob", "gap"}:
            raise ValueError(
                f"kernel_method='{kernel_method}' is not supported for "
                f"model_type='{model_type}' because boosted trees do not provide "
                "the RF-style OOB / in-bag structure required by these kernels."
            )
        return


def get_base_model(model_type, prediction_type):
    """
    Return the base estimator class corresponding to model_type / prediction_type.
    """
    if model_type == "rf":
        return (
            RandomForestClassifier
            if prediction_type == "classification"
            else RandomForestRegressor
        )

    if model_type == "et":
        return (
            ExtraTreesClassifier
            if prediction_type == "classification"
            else ExtraTreesRegressor
        )

    if model_type == "gbt":
        return (
            GradientBoostingClassifier
            if prediction_type == "classification"
            else GradientBoostingRegressor
        )

    if model_type == "lgbm":
        return (
            LGBMClassifier
            if prediction_type == "classification"
            else LGBMRegressor
        )

    if model_type == "xgb":
        return (
            XGBClassifier
            if prediction_type == "classification"
            else XGBRegressor
        )

    raise ValueError(f"Unsupported model_type='{model_type}'")


def validate_model_kwargs(base_model, kwargs, extra_allowed=None):
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
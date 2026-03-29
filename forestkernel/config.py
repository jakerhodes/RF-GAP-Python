import numpy as np
import pandas as pd
import inspect


from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from .wrappers.bagged_rotation_forest import BaggedRotationForest


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


def validate_model_configuration(model_type, prox_method, prediction_type):
    """
    Validate the combination of model type, proximity method, and prediction type.
    """
    if prox_method == "gbt" and model_type != "gbt":
        raise ValueError("prox_method='gbt' requires model_type='gbt'")

    if model_type == "gbt" and prox_method != "gbt":
        raise ValueError("When model_type='gbt', prox_method must be 'gbt'")

    if model_type == "rotf" and prediction_type != "classification":
        raise ValueError("model_type='rotf' currently supports classification only.")

    if model_type not in {"rf", "et", "gbt", "rotf"}:
        raise ValueError(
            "model_type must be one of {'rf', 'et', 'gbt', 'rotf'}"
        )


def get_base_model(model_type, prediction_type):
    """
    Return the base estimator class corresponding to model_type / prediction_type.
    """
    if model_type == "rf":
        return RandomForestClassifier if prediction_type == "classification" else RandomForestRegressor

    if model_type == "et":
        return ExtraTreesClassifier if prediction_type == "classification" else ExtraTreesRegressor

    if model_type == "gbt":
        return GradientBoostingClassifier if prediction_type == "classification" else GradientBoostingRegressor

    if model_type == "rotf":
        return BaggedRotationForest

    raise ValueError(f"Unsupported model_type='{model_type}'")


def sanitize_model_kwargs(model_type, prox_method, kwargs):
    """
    Clean / augment kwargs before passing them to the base estimator.
    """
    kwargs = dict(kwargs)

    # OOB- and RF-GAP-based proximities require bootstrap sampling for RF / ET.
    if prox_method in ["oob", "gap"] and model_type in ["rf", "et"]:
        kwargs["bootstrap"] = True

    # Rotation Forest has a different constructor signature from sklearn forests.
    if model_type == "rotf":
        kwargs.pop("oob_score", None)
        kwargs.pop("verbose", None)
        kwargs.pop("bootstrap", None)
        kwargs.pop("max_samples", None)

    return kwargs


def validate_model_kwargs(base_model, kwargs, extra_allowed=None):
    """
    Validate kwargs against the constructor signature of the selected base model.

    Parameters
    ----------
    base_model : class
        Estimator class that will be instantiated.
    kwargs : dict
        Final kwargs after sanitization.
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

    # If constructor accepts **kwargs, no strict validation is possible/needed.
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return

    allowed = {
        name for name, p in params.items()
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
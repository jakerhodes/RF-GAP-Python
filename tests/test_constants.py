RF_ET_FORESTS_AND_DATA = [
    ("rf_classifier", "classification_data"),
    ("et_classifier", "classification_data"),
    ("rf_regressor", "regression_data"),
    ("et_regressor", "regression_data"),
]

BOOSTED_FORESTS_AND_DATA = [
    ("gbt_classifier", "classification_data"),
    ("lgbm_classifier", "classification_data"),
    ("xgb_classifier", "classification_data"),
]

RF_ET_WEIGHT_SCHEMES = ["uniform", "kerf", "oob", "gap"]

BOOSTED_WEIGHT_SCHEMES = ["uniform", "kerf", "boosted"]

ALL_SUPPORTED_CASES = [
    *[(f, d, s) for f, d in RF_ET_FORESTS_AND_DATA for s in RF_ET_WEIGHT_SCHEMES],
    *[(f, d, s) for f, d in BOOSTED_FORESTS_AND_DATA for s in BOOSTED_WEIGHT_SCHEMES],
]

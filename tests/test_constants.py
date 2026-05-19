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

RF_ET_CLASSIFICATION_CASES = [
    (forest_fixture, data_fixture, weight_scheme)
    for forest_fixture, data_fixture in RF_ET_FORESTS_AND_DATA
    if data_fixture == "classification_data"
    for weight_scheme in RF_ET_WEIGHT_SCHEMES
]

RF_ET_CLASSIFICATION_INDUCTIVE_CASES = [
    (forest_fixture, data_fixture, weight_scheme)
    for forest_fixture, data_fixture in RF_ET_FORESTS_AND_DATA
    if data_fixture == "classification_data"
    for weight_scheme in RF_ET_WEIGHT_SCHEMES
    if weight_scheme not in {"gap", "oob"}  # test leaf map != training query map for them
]

ALL_SUPPORTED_CASES = [
    *[
        (forest_fixture, data_fixture, weight_scheme)
        for forest_fixture, data_fixture in RF_ET_FORESTS_AND_DATA
        for weight_scheme in RF_ET_WEIGHT_SCHEMES
    ],
    *[
        (forest_fixture, data_fixture, weight_scheme)
        for forest_fixture, data_fixture in BOOSTED_FORESTS_AND_DATA
        for weight_scheme in BOOSTED_WEIGHT_SCHEMES
    ],
]

import builtins
import importlib
import sys

import pytest


ADAPTER_MODULE = "forestgeom.adapters"
OPTIONAL_ADAPTER_MODULES = (
    "forestgeom.adapters.lgbm",
    "forestgeom.adapters.xgb",
)


def _import_adapters_with_missing_dependency(monkeypatch, missing_module):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == missing_module or name.startswith(f"{missing_module}."):
            raise ImportError(f"No module named '{missing_module}'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for module_name in (ADAPTER_MODULE, *OPTIONAL_ADAPTER_MODULES):
        sys.modules.pop(module_name, None)

    return importlib.import_module(ADAPTER_MODULE)


@pytest.mark.parametrize(
    "missing_module, expected_adapter_attr, expected_classes_attr",
    [
        ("lightgbm", "LightGBMAdapter", "_LGBM_CLASSES"),
        ("xgboost", "XGBoostAdapter", "_XGB_CLASSES"),
    ],
)
def test_optional_dependency_import_error_is_caught(monkeypatch, missing_module, expected_adapter_attr, expected_classes_attr):
    adapters = _import_adapters_with_missing_dependency(monkeypatch, missing_module)

    assert getattr(adapters, expected_adapter_attr) is None
    assert getattr(adapters, expected_classes_attr) == ()

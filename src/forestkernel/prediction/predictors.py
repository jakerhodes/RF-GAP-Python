from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from ..encoder import LeafEncoder
from .functional import kernel_predict
from .diagnostics import KernelDiagnostics


class KernelClassifier(ClassifierMixin, LeafEncoder):
    def __init__(self, forest=None, weight_scheme="uniform"):
        super().__init__(forest=forest, weight_scheme=weight_scheme)

    def fit(self, X, y, **fit_kwargs):
        if not is_classifier(self.forest):
            raise TypeError("KernelClassifier requires a classifier forest estimator.")

        if hasattr(self, "_diagnostics"):
            del self._diagnostics

        return super().fit(X, y, **fit_kwargs)

    @property
    def diagnostics(self):
        if not hasattr(self, "_diagnostics"):
            self._diagnostics = KernelDiagnostics(self)
        return self._diagnostics

    def predict(self, X):
        self._check_fitted()
        Q = self.transform(X, return_dense=False)

        return kernel_predict(
            Q,
            self.cache_.W_mat,
            self.y_,
            self.weight_scheme,
            prediction_type="classification",
        )

    def predict_proba(self, X):
        self._check_fitted()
        Q = self.transform(X, return_dense=False)

        proba, _ = kernel_predict(
            Q,
            self.cache_.W_mat,
            self.y_,
            self.weight_scheme,
            prediction_type="classification",
            return_proba=True,
        )

        return proba


class KernelRegressor(RegressorMixin, LeafEncoder):
    def __init__(self, forest=None, weight_scheme="uniform"):
        super().__init__(forest=forest, weight_scheme=weight_scheme)

    def fit(self, X, y, **fit_kwargs):
        if not is_regressor(self.forest):
            raise TypeError("KernelRegressor requires a regressor forest estimator.")

        if hasattr(self, "_diagnostics"):
            del self._diagnostics

        return super().fit(X, y, **fit_kwargs)

    @property
    def diagnostics(self):
        if not hasattr(self, "_diagnostics"):
            self._diagnostics = KernelDiagnostics(self)
        return self._diagnostics

    def predict(self, X):
        self._check_fitted()
        Q = self.transform(X, return_dense=False)

        return kernel_predict(
            Q,
            self.cache_.W_mat,
            self.y_,
            self.weight_scheme,
            prediction_type="regression",
        )
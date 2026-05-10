from sklearn.base import ClassifierMixin, RegressorMixin

from .encoder import LeafEncoder
from .prediction import kernel_predict
from .extras.diagnostics import KernelDiagnostics


class KernelClassifier(LeafEncoder, ClassifierMixin):
    def __init__(
        self,
        weight_scheme="original",
        forest_type="rf",
        **forest_kwargs,
    ):
        super().__init__(
            prediction_type="classification",
            weight_scheme=weight_scheme,
            forest_type=forest_type,
            **forest_kwargs,
        )
        self._diagnostics = None

    @property
    def diagnostics(self):
        if self._diagnostics is None:
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


class KernelRegressor(LeafEncoder, RegressorMixin):
    def __init__(
        self,
        weight_scheme="original",
        forest_type="rf",
        **forest_kwargs,
    ):
        super().__init__(
            prediction_type="regression",
            weight_scheme=weight_scheme,
            forest_type=forest_type,
            **forest_kwargs,
        )
        self._diagnostics = None

    @property
    def diagnostics(self):
        if self._diagnostics is None:
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
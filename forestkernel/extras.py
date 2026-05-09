import warnings

import numpy as np
from scipy import sparse
from sklearn import metrics


class KernelDiagnosticsMixin:
    """
    Extra diagnostics built on top of fitted forest kernels.

    These utilities assume the main estimator exposes the LeafEncoder API:
    `kernel`, `kernel_extend`, `training_query_map`, `reference_map`,
    `transform`, `weight_scheme`, `prediction_type`, `forest_`, and `y_`.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_gap(self):
        if self.weight_scheme != "gap":
            raise ValueError("This diagnostic is only available for GAP kernels.")

    def _check_classification(self):
        if self.prediction_type != "classification":
            raise ValueError("This diagnostic is only available for classification.")

    def _check_regression(self):
        if self.prediction_type == "classification":
            raise ValueError("This diagnostic is only available for regression.")

    def _get_oob_decision_function(self):
        if not hasattr(self.forest_, "oob_decision_function_"):
            raise ValueError(
                "The underlying forest must be fit with `oob_score=True`."
            )
        return self.forest_.oob_decision_function_

    def _get_oob_prediction(self):
        if not hasattr(self.forest_, "oob_prediction_"):
            raise ValueError(
                "The underlying forest must be fit with `oob_score=True`."
            )
        return self.forest_.oob_prediction_

    def _oob_correctness(self):
        oob_proba = self._get_oob_decision_function()
        oob_predictions = np.argmax(oob_proba, axis=1)
        is_correct_oob = oob_predictions == self.y_

        self.oob_proba = oob_proba
        self.oob_predictions = oob_predictions
        self.is_correct_oob = is_correct_oob

        return is_correct_oob

    @staticmethod
    def _as_dense(matrix):
        if sparse.issparse(matrix):
            return matrix.toarray()
        return np.asarray(matrix)

    # ------------------------------------------------------------------
    # RF-ICE / trust scores
    # ------------------------------------------------------------------
    def get_instance_classification_expectation(self):
        """
        Compute RF-ICE trust scores for fitted training samples.

        Uses the factorization

            P c = Q (W^T c)

        without materializing the full train-train kernel matrix.
        """
        self._check_fitted()
        self._check_gap()
        self._check_classification()

        is_correct_oob = self._oob_correctness().astype(float)

        Q = self.cache_.Q_mat
        W = self.cache_.W_mat

        self.trust_scores = np.asarray(Q @ (W.T @ is_correct_oob)).ravel()

        quantile_levels = np.linspace(0, 0.99, 100)
        self.trust_quantiles = np.quantile(self.trust_scores, quantile_levels)

        (
            self.trust_auc,
            self.trust_accuracy_drop,
            self.trust_n_drop,
        ) = self.accuracy_rejection_auc(
            self.trust_quantiles,
            self.trust_scores,
        )

        return self.trust_scores

    def get_test_trust(self, X_test):
        """
        Compute RF-ICE trust scores for new samples.

        Uses the factorization

            P_test c = Q_test (W^T c)

        without materializing the full test-train kernel block.
        """
        self._check_fitted()
        self._check_gap()
        self._check_classification()

        is_correct_oob = self._oob_correctness().astype(float)

        Q_test = self.transform(X_test, return_dense=False)
        W = self.cache_.W_mat

        self.trust_scores_test = np.asarray(Q_test @ (W.T @ is_correct_oob)).ravel()

        quantile_levels = np.linspace(0, 0.99, 100)
        self.trust_quantiles_test = np.quantile(
            self.trust_scores_test,
            quantile_levels,
        )

        return self.trust_scores_test

    # ------------------------------------------------------------------
    # Prediction intervals
    # ------------------------------------------------------------------
    def predict_with_intervals(
        self,
        X_test,
        n_neighbors="auto",
        level=0.95,
        verbose=True,
    ):
        """
        Generate point predictions with prediction intervals for regression.

        Prediction intervals are based on the distribution of OOB residuals
        weighted by GAP kernel neighborhoods.
        """
        self._check_fitted()
        self._check_gap()
        self._check_regression()

        if not 0.0 < level < 1.0:
            raise ValueError("`level` must be between 0 and 1.")

        self.interval_level = level

        test_kernel = self.kernel_extend(X_test, return_dense=True)
        self.test_kernel_ = test_kernel
        self.x_test = X_test

        oob_prediction = self._get_oob_prediction()
        oob_residuals = self.y_ - oob_prediction

        residuals_tiled = np.tile(oob_residuals, (test_kernel.shape[0], 1))

        nearest_indices = np.flip(test_kernel.argsort(axis=1), axis=1)
        nearest_residuals = np.take_along_axis(
            residuals_tiled,
            nearest_indices,
            axis=1,
        )
        self.nearest_neighbor_residuals_ = nearest_residuals

        match n_neighbors:
            case int() if n_neighbors > 0:
                pass

            case float() if n_neighbors > 0:
                n_neighbors = round(n_neighbors)
                if verbose:
                    warnings.warn(
                        "`n_neighbors` must be an integer, 'auto', or 'all'. "
                        f"Using {n_neighbors} nearest neighbors.",
                        category=UserWarning,
                    )

            case "auto":
                test_kernel_sorted = np.take_along_axis(
                    test_kernel,
                    nearest_indices,
                    axis=1,
                )
                self.test_kernel_sorted_ = test_kernel_sorted
                nearest_residuals[test_kernel_sorted < 1e-10] = np.nan

            case "all":
                n_neighbors = nearest_residuals.shape[1]

            case _:
                raise ValueError(
                    "`n_neighbors` must be a positive integer, 'auto', or 'all'."
                )

        self.interval_n_neighbors_ = n_neighbors

        alpha = (1.0 - level) / 2.0

        if n_neighbors == "auto":
            resid_lwr = np.nanquantile(nearest_residuals, alpha, axis=1)
            resid_upr = np.nanquantile(nearest_residuals, 1.0 - alpha, axis=1)
        else:
            resid_lwr = np.quantile(
                nearest_residuals[:, :n_neighbors],
                alpha,
                axis=1,
            )
            resid_upr = np.quantile(
                nearest_residuals[:, :n_neighbors],
                1.0 - alpha,
                axis=1,
            )

        y_pred = self.forest_.predict(X_test)

        y_pred_lwr = y_pred + resid_lwr
        y_pred_upr = y_pred + resid_upr

        return y_pred, y_pred_lwr, y_pred_upr

    # ------------------------------------------------------------------
    # Nonconformity / conformity scores
    # ------------------------------------------------------------------
    def get_nonconformity(self, k=5, X_test=None, weight_scheme=None):
        """
        Compute class-wise nonconformity scores from forest kernels.

        If `X_test` is provided, test nonconformity scores are computed using
        predicted test labels.
        """
        self._check_fitted()
        self._check_classification()

        if not isinstance(k, int) or k <= 0:
            raise ValueError("`k` must be a positive integer.")

        original_weight_scheme = self.weight_scheme

        try:
            if weight_scheme is not None:
                self.set_weight_scheme(weight_scheme)

            oob_proba = self._get_oob_decision_function()
            self.oob_proba = oob_proba
            self.oob_predictions = np.argmax(oob_proba, axis=1)

            K = self.kernel(return_dense=True)

            row_max = np.max(K, axis=1, keepdims=True)
            row_max[row_max == 0.0] = 1.0
            K = K / row_max

            y = self.y_
            self.nonconformity_scores = np.zeros_like(y, dtype=float)

            for label in np.unique(y):
                mask = y == label

                same_K = K[:, mask]
                diff_K = K[:, ~mask]

                same_k = np.partition(same_K, -k, axis=1)[:, -k:]
                diff_k = np.partition(diff_K, -k, axis=1)[:, -k:]

                same_mean = np.mean(same_k, axis=1)[mask]
                diff_mean = np.mean(diff_k, axis=1)[mask]

                min_nonzero = np.min(same_mean[same_mean > 0], initial=1e-10)
                same_mean = np.where(same_mean == 0.0, min_nonzero, same_mean)

                self.nonconformity_scores[mask] = diff_mean / same_mean

            self.conformity_scores = (
                np.max(self.nonconformity_scores) - self.nonconformity_scores
            )
            self.conformity_quantiles = np.quantile(
                self.conformity_scores,
                np.linspace(0, 0.99, 100),
            )

            (
                self.conformity_auc,
                self.conformity_accuracy_drop,
                self.conformity_n_drop,
            ) = self.accuracy_rejection_auc(
                self.conformity_quantiles,
                self.conformity_scores,
            )

            if X_test is not None:
                self.test_preds = self.forest_.predict(X_test)

                K_test = self.kernel_extend(X_test, return_dense=True)

                row_max_test = np.max(K_test, axis=1, keepdims=True)
                row_max_test[row_max_test == 0.0] = 1.0
                K_test = K_test / row_max_test

                self.nonconformity_scores_test = np.zeros_like(
                    self.test_preds,
                    dtype=float,
                )

                for label in np.unique(self.test_preds):
                    mask_test = self.test_preds == label
                    mask_train_same = y == label
                    mask_train_diff = y != label

                    same_K = K_test[:, mask_train_same]
                    diff_K = K_test[:, mask_train_diff]

                    same_k = np.partition(same_K, -k, axis=1)[:, -k:]
                    diff_k = np.partition(diff_K, -k, axis=1)[:, -k:]

                    same_mean_all = np.mean(same_k, axis=1)
                    diff_mean_all = np.mean(diff_k, axis=1)

                    same_mean = same_mean_all[mask_test]
                    diff_mean = diff_mean_all[mask_test]

                    min_nonzero = np.min(same_mean[same_mean > 0], initial=1e-10)
                    same_mean = np.where(same_mean == 0.0, min_nonzero, same_mean)

                    self.nonconformity_scores_test[mask_test] = (
                        diff_mean / same_mean
                    )

                self.conformity_scores_test = (
                    np.max(self.nonconformity_scores_test)
                    - self.nonconformity_scores_test
                )
                self.conformity_quantiles_test = np.quantile(
                    self.conformity_scores_test,
                    np.linspace(0, 0.99, 100),
                )

            return self.nonconformity_scores

        finally:
            if self.weight_scheme != original_weight_scheme:
                self.set_weight_scheme(original_weight_scheme)

    # ------------------------------------------------------------------
    # Accuracy rejection AUC
    # ------------------------------------------------------------------
    def accuracy_rejection_auc(self, quantiles, scores):
        """
        Compute area under the accuracy-rejection curve.
        """
        self._check_fitted()
        self._check_classification()

        quantiles = np.asarray(quantiles)
        scores = np.asarray(scores)

        if scores.shape[0] != self.y_.shape[0]:
            raise ValueError(
                "Mismatch between `scores` length and number of fitted labels."
            )

        oob_proba = self._get_oob_decision_function()
        oob_preds = np.argmax(oob_proba, axis=1)

        n_dropped = np.array(
            [np.sum(scores <= q) / len(scores) for q in quantiles]
        )

        accuracy_drop = np.array(
            [
                np.mean(self.y_[scores >= q] == oob_preds[scores >= q])
                if np.any(scores >= q)
                else 1.0
                for q in quantiles
            ]
        )

        auc = np.trapz(accuracy_drop, n_dropped)

        return auc, accuracy_drop, n_dropped

    # ------------------------------------------------------------------
    # Outlier scores
    # ------------------------------------------------------------------
    def get_outlier_scores(
        self,
        y=None,
        scaling="normalize",
        force_symmetric=False,
        adjust_diagonal=False,
    ):
        """
        Compute class-relative outlier scores from the fitted kernel matrix.
        """
        self._check_fitted()

        if y is None:
            y_arr = self.y_
        else:
            try:
                y_arr = y.to_numpy()
            except AttributeError:
                y_arr = np.asarray(y)

        y_arr = np.asarray(y_arr).ravel()
        n_samples = len(y_arr)

        if n_samples != self.cache_.Q_mat.shape[0]:
            raise ValueError(
                "`y` must have the same length as the fitted training set."
            )

        K = self.kernel(
            force_symmetric=force_symmetric,
            adjust_diagonal=adjust_diagonal,
            return_dense=True,
        )

        avg_prox = np.zeros(n_samples, dtype=float)

        for cls in np.unique(y_arr):
            idx = np.where(y_arr == cls)[0]
            K_sub = K[np.ix_(idx, idx)]
            avg_prox[idx] = np.sum(K_sub**2, axis=1)

        if np.any(avg_prox == 0.0):
            warnings.warn(
                "Some samples have zero average same-class kernel mass. "
                "Outlier scores may be unstable.",
                category=UserWarning,
            )

        avg_prox[avg_prox == 0.0] = 1e-10

        raw_scores = n_samples / avg_prox

        outlier_scores = np.zeros_like(raw_scores)

        for cls in np.unique(y_arr):
            idx = np.where(y_arr == cls)[0]
            class_scores = raw_scores[idx]

            median = np.median(class_scores)
            mad = np.median(np.abs(class_scores - median))

            if mad == 0.0:
                outlier_scores[idx] = 0.0
            else:
                outlier_scores[idx] = np.abs(class_scores - median) / mad

        if scaling == "log":
            outlier_scores = np.log1p(outlier_scores)

        elif scaling == "normalize":
            min_score = np.min(outlier_scores)
            max_score = np.max(outlier_scores)

            if max_score > min_score:
                outlier_scores = (outlier_scores - min_score) / (
                    max_score - min_score
                )
            else:
                outlier_scores = np.zeros_like(outlier_scores)

        elif scaling is None or scaling == "none":
            pass

        else:
            raise ValueError("`scaling` must be 'normalize', 'log', 'none', or None.")

        return outlier_scores
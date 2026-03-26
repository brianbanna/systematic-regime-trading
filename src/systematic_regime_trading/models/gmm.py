"""
Gaussian Mixture Model for regime detection.

Replaces the K-Means inverse-distance probability hack with proper
Bayesian posterior probabilities from GMM. Labels ordered by
component mean (ascending: 0=calm, 2=turbulent).
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


class GMMRegimeDetector:
    """
    Gaussian Mixture Model on market features.
    Proper posterior probabilities, no inverse-distance hack.
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config("models").get("gmm", {})

        self.n_components = config.get("n_components", 3)
        self.random_state = config.get("random_state", 42)
        self.covariance_type = config.get("covariance_type", "full")

        self.scaler_ = None
        self.model_ = None
        self.component_order_ = None
        self.is_fitted_ = False

    def fit(self, X) -> "GMMRegimeDetector":
        """
        Fit scaler + GMM on training data.

        Args:
            X: Feature array or DataFrame

        Returns:
            self
        """
        X_arr = self._to_array(X)
        self._validate_input(X_arr)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)

        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5,
            max_iter=200,
        )
        self.model_.fit(X_scaled)

        # Order components by mean (ascending = calm -> turbulent)
        means_scaled = self.model_.means_
        means_original = self.scaler_.inverse_transform(means_scaled)
        centroid_means = means_original.mean(axis=1)
        self.component_order_ = np.argsort(centroid_means)

        self.is_fitted_ = True

        bic = self.model_.bic(X_scaled)
        logger.info(
            f"GMM fitted: {self.n_components} components, "
            f"BIC={bic:.1f}, converged={self.model_.converged_}"
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Regime labels ordered by volatility (0=calm, 2=turbulent)."""
        self._check_fitted()
        X_scaled = self.scaler_.transform(self._to_array(X))
        raw_labels = self.model_.predict(X_scaled)
        return self._reorder_labels(raw_labels)

    def predict_proba(self, X) -> np.ndarray:
        """
        Proper posterior probabilities P(regime|X) from GMM.

        Returns:
            Array of shape (n_samples, n_components)
        """
        self._check_fitted()
        X_scaled = self.scaler_.transform(self._to_array(X))
        raw_probs = self.model_.predict_proba(X_scaled)
        # Reorder columns to match sorted labels
        return raw_probs[:, self.component_order_]

    def bic(self, X) -> float:
        """Bayesian Information Criterion (lower is better)."""
        self._check_fitted()
        X_scaled = self.scaler_.transform(self._to_array(X))
        return self.model_.bic(X_scaled)

    def aic(self, X) -> float:
        """Akaike Information Criterion."""
        self._check_fitted()
        X_scaled = self.scaler_.transform(self._to_array(X))
        return self.model_.aic(X_scaled)

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, pd.Series):
            return X.values.reshape(-1, 1)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def _validate_input(self, X: np.ndarray):
        if np.isnan(X).any():
            raise ValueError(f"Input contains {np.isnan(X).sum()} NaN values.")

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def _reorder_labels(self, labels: np.ndarray) -> np.ndarray:
        reverse_map = {old: new for new, old in enumerate(self.component_order_)}
        return np.array([reverse_map[l] for l in labels])

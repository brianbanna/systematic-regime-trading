"""
Markov-Switching regime detection (Hamilton 1989).

Uses statsmodels MarkovRegression where the mean and variance
of returns switch between regimes. The gold standard in academic
finance for regime detection.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Optional

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


class MarkovSwitchingRegimeDetector:
    """
    Hamilton (1989) Markov-Switching model.
    Regime-switching mean with switching variance on return series.
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config("models").get("markov_switching", {})

        self.n_regimes = config.get("n_regimes", 3)
        self.switching_variance = config.get("switching_variance", True)
        self.random_state = config.get("random_state", 42)

        self.model_ = None
        self.results_ = None
        self.regime_order_ = None
        self.is_fitted_ = False

    def fit(self, returns) -> "MarkovSwitchingRegimeDetector":
        """
        Fit Markov-Switching model on return series.

        Args:
            returns: Daily return series (pd.Series or 1D array)

        Returns:
            self
        """
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        if isinstance(returns, np.ndarray):
            if returns.ndim > 1:
                returns = returns.flatten()
            returns = pd.Series(returns)

        # Scale returns for numerical stability
        ret_scaled = returns.dropna() * 100

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            try:
                self.model_ = MarkovRegression(
                    ret_scaled,
                    k_regimes=self.n_regimes,
                    switching_variance=self.switching_variance,
                )

                np.random.seed(self.random_state)
                self.results_ = self.model_.fit(
                    maxiter=200,
                    disp=False,
                )

                # Order regimes by intercept (ascending = calm -> turbulent vol)
                # For switching variance, sort by regime variance
                if self.switching_variance:
                    regime_vars = []
                    for i in range(self.n_regimes):
                        try:
                            var_param = self.results_.params[f"sigma2[{i}]"]
                            regime_vars.append(var_param)
                        except (KeyError, IndexError):
                            regime_vars.append(i)
                    self.regime_order_ = np.argsort(regime_vars)
                else:
                    self.regime_order_ = np.arange(self.n_regimes)

                self.is_fitted_ = True
                logger.info(
                    f"MarkovSwitching fitted: {self.n_regimes} regimes, "
                    f"switching_variance={self.switching_variance}"
                )

            except Exception as e:
                logger.warning(f"MarkovSwitching failed: {e}, using 2-regime fallback")
                # Fallback to 2 regimes (more stable)
                try:
                    self.model_ = MarkovRegression(
                        ret_scaled,
                        k_regimes=2,
                        switching_variance=self.switching_variance,
                    )
                    np.random.seed(self.random_state)
                    self.results_ = self.model_.fit(maxiter=200, disp=False)
                    self.n_regimes = 2
                    self.regime_order_ = np.arange(2)
                    self.is_fitted_ = True
                    logger.info("MarkovSwitching: 2-regime fallback succeeded")
                except Exception as e2:
                    logger.warning(f"MarkovSwitching 2-regime also failed: {e2}")
                    self.is_fitted_ = False

        return self

    def predict(self, returns=None) -> np.ndarray:
        """
        Hard regime labels from smoothed probabilities.

        Args:
            returns: If None, uses training data

        Returns:
            Array of regime labels (0=calm, N=turbulent)
        """
        probs = self.predict_proba(returns)
        return np.argmax(probs, axis=1)

    def predict_proba(self, returns=None) -> np.ndarray:
        """
        Smoothed state probabilities from the fitted model.

        Returns:
            Array of shape (n_samples, n_regimes) with probabilities
        """
        self._check_fitted()

        if returns is not None:
            # For new data, use filtered probabilities from training model
            # (statsmodels doesn't easily support out-of-sample prediction,
            # so we refit on the new data)
            try:
                from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

                if isinstance(returns, np.ndarray):
                    returns = pd.Series(returns.flatten())
                ret_scaled = returns.dropna() * 100

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = MarkovRegression(
                        ret_scaled,
                        k_regimes=self.n_regimes,
                        switching_variance=self.switching_variance,
                    )
                    np.random.seed(self.random_state)
                    results = model.fit(maxiter=200, disp=False)

                probs = results.smoothed_marginal_probabilities.values

                # Pad to n_regimes=3 if we fell back to 2
                if probs.shape[1] < 3:
                    padded = np.zeros((probs.shape[0], 3))
                    padded[:, :probs.shape[1]] = probs
                    probs = padded

                return probs[:, self.regime_order_] if len(self.regime_order_) == probs.shape[1] else probs

            except Exception:
                return np.full((len(returns), 3), 1/3)

        probs = self.results_.smoothed_marginal_probabilities.values

        if probs.shape[1] < 3:
            padded = np.zeros((probs.shape[0], 3))
            padded[:, :probs.shape[1]] = probs
            probs = padded
            return probs

        return probs[:, self.regime_order_]

    def get_transition_matrix(self) -> np.ndarray:
        """Regime transition probability matrix."""
        self._check_fitted()
        return self.results_.regime_transition.T

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

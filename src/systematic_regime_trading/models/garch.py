"""
GARCH model for conditional volatility estimation and regime detection.

Key improvements over ADA:
- Student-t distribution (not normal) to handle fat tails
- Expanding-window quantile thresholds (not full-sample) to prevent lookahead
- predict_proba via distance from quantile thresholds
- Class-based interface with .fit(), .predict(), .predict_proba()
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Tuple, Dict, Optional
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


class GARCHRegimeDetector:
    """
    GARCH(p,q) conditional volatility -> quantile-based regime classification.
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config("models")["garch"]

        self.p = config.get("p", 1)
        self.q = config.get("q", 1)
        self.mean_model = config.get("mean_model", "Constant")
        self.vol_model = config.get("vol_model", "GARCH")
        self.dist = config.get("distribution", "t")
        self.return_scaling = config.get("return_scaling", 100)
        self.quantiles = config.get("regime_quantiles", [0.33, 0.67])

        self.model_ = None
        self.results_ = None
        self.cond_vol_ = None
        self.is_fitted_ = False

    def fit(self, returns: pd.Series) -> "GARCHRegimeDetector":
        """
        Fit GARCH model on training returns.

        Args:
            returns: Time series of returns (decimal form)

        Returns:
            self
        """
        returns_scaled = returns * self.return_scaling

        self.model_ = arch_model(
            returns_scaled,
            mean=self.mean_model,
            vol=self.vol_model,
            p=self.p,
            q=self.q,
            dist=self.dist,
        )

        self.results_ = self.model_.fit(disp="off")
        self.cond_vol_ = self.results_.conditional_volatility / self.return_scaling
        self.is_fitted_ = True

        pers = self.get_persistence()
        logger.info(
            f"GARCH({self.p},{self.q}) fitted: dist={self.dist}, "
            f"persistence={pers['persistence']:.4f}"
        )
        return self

    def predict(self, returns: pd.Series = None) -> np.ndarray:
        """
        Regime labels using expanding-window quantile thresholds.

        Uses only past data for each threshold to prevent lookahead.

        Args:
            returns: If None, uses training data conditional volatility

        Returns:
            Array of regime labels (0=calm, 1=moderate, 2=turbulent)
        """
        self._check_fitted()
        vol = self._get_vol(returns)
        return self._classify_expanding(vol)

    def predict_proba(self, returns: pd.Series = None) -> np.ndarray:
        """
        Pseudo-probabilities based on distance from quantile thresholds.

        For each time step, computes distance to expanding quantile boundaries
        and converts to soft probabilities via sigmoid-like transformation.

        Returns:
            Array of shape (n_samples, 3) with [P(calm), P(moderate), P(turbulent)]
        """
        self._check_fitted()
        vol = self._get_vol(returns)

        n = len(vol)
        probs = np.zeros((n, 3))

        q_low, q_high = self.quantiles

        for t in range(1, n):
            # Expanding window quantiles up to time t (no lookahead)
            vol_history = vol[:t + 1]
            threshold_low = np.quantile(vol_history, q_low)
            threshold_high = np.quantile(vol_history, q_high)

            v = vol[t]
            span = max(threshold_high - threshold_low, 1e-10)

            if v <= threshold_low:
                # Below low threshold: mostly calm
                dist_ratio = (threshold_low - v) / span
                p_calm = min(0.7 + 0.3 * dist_ratio, 0.95)
                p_turb = 0.02
                p_mod = 1.0 - p_calm - p_turb
            elif v >= threshold_high:
                # Above high threshold: mostly turbulent
                dist_ratio = (v - threshold_high) / span
                p_turb = min(0.7 + 0.3 * dist_ratio, 0.95)
                p_calm = 0.02
                p_mod = 1.0 - p_calm - p_turb
            else:
                # Between thresholds: moderate
                position = (v - threshold_low) / span
                p_calm = max(0.1, 0.4 * (1 - position))
                p_turb = max(0.1, 0.4 * position)
                p_mod = 1.0 - p_calm - p_turb

            probs[t] = [p_calm, p_mod, p_turb]

        # First observation: uniform
        probs[0] = [1/3, 1/3, 1/3]

        return probs

    def get_conditional_volatility(self, annualize: bool = True) -> pd.Series:
        """Time-varying conditional volatility."""
        self._check_fitted()
        if annualize:
            return self.cond_vol_ * np.sqrt(252)
        return self.cond_vol_.copy()

    def get_persistence(self) -> Dict[str, float]:
        """Volatility persistence metrics."""
        self._check_fitted()
        params = self.results_.params
        alpha = params.get("alpha[1]", np.nan)
        beta = params.get("beta[1]", np.nan)
        persistence = alpha + beta

        if 0 < persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf if persistence >= 1 else np.nan

        return {
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "half_life": half_life,
            "is_stationary": persistence < 1,
        }

    def forecast(self, horizon: int = 1) -> pd.DataFrame:
        """Multi-step volatility forecast."""
        self._check_fitted()
        fc = self.results_.forecast(horizon=horizon)
        var_fc = fc.variance.iloc[-1]
        vol_fc = np.sqrt(var_fc) / self.return_scaling
        vol_annual = vol_fc * np.sqrt(252)

        return pd.DataFrame({
            "horizon": range(1, horizon + 1),
            "variance": var_fc.values,
            "volatility": vol_fc.values,
            "volatility_annual": vol_annual.values,
        })

    def bic(self) -> float:
        """Bayesian Information Criterion from arch results."""
        self._check_fitted()
        return self.results_.bic

    def summary(self) -> pd.DataFrame:
        """Model parameter summary."""
        self._check_fitted()
        r = self.results_
        return pd.DataFrame({
            "Parameter": r.params.index,
            "Coefficient": r.params.values,
            "Std Error": r.std_err.values,
            "t-stat": r.tvalues.values,
            "p-value": r.pvalues.values,
        })

    # --- Private helpers ---

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def _get_vol(self, returns: pd.Series = None) -> np.ndarray:
        """Get conditional volatility for given returns or training data."""
        if returns is not None and self.results_ is not None:
            # Re-fit on new data to get conditional vol
            scaled = returns * self.return_scaling
            model = arch_model(
                scaled, mean=self.mean_model, vol=self.vol_model,
                p=self.p, q=self.q, dist=self.dist,
            )
            res = model.fit(disp="off", last_obs=len(scaled))
            return (res.conditional_volatility / self.return_scaling).values
        return self.cond_vol_.values

    def _classify_expanding(self, vol: np.ndarray) -> np.ndarray:
        """Classify volatility into regimes using expanding quantile thresholds."""
        n = len(vol)
        labels = np.ones(n, dtype=int)  # default: moderate
        q_low, q_high = self.quantiles

        for t in range(1, n):
            vol_history = vol[:t + 1]
            threshold_low = np.quantile(vol_history, q_low)
            threshold_high = np.quantile(vol_history, q_high)

            if vol[t] <= threshold_low:
                labels[t] = 0  # calm
            elif vol[t] >= threshold_high:
                labels[t] = 2  # turbulent

        return labels


# --- Backwards-compatible function wrappers ---

def fit_garch(
    returns: pd.Series,
    p: int = 1, q: int = 1,
    mean_model: str = "Constant",
    vol_model: str = "GARCH",
    dist: str = "t",
    return_scaling: float = 100,
) -> Tuple:
    config = {
        "p": p, "q": q, "mean_model": mean_model,
        "vol_model": vol_model, "distribution": dist,
        "return_scaling": return_scaling,
    }
    detector = GARCHRegimeDetector(config)
    detector.fit(returns)
    return detector.model_, detector.results_


def extract_conditional_volatility(
    results, return_scaling: float = 100, annualization_factor: int = 252,
) -> pd.Series:
    cond_vol = results.conditional_volatility
    return (cond_vol / return_scaling) * np.sqrt(annualization_factor)


def compute_persistence(results) -> Dict[str, float]:
    params = results.params
    alpha = params.get("alpha[1]", np.nan)
    beta = params.get("beta[1]", np.nan)
    persistence = alpha + beta
    if 0 < persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        half_life = np.inf if persistence >= 1 else np.nan
    return {
        "alpha": alpha, "beta": beta,
        "persistence": persistence, "half_life": half_life,
        "is_stationary": persistence < 1,
    }


def get_model_summary(results) -> pd.DataFrame:
    return pd.DataFrame({
        "Parameter": results.params.index,
        "Coefficient": results.params.values,
        "Std Error": results.std_err.values,
        "t-stat": results.tvalues.values,
        "p-value": results.pvalues.values,
    })


def forecast_volatility(
    results, horizon: int = 1, return_scaling: float = 100,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    fc = results.forecast(horizon=horizon)
    var_fc = fc.variance.iloc[-1]
    vol_fc = np.sqrt(var_fc)
    vol_annual = (vol_fc / return_scaling) * np.sqrt(annualization_factor)
    return pd.DataFrame({
        "horizon": range(1, horizon + 1),
        "variance": var_fc.values,
        "volatility": vol_fc.values,
        "volatility_annual": vol_annual.values,
    })

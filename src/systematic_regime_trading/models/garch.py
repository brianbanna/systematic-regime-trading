"""
GARCH model for conditional volatility estimation and regime detection.

Fitting, volatility extraction, persistence metrics, and forecasting.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Tuple, Dict


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean_model: str = "Constant",
    vol_model: str = "GARCH",
    dist: str = "t",
    return_scaling: float = 100,
) -> Tuple:
    """
    Fit a GARCH model to a return series.

    Args:
        returns: Time series of returns (decimal form)
        p: ARCH order (lags of squared residuals)
        q: GARCH order (lags of conditional variance)
        mean_model: Mean equation type ('Constant', 'Zero', 'AR')
        vol_model: Volatility model type ('GARCH', 'EGARCH')
        dist: Error distribution ('normal', 't', 'skewt')
        return_scaling: Scale factor for returns (default: 100 for percentage)

    Returns:
        Tuple of (model, fitted_results)
    """
    returns_scaled = returns * return_scaling

    model = arch_model(
        returns_scaled, mean=mean_model, vol=vol_model, p=p, q=q, dist=dist,
    )

    results = model.fit(disp="off")

    return model, results


def extract_conditional_volatility(
    results, return_scaling: float = 100, annualization_factor: int = 252
) -> pd.Series:
    """
    Extract conditional volatility from fitted GARCH model.

    Args:
        results: Fitted GARCH model results
        return_scaling: Scale factor used during fitting
        annualization_factor: Trading days per year

    Returns:
        Series of annualized conditional volatility (decimal form)
    """
    cond_vol = results.conditional_volatility
    cond_vol_annual = (cond_vol / return_scaling) * np.sqrt(annualization_factor)
    return cond_vol_annual


def compute_persistence(results) -> Dict[str, float]:
    """
    Compute volatility persistence metrics.

    Persistence = alpha + beta. Values close to 1 mean shocks decay slowly.
    Half-life = log(0.5) / log(persistence).

    Args:
        results: Fitted GARCH model results

    Returns:
        Dict with alpha, beta, persistence, half_life, is_stationary
    """
    params = results.params

    alpha = params.get("alpha[1]", np.nan)
    beta = params.get("beta[1]", np.nan)
    persistence = alpha + beta

    if 0 < persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        half_life = np.inf if persistence >= 1 else np.nan

    is_stationary = persistence < 1

    return {
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "half_life": half_life,
        "is_stationary": is_stationary,
    }


def get_model_summary(results) -> pd.DataFrame:
    """
    Extract key statistics from GARCH model results.

    Args:
        results: Fitted GARCH model results

    Returns:
        DataFrame with Parameter, Coefficient, Std Error, t-stat, p-value
    """
    return pd.DataFrame(
        {
            "Parameter": results.params.index,
            "Coefficient": results.params.values,
            "Std Error": results.std_err.values,
            "t-stat": results.tvalues.values,
            "p-value": results.pvalues.values,
        }
    )


def forecast_volatility(
    results, horizon: int = 1, return_scaling: float = 100,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Generate multi-step volatility forecasts.

    Args:
        results: Fitted GARCH model results
        horizon: Number of periods ahead to forecast
        return_scaling: Scale factor used during fitting
        annualization_factor: Trading days per year

    Returns:
        DataFrame with horizon, variance, volatility, volatility_annual
    """
    forecasts = results.forecast(horizon=horizon)
    variance_forecast = forecasts.variance.iloc[-1]
    vol_forecast = np.sqrt(variance_forecast)
    vol_forecast_annual = (vol_forecast / return_scaling) * np.sqrt(
        annualization_factor
    )

    return pd.DataFrame(
        {
            "horizon": range(1, horizon + 1),
            "variance": variance_forecast.values,
            "volatility": vol_forecast.values,
            "volatility_annual": vol_forecast_annual.values,
        }
    )

"""
Functions for GARCH model volatility persistence analysis.
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
    dist: str = "normal",
) -> Tuple:
    """
    Fit a GARCH model to a return series.

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) captures
    volatility clustering: high volatility periods tend to persist, as do low
    volatility periods. This models the time-varying variance of returns.

    GARCH(1,1) specification:
    - Mean equation: r_t = mu + epsilon_t
    - Variance equation: sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2

    Where:
    - omega: baseline variance level (constant term)
    - alpha: ARCH coefficient (impact of past squared shocks)
    - beta: GARCH coefficient (impact of past conditional variance)

    Args:
        returns: Time series of returns (log returns or percentage returns)
        p: ARCH order (lags of squared residuals, default: 1)
        q: GARCH order (lags of conditional variance, default: 1)
        mean_model: Mean equation type ('Constant', 'Zero', 'AR', etc.)
        vol_model: Volatility model type ('GARCH', 'EGARCH', etc.)
        dist: Error distribution ('normal', 't', 'skewt')

    Returns:
        Tuple of (fitted_model, model_results)
        - fitted_model: The arch_model object with model specification
        - model_results: Fitted results with estimated parameters and diagnostics
    """
    # Scale returns to percentage for numerical stability
    # GARCH models work better with percentage returns (avoids tiny numbers)
    returns_pct = returns * 100

    # Initialize GARCH model with specified parameters
    model = arch_model(returns_pct, mean=mean_model, vol=vol_model, p=p, q=q, dist=dist)

    # Fit model using maximum likelihood estimation
    # disp='off': suppress optimization output
    results = model.fit(disp="off")

    return model, results


def extract_conditional_volatility(results) -> pd.Series:
    """
    Extract conditional volatility from fitted GARCH model.

    Conditional volatility is the model's time-varying estimate of volatility
    based on past returns. Unlike unconditional (constant) volatility, this
    adapts to recent market conditions and captures volatility clustering.

    Args:
        results: Fitted GARCH model results from fit_garch()

    Returns:
        Series of conditional volatility (annualized, in decimal form)
    """
    # Extract conditional volatility from model (in percentage terms)
    cond_vol = results.conditional_volatility

    # Convert to decimal form and annualize for interpretability
    # Step 1: Divide by 100 to convert from percentage to decimal
    # Step 2: Multiply by sqrt(252) to annualize (252 trading days per year)
    cond_vol_annual = (cond_vol / 100) * np.sqrt(252)

    return cond_vol_annual


def compute_persistence(results) -> Dict[str, float]:
    """
    Compute volatility persistence metrics from GARCH model.

    Persistence measures how long volatility shocks last in the market:
    - Persistence = alpha + beta (sum of ARCH and GARCH coefficients)
    - Values close to 1: shocks decay very slowly (high persistence)
    - Values close to 0: shocks decay quickly (low persistence)
    - Values >= 1: non-stationary process (infinite persistence)

    Half-life measures shock decay rate:
    - Formula: log(0.5) / log(persistence)
    - Interpretation: number of periods for a shock to decay by 50%
    - Example: half-life of 10 means shock loses half its impact in 10 days

    Args:
        results: Fitted GARCH model results

    Returns:
        Dict with keys:
        - alpha: ARCH coefficient (impact of past squared shocks)
        - beta: GARCH coefficient (impact of past conditional variance)
        - persistence: alpha + beta (total persistence)
        - half_life: periods for shock to decay by half
        - is_stationary: whether variance process is stationary
    """
    # Extract estimated parameters from model
    params = results.params

    # Get ARCH and GARCH coefficients
    # alpha[1]: coefficient on lagged squared residual (ARCH term)
    # beta[1]: coefficient on lagged conditional variance (GARCH term)
    alpha = params.get("alpha[1]", np.nan)
    beta = params.get("beta[1]", np.nan)

    # Compute total persistence (sum of ARCH and GARCH effects)
    persistence = alpha + beta

    # Compute half-life of volatility shocks
    # Only meaningful for stationary processes (persistence < 1)
    if persistence > 0 and persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        # Non-stationary: shocks never decay (infinite half-life)
        half_life = np.inf if persistence >= 1 else np.nan

    # Check stationarity condition
    # Stationary if persistence < 1 (variance process returns to long-run mean)
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

    Provides a clean summary table of all estimated parameters with their
    statistical significance. Useful for model interpretation and reporting.

    Args:
        results: Fitted GARCH model results

    Returns:
        DataFrame with columns:
        - Parameter: parameter names (mu, omega, alpha[1], beta[1])
        - Coefficient: estimated values
        - Std Error: standard errors of estimates
        - t-stat: t-statistics for hypothesis testing
        - p-value: significance levels
    """
    # Compile all key statistics into a single DataFrame
    summary_df = pd.DataFrame(
        {
            "Parameter": results.params.index,
            "Coefficient": results.params.values,
            "Std Error": results.std_err.values,
            "t-stat": results.tvalues.values,
            "p-value": results.pvalues.values,
        }
    )

    return summary_df


def forecast_volatility(results, horizon: int = 1) -> pd.DataFrame:
    """
    Generate volatility forecasts from fitted GARCH model.

    Uses the estimated GARCH parameters to project future volatility levels.
    Forecasts converge to long-run unconditional variance as horizon increases.

    Args:
        results: Fitted GARCH model results
        horizon: Number of periods ahead to forecast (default: 1)

    Returns:
        DataFrame with columns:
        - horizon: forecast period (1, 2, 3, ...)
        - variance: forecasted variance (percentage^2)
        - volatility: forecasted volatility (percentage)
        - volatility_annual: annualized volatility (decimal)
    """
    # Generate multi-step ahead forecasts
    forecasts = results.forecast(horizon=horizon)

    # Extract variance forecasts for the last observation
    # This gives the forecast path starting from the end of the sample
    variance_forecast = forecasts.variance.iloc[-1]

    # Convert variance to volatility (standard deviation)
    vol_forecast = np.sqrt(variance_forecast)

    # Convert from percentage to decimal and annualize
    # Step 1: Divide by 100 to convert from percentage
    # Step 2: Multiply by sqrt(252) to annualize
    vol_forecast_annual = (vol_forecast / 100) * np.sqrt(252)

    # Compile forecasts into a clean DataFrame
    forecast_df = pd.DataFrame(
        {
            "horizon": range(1, horizon + 1),
            "variance": variance_forecast.values,
            "volatility": vol_forecast.values,
            "volatility_annual": vol_forecast_annual.values,
        }
    )

    return forecast_df

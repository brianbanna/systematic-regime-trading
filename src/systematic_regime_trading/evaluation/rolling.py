"""
Rolling performance metrics.

Rolling Sharpe, volatility, drawdown, and beta for time-varying
performance analysis.
"""

import pandas as pd
import numpy as np


TRADING_DAYS = 252


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Rolling annualized Sharpe ratio.

    Args:
        returns: Daily return series
        window: Rolling window in trading days (default 252 = 1 year)
        risk_free_rate: Annualized risk-free rate

    Returns:
        Series of rolling Sharpe values
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    roll_mean = excess.rolling(window, min_periods=window).mean()
    roll_std = excess.rolling(window, min_periods=window).std()
    return (roll_mean / roll_std.clip(lower=1e-10)) * np.sqrt(TRADING_DAYS)


def rolling_volatility(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """
    Rolling annualized volatility.

    Args:
        returns: Daily return series
        window: Rolling window (default 63 = quarterly)

    Returns:
        Series of rolling annualized vol
    """
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def rolling_drawdown(returns: pd.Series) -> pd.Series:
    """
    Continuous drawdown series (underwater curve).

    Returns:
        Series of drawdown values (negative = underwater)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Rolling beta to benchmark.

    beta = cov(r, b) / var(b)

    Args:
        returns: Strategy daily returns
        benchmark: Benchmark daily returns
        window: Rolling window

    Returns:
        Series of rolling beta values
    """
    common = returns.index.intersection(benchmark.index)
    r = returns.loc[common]
    b = benchmark.loc[common]

    rolling_cov = r.rolling(window, min_periods=window).cov(b)
    rolling_var = b.rolling(window, min_periods=window).var()

    return rolling_cov / rolling_var.clip(lower=1e-10)

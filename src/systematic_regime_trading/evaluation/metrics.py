"""
Core performance metrics.

Sharpe, Sortino, Calmar, max drawdown, CAGR, hit rate, profit factor.
All functions work on daily return series.
"""

import pandas as pd
import numpy as np
from typing import Optional


TRADING_DAYS = 252


def compute_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    positions: pd.Series = None,
) -> dict:
    """
    Compute all standard performance metrics.

    Args:
        returns: Daily return series (net of costs)
        risk_free_rate: Annualized risk-free rate (decimal)
        positions: Optional position series for turnover calculation

    Returns:
        Dict with all metrics
    """
    n_days = len(returns)
    n_years = n_days / TRADING_DAYS

    metrics = {
        "cagr": compute_cagr(returns),
        "annual_vol": compute_annual_vol(returns),
        "sharpe": compute_sharpe(returns, risk_free_rate),
        "sortino": compute_sortino(returns, risk_free_rate),
        "max_drawdown": compute_max_drawdown(returns)[0],
        "max_drawdown_duration": compute_max_drawdown(returns)[1],
        "calmar": compute_calmar(returns),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "hit_rate_daily": (returns > 0).mean(),
        "hit_rate_monthly": compute_monthly_hit_rate(returns),
        "profit_factor": compute_profit_factor(returns),
        "avg_daily_return": returns.mean(),
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "n_days": n_days,
        "n_years": round(n_years, 2),
    }

    if positions is not None:
        daily_turnover = positions.diff().abs()
        metrics["turnover_annual"] = daily_turnover.mean() * TRADING_DAYS

    # Monthly stats
    if isinstance(returns.index, pd.DatetimeIndex):
        monthly = returns.resample("ME").sum()
        metrics["avg_monthly_return"] = monthly.mean()
        metrics["best_month"] = monthly.max()
        metrics["worst_month"] = monthly.min()

    return metrics


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio."""
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def compute_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Sortino ratio: penalizes downside vol only."""
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if excess.mean() > 0 else 0.0
    return (excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS)


def compute_calmar(returns: pd.Series) -> float:
    """CAGR / |Max Drawdown|."""
    cagr = compute_cagr(returns)
    max_dd = abs(compute_max_drawdown(returns)[0])
    if max_dd == 0:
        return np.inf if cagr > 0 else 0.0
    return cagr / max_dd


def compute_cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    cumulative = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    if n_years == 0 or cumulative <= 0:
        return 0.0
    return cumulative ** (1 / n_years) - 1


def compute_annual_vol(returns: pd.Series) -> float:
    """Annualized volatility."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def compute_max_drawdown(returns: pd.Series) -> tuple:
    """
    Max drawdown magnitude and duration.

    Returns:
        (max_dd_magnitude, max_dd_duration_days)
        max_dd is negative (e.g., -0.35 = 35% drawdown)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    max_dd = drawdown.min()

    # Duration: longest time between peaks
    is_at_peak = cumulative == running_max
    if is_at_peak.sum() <= 1:
        duration = len(returns)
    else:
        gaps = np.diff(np.where(is_at_peak)[0])
        duration = int(gaps.max()) if len(gaps) > 0 else 0

    return max_dd, duration


def compute_monthly_hit_rate(returns: pd.Series) -> float:
    """Fraction of months with positive returns."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        return np.nan
    monthly = returns.resample("ME").sum()
    if len(monthly) == 0:
        return 0.0
    return (monthly > 0).mean()


def compute_profit_factor(returns: pd.Series) -> float:
    """Sum of gains / |sum of losses|."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return gains / losses

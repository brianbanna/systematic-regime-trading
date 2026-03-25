"""
Portfolio constraints module.

Applies leverage limits, allocation floors, and turnover caps
to raw allocation signals before backtesting.
"""

import pandas as pd
import numpy as np

from systematic_regime_trading.utils.config import load_config


def construct_portfolio(
    signal: pd.Series,
    config: dict = None,
) -> pd.Series:
    """
    Apply portfolio constraints to raw allocation signal.

    Constraints (from config):
    - Max leverage: cap allocation at max_leverage
    - Min allocation: floor at min_allocation (0 = no shorting)
    - Turnover cap: if annualized turnover exceeds limit, smooth

    Args:
        signal: Raw allocation signal (daily)
        config: Constraints config dict. If None, loads from backtest.yaml

    Returns:
        Constrained allocation signal
    """
    if config is None:
        config = load_config("backtest")["constraints"]

    max_leverage = config.get("max_leverage", 1.0)
    min_allocation = config.get("min_allocation", 0.0)
    max_turnover = config.get("max_turnover_annual", 20.0)

    # Apply allocation bounds
    constrained = signal.clip(lower=min_allocation, upper=max_leverage)

    # Turnover cap: smooth signal if turnover is too high
    daily_turnover = constrained.diff().abs()
    current_annual = daily_turnover.mean() * 252

    if current_annual > max_turnover and max_turnover > 0:
        # Exponential smoothing to reduce turnover
        alpha = _find_smoothing_alpha(constrained, max_turnover)
        constrained = constrained.ewm(alpha=alpha).mean()
        constrained = constrained.clip(lower=min_allocation, upper=max_leverage)

    return constrained


def _find_smoothing_alpha(signal: pd.Series, target_turnover: float) -> float:
    """Binary search for EWM alpha that achieves target turnover."""
    low, high = 0.01, 1.0

    for _ in range(20):
        mid = (low + high) / 2
        smoothed = signal.ewm(alpha=mid).mean()
        turnover = smoothed.diff().abs().mean() * 252

        if turnover > target_turnover:
            high = mid
        else:
            low = mid

    return (low + high) / 2

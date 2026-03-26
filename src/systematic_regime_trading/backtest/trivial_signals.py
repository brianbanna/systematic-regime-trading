"""
Trivial signal benchmarks.

Simple, zero-cost indicators to benchmark against the regime ensemble.
If the ensemble can't beat these, the complexity isn't justified.
"""

import pandas as pd
import numpy as np


def sma_crossover_signal(
    prices: pd.Series,
    window: int = 200,
) -> pd.Series:
    """
    Simple moving average crossover signal.

    Allocation = 1.0 when price > 200-day SMA, 0.0 otherwise.
    This is the simplest possible trend-following signal.

    Args:
        prices: Daily price series (e.g., SPY Adj Close)
        window: SMA lookback window (default 200 days)

    Returns:
        Series of allocation signals (0.0 or 1.0)
    """
    sma = prices.rolling(window, min_periods=window).mean()
    signal = (prices > sma).astype(float)
    signal.name = "sma_crossover"
    return signal


def vix_threshold_signal(
    vix: pd.Series,
    threshold: float = 20.0,
) -> pd.Series:
    """
    VIX threshold signal.

    Allocation = 1.0 when VIX < threshold, 0.0 otherwise.
    A $0 indicator from Yahoo Finance.

    Args:
        vix: Daily VIX closing values
        threshold: VIX level above which to go defensive

    Returns:
        Series of allocation signals (0.0 or 1.0)
    """
    signal = (vix < threshold).astype(float)
    signal.name = "vix_threshold"
    return signal


def vol_managed_signal(
    returns: pd.Series,
    vol_target: float = 0.10,
    lookback: int = 63,
) -> pd.Series:
    """
    Volatility-managed portfolio (no regime detection).

    Scales allocation inversely to realized vol to target a fixed
    annualized volatility. This is the simplest risk management
    approach and serves as a baseline to test whether regime
    detection adds value beyond simple vol-scaling.

    Args:
        returns: Daily market returns
        vol_target: Target annualized volatility (default 10%)
        lookback: Rolling window for vol estimation (default 63 = quarterly)

    Returns:
        Series of allocation signals (clipped to [0, 1])
    """
    realized_vol = returns.rolling(lookback, min_periods=lookback).std() * np.sqrt(252)
    allocation = vol_target / realized_vol.clip(lower=0.01)
    allocation = allocation.clip(lower=0.0, upper=1.0)
    allocation = allocation.shift(1)  # execution lag
    allocation.name = "vol_managed"
    return allocation

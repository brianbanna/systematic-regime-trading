"""
Signal filters: confirmation filter and rate limiter.

Anti-whipsaw and smoothing to reduce turnover from noisy regime transitions.
"""

import pandas as pd
import numpy as np
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


def apply_confirmation_filter(
    raw_signal: pd.Series,
    regime_labels: pd.Series,
    confirmation_days: int = None,
    config: dict = None,
) -> pd.Series:
    """
    Anti-whipsaw filter: only switch allocation when new regime
    persists for N consecutive days.

    Prevents: calm -> turbulent (1 day) -> calm from triggering 2 trades.
    The signal holds the previous allocation until confirmation.

    Args:
        raw_signal: Raw allocation signal
        regime_labels: Hard regime labels (0, 1, 2)
        confirmation_days: Days required in new regime before switching
        config: Strategy config. If None, loads from strategy.yaml

    Returns:
        Filtered allocation signal
    """
    if confirmation_days is None:
        if config is None:
            config = load_config("strategy")
        confirmation_days = config["signal_filters"]["confirmation_days"]

    labels = regime_labels.values
    signal = raw_signal.values.copy()
    n = len(labels)

    confirmed_regime = labels[0]
    consecutive = 1
    confirmed_signal = signal[0]

    result = np.zeros(n)
    result[0] = signal[0]

    for t in range(1, n):
        if labels[t] == confirmed_regime:
            consecutive += 1
            result[t] = signal[t]
        else:
            consecutive = 1 if labels[t] != labels[t - 1] else consecutive + 1

            if consecutive >= confirmation_days:
                confirmed_regime = labels[t]
                confirmed_signal = signal[t]
                result[t] = signal[t]
            else:
                result[t] = confirmed_signal

    return pd.Series(result, index=raw_signal.index, name=raw_signal.name)


def apply_rate_limit(
    signal: pd.Series,
    max_daily_change: float = None,
    config: dict = None,
) -> pd.Series:
    """
    Limit how fast allocation can change per day.

    Prevents jumping from 100% to 0% in a single day.
    Smooths the transition over multiple days.

    Args:
        signal: Allocation signal
        max_daily_change: Maximum allowed change per day (e.g., 0.25)
        config: Strategy config

    Returns:
        Rate-limited signal
    """
    if max_daily_change is None:
        if config is None:
            config = load_config("strategy")
        max_daily_change = config["signal_filters"]["max_daily_allocation_change"]

    values = signal.values.copy()
    n = len(values)
    result = np.zeros(n)
    result[0] = values[0]

    for t in range(1, n):
        target = values[t]
        current = result[t - 1]
        change = target - current

        if abs(change) > max_daily_change:
            change = np.sign(change) * max_daily_change

        result[t] = current + change

    return pd.Series(result, index=signal.index, name=signal.name)


def apply_execution_lag(
    signal: pd.Series,
    lag_days: int = None,
    config: dict = None,
) -> pd.Series:
    """
    Shift signal by lag_days to simulate execution delay.

    Signal generated at close on day T -> position entered at close on day T+1.
    This is the single most important anti-lookahead measure.

    Args:
        signal: Allocation signal
        lag_days: Number of days to lag (default: 1)
        config: Backtest config

    Returns:
        Lagged signal
    """
    if lag_days is None:
        if config is None:
            config = load_config("backtest")
        lag_days = config["execution"]["lag_days"]

    return signal.shift(lag_days)

"""
Statistical significance tests for strategy performance.

Bootstrap confidence intervals for Sharpe ratio and
Sharpe difference tests between strategy and benchmark.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from systematic_regime_trading.evaluation.metrics import compute_sharpe

TRADING_DAYS = 252


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Uses circular block bootstrap to preserve autocorrelation structure.

    Args:
        returns: Daily return series
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95)
        random_state: Random seed

    Returns:
        (sharpe_point_estimate, lower_ci, upper_ci)
        If lower_ci > 0: strategy is statistically significant
    """
    rng = np.random.default_rng(random_state)
    n = len(returns)
    block_size = max(int(np.sqrt(n)), 5)

    returns_arr = returns.values
    point_estimate = compute_sharpe(returns)

    bootstrap_sharpes = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        n_blocks = n // block_size + 1
        start_indices = rng.integers(0, n, size=n_blocks)

        sample = []
        for start in start_indices:
            block = np.take(returns_arr, range(start, start + block_size), mode="wrap")
            sample.extend(block)

        sample = np.array(sample[:n])

        if sample.std() > 0:
            bootstrap_sharpes[i] = (sample.mean() / sample.std()) * np.sqrt(TRADING_DAYS)
        else:
            bootstrap_sharpes[i] = 0.0

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_sharpes, 100 * alpha / 2)
    upper = np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper


def sharpe_difference_test(
    returns_strategy: pd.Series,
    returns_benchmark: pd.Series,
    n_bootstrap: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Test whether strategy Sharpe is significantly different from benchmark.

    Uses paired bootstrap: resample same indices for both series
    to preserve correlation structure.

    Args:
        returns_strategy: Strategy daily returns
        returns_benchmark: Benchmark daily returns
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        (sharpe_diff, p_value)
        sharpe_diff = strategy_sharpe - benchmark_sharpe
        p_value < 0.05 means significantly different
    """
    common = returns_strategy.index.intersection(returns_benchmark.index)
    strat = returns_strategy.loc[common].values
    bench = returns_benchmark.loc[common].values
    n = len(common)

    rng = np.random.default_rng(random_state)

    point_diff = _sharpe(strat) - _sharpe(bench)

    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_boot = strat[idx]
        b_boot = bench[idx]
        bootstrap_diffs[i] = _sharpe(s_boot) - _sharpe(b_boot)

    # Two-sided p-value
    if point_diff >= 0:
        p_value = (bootstrap_diffs <= 0).mean()
    else:
        p_value = (bootstrap_diffs >= 0).mean()

    p_value = min(p_value * 2, 1.0)

    return point_diff, p_value


def _sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe from array."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)

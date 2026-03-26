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


def bonferroni_correction(p_values: dict, n_tests: int = None) -> dict:
    """
    Apply Bonferroni correction to multiple p-values.

    Args:
        p_values: Dict mapping strategy_name -> raw p-value
        n_tests: Number of tests (default: len(p_values))

    Returns:
        Dict mapping strategy_name -> corrected p-value
    """
    if n_tests is None:
        n_tests = len(p_values)
    return {k: min(v * n_tests, 1.0) for k, v in p_values.items()}


def hac_sharpe_se(
    returns: pd.Series,
    max_lag: int = None,
) -> Tuple[float, float, float, float]:
    """
    Newey-West HAC-adjusted standard error for Sharpe ratio.

    Accounts for autocorrelation in strategy returns, which inflates
    naive Sharpe standard errors.

    Args:
        returns: Daily return series
        max_lag: Maximum lag for HAC kernel (default: floor(4*(T/100)^(2/9)))

    Returns:
        (sharpe, hac_se, hac_ci_lower, hac_ci_upper) at 95% confidence
    """
    r = returns.values
    n = len(r)

    if max_lag is None:
        max_lag = int(4 * (n / 100) ** (2 / 9))

    mu = r.mean()
    sigma = r.std()
    sharpe_daily = mu / sigma if sigma > 0 else 0.0

    # Naive variance of Sharpe: (1 + 0.5*SR^2) / T
    naive_var = (1 + 0.5 * sharpe_daily ** 2) / n

    # HAC correction: add autocovariance terms
    gamma_0 = np.var(r)
    hac_sum = 0.0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.cov(r[lag:], r[:-lag])[0, 1]
        hac_sum += 2 * weight * gamma_lag / gamma_0

    hac_var = naive_var * (1 + hac_sum)
    hac_var = max(hac_var, 1e-10)  # floor to avoid sqrt of negative

    sharpe_annual = sharpe_daily * np.sqrt(TRADING_DAYS)
    hac_se = np.sqrt(hac_var) * np.sqrt(TRADING_DAYS)

    ci_lower = sharpe_annual - 1.96 * hac_se
    ci_upper = sharpe_annual + 1.96 * hac_se

    return sharpe_annual, hac_se, ci_lower, ci_upper


def _sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe from array."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)

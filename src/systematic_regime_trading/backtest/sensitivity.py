"""
Transaction cost sensitivity analysis.

Sweeps cost levels (0-20 bps) to find breakeven cost where
strategy Sharpe drops to 0 or equals benchmark.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

from systematic_regime_trading.backtest.engine import run_backtest
from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


def cost_sensitivity(
    market_returns: pd.Series,
    signal: pd.Series,
    cost_levels_bps: list = None,
    constraint_config: dict = None,
) -> pd.DataFrame:
    """
    Run backtest at multiple cost levels.

    Args:
        market_returns: Daily market returns
        signal: Allocation signal (already lagged)
        cost_levels_bps: List of one-way cost levels in bps
        constraint_config: Portfolio constraint config

    Returns:
        DataFrame with columns: cost_bps, sharpe, cagr, max_dd,
        annual_turnover, total_cost
    """
    if cost_levels_bps is None:
        cost_levels_bps = [0, 2, 5, 10, 15, 20]

    if constraint_config is None:
        constraint_config = load_config("backtest")["constraints"]

    results = []

    for cost_bps in cost_levels_bps:
        cost_config = {
            "cost_bps": cost_bps,
            "slippage_bps": 0,  # Already included in cost_bps for sweep
            "min_trade_bps": 0,
        }

        bt = run_backtest(
            market_returns, signal,
            cost_config=cost_config,
            constraint_config=constraint_config,
        )

        net_returns = bt["net_return"].dropna()
        n_years = len(net_returns) / 252

        # Annualized metrics
        if net_returns.std() > 0:
            sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        cum_ret = bt["cumulative_return"].iloc[-1]
        cagr = cum_ret ** (1 / max(n_years, 0.01)) - 1
        max_dd = bt["drawdown"].min()
        annual_turnover = bt["turnover"].mean() * 252
        total_cost = bt["cost"].sum()

        results.append({
            "cost_bps": cost_bps,
            "sharpe": sharpe,
            "cagr": cagr,
            "max_dd": max_dd,
            "annual_turnover": annual_turnover,
            "total_cost": total_cost,
            "cumulative_return": cum_ret,
        })

    return pd.DataFrame(results)


def find_breakeven_cost(
    sensitivity_df: pd.DataFrame,
    metric: str = "sharpe",
    threshold: float = 0.0,
) -> float:
    """
    Find the cost level where a metric drops below threshold.

    Uses linear interpolation between cost levels.

    Args:
        sensitivity_df: Output from cost_sensitivity()
        metric: Column name to check ('sharpe', 'cagr')
        threshold: Value below which the strategy "breaks"

    Returns:
        Breakeven cost in bps (float). Returns max cost if never breaks.
    """
    df = sensitivity_df.sort_values("cost_bps")

    for i in range(len(df) - 1):
        v_curr = df.iloc[i][metric]
        v_next = df.iloc[i + 1][metric]

        if v_curr >= threshold and v_next < threshold:
            # Linear interpolation
            c_curr = df.iloc[i]["cost_bps"]
            c_next = df.iloc[i + 1]["cost_bps"]
            slope = (v_next - v_curr) / (c_next - c_curr)
            breakeven = c_curr + (threshold - v_curr) / slope
            return breakeven

    # Never breaks even at max cost, or already broken at 0 cost
    if df.iloc[0][metric] < threshold:
        return 0.0
    return df["cost_bps"].max()


def run_all_sensitivity(
    market_returns: pd.Series,
    signals: pd.DataFrame,
    cost_levels_bps: list = None,
) -> dict:
    """
    Run cost sensitivity for all strategies.

    Args:
        market_returns: Daily market returns
        signals: DataFrame with strategy signals as columns
        cost_levels_bps: Cost levels to sweep

    Returns:
        Dict mapping strategy_name -> sensitivity DataFrame
    """
    results = {}

    for strategy_name in signals.columns:
        logger.info(f"Cost sensitivity: {strategy_name}")
        results[strategy_name] = cost_sensitivity(
            market_returns, signals[strategy_name],
            cost_levels_bps=cost_levels_bps,
        )

    return results

"""
Benchmark strategies for comparison.

Buy-and-hold, 60/40, and risk parity. Each produces a backtest
DataFrame matching the engine output format.
"""

import pandas as pd
import numpy as np
import logging

from systematic_regime_trading.backtest.costs import (
    compute_transaction_costs,
    compute_turnover,
)

logger = logging.getLogger(__name__)


def buy_and_hold(
    market_returns: pd.Series,
    cost_config: dict = None,
) -> pd.DataFrame:
    """
    100% equity, always. The simplest benchmark.

    Args:
        market_returns: Daily market returns
        cost_config: Transaction cost config (only initial trade cost applies)

    Returns:
        Backtest results DataFrame
    """
    position = pd.Series(1.0, index=market_returns.index)

    gross_return = market_returns.copy()
    turnover = compute_turnover(position)

    if cost_config is not None:
        costs = compute_transaction_costs(position, cost_config)
    else:
        costs = pd.Series(0.0, index=market_returns.index)

    net_return = gross_return - costs
    cumulative = (1 + net_return).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    return pd.DataFrame({
        "position": position,
        "gross_return": gross_return,
        "turnover": turnover,
        "cost": costs,
        "net_return": net_return,
        "cumulative_return": cumulative,
        "drawdown": drawdown,
    }, index=market_returns.index)


def sixty_forty(
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    cost_config: dict = None,
    rebalance: str = "monthly",
) -> pd.DataFrame:
    """
    Classic 60/40 portfolio, rebalanced monthly.

    Args:
        equity_returns: Daily equity market returns
        bond_returns: Daily bond (TLT) returns
        cost_config: Transaction cost config
        rebalance: Rebalance frequency ('daily', 'monthly')

    Returns:
        Backtest results DataFrame
    """
    # Align indices
    common_idx = equity_returns.index.intersection(bond_returns.index)
    eq_ret = equity_returns.loc[common_idx]
    bd_ret = bond_returns.loc[common_idx]

    if rebalance == "daily":
        # Fixed 60/40 weights every day
        portfolio_return = 0.60 * eq_ret + 0.40 * bd_ret
        position = pd.Series(0.60, index=common_idx)  # equity weight
    else:
        # Monthly rebalance: track drifting weights
        eq_weight = 0.60
        bd_weight = 0.40
        portfolio_returns = []
        positions = []

        for i, date in enumerate(common_idx):
            port_ret = eq_weight * eq_ret.iloc[i] + bd_weight * bd_ret.iloc[i]
            portfolio_returns.append(port_ret)
            positions.append(eq_weight)

            # Update weights based on returns (drift)
            eq_value = eq_weight * (1 + eq_ret.iloc[i])
            bd_value = bd_weight * (1 + bd_ret.iloc[i])
            total = eq_value + bd_value

            if total > 0:
                eq_weight = eq_value / total
                bd_weight = bd_value / total

            # Rebalance on first trading day of month
            if i + 1 < len(common_idx):
                next_date = common_idx[i + 1]
                if hasattr(next_date, 'month') and next_date.month != date.month:
                    eq_weight = 0.60
                    bd_weight = 0.40

        portfolio_return = pd.Series(portfolio_returns, index=common_idx)
        position = pd.Series(positions, index=common_idx)

    turnover = compute_turnover(position)

    if cost_config is not None:
        costs = compute_transaction_costs(position, cost_config)
    else:
        costs = pd.Series(0.0, index=common_idx)

    net_return = portfolio_return - costs
    cumulative = (1 + net_return).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    return pd.DataFrame({
        "position": position,
        "gross_return": portfolio_return,
        "turnover": turnover,
        "cost": costs,
        "net_return": net_return,
        "cumulative_return": cumulative,
        "drawdown": drawdown,
    }, index=common_idx)


def risk_parity(
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    cost_config: dict = None,
    lookback: int = 63,
) -> pd.DataFrame:
    """
    Inverse-vol weighted equity/bond allocation.

    Allocates more to the lower-vol asset so each contributes
    roughly equal risk to the portfolio.

    Args:
        equity_returns: Daily equity returns
        bond_returns: Daily bond returns
        cost_config: Transaction cost config
        lookback: Rolling window for vol estimation (default 63 = quarterly)

    Returns:
        Backtest results DataFrame
    """
    common_idx = equity_returns.index.intersection(bond_returns.index)
    eq_ret = equity_returns.loc[common_idx]
    bd_ret = bond_returns.loc[common_idx]

    # Rolling vol estimates
    eq_vol = eq_ret.rolling(lookback, min_periods=lookback).std()
    bd_vol = bd_ret.rolling(lookback, min_periods=lookback).std()

    # Inverse vol weights
    inv_eq = 1.0 / eq_vol.clip(lower=1e-8)
    inv_bd = 1.0 / bd_vol.clip(lower=1e-8)
    total_inv = inv_eq + inv_bd

    eq_weight = (inv_eq / total_inv).fillna(0.5)
    bd_weight = (inv_bd / total_inv).fillna(0.5)

    # Portfolio return
    portfolio_return = eq_weight * eq_ret + bd_weight * bd_ret

    # Fill warmup period with 50/50
    warmup_mask = eq_vol.isna()
    portfolio_return.loc[warmup_mask] = (
        0.5 * eq_ret.loc[warmup_mask] + 0.5 * bd_ret.loc[warmup_mask]
    )
    eq_weight.loc[warmup_mask] = 0.5

    turnover = compute_turnover(eq_weight)

    if cost_config is not None:
        costs = compute_transaction_costs(eq_weight, cost_config)
    else:
        costs = pd.Series(0.0, index=common_idx)

    net_return = portfolio_return - costs
    cumulative = (1 + net_return).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    return pd.DataFrame({
        "position": eq_weight,
        "gross_return": portfolio_return,
        "turnover": turnover,
        "cost": costs,
        "net_return": net_return,
        "cumulative_return": cumulative,
        "drawdown": drawdown,
    }, index=common_idx)

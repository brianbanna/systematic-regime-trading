"""
Vectorized backtesting engine.

Computes portfolio returns from allocation signals and market returns,
applying transaction costs. Intentionally simple: vectorized daily
computation, not an event-driven simulator.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

from systematic_regime_trading.utils.config import load_config
from systematic_regime_trading.backtest.costs import (
    compute_transaction_costs,
    compute_turnover,
)
from systematic_regime_trading.backtest.portfolio import construct_portfolio

logger = logging.getLogger(__name__)


def run_backtest(
    market_returns: pd.Series,
    allocation_signal: pd.Series,
    cost_config: dict = None,
    constraint_config: dict = None,
    apply_constraints: bool = True,
) -> pd.DataFrame:
    """
    Vectorized backtest.

    Steps:
    1. Position = allocation_signal (already lagged by signal pipeline)
    2. Gross return = position * market_return (per day)
    3. Position change = |position_t - position_{t-1}|
    4. Transaction cost = position_change * (cost_bps + slippage_bps) / 10000
    5. Net return = gross_return - transaction_cost
    6. Cumulative return = (1 + net_return).cumprod()
    7. Drawdown = cumulative / cumulative.cummax() - 1

    Args:
        market_returns: Daily market returns (index-level or single asset)
        allocation_signal: Daily allocation target [0, max_leverage]
            Should already have execution lag applied
        cost_config: Transaction cost config. If None, loads from backtest.yaml
        constraint_config: Portfolio constraint config. If None, loads from backtest.yaml
        apply_constraints: Whether to apply portfolio constraints

    Returns:
        DataFrame with columns:
            position, gross_return, turnover, cost, net_return,
            cumulative_return, drawdown
    """
    if cost_config is None:
        cost_config = load_config("backtest")["execution"]
    if constraint_config is None:
        constraint_config = load_config("backtest")["constraints"]

    # Align indices
    common_idx = market_returns.index.intersection(allocation_signal.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between returns and signal")

    returns = market_returns.loc[common_idx].copy()
    signal = allocation_signal.loc[common_idx].copy()

    # Drop leading NaNs from signal (execution lag)
    first_valid = signal.first_valid_index()
    if first_valid is not None:
        mask = signal.index >= first_valid
        returns = returns.loc[mask]
        signal = signal.loc[mask]

    # Fill any remaining NaN signals with 0 (no position)
    signal = signal.fillna(0.0)

    # Apply portfolio constraints
    if apply_constraints:
        position = construct_portfolio(signal, constraint_config)
    else:
        position = signal.copy()

    # Gross return
    gross_return = position * returns

    # Transaction costs
    turnover = compute_turnover(position)
    costs = compute_transaction_costs(position, cost_config)

    # Net return
    net_return = gross_return - costs

    # Cumulative return (wealth curve)
    cumulative_return = (1 + net_return).cumprod()

    # Drawdown
    running_max = cumulative_return.cummax()
    drawdown = cumulative_return / running_max - 1

    result = pd.DataFrame({
        "position": position,
        "gross_return": gross_return,
        "turnover": turnover,
        "cost": costs,
        "net_return": net_return,
        "cumulative_return": cumulative_return,
        "drawdown": drawdown,
    }, index=returns.index)

    logger.info(
        f"Backtest: {len(result)} days, "
        f"cum_return={cumulative_return.iloc[-1]:.4f}, "
        f"max_dd={drawdown.min():.4f}, "
        f"total_cost={costs.sum():.6f}"
    )

    return result


def run_all_backtests(
    signals: pd.DataFrame,
    market_returns: pd.Series,
    bond_returns: pd.Series = None,
    config: dict = None,
) -> dict:
    """
    Run backtest for each strategy + benchmarks.

    Args:
        signals: DataFrame with columns = strategy names, values = allocation signals
        market_returns: Daily market (equity index) returns
        bond_returns: Daily bond returns (for 60/40 and risk parity benchmarks)
        config: Full backtest config. If None, loads from backtest.yaml

    Returns:
        Dict mapping strategy_name -> backtest_results DataFrame
    """
    from systematic_regime_trading.backtest.benchmarks import (
        buy_and_hold,
        sixty_forty,
        risk_parity,
    )

    if config is None:
        config = load_config("backtest")

    cost_config = config["execution"]
    constraint_config = config["constraints"]

    results = {}

    # Run each strategy
    for strategy_name in signals.columns:
        logger.info(f"Backtesting: {strategy_name}")
        signal = signals[strategy_name]
        bt = run_backtest(
            market_returns, signal, cost_config, constraint_config,
        )
        results[strategy_name] = bt

    # Benchmarks
    logger.info("Backtesting: buy_and_hold")
    results["buy_and_hold"] = buy_and_hold(market_returns, cost_config)

    if bond_returns is not None:
        logger.info("Backtesting: sixty_forty")
        results["sixty_forty"] = sixty_forty(
            market_returns, bond_returns, cost_config,
        )

        logger.info("Backtesting: risk_parity")
        results["risk_parity"] = risk_parity(
            market_returns, bond_returns, cost_config,
        )

    return results

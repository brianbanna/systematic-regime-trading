"""
Transaction cost model.

Computes daily transaction costs from position changes using
a fixed + slippage cost model with minimum trade filter.
"""

import pandas as pd
import numpy as np
from typing import Optional

from systematic_regime_trading.utils.config import load_config


def compute_transaction_costs(
    positions: pd.Series,
    config: dict = None,
) -> pd.Series:
    """
    Compute daily transaction costs from position changes.

    Cost model:
    - Fixed cost: cost_bps per unit traded (one-way)
    - Slippage: slippage_bps per unit traded
    - Minimum trade filter: skip trades smaller than min_trade_bps

    Total one-way cost = (cost_bps + slippage_bps) / 10000
    Daily cost = |position_change| * total_one_way_cost

    Args:
        positions: Series of daily position sizes (0 to 1)
        config: Execution config dict. If None, loads from backtest.yaml

    Returns:
        Series of daily transaction costs (as return drag)
    """
    if config is None:
        config = load_config("backtest")["execution"]

    cost_bps = config.get("cost_bps", 5)
    slippage_bps = config.get("slippage_bps", 2)
    min_trade_bps = config.get("min_trade_bps", 1)

    total_cost_rate = (cost_bps + slippage_bps) / 10_000

    # Position changes (turnover)
    position_change = positions.diff().abs()
    position_change.iloc[0] = positions.iloc[0].copy()  # initial trade

    # Minimum trade filter: zero out tiny trades
    min_trade_threshold = min_trade_bps / 10_000
    position_change = position_change.where(
        position_change >= min_trade_threshold, 0.0,
    )

    # Cost model selection
    cost_model = config.get("cost_model", "linear")

    if cost_model == "sqrt_impact":
        # Square-root market impact: cost scales with sqrt(trade size)
        # More realistic for larger trades where market impact is convex
        impact_coeff = config.get("impact_coefficient", 0.1)
        linear_cost = position_change * total_cost_rate
        impact_cost = impact_coeff * np.sqrt(position_change) / 10_000
        costs = linear_cost + impact_cost
    else:
        costs = position_change * total_cost_rate

    return costs


def compute_turnover(positions: pd.Series) -> pd.Series:
    """
    Compute daily turnover (absolute position change).

    Args:
        positions: Series of daily position sizes

    Returns:
        Series of daily turnover values
    """
    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0].copy()
    return turnover


def annualized_turnover(positions: pd.Series) -> float:
    """
    Compute annualized turnover from position series.

    Returns:
        Annual turnover as a multiple (e.g., 5.0 = 500% annual)
    """
    daily_turnover = compute_turnover(positions)
    return daily_turnover.mean() * 252

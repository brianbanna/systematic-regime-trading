"""
Volatility targeting overlay.

Scales allocation so portfolio volatility targets a fixed level.
Uses only past data (rolling window) for vol estimate.
"""

import pandas as pd
import numpy as np
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


def apply_vol_target(
    allocation: pd.Series,
    returns: pd.Series,
    vol_target: float = None,
    lookback_days: int = None,
    max_leverage: float = None,
    config: dict = None,
) -> pd.Series:
    """
    Scale allocation so that portfolio volatility targets a fixed level.

    realized_vol = returns.rolling(lookback).std() * sqrt(252)
    vol_scalar = vol_target / realized_vol
    adjusted_allocation = allocation * vol_scalar

    Clipped to [0, max_leverage].

    Args:
        allocation: Base allocation signal
        returns: Market return series (decimal, same index as allocation)
        vol_target: Target annualized volatility (e.g., 0.10 for 10%)
        lookback_days: Rolling window for vol estimate
        max_leverage: Maximum allocation (default: 1.0)
        config: Strategy config

    Returns:
        Vol-targeted allocation signal
    """
    if config is None:
        strategy_cfg = load_config("strategy")
        backtest_cfg = load_config("backtest")
    else:
        strategy_cfg = config.get("strategy", load_config("strategy"))
        backtest_cfg = config.get("backtest", load_config("backtest"))

    vt_config = strategy_cfg["strategies"]["vol_targeted"]

    if vol_target is None:
        vol_target = vt_config["vol_target"]
    if lookback_days is None:
        lookback_days = vt_config["vol_lookback_days"]
    if max_leverage is None:
        max_leverage = backtest_cfg["constraints"]["max_leverage"]

    # Realized volatility using only past data (rolling, not expanding)
    realized_vol = returns.rolling(window=lookback_days, min_periods=lookback_days).std() * np.sqrt(252)

    # Vol scalar
    vol_scalar = vol_target / realized_vol.clip(lower=1e-6)

    # Adjusted allocation
    adjusted = allocation * vol_scalar

    # Apply constraints
    adjusted = adjusted.clip(lower=0.0, upper=max_leverage)

    logger.info(
        f"Vol targeting: target={vol_target:.1%}, "
        f"mean scalar={vol_scalar.dropna().mean():.2f}, "
        f"mean adjusted alloc={adjusted.dropna().mean():.2f}"
    )

    return adjusted

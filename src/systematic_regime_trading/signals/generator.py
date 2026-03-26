"""
Signal generation pipeline.

Produces allocation signals for all strategy variants defined in config.
Applies filters, vol-targeting, and execution lag.
"""

import pandas as pd
import numpy as np
import logging

from systematic_regime_trading.utils.config import load_config
from systematic_regime_trading.signals.regime_signal import regime_to_allocation
from systematic_regime_trading.signals.filters import (
    apply_confirmation_filter,
    apply_rate_limit,
    apply_execution_lag,
)
from systematic_regime_trading.signals.vol_target import apply_vol_target, apply_regime_vol_target

logger = logging.getLogger(__name__)


def generate_all_signals(
    regime_probs: pd.DataFrame,
    regime_labels: pd.Series,
    market_returns: pd.Series,
    strategy_config: dict = None,
    backtest_config: dict = None,
) -> pd.DataFrame:
    """
    Generate allocation signals for each strategy defined in config.
    Apply filters, vol-targeting, and execution lag.

    Args:
        regime_probs: DataFrame with prob_calm, prob_moderate, prob_turbulent
        regime_labels: Hard regime labels (0, 1, 2)
        market_returns: Daily market return series
        strategy_config: Strategy config (from strategy.yaml)
        backtest_config: Backtest config (from backtest.yaml)

    Returns:
        DataFrame with columns: date + one column per strategy variant.
        Each column is a daily allocation target [0, max_leverage].
        All signals have execution lag applied.
    """
    if strategy_config is None:
        strategy_config = load_config("strategy")
    if backtest_config is None:
        backtest_config = load_config("backtest")

    lag_days = backtest_config["execution"]["lag_days"]
    filter_config = strategy_config["signal_filters"]

    signals = {}

    for name in strategy_config["strategies"]:
        logger.info(f"Generating signal: {name}")

        # Step 1: Raw allocation from regime probabilities
        raw = regime_to_allocation(regime_probs, name, strategy_config)

        # Step 2: Apply confirmation filter
        filtered = apply_confirmation_filter(
            raw, regime_labels,
            confirmation_days=filter_config["confirmation_days"],
        )

        # Step 3: Apply rate limiter
        smoothed = apply_rate_limit(
            filtered,
            max_daily_change=filter_config["max_daily_allocation_change"],
        )

        # Step 4: Vol-targeting
        if name == "vol_targeted":
            smoothed = apply_vol_target(
                smoothed, market_returns,
                config={"strategy": strategy_config, "backtest": backtest_config},
            )
        elif name == "regime_vol_targeted":
            rvt_cfg = strategy_config["strategies"]["regime_vol_targeted"]
            vol_targets = {int(k): v for k, v in rvt_cfg.get("vol_targets", {}).items()}
            if not vol_targets:
                vol_targets = {0: 0.12, 1: 0.08, 2: 0.04}
            smoothed = apply_regime_vol_target(
                smoothed, market_returns, regime_labels,
                vol_targets=vol_targets,
                lookback_days=rvt_cfg.get("vol_lookback_days", 63),
                max_leverage=backtest_config["constraints"]["max_leverage"],
            )

        # Step 5: Apply execution lag
        lagged = apply_execution_lag(smoothed, lag_days=lag_days)

        # Step 6: Clip to constraints
        max_leverage = backtest_config["constraints"]["max_leverage"]
        min_alloc = backtest_config["constraints"]["min_allocation"]
        lagged = lagged.clip(lower=min_alloc, upper=max_leverage)

        signals[name] = lagged

    result = pd.DataFrame(signals, index=regime_probs.index)

    logger.info(
        f"Generated {len(signals)} strategy signals, "
        f"{result.notna().sum().min()} valid observations (after lag)"
    )

    return result

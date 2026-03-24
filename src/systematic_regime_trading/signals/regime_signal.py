"""
Regime signal module: convert regime probabilities to allocation targets.

Each strategy variant in configs/strategy.yaml produces a different
allocation series from the same regime probability inputs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


def regime_to_allocation(
    regime_probs: pd.DataFrame,
    strategy_name: str,
    config: dict = None,
) -> pd.Series:
    """
    Convert regime probabilities to equity allocation target [0, max_leverage].

    Args:
        regime_probs: DataFrame with columns containing 'calm', 'moderate',
                      'turbulent' probability columns and DatetimeIndex
        strategy_name: Key from strategy.yaml (e.g., 'binary_regime')
        config: Strategy config. If None, loads from strategy.yaml

    Returns:
        Series of daily allocation targets
    """
    if config is None:
        config = load_config("strategy")

    strategy = config["strategies"][strategy_name]

    # Normalize column names
    prob_cols = _get_prob_columns(regime_probs)
    p_calm = regime_probs[prob_cols["calm"]]
    p_moderate = regime_probs[prob_cols["moderate"]]
    p_turbulent = regime_probs[prob_cols["turbulent"]]

    if strategy_name == "binary_regime":
        alloc = strategy["allocation"]
        allocation = (
            p_calm * alloc["calm"]
            + p_moderate * alloc["moderate"]
            + p_turbulent * alloc["turbulent"]
        )

    elif strategy_name == "proportional_regime":
        allocation = p_calm

    elif strategy_name == "vol_targeted":
        # Base allocation from proportional, vol-targeting applied later
        allocation = p_calm

    elif strategy_name == "regime_momentum":
        allocation = _compute_momentum_allocation(
            regime_probs, prob_cols, strategy,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    allocation.name = strategy_name
    return allocation


def regime_labels_to_allocation(
    regime_labels: pd.Series,
    allocation_map: dict,
) -> pd.Series:
    """
    Convert hard regime labels to allocations using a mapping.

    Args:
        regime_labels: Series of regime labels (0, 1, 2)
        allocation_map: Dict mapping label -> allocation

    Returns:
        Series of allocation targets
    """
    return regime_labels.map(allocation_map)


def _compute_momentum_allocation(
    regime_probs: pd.DataFrame,
    prob_cols: dict,
    strategy: dict,
) -> pd.Series:
    """Overweight when calm regime is persistent."""
    threshold = strategy.get("persistence_threshold_days", 10)
    alloc_config = strategy["allocation"]

    probs_array = regime_probs[
        [prob_cols["calm"], prob_cols["moderate"], prob_cols["turbulent"]]
    ].values
    dominant = np.argmax(probs_array, axis=1)

    n = len(dominant)
    consecutive = np.ones(n, dtype=int)
    for t in range(1, n):
        if dominant[t] == dominant[t - 1]:
            consecutive[t] = consecutive[t - 1] + 1

    allocation = np.zeros(n)
    for t in range(n):
        if dominant[t] == 2:
            allocation[t] = alloc_config["turbulent"]
        elif dominant[t] == 1:
            allocation[t] = alloc_config["moderate"]
        elif dominant[t] == 0:
            if consecutive[t] >= threshold:
                allocation[t] = alloc_config["calm_persistent"]
            else:
                allocation[t] = alloc_config["calm_new"]

    return pd.Series(allocation, index=regime_probs.index, name="regime_momentum")


def _get_prob_columns(df: pd.DataFrame) -> dict:
    """Identify probability column names in DataFrame."""
    col_map = {"calm": None, "moderate": None, "turbulent": None}

    for col in df.columns:
        col_lower = col.lower()
        if "calm" in col_lower:
            col_map["calm"] = col
        elif "moderate" in col_lower:
            col_map["moderate"] = col
        elif "turbulent" in col_lower:
            col_map["turbulent"] = col

    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(
            f"Cannot find probability columns for: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return col_map

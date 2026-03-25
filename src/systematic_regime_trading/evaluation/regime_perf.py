"""
Regime-conditional performance analysis.

Computes metrics separately for each regime to answer:
"Where does the alpha come from?"
"""

import pandas as pd
import numpy as np

from systematic_regime_trading.evaluation.metrics import (
    compute_sharpe,
    compute_max_drawdown,
)

TRADING_DAYS = 252


def performance_by_regime(
    returns: pd.Series,
    regime_labels: pd.Series,
    regime_names: dict = None,
) -> pd.DataFrame:
    """
    Compute performance metrics separately for each regime.

    Args:
        returns: Daily return series
        regime_labels: Series of regime labels (0, 1, 2) aligned with returns
        regime_names: Optional mapping {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    Returns:
        DataFrame indexed by regime with columns:
            mean_daily_return, annual_return, vol, sharpe, max_dd,
            pct_days, contribution, n_days
    """
    if regime_names is None:
        regime_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    common = returns.index.intersection(regime_labels.index)
    ret = returns.loc[common]
    labels = regime_labels.loc[common]

    total_cum_return = (1 + ret).prod() - 1
    results = []

    for regime_id in sorted(labels.unique()):
        mask = labels == regime_id
        regime_ret = ret[mask]

        n_days = len(regime_ret)
        pct_days = n_days / len(ret)

        if n_days < 2:
            results.append({
                "regime": regime_names.get(regime_id, f"Regime {regime_id}"),
                "regime_id": regime_id,
                "mean_daily_return": 0.0,
                "annual_return": 0.0,
                "vol": 0.0,
                "sharpe": 0.0,
                "max_dd": 0.0,
                "pct_days": pct_days,
                "contribution": 0.0,
                "n_days": n_days,
            })
            continue

        mean_ret = regime_ret.mean()
        annual_ret = mean_ret * TRADING_DAYS
        vol = regime_ret.std() * np.sqrt(TRADING_DAYS)
        sharpe = compute_sharpe(regime_ret) if vol > 0 else 0.0
        max_dd = compute_max_drawdown(regime_ret)[0]

        regime_cum = (1 + regime_ret).prod() - 1
        contribution = regime_cum / total_cum_return if total_cum_return != 0 else 0

        results.append({
            "regime": regime_names.get(regime_id, f"Regime {regime_id}"),
            "regime_id": regime_id,
            "mean_daily_return": mean_ret,
            "annual_return": annual_ret,
            "vol": vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "pct_days": pct_days,
            "contribution": contribution,
            "n_days": n_days,
        })

    return pd.DataFrame(results).set_index("regime")


def crisis_performance(
    returns: pd.Series,
    crisis_periods: dict,
) -> pd.DataFrame:
    """
    Compute performance during defined crisis periods.

    Args:
        returns: Daily return series
        crisis_periods: Dict mapping name -> {"start": str, "end": str}

    Returns:
        DataFrame with crisis name, return, max_dd, vol for each period
    """
    results = []

    for name, period in crisis_periods.items():
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])

        mask = (returns.index >= start) & (returns.index <= end)
        crisis_ret = returns[mask]

        if len(crisis_ret) < 2:
            continue

        cum_ret = (1 + crisis_ret).prod() - 1
        max_dd = compute_max_drawdown(crisis_ret)[0]
        vol = crisis_ret.std() * np.sqrt(TRADING_DAYS)

        results.append({
            "period": name,
            "start": start,
            "end": end,
            "n_days": len(crisis_ret),
            "cumulative_return": cum_ret,
            "max_drawdown": max_dd,
            "annualized_vol": vol,
        })

    return pd.DataFrame(results).set_index("period")

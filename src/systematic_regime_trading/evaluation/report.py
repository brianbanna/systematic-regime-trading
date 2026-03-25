"""
Performance comparison table and report generation.

The performance table is the single most important output of the project.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from systematic_regime_trading.evaluation.metrics import compute_metrics
from systematic_regime_trading.evaluation.significance import bootstrap_sharpe_ci
from systematic_regime_trading.utils.config import get_path

logger = logging.getLogger(__name__)


def performance_table(
    backtest_results: dict,
    risk_free_rate: float = 0.02,
    include_ci: bool = True,
    n_bootstrap: int = 10000,
) -> pd.DataFrame:
    """
    Master comparison table across all strategies and benchmarks.

    Args:
        backtest_results: Dict mapping strategy_name -> backtest DataFrame
        risk_free_rate: Annualized risk-free rate
        include_ci: Whether to compute bootstrap Sharpe CIs
        n_bootstrap: Number of bootstrap samples for CI

    Returns:
        DataFrame with one row per strategy, sorted by Sharpe descending
    """
    rows = []

    for name, bt in backtest_results.items():
        returns = bt["net_return"]
        metrics = compute_metrics(
            returns,
            risk_free_rate=risk_free_rate,
            positions=bt.get("position"),
        )

        row = {
            "Strategy": name,
            "CAGR": metrics["cagr"],
            "Annual_Vol": metrics["annual_vol"],
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "Calmar": metrics["calmar"],
            "Max_DD": metrics["max_drawdown"],
            "Max_DD_Duration": metrics["max_drawdown_duration"],
            "Skewness": metrics["skewness"],
            "Kurtosis": metrics["kurtosis"],
            "Hit_Rate_Monthly": metrics.get("hit_rate_monthly", np.nan),
            "Profit_Factor": metrics["profit_factor"],
            "Turnover": metrics.get("turnover_annual", np.nan),
        }

        if include_ci:
            try:
                _, ci_low, ci_high = bootstrap_sharpe_ci(
                    returns, n_bootstrap=n_bootstrap,
                )
                row["Sharpe_CI_Lower"] = ci_low
                row["Sharpe_CI_Upper"] = ci_high
            except Exception:
                row["Sharpe_CI_Lower"] = np.nan
                row["Sharpe_CI_Upper"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")
    df = df.sort_values("Sharpe", ascending=False)

    return df


def save_performance_table(
    table: pd.DataFrame,
    output_dir: str = "results",
) -> Path:
    """Save performance table to CSV."""
    output_path = get_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "performance_table.csv"
    table.to_csv(csv_path, float_format="%.4f")
    logger.info(f"Saved performance table to {csv_path}")

    return csv_path


def format_performance_table(table: pd.DataFrame) -> str:
    """Format table for console/report display."""
    formatters = {
        "CAGR": "{:.2%}".format,
        "Annual_Vol": "{:.2%}".format,
        "Sharpe": "{:.3f}".format,
        "Sortino": "{:.3f}".format,
        "Calmar": "{:.3f}".format,
        "Max_DD": "{:.2%}".format,
        "Max_DD_Duration": "{:.0f}".format,
        "Hit_Rate_Monthly": "{:.1%}".format,
        "Profit_Factor": "{:.2f}".format,
        "Turnover": "{:.2f}".format,
        "Sharpe_CI_Lower": "{:.3f}".format,
        "Sharpe_CI_Upper": "{:.3f}".format,
    }

    formatted = table.copy()
    for col, fmt in formatters.items():
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: fmt(x) if pd.notna(x) and np.isfinite(x) else "N/A"
            )

    return formatted.to_string()

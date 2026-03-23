"""
Performance visualization functions.

Cumulative returns, drawdowns, monthly heatmaps, rolling metrics,
and volatility charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


def plot_price_time_series(
    series: pd.Series,
    title: str = "Price Time Series",
    ylabel: str = "Price",
    figsize: tuple = (15, 6),
    save_path: str = None,
):
    """
    Plot a price or index time series.

    Args:
        series: Time series with DatetimeIndex
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_return_distribution(
    returns: pd.Series,
    title: str = "Return Distribution",
    bins: int = 100,
    figsize: tuple = (10, 6),
    save_path: str = None,
):
    """
    Plot histogram of returns with density curve.

    Args:
        returns: Series of returns
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(returns.dropna(), bins=bins, density=True, alpha=0.7, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_volatility_time_series(
    vol_series: pd.Series,
    title: str = "Market Volatility",
    figsize: tuple = (15, 6),
    save_path: str = None,
):
    """
    Plot volatility over time.

    Args:
        vol_series: Volatility time series
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(vol_series.index, vol_series.values, linewidth=0.8, color="darkred")
    ax.set_title(title)
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_volatility_comparison(
    vol_dict: Dict[str, pd.Series],
    title: str = "Volatility Comparison",
    figsize: tuple = (15, 6),
    save_path: str = None,
):
    """
    Compare multiple volatility series on the same plot.

    Args:
        vol_dict: Dict mapping series name -> volatility Series
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, series in vol_dict.items():
        ax.plot(series.index, series.values, linewidth=0.8, label=name, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_smoothed_volatility(
    smoothed_df: pd.DataFrame,
    title: str = "Volatility with Rolling Averages",
    figsize: tuple = (15, 6),
    save_path: str = None,
):
    """
    Plot raw volatility with smoothed rolling averages.

    Args:
        smoothed_df: DataFrame with 'market_volatility' and rolling avg columns
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        smoothed_df.index, smoothed_df["market_volatility"],
        alpha=0.3, linewidth=0.5, label="Raw", color="gray",
    )

    ma_cols = [c for c in smoothed_df.columns if c.startswith("volatility_ma_")]
    for col in ma_cols:
        ax.plot(smoothed_df.index, smoothed_df[col], linewidth=1, label=col, alpha=0.8)

    ax.set_title(title)
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

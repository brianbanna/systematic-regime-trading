"""
Diagnostic visualization functions.

ACF plots, model residuals, QQ plots, feature importance,
and regime validation charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def plot_acf(
    series: pd.Series,
    lags: int = 40,
    title: str = "Autocorrelation Function",
    figsize: tuple = (12, 5),
    save_path: str = None,
):
    """
    Plot autocorrelation function.

    Args:
        series: Time series to analyze
        lags: Number of lags
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    acf_values = [series.autocorr(lag=i) for i in range(1, lags + 1)]
    ax.bar(range(1, lags + 1), acf_values, color="steelblue", alpha=0.7)

    n = len(series)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(y=ci, linestyle="--", color="red", alpha=0.5, label="95% CI")
    ax.axhline(y=-ci, linestyle="--", color="red", alpha=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    feature_importance: List[tuple],
    title: str = "Feature Importance for Clustering",
    top_n: int = 5,
    figsize: tuple = (10, 6),
    save_path: str = None,
):
    """
    Plot feature importance from PCA-based clustering.

    Args:
        feature_importance: List of (feature_name, importance) tuples
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    top_n = min(top_n, len(feature_importance))
    features = feature_importance[:top_n]
    names = [f[0] for f in features]
    values = [f[1] for f in features]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(top_n), values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Relative Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom",
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_regime_boxplots(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature: str,
    regime_names: Dict = None,
    title: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
):
    """
    Box plots of a feature by regime.

    Args:
        df: DataFrame containing the feature
        labels: Regime labels
        feature: Feature column name
        regime_names: Dict mapping label -> name
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if regime_names is None:
        regime_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}
    if title is None:
        title = f"{feature.replace('_', ' ').title()} by Regime"

    plot_df = df[[feature]].copy()
    plot_df["Regime"] = [regime_names.get(l, str(l)) for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=plot_df, x="Regime", y=feature, ax=ax, palette="Set2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_drawdown_validation(
    drawdown: pd.Series,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    regime_names: Dict = None,
    title: str = "Drawdown vs Regime Predictions",
    figsize: tuple = (15, 8),
    save_path: str = None,
):
    """
    Plot drawdown with regime predictions overlaid.

    Args:
        drawdown: Drawdown series (negative values)
        labels: Regime predictions
        dates: DatetimeIndex
        regime_names: Dict mapping label -> name
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if regime_names is None:
        regime_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.fill_between(dates, drawdown, 0, alpha=0.5, color="red")
    ax1.set_ylabel("Drawdown")
    ax1.set_title("Market Drawdown")
    ax1.grid(True, alpha=0.3)

    for regime_id in sorted(np.unique(labels)):
        mask = labels == regime_id
        ax2.scatter(
            dates[mask], np.ones(mask.sum()) * regime_id,
            c=colors.get(regime_id, "gray"),
            label=regime_names.get(regime_id, str(regime_id)),
            alpha=0.6, s=10,
        )

    ax2.set_ylabel("Regime")
    ax2.set_title("Regime Predictions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

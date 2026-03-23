"""
Regime visualization functions.

Timeline plots, transition matrices, regime distributions,
crisis period overlays, and regime comparison charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


# ---------------------------------------------------------------------------
# Regime timeline
# ---------------------------------------------------------------------------

def plot_regime_timeline(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature: str = "market_volatility",
    label_names: Dict = None,
    title: str = "Market Volatility Over Time by Regime",
    figsize: tuple = (15, 6),
    save_path: str = None,
):
    """
    Plot feature over time colored by regime.

    Args:
        df: DataFrame with DatetimeIndex
        labels: Regime labels for each data point
        feature: Feature column to plot on y-axis
        label_names: Dict mapping label -> name (e.g., {0: 'Calm'})
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if label_names is None:
        label_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}

    fig, ax = plt.subplots(figsize=figsize)

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(
            df.index[mask],
            df[feature][mask],
            c=colors.get(cluster_id, "gray"),
            label=label_names.get(cluster_id, f"Cluster {cluster_id}"),
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_regime_shaded_timeseries(
    series: pd.Series,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    label_names: Dict = None,
    title: str = "Time Series with Regime Shading",
    figsize: tuple = (15, 6),
    save_path: str = None,
):
    """
    Plot time series with background shaded by regime.

    Args:
        series: Time series to plot
        labels: Regime labels
        dates: DatetimeIndex for x-axis
        label_names: Dict mapping label -> name
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if label_names is None:
        label_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, series, color="black", linewidth=0.8, alpha=0.8)

    prev_label = labels[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != prev_label or i == len(labels) - 1:
            end_idx = i if labels[i] != prev_label else i + 1
            ax.axvspan(
                dates[start_idx], dates[min(end_idx, len(dates) - 1)],
                alpha=0.2, color=colors.get(prev_label, "gray"),
            )
            start_idx = i
            prev_label = labels[i]

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors.get(k, "gray"), alpha=0.3, label=v)
        for k, v in label_names.items()
    ]
    ax.legend(handles=legend_elements)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def plot_transition_matrix_heatmap(
    labels: np.ndarray,
    regime_names: Dict,
    title: str = "Regime Transition Matrix",
    figsize: tuple = (8, 6),
    save_path: str = None,
) -> pd.DataFrame:
    """
    Compute and plot regime transition probability matrix.

    Args:
        labels: Array of regime labels
        regime_names: Dict mapping label -> name
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Transition probability DataFrame
    """
    transitions = pd.DataFrame({
        "from_regime": labels[:-1],
        "to_regime": labels[1:],
    })

    transition_matrix = pd.crosstab(
        transitions["from_regime"],
        transitions["to_regime"],
        normalize="index",
    )

    transition_matrix.index = transition_matrix.index.map(regime_names)
    transition_matrix.columns = transition_matrix.columns.map(regime_names)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        transition_matrix, annot=True, cmap="Reds", fmt=".2f",
        vmin=0, vmax=1, cbar_kws={"label": "Transition Probability"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return transition_matrix


# ---------------------------------------------------------------------------
# Regime distributions
# ---------------------------------------------------------------------------

def plot_regime_distribution_bar(
    labels: np.ndarray,
    regime_names: Dict = None,
    title: str = "Regime Distribution",
    figsize: tuple = (8, 5),
    save_path: str = None,
):
    """
    Bar chart of regime frequencies.

    Args:
        labels: Array of regime labels
        regime_names: Dict mapping label -> name
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if regime_names is None:
        regime_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}

    counts = pd.Series(labels).value_counts().sort_index()
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [regime_names.get(k, str(k)) for k in counts.index],
        pcts.values,
        color=[colors.get(k, "gray") for k in counts.index],
        alpha=0.8,
    )

    for bar, pct in zip(bars, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{pct:.1f}%", ha="center", va="bottom",
        )

    ax.set_ylabel("Percentage of Time (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Crisis period overlay
# ---------------------------------------------------------------------------

def plot_crisis_period(
    series: pd.Series,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    crisis_start: str,
    crisis_end: str,
    label_names: Dict = None,
    title: str = "Crisis Period",
    figsize: tuple = (12, 5),
    save_path: str = None,
):
    """
    Zoom into a crisis period with regime coloring.

    Args:
        series: Time series to plot
        labels: Regime labels
        dates: DatetimeIndex
        crisis_start: Start date string
        crisis_end: End date string
        label_names: Dict mapping label -> name
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if label_names is None:
        label_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}

    mask = (dates >= pd.to_datetime(crisis_start)) & (
        dates <= pd.to_datetime(crisis_end)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates[mask], series[mask], color="black", linewidth=1, alpha=0.8)

    crisis_labels = labels[mask]
    crisis_dates = dates[mask]

    for cluster_id in np.unique(crisis_labels):
        cmask = crisis_labels == cluster_id
        ax.scatter(
            crisis_dates[cmask], series[mask][cmask],
            c=colors.get(cluster_id, "gray"),
            label=label_names.get(cluster_id, f"State {cluster_id}"),
            alpha=0.7, s=40,
        )

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Regime comparison
# ---------------------------------------------------------------------------

def plot_regime_distributions_comparison(
    labels_dict: Dict[str, np.ndarray],
    regime_names: Dict = None,
    figsize: tuple = (14, 5),
    save_path: str = None,
):
    """
    Side-by-side bar charts comparing regime distributions across methods.

    Args:
        labels_dict: Dict mapping method name -> labels array
        regime_names: Dict mapping label -> name
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if regime_names is None:
        regime_names = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    n_methods = len(labels_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)
    if n_methods == 1:
        axes = [axes]

    colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}

    for ax, (method_name, labels) in zip(axes, labels_dict.items()):
        counts = pd.Series(labels).value_counts().sort_index()
        pcts = counts / counts.sum() * 100

        ax.bar(
            [regime_names.get(k, str(k)) for k in counts.index],
            pcts.values,
            color=[colors.get(k, "gray") for k in counts.index],
            alpha=0.8,
        )
        ax.set_title(method_name)
        ax.set_ylabel("%" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Regime Distribution Comparison")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

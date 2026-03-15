"""
Plotting utilities: timelines, histograms, heatmaps, and regime comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure plot styling for consistency across all notebooks
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 16


def plot_price_time_series(
    df: pd.DataFrame, ticker: str, price_col="Adj Close", figsize=(14, 6), dpi=300
):
    """Plot price over time as a line chart."""
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(df["Date"], df[price_col], color="blue", label=f"{ticker} {price_col}")
    plt.title(f"{ticker} {price_col} Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_price_distribution(
    df: pd.DataFrame, price_col="Adj Close", figsize=(12, 5), dpi=300
):
    """Plot histogram showing how prices are distributed."""
    plt.figure(figsize=figsize, dpi=dpi)
    sns.histplot(df[price_col], bins=100, kde=True, color="blue")
    plt.title(f"{price_col} Distribution")
    plt.xlabel(price_col)
    plt.ylabel("Count")
    plt.show()


def plot_return_distribution(
    df: pd.DataFrame, return_col="log_return", figsize=(8, 6), dpi=300
):
    """Plot histogram showing how returns are distributed."""
    plt.figure(figsize=figsize, dpi=dpi)
    sns.histplot(df[return_col].dropna(), bins=100, kde=True, color="green")
    plt.title(f'Daily {return_col.replace("_", " ").title()} Distribution')
    plt.xlabel(return_col.replace("_", " ").title())
    plt.ylabel("Count")
    plt.show()


def plot_return_time_series(
    df: pd.DataFrame, return_col="log_return", figsize=(14, 6), dpi=300
):
    """Plot time series of returns."""
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(
        df["Date"],
        df[return_col],
        color="green",
        label=f'Daily {return_col.replace("_", " ").title()}',
    )
    plt.title(f'Daily {return_col.replace("_", " ").title()} Time Series')
    plt.xlabel("Date")
    plt.ylabel(return_col.replace("_", " ").title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_volatility_time_series(volatility_df: pd.DataFrame, figsize=(12, 6), dpi=300):
    """Plot market volatility over time."""
    plt.figure(figsize=figsize, dpi=dpi)

    # Handle both Date as column or index
    if "Date" in volatility_df.columns:
        dates = volatility_df["Date"]
    else:
        dates = volatility_df.index

    plt.plot(dates, volatility_df["market_volatility"], color="red")
    plt.title("Daily Market Volatility Index")
    plt.xlabel("Date")
    plt.ylabel("Volatility (std of log returns)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_smoothed_volatility(smoothed_df: pd.DataFrame, figsize=(12, 6), dpi=300):
    """
    Plot raw market volatility together with smoothed rolling averages.

    Args:
        smoothed_df: DataFrame with 'market_volatility' and rolling average columns.
        figsize: Figure size for the plot.
    """
    plt.figure(figsize=figsize, dpi=dpi)

    if "Date" in smoothed_df.columns:
        dates = smoothed_df["Date"]
    else:
        dates = smoothed_df.index

    plt.plot(
        dates,
        smoothed_df["market_volatility"],
        color="black",
        linewidth=1.5,
        label="Raw volatility",
    )

    for window in [5, 10, 20, 30]:
        col_name = f"volatility_ma_{window}d"
        if col_name in smoothed_df.columns:
            plt.plot(dates, smoothed_df[col_name], label=f"{window}-day MA")

    plt.title("Market Volatility: Raw vs Smoothed")
    plt.xlabel("Date")
    plt.ylabel("Market Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_market(
    df,
    feature: str,
    smoothing_windows=[5, 10, 20, 30],
    start_date=None,
    end_date=None,
    dpi=300,
):
    """
    Plot a market feature with moving averages.

    Args:
        df: DataFrame with DatetimeIndex and the feature column.
        feature: Name of the column to plot.
        smoothing_windows: Window sizes for moving averages. Defaults to [3, 5, 10].
        start_date: Start date (e.g., '2020-01-01'). If None, uses start of data.
        end_date: End date (e.g., '2021-12-31'). If None, uses end of data.
    """
    plot_data = df.loc[start_date:end_date, [feature]].copy()

    # create a variable without underscore to display properly
    display_name = feature.replace("_", " ")

    for window in smoothing_windows:
        col_name = f"{window}-Day MA"
        plot_data[col_name] = plot_data[feature].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(15, 7), dpi=dpi)

    ax.plot(
        plot_data.index,
        plot_data[feature],
        color="grey",
        alpha=0.4,
        label="Daily " + display_name,
    )

    colors = ["#26a69a", "#ef5350", "#ff9800", "purple"]
    for i, window in enumerate(smoothing_windows):
        col_name = f"{window}-Day MA"
        ax.plot(
            plot_data.index,
            plot_data[col_name],
            color=colors[i % len(colors)],
            linewidth=2,
            label=col_name,
        )

    ax.set_title(f"{display_name} Index", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"{display_name}", fontsize=12)
    ax.legend()

    plt.show()


def plot_boxplot(df: pd.DataFrame, col_name="Adj Close", figsize=(8, 6), dpi=300):
    """Plot boxplot showing price distribution and outliers."""

    # Create a clean name for display
    display_name = col_name.replace("_", " ")

    plt.figure(figsize=figsize, dpi=dpi)
    sns.boxplot(y=df[col_name], color="#26a69a")
    plt.title(f"{display_name} Distribution")
    plt.ylabel(f"{display_name} Value")
    plt.show()


def plot_volatility_comparison(
    df: pd.DataFrame,
    vol_cols=("vol_5d", "vol_10d", "vol_20d", "vol_30d"),
    figsize=(12, 6),
    dpi=300,
):
    """
    Plots volatility distribution with a trading aesthetic.
    Features:
    - Trading Palette for the boxes.
    - RED markers for outliers (highlighting tail risk).
    """
    # 1. Prepare Data
    plot_df = df[list(vol_cols)].melt(var_name="Window", value_name="Volatility")

    # Clean up the names for display: remove underscores
    plot_df["Window"] = plot_df["Window"].str.replace("_", " ")

    # 2. Define Trading Paletterolli
    # Green (5d), Red (10d), Purple (20d), Orange (30d)
    trading_palette = ["#26a69a", "#ef5350", "#ab47bc", "#ff9800"]

    # 3. Define Outlier Style (Tail Risk = Red)
    flier_props = dict(
        marker="o", markersize=4, linestyle="none", markeredgecolor="#ef5350", alpha=0.6
    )

    plt.figure(figsize=figsize, dpi=dpi)

    # 4. Plot Boxplot
    ax = sns.boxplot(
        x="Window",
        y="Volatility",
        data=plot_df,
        hue="Window",
        palette=trading_palette,
        flierprops=flier_props,  # Apply the Red outlier style
        linewidth=1.5,
    )

    # Manually delete legend if generated by 'hue'
    if ax.legend_:
        ax.legend_.remove()

    # 5. Styling
    plt.title(
        "Volatility Distribution (Tail Risk in Red)", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Volatility (Standardized)", fontsize=10)
    plt.xlabel("Lookback Window", fontsize=10)

    # Grid configuration (Horizontal only, subtle)
    plt.grid(axis="y", color="#b0b0b0", linestyle="--", linewidth=0.5, alpha=0.5)

    # Remove top and right spines for a cleaner look
    sns.despine()

    plt.show()


def display_summary_stats(df: pd.DataFrame, ticker: str, col="Adj Close"):
    """Print summary statistics for a column (mean, std, skewness, kurtosis)."""
    print(f"--- {ticker} {col} ---")
    print("\nStatistics:")
    print(df[col].describe())
    print("\nSkewness:", df[col].skew())
    print("Kurtosis:", df[col].kurtosis())


def plot_full_period_with_events(
    df: pd.DataFrame, events: dict, figsize=(14, 6), dpi=300
):
    """
    Plot standardized indicators over the full period with red shaded areas for known events.

    Args:
        df: DataFrame with standardized indicators as columns (e.g., features_for_kmeans).
        events: Dictionary with event name as key and tuple(start_date, end_date) as value.
        figsize: Figure size for plot.
    """
    selected_cols = ["market_volatility", "market_correlation", "market_atr"]
    existing_cols = [col for col in selected_cols if col in df.columns]
    df_subset = df[existing_cols]

    plt.figure(figsize=figsize, dpi=dpi)
    for col in df_subset.columns:
        plt.plot(df_subset.index, df_subset[col], label=col)

    # Highlight all events
    for i, (event_name, (start, end)) in enumerate(events.items()):
        label = "Event Periods" if i == 0 else None
        plt.axvspan(
            pd.to_datetime(start),
            pd.to_datetime(end),
            color="red",
            alpha=0.1,
            label=label,
        )

    plt.title("Mood Indicators with Event Periods Highlighted")
    plt.xlabel("Date")
    plt.ylabel("Standardized Indicator (z-score)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_rolling_with_events(
    df: pd.DataFrame,
    events: dict,
    selected_cols: list,
    window: int = 20,
    figsize=(15, 7),
    dpi=300,
):
    """
    Plots rolling averages with a 'Trading Board' aesthetic.
    """
    # 1. Validation
    existing_cols = [col for col in selected_cols if col in df.columns]
    if not existing_cols:
        print(f"Warning: None of the specified columns {selected_cols} were found.")
        return

    # 2. Data Preparation
    df_subset = df[existing_cols]
    df_rolling = df_subset.rolling(window=window, min_periods=1).mean()

    # 3. Define Trading Palette (to cycle through for lines)
    # Order: Green (Primary), Blue (Secondary), Orange (Warning), Purple (Alt)
    trading_colors = ["#26a69a", "#2962ff", "#ff9800", "#9c27b0", "#546e7a"]

    # Crisis/Event Color
    c_crisis = "#ef5350"  # Trading Red

    # 4. Setup Plot
    plt.figure(figsize=figsize, dpi=dpi)

    # 5. Plot Lines
    for i, col in enumerate(df_rolling.columns):
        # Pick color from our palette (cycle if there are many lines)
        line_color = trading_colors[i % len(trading_colors)]

        # Clean up column name for display (remove underscores)
        display_col = col.replace("_", " ")

        plt.plot(
            df_rolling.index,
            df_rolling[col],
            label=f"{display_col} (SMA {window})",
            color=line_color,
            linewidth=2,  # Slightly thicker for readability
            alpha=0.9,
        )

    # 6. Highlight Events (Crisis Zones)
    for i, (event_name, (start, end)) in enumerate(events.items()):
        # Only label the first event to keep legend clean
        label = "Crisis / Event" if i == 0 else ""

        # Convert to datetime to avoid indexing errors
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        plt.axvspan(start_dt, end_dt, color=c_crisis, alpha=0.15, label=label)

    # 7. Styling & Labels
    plt.title(
        f"Market Indicators ({window}-Day Rolling)", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Value (Standardized)", fontsize=10)

    # Professional Grid (subtle)
    plt.grid(
        True, which="both", color="#b0b0b0", linestyle="--", linewidth=0.5, alpha=0.5
    )

    # Legend: remove frame for cleaner look
    plt.legend(loc="upper left", frameon=False, fontsize=10)

    # Remove empty whitespace on left/right of the plot
    plt.autoscale(enable=True, axis="x", tight=True)

    plt.show()


def plot_acf(
    df: pd.DataFrame,
    ticker: str,
    return_col: str = "log_return",
    lags: int = 40,
    dpi=300,
):
    """
    Plots ACF with a Trading Board aesthetic (Green/Red bars, Orange thresholds).
    """
    # 1. Prepare data
    subset = df[df["ticker"] == ticker].copy()
    series = subset[return_col].dropna()

    if series.empty:
        print(f"No data for {ticker}")
        return

    # 2. Calculate ACF manually
    acf_values = [1.0]
    for lag in range(1, lags + 1):
        acf_values.append(series.autocorr(lag=lag))

    # 3. Calculate Confidence Interval (95%)
    n = len(series)
    conf_level = 1.96 / np.sqrt(n)

    # 4. Setup Plot
    plt.figure(figsize=(10, 6), dpi=dpi)

    lags_indices = range(len(acf_values))

    # --- THE TRADING VISUALS ---

    # A. Conditional Colors: Green if > 0, Red if < 0
    # Uses specific hex codes for "TradingView" style Green/Red
    c_pos = "#26a69a"  # Teal/Green
    c_neg = "#ef5350"  # Red
    colors = [c_pos if x >= 0 else c_neg for x in acf_values]

    # B. Plot Bars (Thinner width looks more technical/precise)
    plt.bar(
        lags_indices,
        acf_values,
        width=0.2,
        color=colors,
        alpha=0.9,
        label="Autocorrelation",
    )

    # C. Confidence Interval (The "Noise" Zone) -> Orange
    # We use orange to signal "Warning: Data in this zone is insignificant"
    c_warn = "#ff9800"
    plt.axhline(y=conf_level, color=c_warn, linestyle="--", linewidth=1, alpha=0.8)
    plt.axhline(y=-conf_level, color=c_warn, linestyle="--", linewidth=1, alpha=0.8)
    plt.fill_between(lags_indices, -conf_level, conf_level, color=c_warn, alpha=0.1)

    # D. Zero Line (Neutral baseline)
    plt.axhline(y=0, color="#9e9e9e", linewidth=1)

    # 5. Labels and Clean up
    plt.title(f"ACF: {ticker} (log return)", fontsize=14, fontweight="bold")
    plt.xlabel("Lag (Days)")
    plt.ylabel("Correlation")

    # Optional: Customize Grid to be subtle
    plt.grid(True, linestyle=":", alpha=0.4)

    # We remove the legend because the colors explain themselves (Up/Down)
    # But if you really want it, you can uncomment below:
    # plt.legend(['Threshold', 'Threshold', 'Correlation'])

    plt.show()


def plot_volatility_time_series(
    df: pd.DataFrame, col: str = "market_volatility", dpi=300
):
    """
    Plots the volatility index over time.
    """
    plt.figure(figsize=(12, 6), dpi=dpi)
    plt.plot(df.index, df[col], color="blue", linewidth=1)
    plt.title("Market Volatility Index Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volatility (Std Dev)")
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# REGIME VISUALIZATION FUNCTIONS
# Generic plotting functions for single-method regime analysis
# Used by notebooks 04 (K-means), 05 (GARCH), 06 (HMM)
# ============================================================================


def plot_regime_timeline(
    dates,
    regimes,
    volatility,
    thresholds=None,
    title="Regime Timeline",
    ylabel="Volatility",
    save_path=None,
    figsize=(16, 6),
):
    """
    Plot regime evolution over time with colored scatter points.

    Args:
        dates: Array-like or DatetimeIndex
        regimes: Array of regime labels (0, 1, 2)
        volatility: Array of volatility values
        thresholds: Optional tuple of (low, high) threshold lines (None for HMM)
        title: Plot title
        ylabel: Y-axis label (default: "Volatility")
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    colors_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    color_list = [colors_map[r] for r in regimes]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(dates, volatility, c=color_list, alpha=0.6, s=10)
    ax.plot(dates, volatility, color="gray", alpha=0.3, linewidth=0.5)

    # Add threshold lines if provided (only for quantile-based methods like GARCH)
    if thresholds is not None:
        ax.axhline(
            thresholds[0],
            color="green",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label=f"Low threshold ({thresholds[0]:.2f})",
        )
        ax.axhline(
            thresholds[1],
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label=f"High threshold ({thresholds[1]:.2f})",
        )
        ax.legend()

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_distribution_bar(
    regimes,
    regime_labels=None,
    colors=None,
    title="Regime Distribution",
    save_path=None,
    figsize=(10, 6),
):
    """
    Bar chart showing regime frequency distribution.

    Args:
        regimes: Array of regime labels (0, 1, 2)
        regime_labels: Optional dict mapping regime ids to names
        colors: Optional dict mapping regime ids to colors
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    if colors is None:
        colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    # Count regimes
    regime_counts = {i: np.sum(regimes == i) for i in range(3)}
    regime_pcts = {i: count / len(regimes) * 100 for i, count in regime_counts.items()}

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        [regime_labels[i] for i in range(3)],
        [regime_counts[i] for i in range(3)],
        color=[colors[i] for i in range(3)],
        alpha=0.7,
        edgecolor="black",
    )

    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{regime_pcts[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Regime")
    ax.set_ylabel("Number of Days")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_distribution_box(
    volatility,
    regimes,
    regime_labels=None,
    colors=None,
    title="Volatility Distribution by Regime",
    save_path=None,
    figsize=(10, 6),
):
    """
    Box plot showing feature distribution within each regime.

    Args:
        volatility: Array of feature values
        regimes: Array of regime labels (0, 1, 2)
        regime_labels: Optional dict mapping regime ids to names
        colors: Optional dict mapping regime ids to colors
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    if colors is None:
        colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    fig, ax = plt.subplots(figsize=figsize)

    regime_data = [volatility[regimes == i] for i in range(3)]
    bp = ax.boxplot(
        regime_data,
        labels=[regime_labels[i] for i in range(3)],
        patch_artist=True,
        widths=0.6,
    )

    for patch, i in zip(bp["boxes"], range(3)):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)

    ax.set_xlabel("Regime")
    ax.set_ylabel("Volatility")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_transition_matrix_heatmap(
    transition_matrix,
    regime_labels=None,
    title="Regime Transition Matrix",
    save_path=None,
    figsize=(8, 6),
):
    """
    Heatmap visualization of regime transition probabilities.

    Args:
        transition_matrix: 3x3 numpy array of transition probabilities
        regime_labels: Optional dict mapping regime ids to names
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    labels = [regime_labels[i] for i in range(3)]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Transition Probability"},
        ax=ax,
        linewidths=1,
        linecolor="gray",
    )

    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_crisis_period(
    dates,
    regimes,
    volatility,
    thresholds,
    crisis_start,
    crisis_end,
    events=None,
    title="Crisis Period Analysis",
    ylabel="Volatility",
    save_path=None,
    figsize=(16, 6),
):
    """
    Zoom into a specific crisis period with event markers.

    Args:
        dates: Array-like or DatetimeIndex
        regimes: Array of regime labels (0, 1, 2)
        volatility: Array of volatility values
        thresholds: Tuple of (low, high) threshold values (None for HMM)
        crisis_start: Start date (string or datetime)
        crisis_end: End date (string or datetime)
        events: Optional dict of {date: label} for key events
        title: Plot title
        ylabel: Y-axis label (default: "Volatility")
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    # Filter to crisis period
    mask = (dates >= crisis_start) & (dates <= crisis_end)
    crisis_dates = dates[mask]
    crisis_vol = volatility[mask]
    crisis_reg = regimes[mask]

    colors_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    color_list = [colors_map[r] for r in crisis_reg]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(crisis_dates, crisis_vol, c=color_list, alpha=0.7, s=20)
    ax.plot(crisis_dates, crisis_vol, color="gray", alpha=0.4, linewidth=1)

    # Add threshold lines if provided (only for quantile-based methods)
    if thresholds is not None:
        ax.axhline(thresholds[0], color="green", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(thresholds[1], color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Mark events
    if events:
        for date, label in events.items():
            ax.axvline(
                pd.Timestamp(date),
                color="black",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.text(
                pd.Timestamp(date),
                ax.get_ylim()[1] * 0.95,
                label,
                rotation=90,
                verticalalignment="top",
                fontsize=9,
            )

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_facet_boxplots(
    df,
    regime_col,
    indicator_cols,
    regime_labels=None,
    colors=None,
    title="Regime Characteristics",
    save_path=None,
    figsize=(16, 12),
):
    """
    Create faceted boxplots showing indicator distributions by regime.

    Args:
        df: DataFrame with regime and indicator columns
        regime_col: Name of the regime column
        indicator_cols: List of indicator column names
        regime_labels: Dict mapping regime integers to labels
        colors: Dict mapping regime integers to colors
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    if colors is None:
        colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    n_indicators = len(indicator_cols)
    n_cols = 3
    n_rows = (n_indicators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_indicators > 1 else [axes]

    for idx, indicator in enumerate(indicator_cols):
        ax = axes[idx]

        # Prepare data for boxplot
        data_by_regime = [
            df.loc[df[regime_col] == regime, indicator].values
            for regime in sorted(df[regime_col].unique())
        ]

        bp = ax.boxplot(
            data_by_regime,
            labels=[regime_labels[r] for r in sorted(df[regime_col].unique())],
            patch_artist=True,
            widths=0.6,
        )

        # Color boxes
        for patch, regime in zip(bp["boxes"], sorted(df[regime_col].unique())):
            patch.set_facecolor(colors[regime])
            patch.set_alpha(0.7)

        ax.set_xlabel("Regime")
        ax.set_ylabel(indicator.replace("_", " ").title())
        ax.set_title(f"{indicator.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(n_indicators, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_facet_violinplots(
    df,
    regime_col,
    indicator_cols,
    regime_labels=None,
    colors=None,
    title="Regime Characteristics",
    save_path=None,
    figsize=(16, 12),
):
    """
    Create faceted violin plots showing indicator distributions by regime.

    Args:
        df: DataFrame with regime and indicator columns
        regime_col: Name of the regime column
        indicator_cols: List of indicator column names
        regime_labels: Dict mapping regime integers to labels
        colors: Dict mapping regime integers to colors
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    if colors is None:
        colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    n_indicators = len(indicator_cols)
    n_cols = 3
    n_rows = (n_indicators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_indicators > 1 else [axes]

    # Prepare data for seaborn
    df_plot = df[[regime_col] + indicator_cols].copy()
    df_plot["regime_label"] = df_plot[regime_col].map(regime_labels)

    for idx, indicator in enumerate(indicator_cols):
        ax = axes[idx]

        # Create violin plot
        parts = ax.violinplot(
            [
                df_plot.loc[df_plot[regime_col] == regime, indicator].values
                for regime in sorted(df_plot[regime_col].unique())
            ],
            positions=range(len(df_plot[regime_col].unique())),
            showmeans=True,
            showmedians=True,
            widths=0.7,
        )

        # Color violins
        for pc, regime in zip(parts["bodies"], sorted(df_plot[regime_col].unique())):
            pc.set_facecolor(colors[regime])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(df_plot[regime_col].unique())))
        ax.set_xticklabels(
            [regime_labels[r] for r in sorted(df_plot[regime_col].unique())]
        )
        ax.set_xlabel("Regime")
        ax.set_ylabel(indicator.replace("_", " ").title())
        ax.set_title(f"{indicator.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(n_indicators, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_heatmap(
    regime_means,
    regime_labels=None,
    title="Regime Characteristics Heatmap (Standardized)",
    save_path=None,
    figsize=(10, 8),
):
    """
    Plot heatmap of standardized regime means by indicator.

    Args:
        regime_means: DataFrame with regimes as rows and indicators as columns
        regime_labels: Dict mapping regime integers to labels
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        regime_means,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        center=0,
        cbar_kws={"label": "Standardized Mean"},
        ax=ax,
        linewidths=1,
        linecolor="gray",
    )

    ax.set_xlabel("Indicator")
    ax.set_ylabel("Regime")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_means_errorbars(
    ci_df,
    regime_labels=None,
    colors=None,
    title="Regime Means with 95% Confidence Intervals",
    save_path=None,
    figsize=(16, 12),
):
    """
    Plot regime means with confidence interval error bars.

    Args:
        ci_df: DataFrame with columns: indicator, regime, mean, ci_low, ci_high
        regime_labels: Dict mapping regime integers to labels
        colors: Dict mapping regime integers to colors
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    if colors is None:
        colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    indicators = ci_df["indicator"].unique()
    n_indicators = len(indicators)
    n_cols = 3
    n_rows = (n_indicators + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_indicators > 1 else [axes]

    for idx, indicator in enumerate(indicators):
        ax = axes[idx]

        df_ind = ci_df[ci_df["indicator"] == indicator].sort_values("regime")

        x_positions = np.arange(len(df_ind))
        means = df_ind["mean"].values
        errors_low = means - df_ind["ci_low"].values
        errors_high = df_ind["ci_high"].values - means
        errors = [errors_low, errors_high]

        colors_list = [colors[r] for r in df_ind["regime"].values]

        ax.errorbar(
            x_positions,
            means,
            yerr=errors,
            fmt="o",
            capsize=5,
            capthick=2,
            markersize=8,
            linewidth=2,
            ecolor="gray",
        )

        # Color the points
        for i, (x, y, c) in enumerate(zip(x_positions, means, colors_list)):
            ax.plot(x, y, "o", color=c, markersize=10, alpha=0.8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([regime_labels[r] for r in df_ind["regime"].values])
        ax.set_xlabel("Regime")
        ax.set_ylabel(indicator.replace("_", " ").title())
        ax.set_title(f"{indicator.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(n_indicators, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_correlation_matrices(
    df,
    regime_col,
    indicator_cols,
    regime_labels=None,
    title="Indicator Correlations by Regime",
    save_path=None,
    figsize=(16, 5),
):
    """
    Plot correlation matrices for each regime side by side.

    Args:
        df: DataFrame with regime and indicator columns
        regime_col: Name of the regime column
        indicator_cols: List of indicator column names
        regime_labels: Dict mapping regime integers to labels
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    regimes = sorted(df[regime_col].unique())
    n_regimes = len(regimes)

    fig, axes = plt.subplots(1, n_regimes, figsize=figsize)

    for idx, regime in enumerate(regimes):
        ax = axes[idx] if n_regimes > 1 else axes

        # Compute correlation matrix for this regime
        regime_data = df.loc[df[regime_col] == regime, indicator_cols]
        corr_matrix = regime_data.corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            cbar=(idx == n_regimes - 1),  # Only show colorbar on last plot
            ax=ax,
            square=True,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_title(f"{regime_labels[regime]}")

        if idx == 0:
            ax.set_ylabel("Indicator")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Indicator")

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regime_shaded_timeseries(
    df,
    date_col,
    regime_col,
    indicator_cols,
    regime_labels=None,
    colors=None,
    title="Indicator Time Series with Regime Shading",
    save_path=None,
    figsize=(16, 10),
):
    """
    Plot indicator time series with background shading by regime.

    Args:
        df: DataFrame with date, regime, and indicator columns
        date_col: Name of the date column (or use index if None)
        regime_col: Name of the regime column
        indicator_cols: List of indicator column names to plot
        regime_labels: Dict mapping regime integers to labels
        colors: Dict mapping regime integers to colors
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    if colors is None:
        colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    n_indicators = len(indicator_cols)
    fig, axes = plt.subplots(n_indicators, 1, figsize=figsize, sharex=True)

    if n_indicators == 1:
        axes = [axes]

    # Get dates
    dates = df.index if date_col is None else df[date_col]

    for idx, indicator in enumerate(indicator_cols):
        ax = axes[idx]

        # Plot indicator line
        ax.plot(dates, df[indicator], color="black", linewidth=1, alpha=0.7)

        # Add regime shading
        current_regime = df[regime_col].iloc[0]
        start_idx = 0

        for i in range(1, len(df)):
            if df[regime_col].iloc[i] != current_regime or i == len(df) - 1:
                end_idx = i if i < len(df) - 1 else i + 1
                ax.axvspan(
                    dates.iloc[start_idx],
                    dates.iloc[end_idx - 1],
                    alpha=0.2,
                    color=colors[current_regime],
                )
                current_regime = df[regime_col].iloc[i]
                start_idx = i

        ax.set_ylabel(indicator.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

        if idx == 0:
            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=colors[r], alpha=0.3, label=regime_labels[r])
                for r in sorted(df[regime_col].unique())
            ]
            ax.legend(handles=legend_elements, loc="upper right")

    axes[-1].set_xlabel("Date")
    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_transition_matrix_and_durations(
    transition_matrix,
    run_lengths,
    regime_labels=None,
    title="Regime Transitions and Durations",
    save_path=None,
    figsize=(14, 6),
):
    """
    Plot transition matrix heatmap and duration distributions side by side.

    Args:
        transition_matrix: Square array of transition probabilities
        run_lengths: Dict mapping regime -> list of run lengths
        regime_labels: Dict mapping regime integers to labels
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Transition matrix heatmap
    labels = [regime_labels[i] for i in range(len(transition_matrix))]
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Probability"},
        ax=ax1,
        linewidths=1,
        linecolor="gray",
    )
    ax1.set_xlabel("Next State")
    ax1.set_ylabel("Current State")
    ax1.set_title("Transition Matrix")

    # Duration distributions
    for regime, lengths in run_lengths.items():
        if len(lengths) > 0:
            ax2.hist(
                lengths,
                bins=30,
                alpha=0.6,
                label=f"{regime_labels[regime]} (avg={np.mean(lengths):.1f})",
                color=colors[regime],
                edgecolor="black",
            )

    ax2.set_xlabel("Duration (days)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Regime Duration Distributions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_drawdown_validation(
    dates,
    regimes,
    drawdown,
    market_state,
    regime_labels=None,
    title="Ensemble Regimes vs Market Drawdown Validation",
    save_path=None,
    figsize=(16, 10),
):
    """
    Plot ensemble regimes against market drawdown ground truth.

    Args:
        dates: DatetimeIndex or array of dates
        regimes: Array of regime predictions (0=Calm, 1=Moderate, 2=Turbulent)
        drawdown: Array of drawdown values (expanding window)
        market_state: Array of ground truth states (0=Bull, 1=Correction, 2=Bear)
        regime_labels: Dict mapping regime integers to labels
        title: Overall plot title
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    regime_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    state_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Panel 1: Ensemble regime timeline with drawdown background
    ax1_twin = ax1.twinx()

    # Drawdown as background shading
    ax1.fill_between(
        dates, 0, drawdown * 100, alpha=0.2, color="gray", label="Drawdown"
    )
    ax1.axhline(
        -10,
        color="orange",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Correction (-10%)",
    )
    ax1.axhline(
        -20,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Bear Market (-20%)",
    )

    # Regime scatter on twin axis
    regime_colors_list = [regime_colors[r] for r in regimes]
    ax1_twin.scatter(dates, regimes, c=regime_colors_list, alpha=0.6, s=10)

    ax1.set_ylabel("Drawdown (%)")
    ax1_twin.set_ylabel("Ensemble Regime")
    ax1_twin.set_yticks([0, 1, 2])
    ax1_twin.set_yticklabels([regime_labels[i] for i in range(3)])
    ax1.set_title("Panel 1: Ensemble Regimes with Drawdown Background")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Drawdown with ground truth zones
    for state in [0, 1, 2]:
        mask = market_state == state
        if np.any(mask):
            ax2.scatter(
                dates[mask],
                drawdown[mask] * 100,
                c=state_colors[state],
                alpha=0.4,
                s=15,
                label=["Bull Market", "Correction", "Bear Market"][state],
            )

    ax2.axhline(-10, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(-20, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.plot(dates, drawdown * 100, color="black", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Panel 2: Market Drawdown with Ground Truth Zones")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

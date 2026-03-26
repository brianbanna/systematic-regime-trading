"""
Generate all publication-quality figures from saved results.

Produces 13 charts for the research website and tearsheet.
All figures saved as 300 DPI PNGs in results/figures/.

Usage:
    python scripts/generate_figures.py
    make figures
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from systematic_regime_trading.utils.config import get_path, load_config
from systematic_regime_trading.evaluation.rolling import (
    rolling_sharpe, rolling_volatility, rolling_drawdown,
)

RESULTS_DIR = get_path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

# Style
REGIME_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
REGIME_NAMES = {0: "Calm", 1: "Moderate", 2: "Turbulent"}
STRATEGY_COLORS = {
    "regime_momentum": "#4fc3f7",
    "binary_regime": "#81c784",
    "proportional_regime": "#ffb74d",
    "vol_targeted": "#ce93d8",
    "buy_and_hold": "#90a4ae",
    "sixty_forty": "#a1887f",
    "risk_parity": "#80cbc4",
}
STRATEGY_LABELS = {
    "regime_momentum": "Regime Momentum",
    "binary_regime": "Binary Regime",
    "proportional_regime": "Proportional",
    "vol_targeted": "Vol-Targeted",
    "buy_and_hold": "Buy & Hold",
    "sixty_forty": "60/40",
    "risk_parity": "Risk Parity",
}

DPI = 300


STRATEGY_COLORS["sma_200"] = "#fff176"
STRATEGY_COLORS["vix_20"] = "#ef9a9a"
STRATEGY_COLORS["vol_managed"] = "#b0bec5"
STRATEGY_COLORS["regime_vol_targeted"] = "#80deea"

STRATEGY_LABELS["sma_200"] = "SMA 200"
STRATEGY_LABELS["vix_20"] = "VIX > 20"
STRATEGY_LABELS["vol_managed"] = "Vol-Managed"
STRATEGY_LABELS["regime_vol_targeted"] = "Regime Vol-Target"

# Dark theme colors
BG_COLOR = "#0a0a0a"
FG_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a2a"
ACCENT = "#4fc3f7"


def setup():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#141414",
        "text.color": FG_COLOR,
        "axes.labelcolor": FG_COLOR,
        "xtick.color": FG_COLOR,
        "ytick.color": FG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.facecolor": "#1a1a1a",
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": FG_COLOR,
        "savefig.facecolor": BG_COLOR,
    })


def load_results():
    """Load all saved results."""
    results = {}

    # Backtest results
    bt_dir = RESULTS_DIR / "backtest_results"
    backtest = {}
    for f in bt_dir.glob("*.parquet"):
        backtest[f.stem] = pd.read_parquet(f)
    results["backtest"] = backtest

    # Regime predictions
    results["predictions"] = pd.read_parquet(RESULTS_DIR / "regime_predictions.parquet")

    # Strategy signals
    results["signals"] = pd.read_parquet(RESULTS_DIR / "strategy_signals.parquet")

    # Performance table
    results["perf_table"] = pd.read_csv(RESULTS_DIR / "performance_table.csv", index_col=0)

    # Cost sensitivity
    results["cost_sens"] = pd.read_csv(RESULTS_DIR / "cost_sensitivity.csv")

    # Regime performance
    results["regime_perf"] = pd.read_csv(RESULTS_DIR / "regime_performance.csv", index_col=0)

    return results


# =========================================================================
# Chart 1: Cumulative Return Comparison
# =========================================================================
def chart_cumulative_returns(results):
    logger.info("Chart 1: Cumulative returns")
    backtest = results["backtest"]
    predictions = results["predictions"]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Add regime shading
    _add_regime_shading(ax, predictions)

    # Plot strategies
    plot_order = ["regime_momentum", "binary_regime", "proportional_regime",
                  "vol_targeted", "buy_and_hold"]
    for name in plot_order:
        if name not in backtest:
            continue
        bt = backtest[name]
        label = STRATEGY_LABELS.get(name, name)
        color = STRATEGY_COLORS.get(name, "gray")
        lw = 2.0 if name == "regime_momentum" else 1.2
        ls = "-" if name != "buy_and_hold" else "--"
        ax.plot(bt.index, bt["cumulative_return"], label=label,
                color=color, linewidth=lw, linestyle=ls, alpha=0.9)

    # Annotations
    ax.annotate("2008 Crisis", xy=(pd.Timestamp("2008-09-15"), 1.0),
                fontsize=9, color="gray", ha="center")
    ax.annotate("COVID", xy=(pd.Timestamp("2020-03-15"), 2.0),
                fontsize=9, color="gray", ha="center")

    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Return Comparison (Out-of-Sample)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(backtest["buy_and_hold"].index[0], backtest["buy_and_hold"].index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_cumulative_returns.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 2: Drawdown Chart
# =========================================================================
def chart_drawdown(results):
    logger.info("Chart 2: Drawdown")
    backtest = results["backtest"]
    predictions = results["predictions"]

    fig, ax = plt.subplots(figsize=(14, 5))

    _add_regime_shading(ax, predictions)

    # Best strategy
    bt_best = backtest["regime_momentum"]
    ax.fill_between(bt_best.index, bt_best["drawdown"] * 100, 0,
                    alpha=0.4, color=STRATEGY_COLORS["regime_momentum"],
                    label="Regime Momentum")

    # Benchmark
    bt_bah = backtest["buy_and_hold"]
    ax.plot(bt_bah.index, bt_bah["drawdown"] * 100, color=STRATEGY_COLORS["buy_and_hold"],
            linewidth=1.0, linestyle="--", label="Buy & Hold", alpha=0.8)

    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown from Peak")
    ax.legend(loc="lower left")
    ax.set_xlim(bt_bah.index[0], bt_bah.index[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_drawdown.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 3: Monthly Return Heatmap
# =========================================================================
def chart_monthly_heatmap(results):
    logger.info("Chart 3: Monthly return heatmap")
    bt = results["backtest"]["regime_momentum"]

    monthly = bt["net_return"].resample("ME").sum() * 100
    pivot = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    heatmap_data = pivot.pivot_table(index="Year", columns="Month", values="Return")
    heatmap_data.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Monthly Return (%)"})
    ax.set_title("Regime Momentum: Monthly Returns (%)")
    ax.set_ylabel("Year")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_monthly_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 4: Rolling 1-Year Sharpe
# =========================================================================
def chart_rolling_sharpe(results):
    logger.info("Chart 4: Rolling Sharpe")
    backtest = results["backtest"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for name in ["regime_momentum", "buy_and_hold"]:
        if name not in backtest:
            continue
        bt = backtest[name]
        rs = rolling_sharpe(bt["net_return"], window=252)
        label = STRATEGY_LABELS.get(name, name)
        color = STRATEGY_COLORS.get(name, "gray")
        lw = 2.0 if name == "regime_momentum" else 1.0
        ls = "-" if name != "buy_and_hold" else "--"
        ax.plot(rs.index, rs, label=label, color=color, linewidth=lw, linestyle=ls)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.axhline(y=1, color="gray", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Rolling 1Y Sharpe")
    ax.set_title("Rolling 1-Year Sharpe Ratio")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_rolling_sharpe.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 5: Cost Sensitivity
# =========================================================================
def chart_cost_sensitivity(results):
    logger.info("Chart 5: Cost sensitivity")
    df = results["cost_sens"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in df["strategy"].unique():
        sdf = df[df["strategy"] == strategy].sort_values("cost_bps")
        label = STRATEGY_LABELS.get(strategy, strategy)
        color = STRATEGY_COLORS.get(strategy, "gray")
        ax.plot(sdf["cost_bps"], sdf["sharpe"], marker="o", label=label,
                color=color, linewidth=2, markersize=5)

    # Benchmark Sharpe line
    perf = results["perf_table"]
    if "buy_and_hold" in perf.index:
        bah_sharpe = perf.loc["buy_and_hold", "Sharpe"]
        ax.axhline(y=bah_sharpe, color=STRATEGY_COLORS["buy_and_hold"],
                   linestyle="--", linewidth=1, label=f"Buy & Hold ({bah_sharpe:.2f})")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Transaction Cost (bps, one-way)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Cost Sensitivity Analysis")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_cost_sensitivity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 6: Regime Timeline
# =========================================================================
def chart_regime_timeline(results):
    logger.info("Chart 6: Regime timeline")
    pred = results["predictions"]
    bt_bah = results["backtest"]["buy_and_hold"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1],
                                    sharex=True)

    # Top: Equity curve with regime shading
    ax1.plot(bt_bah.index, bt_bah["cumulative_return"], color="black", linewidth=0.8)
    _add_regime_shading(ax1, pred)
    ax1.set_ylabel("Growth of $1")
    ax1.set_title("Market Performance with Regime Classification")

    # Bottom: Regime probabilities stacked area
    dates = pd.to_datetime(pred["Date"])
    ax2.fill_between(dates, 0, pred["prob_calm"],
                     color=REGIME_COLORS[0], alpha=0.7, label="P(Calm)")
    ax2.fill_between(dates, pred["prob_calm"],
                     pred["prob_calm"] + pred["prob_moderate"],
                     color=REGIME_COLORS[1], alpha=0.7, label="P(Moderate)")
    ax2.fill_between(dates, pred["prob_calm"] + pred["prob_moderate"], 1,
                     color=REGIME_COLORS[2], alpha=0.7, label="P(Turbulent)")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", ncol=3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_regime_timeline.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 7: Transition Matrix
# =========================================================================
def chart_transition_matrix(results):
    logger.info("Chart 7: Transition matrix")
    pred = results["predictions"]
    labels = pred["regime_label"].values

    # Compute transition matrix
    n_states = 3
    counts = np.zeros((n_states, n_states))
    for i in range(len(labels) - 1):
        counts[int(labels[i]), int(labels[i + 1])] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    trans = np.divide(counts, row_sums, where=row_sums > 0, out=np.zeros_like(counts))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(trans, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=["Calm", "Moderate", "Turbulent"],
                yticklabels=["Calm", "Moderate", "Turbulent"],
                vmin=0, vmax=1, ax=ax, linewidths=1,
                cbar_kws={"label": "Transition Probability"})
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    ax.set_title("Regime Transition Probabilities")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_transition_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 8: Regime Distribution
# =========================================================================
def chart_regime_distribution(results):
    logger.info("Chart 8: Regime distribution")
    pred = results["predictions"]

    dates = pd.to_datetime(pred["Date"])
    labels = pred["regime_label"].values

    # Split by period
    periods = {
        "2006-2007": (dates < "2008-01-01"),
        "2008-2009": (dates >= "2008-01-01") & (dates < "2010-01-01"),
        "2010-2019": (dates >= "2010-01-01") & (dates < "2020-01-01"),
        "2020": (dates >= "2020-01-01"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.2
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(periods))

    for (period_name, mask), offset in zip(periods.items(), offsets):
        period_labels = labels[mask]
        if len(period_labels) == 0:
            continue
        pcts = []
        for regime in range(3):
            pcts.append((period_labels == regime).sum() / len(period_labels) * 100)
        ax.bar(x + offset, pcts, width, label=period_name, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Calm", "Moderate", "Turbulent"])
    ax.set_ylabel("% of Days")
    ax.set_title("Regime Distribution by Period")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_regime_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 9: Regime-Conditional Performance
# =========================================================================
def chart_regime_performance(results):
    logger.info("Chart 9: Regime-conditional performance")
    rp = results["regime_perf"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ("annual_return", "Annualized Return (%)", 100),
        ("vol", "Annualized Volatility (%)", 100),
        ("sharpe", "Sharpe Ratio", 1),
    ]

    for ax, (col, ylabel, mult) in zip(axes, metrics):
        colors_list = [REGIME_COLORS[rp.loc[regime, "regime_id"]] for regime in rp.index]
        vals = rp[col].values * mult
        bars = ax.bar(rp.index, vals, color=colors_list, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Regime-Conditional Performance (Regime Momentum)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_regime_performance.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 10: Allocation Over Time
# =========================================================================
def chart_allocation(results):
    logger.info("Chart 10: Allocation over time")
    signals = results["signals"]
    predictions = results["predictions"]

    fig, ax = plt.subplots(figsize=(14, 5))

    _add_regime_shading(ax, predictions)

    if "regime_momentum" in signals.columns:
        sig = signals["regime_momentum"].dropna()
        ax.fill_between(sig.index, sig.values, 0,
                        color=STRATEGY_COLORS["regime_momentum"], alpha=0.5)
        ax.plot(sig.index, sig.values, color=STRATEGY_COLORS["regime_momentum"],
                linewidth=0.8)

    ax.set_ylabel("Equity Allocation")
    ax.set_title("Regime Momentum: Allocation Over Time")
    ax.set_ylim(-0.05, 1.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_allocation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 11: Signal vs Drawdown
# =========================================================================
def chart_signal_vs_drawdown(results):
    logger.info("Chart 11: Signal vs drawdown")
    signals = results["signals"]
    bt_bah = results["backtest"]["buy_and_hold"]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Allocation (left axis)
    if "regime_momentum" in signals.columns:
        sig = signals["regime_momentum"].dropna()
        common = sig.index.intersection(bt_bah.index)
        ax1.fill_between(common, sig.loc[common].values, 0,
                         color=STRATEGY_COLORS["regime_momentum"], alpha=0.3,
                         label="Allocation")
        ax1.set_ylabel("Equity Allocation", color=STRATEGY_COLORS["regime_momentum"])
        ax1.set_ylim(-0.05, 1.3)

    # Market drawdown (right axis, inverted)
    ax2 = ax1.twinx()
    dd = bt_bah["drawdown"].loc[common] * 100
    ax2.fill_between(common, dd, 0, color="red", alpha=0.2)
    ax2.plot(common, dd, color="red", linewidth=0.8, alpha=0.6, label="Market Drawdown")
    ax2.set_ylabel("Market Drawdown (%)", color="red")
    ax2.invert_yaxis()

    ax1.set_title("Signal vs Market Drawdown: Does the strategy go defensive before crashes?")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_signal_vs_drawdown.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 12: Model Agreement
# =========================================================================
def chart_model_agreement(results):
    logger.info("Chart 12: Model agreement")
    pred = results["predictions"]

    # Include all available models
    all_models = [("hmm_label", "HMM"), ("garch_label", "GARCH"), ("kmeans_label", "KMeans"),
                  ("gmm_label", "GMM"), ("ms_label", "Markov-Sw")]
    models = [m for m, _ in all_models if m in pred.columns]
    model_names = [n for m, n in all_models if m in pred.columns]

    # Pairwise agreement
    n = len(models)
    agreement = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agreement[i, j] = (pred[models[i]] == pred[models[j]]).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(agreement, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=model_names, yticklabels=model_names,
                vmin=0, vmax=100, ax=ax, linewidths=1,
                cbar_kws={"label": "Agreement (%)"})
    ax.set_title("Pairwise Model Agreement (%)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_model_agreement.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 13: Performance Summary Table (as figure)
# =========================================================================
def chart_performance_table(results):
    logger.info("Chart 13: Performance table")
    table = results["perf_table"]

    display_cols = ["CAGR", "Annual_Vol", "Sharpe", "Sortino", "Max_DD", "Turnover"]
    available = [c for c in display_cols if c in table.columns]
    display = table[available].copy()

    # Format
    fmt_pct = lambda x: f"{x:.1%}" if abs(x) < 10 else f"{x:.0%}"
    fmt_dec = lambda x: f"{x:.2f}" if np.isfinite(x) else "N/A"

    formatted = display.copy()
    for col in ["CAGR", "Annual_Vol", "Max_DD"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(fmt_pct)
    for col in ["Sharpe", "Sortino", "Turnover"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(fmt_dec)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    tbl = ax.table(
        cellText=formatted.values,
        rowLabels=[STRATEGY_LABELS.get(n, n) for n in formatted.index],
        colLabels=formatted.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Color header
    for j in range(len(formatted.columns)):
        tbl[0, j].set_facecolor("#4fc3f7")
        tbl[0, j].set_text_props(color="white", weight="bold")

    # Highlight best Sharpe row
    best_idx = display["Sharpe"].idxmax() if "Sharpe" in display.columns else None
    if best_idx is not None:
        row_idx = list(formatted.index).index(best_idx) + 1
        for j in range(len(formatted.columns)):
            tbl[row_idx, j].set_facecolor("#e3f2fd")

    ax.set_title("Strategy Performance Comparison", fontsize=14, pad=20)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_performance_table.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Helper
# =========================================================================
def _add_regime_shading(ax, predictions):
    """Add regime background shading to an axes."""
    dates = pd.to_datetime(predictions["Date"])
    labels = predictions["regime_label"].values

    prev_label = labels[0]
    start_date = dates.iloc[0]

    for i in range(1, len(labels)):
        if labels[i] != prev_label or i == len(labels) - 1:
            end_date = dates.iloc[i]
            color = REGIME_COLORS.get(int(prev_label), "gray")
            ax.axvspan(start_date, end_date, alpha=0.08, color=color)
            start_date = end_date
            prev_label = labels[i]


# =========================================================================
# Chart 14: Prediction Accuracy
# =========================================================================
def chart_prediction_accuracy(results):
    logger.info("Chart 14: Prediction accuracy")
    try:
        pred_acc = pd.read_csv(RESULTS_DIR / "prediction_accuracy.csv")
    except FileNotFoundError:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    horizons = pred_acc["horizon_days"].unique()

    x = np.arange(3)
    width = 0.25
    for i, h in enumerate(horizons):
        hdf = pred_acc[pred_acc["horizon_days"] == h].sort_values("regime_id")
        bars = ax.bar(x + i * width, hdf["mean_realized_vol"].values * 100,
                      width, label=f"{h}d horizon", alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["Calm", "Moderate", "Turbulent"])
    ax.set_ylabel("Realized Volatility (%)")
    ax.set_title("Does the model predict future volatility correctly?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_prediction_accuracy.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 15: Signal Decay (IC by horizon)
# =========================================================================
def chart_signal_decay(results):
    logger.info("Chart 15: Signal decay")
    try:
        ic_df = pd.read_csv(RESULTS_DIR / "signal_decay.csv")
    except FileNotFoundError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ic_df["horizon_days"].astype(str), ic_df["ic"], color=ACCENT, alpha=0.8)
    ax.axhline(y=0, color=FG_COLOR, linewidth=0.5)
    ax.set_xlabel("Forward Horizon (days)")
    ax.set_ylabel("Information Coefficient (Spearman)")
    ax.set_title("Signal Decay: How quickly does regime information lose value?")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_signal_decay.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 16: Cumulative Excess Return
# =========================================================================
def chart_cumulative_excess(results):
    logger.info("Chart 16: Cumulative excess return")
    backtest = results["backtest"]

    if "regime_momentum" not in backtest or "buy_and_hold" not in backtest:
        return

    bt_strat = backtest["regime_momentum"]
    bt_bench = backtest["buy_and_hold"]
    common = bt_strat.index.intersection(bt_bench.index)

    excess = bt_strat.loc[common, "net_return"] - bt_bench.loc[common, "net_return"]
    cum_excess = excess.cumsum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cum_excess.index, cum_excess * 100, color=ACCENT, linewidth=1.5)
    ax.fill_between(cum_excess.index, cum_excess * 100, 0,
                    where=cum_excess >= 0, color=ACCENT, alpha=0.15)
    ax.fill_between(cum_excess.index, cum_excess * 100, 0,
                    where=cum_excess < 0, color="#e74c3c", alpha=0.15)
    ax.axhline(y=0, color=FG_COLOR, linewidth=0.5)
    ax.set_ylabel("Cumulative Excess Return (%)")
    ax.set_title("Regime Momentum vs Buy-and-Hold: Cumulative Excess Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "16_cumulative_excess.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 17: OOS Extension Comparison
# =========================================================================
def chart_oos_comparison(results):
    logger.info("Chart 17: OOS 2021-2026 comparison")
    oos_dir = RESULTS_DIR / "oos_extension"
    if not oos_dir.exists():
        return

    try:
        oos_table = pd.read_csv(oos_dir / "oos_performance_table.csv", index_col=0)
    except FileNotFoundError:
        return

    # In-sample vs OOS Sharpe comparison
    is_table = results["perf_table"]

    strategies = ["regime_momentum", "buy_and_hold", "sma_200", "vol_managed"]
    is_sharpes = [is_table.loc[s, "Sharpe"] if s in is_table.index else 0 for s in strategies]
    oos_sharpes = [oos_table.loc[s, "Sharpe"] if s in oos_table.index else 0 for s in strategies]

    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(strategies))
    width = 0.35

    ax.bar(x - width/2, is_sharpes, width, label="In-Sample (2006-2020)", color=ACCENT, alpha=0.8)
    ax.bar(x + width/2, oos_sharpes, width, label="Out-of-Sample (2021-2026)", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("In-Sample vs Out-of-Sample Performance")
    ax.axhline(y=0, color=FG_COLOR, linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "17_oos_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Chart 18: Crude Oil Regimes
# =========================================================================
def chart_crude_oil_regimes(results):
    logger.info("Chart 18: Crude oil regimes")
    comm_dir = RESULTS_DIR / "commodities"
    if not comm_dir.exists():
        return

    try:
        cl_pred = pd.read_parquet(comm_dir / "crude_oil_regimes.parquet")
    except FileNotFoundError:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)

    dates = pd.to_datetime(cl_pred["Date"])

    # Top: Price with regime shading
    ax1.plot(dates, cl_pred["close_price"], color=FG_COLOR, linewidth=0.8)
    labels = cl_pred["regime_label"].values
    prev = labels[0]
    start = dates.iloc[0]
    for i in range(1, len(labels)):
        if labels[i] != prev or i == len(labels) - 1:
            color = REGIME_COLORS.get(int(prev), "gray")
            ax1.axvspan(start, dates.iloc[i], alpha=0.15, color=color)
            start = dates.iloc[i]
            prev = labels[i]
    ax1.set_ylabel("WTI Crude Oil ($)")
    ax1.set_title("Crude Oil Regime Detection")

    # Bottom: Stacked regime probabilities
    ax2.fill_between(dates, 0, cl_pred["prob_calm"],
                     color=REGIME_COLORS[0], alpha=0.7, label="Calm")
    ax2.fill_between(dates, cl_pred["prob_calm"],
                     cl_pred["prob_calm"] + cl_pred["prob_moderate"],
                     color=REGIME_COLORS[1], alpha=0.7, label="Moderate")
    ax2.fill_between(dates, cl_pred["prob_calm"] + cl_pred["prob_moderate"], 1,
                     color=REGIME_COLORS[2], alpha=0.7, label="Turbulent")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", ncol=3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "18_crude_oil_regimes.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Plotly Interactive Export
# =========================================================================
def export_plotly_data(results):
    """Export equity curve data as JSON for interactive Plotly chart on website."""
    logger.info("Exporting Plotly interactive data...")
    import json

    backtest = results["backtest"]
    predictions = results["predictions"]

    chart_data = {"strategies": {}}

    for name in ["regime_momentum", "buy_and_hold", "sma_200", "vol_managed"]:
        if name not in backtest:
            continue
        bt = backtest[name]
        chart_data["strategies"][STRATEGY_LABELS.get(name, name)] = {
            "dates": [d.strftime("%Y-%m-%d") for d in bt.index],
            "cumulative_return": bt["cumulative_return"].round(4).tolist(),
            "daily_return": (bt["net_return"] * 100).round(2).tolist(),
            "position": bt["position"].round(2).tolist(),
            "color": STRATEGY_COLORS.get(name, "#888"),
        }

    # Regime data for hover
    pred_dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(predictions["Date"])]
    chart_data["regimes"] = {
        "dates": pred_dates,
        "labels": predictions["regime_label"].tolist(),
    }

    # Cost sensitivity data
    try:
        cost_sens = results["cost_sens"]
        sens_data = {}
        for strategy in cost_sens["strategy"].unique():
            sdf = cost_sens[cost_sens["strategy"] == strategy].sort_values("cost_bps")
            sens_data[STRATEGY_LABELS.get(strategy, strategy)] = {
                "cost_bps": sdf["cost_bps"].tolist(),
                "sharpe": sdf["sharpe"].round(3).tolist(),
            }
        chart_data["cost_sensitivity"] = sens_data
    except Exception:
        pass

    # Performance table
    try:
        table = results["perf_table"]
        chart_data["performance_table"] = {
            "strategies": [STRATEGY_LABELS.get(s, s) for s in table.index],
            "cagr": (table["CAGR"] * 100).round(1).tolist(),
            "sharpe": table["Sharpe"].round(2).tolist(),
            "max_dd": (table["Max_DD"] * 100).round(1).tolist(),
            "turnover": table["Turnover"].round(1).tolist(),
        }
    except Exception:
        pass

    # Save to website/js
    js_dir = get_path("website/js")
    js_dir.mkdir(parents=True, exist_ok=True)
    with open(js_dir / "chart_data.json", "w") as f:
        json.dump(chart_data, f)

    logger.info(f"Plotly data exported to {js_dir / 'chart_data.json'}")


# =========================================================================
# Main
# =========================================================================
def main():
    setup()
    results = load_results()

    # Original 13 charts (now dark themed)
    chart_cumulative_returns(results)      # 1
    chart_drawdown(results)                # 2
    chart_monthly_heatmap(results)         # 3
    chart_rolling_sharpe(results)          # 4
    chart_cost_sensitivity(results)        # 5
    chart_regime_timeline(results)         # 6
    chart_transition_matrix(results)       # 7
    chart_regime_distribution(results)     # 8
    chart_regime_performance(results)      # 9
    chart_allocation(results)              # 10
    chart_signal_vs_drawdown(results)      # 11
    chart_model_agreement(results)         # 12
    chart_performance_table(results)       # 13

    # New charts from Phase 5-7
    chart_prediction_accuracy(results)     # 14
    chart_signal_decay(results)            # 15
    chart_cumulative_excess(results)       # 16
    chart_oos_comparison(results)          # 17
    chart_crude_oil_regimes(results)       # 18

    # Interactive data export
    export_plotly_data(results)

    logger.info(f"All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

"""
Regime prediction quality analysis.

Measures whether the regime model actually predicts future volatility,
how quickly signals decay, calendar concentration of alpha,
and the cost of defensiveness.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def regime_prediction_accuracy(
    returns: pd.Series,
    regime_labels: pd.Series,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Test whether predicted regimes predict future realized volatility.

    Computes realized vol over the next N days, grouped by predicted regime.
    If calm predictions have lower realized vol than turbulent predictions,
    the model is informative.

    Args:
        returns: Daily market returns
        regime_labels: Predicted regime labels (0=calm, 2=turbulent)
        horizons: Forward-looking horizons in days (default: [5, 10, 20])

    Returns:
        DataFrame with regime, horizon, mean_realized_vol, count
    """
    if horizons is None:
        horizons = [5, 10, 20]

    common = returns.index.intersection(regime_labels.index)
    ret = returns.loc[common]
    labels = regime_labels.loc[common]

    results = []
    for horizon in horizons:
        fwd_vol = ret.rolling(horizon).std().shift(-horizon) * np.sqrt(TRADING_DAYS)

        for regime in sorted(labels.unique()):
            mask = labels == regime
            regime_fwd_vol = fwd_vol[mask].dropna()

            if len(regime_fwd_vol) > 10:
                results.append({
                    "regime": {0: "Calm", 1: "Moderate", 2: "Turbulent"}.get(regime, f"R{regime}"),
                    "regime_id": regime,
                    "horizon_days": horizon,
                    "mean_realized_vol": regime_fwd_vol.mean(),
                    "median_realized_vol": regime_fwd_vol.median(),
                    "count": len(regime_fwd_vol),
                })

    return pd.DataFrame(results)


def regime_lead_time(
    returns: pd.Series,
    regime_labels: pd.Series,
    drawdown_threshold: float = -0.05,
) -> pd.DataFrame:
    """
    How early does the model detect regime shifts before drawdowns?

    Finds market peaks (local maxima in cumulative return), then measures
    days between peak and first turbulent signal.

    Args:
        returns: Daily market returns
        regime_labels: Predicted regime labels
        drawdown_threshold: Minimum drawdown to count as a "real" peak

    Returns:
        DataFrame with peak dates, drawdown magnitude, and lead time
    """
    common = returns.index.intersection(regime_labels.index)
    ret = returns.loc[common]
    labels = regime_labels.loc[common]

    cumret = (1 + ret).cumprod()
    running_max = cumret.cummax()
    drawdown = cumret / running_max - 1

    # Find peaks where subsequent drawdown exceeds threshold
    results = []
    in_drawdown = False
    peak_date = None
    peak_value = 0

    for date in cumret.index:
        if cumret[date] == running_max[date]:
            if in_drawdown and peak_date is not None:
                in_drawdown = False
            peak_date = date
            peak_value = cumret[date]
        elif drawdown[date] < drawdown_threshold and not in_drawdown:
            in_drawdown = True
            # Find first turbulent signal after peak
            mask_after_peak = (labels.index >= peak_date) & (labels == 2)
            turb_dates = labels.index[mask_after_peak]

            if len(turb_dates) > 0:
                first_turb = turb_dates[0]
                lead_days = (first_turb - peak_date).days
            else:
                lead_days = np.nan

            results.append({
                "peak_date": peak_date,
                "drawdown_start": date,
                "max_drawdown": drawdown[date:].min(),
                "lead_time_days": lead_days,
            })

    return pd.DataFrame(results)


def signal_decay_analysis(
    signal: pd.Series,
    returns: pd.Series,
    horizons: list = None,
) -> pd.DataFrame:
    """
    Information coefficient (rank correlation) at different horizons.

    IC = spearman_corr(signal_t, return_{t to t+h})

    Shows how quickly regime information decays. Higher IC at longer
    horizons means the signal is useful for position-holding, not just
    day-trading.

    Args:
        signal: Daily allocation signal (0 to 1)
        returns: Daily market returns
        horizons: Forward horizons in days

    Returns:
        DataFrame with horizon, ic, t_stat
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 60]

    common = signal.dropna().index.intersection(returns.index)
    sig = signal.loc[common]
    ret = returns.loc[common]

    results = []
    for h in horizons:
        fwd_return = ret.rolling(h).sum().shift(-h)
        valid = sig.notna() & fwd_return.notna()

        if valid.sum() < 30:
            continue

        ic = sig[valid].corr(fwd_return[valid], method="spearman")
        n = valid.sum()
        t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 else 0

        results.append({
            "horizon_days": h,
            "ic": ic,
            "t_stat": t_stat,
            "n_obs": int(n),
        })

    return pd.DataFrame(results)


def calendar_decomposition(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    Monthly and yearly decomposition of excess returns.

    Checks if alpha concentrates in specific months or years.

    Returns:
        Dict with 'monthly' and 'yearly' DataFrames
    """
    common = strategy_returns.index.intersection(benchmark_returns.index)
    excess = strategy_returns.loc[common] - benchmark_returns.loc[common]

    # Monthly decomposition
    monthly_excess = excess.groupby(excess.index.month).mean() * TRADING_DAYS / 12
    monthly_df = pd.DataFrame({
        "month": range(1, 13),
        "month_name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "annualized_excess": monthly_excess.values,
    })

    # Yearly decomposition
    yearly_excess = excess.groupby(excess.index.year).sum()
    yearly_df = pd.DataFrame({
        "year": yearly_excess.index,
        "excess_return": yearly_excess.values,
    })

    return {"monthly": monthly_df, "yearly": yearly_df}


def carry_cost_of_defensiveness(
    positions: pd.Series,
    market_returns: pd.Series,
) -> dict:
    """
    Opportunity cost of being underweight.

    For every day the strategy holds less than 100% equity,
    compute the return that was "left on the table."

    Args:
        positions: Daily allocation (0 to 1)
        market_returns: Daily market returns

    Returns:
        Dict with total carry cost, average daily cost, and breakdown
    """
    common = positions.index.intersection(market_returns.index)
    pos = positions.loc[common]
    ret = market_returns.loc[common]

    underweight = (1.0 - pos).clip(lower=0)
    missed_return = underweight * ret

    # Only count positive missed returns (opportunity cost of missing gains)
    positive_missed = missed_return[missed_return > 0]
    negative_missed = missed_return[missed_return < 0]  # avoided losses

    total_missed_gains = positive_missed.sum()
    total_avoided_losses = abs(negative_missed.sum())
    net_value = total_avoided_losses - total_missed_gains

    return {
        "total_missed_gains": total_missed_gains,
        "total_avoided_losses": total_avoided_losses,
        "net_value_of_defensiveness": net_value,
        "avg_daily_position": pos.mean(),
        "pct_days_underweight": (pos < 1.0).mean(),
        "avg_underweight_when_defensive": underweight[underweight > 0].mean(),
        "annualized_carry_cost": total_missed_gains / (len(common) / TRADING_DAYS),
        "annualized_avoided_loss": total_avoided_losses / (len(common) / TRADING_DAYS),
    }


def drawdown_duration_analysis(
    backtest_results: dict,
) -> pd.DataFrame:
    """
    Compare drawdown recovery times across strategies.

    Args:
        backtest_results: Dict mapping name -> backtest DataFrame

    Returns:
        DataFrame with strategy, max_dd, max_duration, avg_duration, n_drawdowns
    """
    results = []

    for name, bt in backtest_results.items():
        cumret = bt["cumulative_return"]
        running_max = cumret.cummax()
        dd = cumret / running_max - 1

        # Find drawdown periods
        in_dd = dd < 0
        dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
        dd_ends = ~in_dd & in_dd.shift(1, fill_value=False)

        start_dates = dd.index[dd_starts]
        end_dates = dd.index[dd_ends]

        if len(start_dates) > len(end_dates):
            end_dates = end_dates.append(pd.DatetimeIndex([dd.index[-1]]))

        durations = []
        for s, e in zip(start_dates, end_dates[:len(start_dates)]):
            dur = len(dd.loc[s:e])
            durations.append(dur)

        results.append({
            "strategy": name,
            "max_drawdown": dd.min(),
            "max_dd_duration": max(durations) if durations else 0,
            "avg_dd_duration": np.mean(durations) if durations else 0,
            "median_dd_duration": np.median(durations) if durations else 0,
            "n_drawdown_periods": len(durations),
        })

    return pd.DataFrame(results).set_index("strategy")


def crisis_walkthrough(
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    market_returns: pd.Series,
    start: str = "2008-09-01",
    end: str = "2008-12-31",
) -> pd.DataFrame:
    """
    Day-by-day walkthrough of model behavior during a specific period.

    Shows: date, regime prediction, probabilities, allocation, market return,
    strategy return, cumulative market, cumulative strategy.

    Args:
        predictions: Regime predictions with Date column
        backtest: Strategy backtest results
        market_returns: Daily market returns
        start: Start date of walkthrough
        end: End date of walkthrough

    Returns:
        DataFrame with day-by-day detail
    """
    pred = predictions.set_index("Date") if "Date" in predictions.columns else predictions
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    dates = backtest.index[(backtest.index >= start_dt) & (backtest.index <= end_dt)]

    rows = []
    for date in dates:
        row = {"date": date}

        if date in pred.index:
            row["regime"] = {0: "Calm", 1: "Moderate", 2: "Turbulent"}.get(
                int(pred.loc[date, "regime_label"]), "Unknown"
            )
            row["prob_calm"] = pred.loc[date, "prob_calm"]
            row["prob_turbulent"] = pred.loc[date, "prob_turbulent"]

        row["allocation"] = backtest.loc[date, "position"]
        row["market_return"] = market_returns.get(date, np.nan)
        row["strategy_return"] = backtest.loc[date, "net_return"]

        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["cum_market"] = (1 + result["market_return"].fillna(0)).cumprod()
        result["cum_strategy"] = (1 + result["strategy_return"].fillna(0)).cumprod()

    return result

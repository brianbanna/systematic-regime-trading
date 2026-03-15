"""
Regime Characterization Analysis Module

Provides functions for statistical analysis and validation of market regimes:
- Regime statistics computation
- ANOVA and pairwise tests
- Effect sizes and confidence intervals
- Risk metrics computation
- Transition and persistence analysis
- Ground truth validation against drawdown

Used by Notebook 08: Regime Characterization
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from typing import Dict, Tuple, List


def compute_regime_statistics(
    df: pd.DataFrame,
    regime_col: str,
    indicator_cols: List[str],
    regime_labels: Dict[int, str] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive statistics for each regime and indicator.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime column and indicator columns
    regime_col : str
        Name of the regime column
    indicator_cols : list
        List of indicator column names
    regime_labels : dict, optional
        Mapping from regime integers to labels

    Returns
    -------
    pd.DataFrame
        Statistics table with columns: indicator, regime, count, mean, std,
        median, q05, q95
    """
    if regime_labels is None:
        regime_labels = {0: "Calm", 1: "Moderate", 2: "Turbulent"}

    results = []

    for indicator in indicator_cols:
        for regime in sorted(df[regime_col].unique()):
            mask = df[regime_col] == regime
            values = df.loc[mask, indicator]

            results.append(
                {
                    "indicator": indicator,
                    "regime": regime,
                    "regime_label": regime_labels.get(regime, str(regime)),
                    "count": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "median": values.median(),
                    "q05": values.quantile(0.05),
                    "q95": values.quantile(0.95),
                }
            )

    return pd.DataFrame(results)


def run_anova_and_pairwise_tests(
    df: pd.DataFrame, regime_col: str, indicator_cols: List[str], alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run ANOVA and pairwise t-tests for each indicator across regimes.
    Apply FDR correction across all comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime and indicator columns
    regime_col : str
        Name of the regime column
    indicator_cols : list
        List of indicator column names
    alpha : float
        Significance level for FDR correction

    Returns
    -------
    pd.DataFrame
        Test results with columns: indicator, anova_f, anova_p, pair,
        t_stat, t_p, cohens_d
    """
    results = []
    all_p_values = []

    regimes = sorted(df[regime_col].unique())
    pairs = [
        (regimes[i], regimes[j])
        for i in range(len(regimes))
        for j in range(i + 1, len(regimes))
    ]

    # Run tests for each indicator
    for indicator in indicator_cols:
        # ANOVA
        groups = [df.loc[df[regime_col] == r, indicator].values for r in regimes]
        f_stat, p_anova = f_oneway(*groups)

        # Pairwise t-tests
        for r1, r2 in pairs:
            g1 = df.loc[df[regime_col] == r1, indicator].values
            g2 = df.loc[df[regime_col] == r2, indicator].values

            t_stat, p_t = ttest_ind(g1, g2)

            # Cohen's d
            pooled_std = np.sqrt(
                (
                    (len(g1) - 1) * np.std(g1, ddof=1) ** 2
                    + (len(g2) - 1) * np.std(g2, ddof=1) ** 2
                )
                / (len(g1) + len(g2) - 2)
            )
            cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

            results.append(
                {
                    "indicator": indicator,
                    "anova_f": f_stat,
                    "anova_p": p_anova,
                    "pair": f"{r1}_vs_{r2}",
                    "t_stat": t_stat,
                    "t_p": p_t,
                    "cohens_d": cohens_d,
                }
            )
            all_p_values.append(p_t)

    # FDR correction across all pairwise tests
    _, p_fdr, _, _ = multipletests(all_p_values, alpha=alpha, method="fdr_bh")

    # Add corrected p-values to results
    df_results = pd.DataFrame(results)
    df_results["p_fdr"] = p_fdr

    return df_results


def bootstrap_mean_ci(
    data: np.ndarray, n_boot: int = 10000, ci: float = 0.95, random_seed: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : np.ndarray
        Data array
    n_boot : int
        Number of bootstrap samples
    ci : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval
    """
    np.random.seed(random_seed)

    boot_means = []
    n = len(data)

    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(boot_means, alpha / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)

    return lower, upper


def compute_confidence_intervals(
    df: pd.DataFrame,
    regime_col: str,
    indicator_cols: List[str],
    ci: float = 0.95,
    method: str = "bootstrap",
    n_boot: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute confidence intervals for regime-indicator means.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime and indicator columns
    regime_col : str
        Name of the regime column
    indicator_cols : list
        List of indicator column names
    ci : float
        Confidence level
    method : str
        'bootstrap' or 'normal'
    n_boot : int
        Number of bootstrap samples (if method='bootstrap')
    random_seed : int
        Random seed for bootstrap

    Returns
    -------
    pd.DataFrame
        CI results with columns: indicator, regime, mean, ci_low, ci_high
    """
    results = []

    for indicator in indicator_cols:
        for regime in sorted(df[regime_col].unique()):
            mask = df[regime_col] == regime
            values = df.loc[mask, indicator].values

            mean_val = np.mean(values)

            if method == "bootstrap":
                ci_low, ci_high = bootstrap_mean_ci(
                    values, n_boot=n_boot, ci=ci, random_seed=random_seed
                )
            elif method == "normal":
                sem = stats.sem(values)
                ci_margin = sem * stats.t.ppf((1 + ci) / 2, len(values) - 1)
                ci_low = mean_val - ci_margin
                ci_high = mean_val + ci_margin
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append(
                {
                    "indicator": indicator,
                    "regime": regime,
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    return pd.DataFrame(results)


def compute_risk_metrics(
    returns: pd.Series, regimes: pd.Series, rf_rate: float = 0.0
) -> pd.DataFrame:
    """
    Compute risk-return metrics for each regime.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series
    regimes : pd.Series
        Regime labels (aligned with returns)
    rf_rate : float
        Risk-free rate (annualized)

    Returns
    -------
    pd.DataFrame
        Risk metrics with columns: regime, mean_return, volatility,
        sharpe_ratio, max_drawdown, var_95, var_99, cvar_95, cvar_99
    """
    results = []
    rf_daily = rf_rate / 252  # Convert to daily

    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        regime_returns = returns[mask]

        if len(regime_returns) == 0:
            continue

        # Basic metrics
        mean_ret = regime_returns.mean()
        vol = regime_returns.std()
        sharpe = (mean_ret - rf_daily) / vol if vol > 0 else 0

        # Maximum drawdown
        cum_returns = (1 + regime_returns).cumprod()
        running_max = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / running_max) - 1
        max_dd = drawdown.min()

        # Value at Risk (VaR)
        var_95 = np.percentile(regime_returns, 5)
        var_99 = np.percentile(regime_returns, 1)

        # Conditional VaR (CVaR)
        cvar_95 = regime_returns[regime_returns <= var_95].mean()
        cvar_99 = regime_returns[regime_returns <= var_99].mean()

        results.append(
            {
                "regime": regime,
                "count": len(regime_returns),
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "cvar_99": cvar_99,
            }
        )

    return pd.DataFrame(results)


def compute_transition_metrics(regimes: pd.Series) -> Dict:
    """
    Compute regime transition matrix and persistence metrics.

    Parameters
    ----------
    regimes : pd.Series
        Time series of regime labels

    Returns
    -------
    dict
        Dictionary containing:
        - transition_matrix: np.ndarray (n_regimes x n_regimes)
        - persistence: np.ndarray (diagonal probabilities)
        - avg_duration: np.ndarray (average run length per regime)
        - transition_counts: np.ndarray (raw counts)
    """
    regimes_array = regimes.values
    n_regimes = len(np.unique(regimes_array))

    # Transition counts
    transition_counts = np.zeros((n_regimes, n_regimes))
    for i in range(len(regimes_array) - 1):
        curr = int(regimes_array[i])
        next_r = int(regimes_array[i + 1])
        transition_counts[curr, next_r] += 1

    # Transition matrix (probabilities)
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.where(row_sums > 0, transition_counts / row_sums, 0)

    # Persistence (diagonal probabilities)
    persistence = np.diag(transition_matrix)

    # Average duration per regime
    avg_duration = np.zeros(n_regimes)
    for regime in range(n_regimes):
        if persistence[regime] > 0 and persistence[regime] < 1:
            avg_duration[regime] = 1 / (1 - persistence[regime])
        else:
            avg_duration[regime] = np.inf

    # Compute run lengths for validation
    run_lengths = {i: [] for i in range(n_regimes)}
    current_regime = regimes_array[0]
    current_length = 1

    for i in range(1, len(regimes_array)):
        if regimes_array[i] == current_regime:
            current_length += 1
        else:
            run_lengths[int(current_regime)].append(current_length)
            current_regime = regimes_array[i]
            current_length = 1

    # Add last run
    run_lengths[int(current_regime)].append(current_length)

    # Compute actual average durations from runs
    actual_avg_duration = np.zeros(n_regimes)
    for regime in range(n_regimes):
        if len(run_lengths[regime]) > 0:
            actual_avg_duration[regime] = np.mean(run_lengths[regime])

    return {
        "transition_matrix": transition_matrix,
        "transition_counts": transition_counts,
        "persistence": persistence,
        "avg_duration": avg_duration,
        "actual_avg_duration": actual_avg_duration,
        "run_lengths": run_lengths,
    }


def compute_drawdown_ground_truth(
    market_prices: pd.Series, drawdown_thresholds: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Compute market drawdown and classify into ground truth states.

    Parameters
    ----------
    market_prices : pd.Series
        Time series of market prices (e.g., average Adj Close)
    drawdown_thresholds : dict, optional
        Thresholds for classification, e.g.,
        {'bear': -0.20, 'correction': -0.10}

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, drawdown, market_state (0=bull, 1=correction, 2=bear)
    """
    if drawdown_thresholds is None:
        drawdown_thresholds = {"bear": -0.20, "correction": -0.10}

    # Compute drawdown using expanding window (all-time peak)
    running_max = market_prices.expanding(min_periods=1).max()
    drawdown = (market_prices / running_max) - 1.0

    # Classify into market states
    market_state = np.zeros(len(drawdown), dtype=int)
    market_state[drawdown < drawdown_thresholds["bear"]] = 2  # Bear market
    market_state[
        (drawdown >= drawdown_thresholds["bear"])
        & (drawdown < drawdown_thresholds["correction"])
    ] = 1  # Correction
    # market_state = 0 is bull market (default)

    return pd.DataFrame(
        {
            "date": drawdown.index,
            "drawdown": drawdown.values,
            "market_state": market_state,
        }
    ).set_index("date")


def validate_regimes_vs_drawdown(regimes: pd.Series, market_state: pd.Series) -> Dict:
    """
    Validate regime predictions against ground truth market states.

    Parameters
    ----------
    regimes : pd.Series
        Predicted regimes (0=Calm, 1=Moderate, 2=Turbulent)
    market_state : pd.Series
        Ground truth market states (0=Bull, 1=Correction, 2=Bear)

    Returns
    -------
    dict
        Validation metrics including precision, recall, F1 for Turbulent detection
    """
    # Align series
    common_idx = regimes.index.intersection(market_state.index)
    regimes_aligned = regimes.loc[common_idx]
    market_aligned = market_state.loc[common_idx]

    # Define "crisis" as Bear or Correction (state 1 or 2)
    true_crisis = (market_aligned >= 1).astype(int)
    pred_crisis = (regimes_aligned == 2).astype(int)  # Turbulent

    # Compute metrics
    tp = ((pred_crisis == 1) & (true_crisis == 1)).sum()
    fp = ((pred_crisis == 1) & (true_crisis == 0)).sum()
    fn = ((pred_crisis == 0) & (true_crisis == 1)).sum()
    tn = ((pred_crisis == 0) & (true_crisis == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Bear-specific metrics
    true_bear = (market_aligned == 2).astype(int)
    bear_tp = ((pred_crisis == 1) & (true_bear == 1)).sum()
    bear_fn = ((pred_crisis == 0) & (true_bear == 1)).sum()

    bear_recall = bear_tp / (bear_tp + bear_fn) if (bear_tp + bear_fn) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "bear_recall": bear_recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_crisis_days": true_crisis.sum(),
        "n_bear_days": true_bear.sum(),
        "n_turbulent_days": pred_crisis.sum(),
    }

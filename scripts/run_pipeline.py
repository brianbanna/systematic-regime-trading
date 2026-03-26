"""
End-to-end pipeline: data -> features -> regimes -> signals -> backtest -> results.

Runs the full systematic regime trading system on real data and saves
all results to results/ directory.

Usage:
    python scripts/run_pipeline.py
    make run
"""

import pandas as pd
import numpy as np
import logging
import time
import warnings
import subprocess
import hashlib
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress convergence warnings from HMM fitting
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from systematic_regime_trading.utils.config import load_config, get_path
from systematic_regime_trading.data.storage import load_parquet, load_parquet_subdir, save_parquet_subdir
from systematic_regime_trading.data.loaders import download_spy
from systematic_regime_trading.features.volatility import (
    compute_daily_returns_unified,
    compute_market_volatility_index_unified,
)
from systematic_regime_trading.features.indicators import (
    compute_market_direction_unified,
    compute_market_volume_unified,
    compute_market_breadth_unified,
    compute_market_atr_unified,
    compute_market_correlation_dynamic,
    compute_mood_index,
)
from systematic_regime_trading.models.hmm import HMMRegimeDetector
from systematic_regime_trading.models.garch import GARCHRegimeDetector
from systematic_regime_trading.models.kmeans import KMeansRegimeDetector
from systematic_regime_trading.models.gmm import GMMRegimeDetector
from systematic_regime_trading.models.markov_switching import MarkovSwitchingRegimeDetector
from systematic_regime_trading.models.ensemble import EnsembleRegimeDetector
from systematic_regime_trading.signals.generator import generate_all_signals
from systematic_regime_trading.backtest.engine import run_backtest, run_all_backtests
from systematic_regime_trading.backtest.sensitivity import cost_sensitivity, find_breakeven_cost
from systematic_regime_trading.backtest.trivial_signals import (
    sma_crossover_signal, vix_threshold_signal, vol_managed_signal,
)
from systematic_regime_trading.evaluation.metrics import compute_metrics
from systematic_regime_trading.evaluation.report import (
    performance_table, save_performance_table, format_performance_table,
)
from systematic_regime_trading.evaluation.regime_perf import (
    performance_by_regime, crisis_performance,
)
from systematic_regime_trading.evaluation.significance import (
    bootstrap_sharpe_ci, sharpe_difference_test, bonferroni_correction, hac_sharpe_se,
)
from systematic_regime_trading.evaluation.factor_regression import (
    download_ff_factors, run_factor_regressions,
)
from systematic_regime_trading.evaluation.prediction_quality import (
    regime_prediction_accuracy, regime_lead_time, signal_decay_analysis,
    calendar_decomposition, carry_cost_of_defensiveness,
    drawdown_duration_analysis, crisis_walkthrough,
)


RESULTS_DIR = get_path("results")
BACKTEST_DIR = RESULTS_DIR / "backtest_results"


def setup_results_dirs():
    """Create results directory structure."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)


def step1_load_data():
    """Load cleaned equity data, SPY benchmark, and auxiliary data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    equity_data = load_parquet("equity_data")
    logger.info(f"Equity data: {len(equity_data):,} rows, "
                f"{equity_data['ticker'].nunique()} tickers")

    data_cfg = load_config("data")
    start = data_cfg["universe"]["date_range"]["start"]
    end = data_cfg["universe"]["date_range"]["end"]

    # Load or download SPY benchmark
    try:
        spy_data = load_parquet_subdir("spy", "auxiliary")
        logger.info(f"SPY: {len(spy_data):,} rows (cached)")
    except FileNotFoundError:
        logger.info("SPY not cached, downloading...")
        spy_data = download_spy(start, end)
        save_parquet_subdir(spy_data, "spy", "auxiliary")

    # Load auxiliary data
    try:
        vix_data = load_parquet_subdir("vix", "auxiliary")
        logger.info(f"VIX: {len(vix_data):,} rows")
    except FileNotFoundError:
        vix_data = None
        logger.warning("VIX data not found")

    try:
        tlt_data = load_parquet_subdir("tlt_bonds", "auxiliary")
        logger.info(f"TLT: {len(tlt_data):,} rows")
    except FileNotFoundError:
        tlt_data = None
        logger.warning("TLT data not found")

    return equity_data, spy_data, vix_data, tlt_data


def step2_compute_features(equity_data, spy_data):
    """Compute market-level features from raw data. Use SPY as market return."""
    logger.info("=" * 60)
    logger.info("STEP 2: Computing features")
    logger.info("=" * 60)

    features_cfg = load_config("features")

    # Daily returns for feature computation (cross-sectional indicators)
    logger.info("Computing daily returns...")
    df_ret = compute_daily_returns_unified(equity_data)

    # Market indicators (from cross-section of stocks)
    logger.info("Computing market indicators...")
    vol_df = compute_market_volatility_index_unified(df_ret).set_index("Date")
    dir_df = compute_market_direction_unified(df_ret)
    vol_obj_df = compute_market_volume_unified(df_ret)
    brd_df = compute_market_breadth_unified(df_ret)
    atr_df = compute_market_atr_unified(df_ret)

    logger.info("Computing market correlation (this may take a minute)...")
    corr_cfg = features_cfg["indicators"]
    corr_df = compute_market_correlation_dynamic(
        df=df_ret,
        window=corr_cfg["correlation_window"],
        sample_size=min(corr_cfg["correlation_sample_size"],
                        equity_data["ticker"].nunique()),
    )

    # VIX term structure (contango/backwardation)
    logger.info("Adding VIX term structure feature...")
    try:
        vix_data = load_parquet_subdir("vix", "auxiliary")
        if "VIX" in vix_data.columns and "VIX3M" in vix_data.columns:
            vix_ts = vix_data.set_index("Date")[["VIX", "VIX3M"]].copy()
            vix_ts["vix_term_structure"] = vix_ts["VIX"] / vix_ts["VIX3M"]
            vix_term = vix_ts[["vix_term_structure"]]
            logger.info(f"VIX term structure: {len(vix_term)} days, "
                        f"mean={vix_term['vix_term_structure'].mean():.3f}")
        else:
            vix_term = None
            logger.warning("VIX3M not available, skipping VIX term structure")
    except Exception:
        vix_term = None
        logger.warning("Could not compute VIX term structure")

    # Mood index (z-score standardization)
    logger.info("Computing mood index...")
    indicators_dict = {
        "market_volatility": vol_df,
        "market_direction": dir_df,
        "market_volume": vol_obj_df,
        "market_breadth": brd_df,
        "market_atr": atr_df,
        "market_correlation": corr_df,
    }
    if vix_term is not None:
        indicators_dict["vix_term_structure"] = vix_term

    mood_df = compute_mood_index(
        indicators_dict,
        window=corr_cfg["standardization_window"],
    )

    # SPY as market return for backtesting (not equal-weight NASDAQ)
    spy_returns = spy_data.set_index("Date")["spy_return"].dropna()
    spy_returns.name = "market_return"
    logger.info(f"SPY benchmark: {len(spy_returns)} days, "
                f"annualized return={spy_returns.mean()*252:.1%}, "
                f"vol={spy_returns.std()*np.sqrt(252):.1%}")

    logger.info(f"Features: {len(mood_df)} days, {mood_df.shape[1]} indicators")

    return df_ret, mood_df, spy_returns, vol_df


def step3_walk_forward_regimes(mood_df, market_return):
    """Run walk-forward regime detection with all 3 models + ensemble."""
    logger.info("=" * 60)
    logger.info("STEP 3: Walk-forward regime detection")
    logger.info("=" * 60)

    models_cfg = load_config("models")
    wf_cfg = models_cfg["walk_forward"]

    min_train_days = wf_cfg["min_train_years"] * 252
    test_window = wf_cfg["test_window_days"]
    step_days = wf_cfg["step_days"]

    # Prepare feature data with Date column
    feature_df = mood_df.copy()
    feature_df["Date"] = feature_df.index
    feature_df = feature_df.dropna(subset=["market_volatility"]).reset_index(drop=True)

    n_total = len(feature_df)
    logger.info(f"Feature data: {n_total} days ({feature_df['Date'].min()} to {feature_df['Date'].max()})")

    # Also prepare market return aligned with features
    market_ret_aligned = market_return.loc[
        market_return.index.isin(feature_df["Date"])
    ].copy()

    all_predictions = []
    window_id = 0
    train_end_idx = min_train_days

    while train_end_idx + step_days <= n_total:
        test_end_idx = min(train_end_idx + test_window, n_total)
        train_data = feature_df.iloc[:train_end_idx]
        test_data = feature_df.iloc[train_end_idx:test_end_idx]

        if len(test_data) == 0:
            break

        train_dates = f"{train_data['Date'].iloc[0].date()} to {train_data['Date'].iloc[-1].date()}"
        test_dates = f"{test_data['Date'].iloc[0].date()} to {test_data['Date'].iloc[-1].date()}"

        if window_id % 5 == 0:
            logger.info(f"Window {window_id}: train [{train_dates}] -> test [{test_dates}] ({len(test_data)} days)")

        # --- HMM ---
        hmm_probs = _fit_predict_hmm(train_data, test_data, models_cfg["hmm"])

        # --- GARCH ---
        garch_probs = _fit_predict_garch(
            train_data, test_data, market_ret_aligned, models_cfg["garch"],
        )

        # --- KMeans ---
        kmeans_probs = _fit_predict_kmeans(train_data, test_data, models_cfg["kmeans"])

        # --- GMM ---
        gmm_probs = _fit_predict_gmm(train_data, test_data, models_cfg.get("gmm", {}))

        # --- Markov-Switching ---
        ms_probs = _fit_predict_markov_switching(
            train_data, test_data, market_ret_aligned, models_cfg.get("markov_switching", {}),
        )

        # --- 5-model Ensemble ---
        ensemble = EnsembleRegimeDetector(
            weights={"hmm": 0.25, "garch": 0.25, "kmeans": 0.15, "gmm": 0.20, "markov_switching": 0.15},
            config=models_cfg["ensemble"],
        )
        model_probs = {
            "hmm": hmm_probs,
            "garch": garch_probs,
            "kmeans": kmeans_probs,
            "gmm": gmm_probs,
            "markov_switching": ms_probs,
        }
        combined_probs = ensemble.combine(model_probs)
        ensemble_labels = np.argmax(combined_probs, axis=1)

        window_result = pd.DataFrame({
            "Date": test_data["Date"].values,
            "regime_label": ensemble_labels,
            "prob_calm": combined_probs[:, 0],
            "prob_moderate": combined_probs[:, 1],
            "prob_turbulent": combined_probs[:, 2],
            "hmm_label": np.argmax(hmm_probs, axis=1),
            "garch_label": np.argmax(garch_probs, axis=1),
            "kmeans_label": np.argmax(kmeans_probs, axis=1),
            "gmm_label": np.argmax(gmm_probs, axis=1),
            "ms_label": np.argmax(ms_probs, axis=1),
            "window_id": window_id,
        })

        all_predictions.append(window_result)
        window_id += 1
        train_end_idx += step_days

    # Combine and deduplicate (keep latest prediction for overlapping dates)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(["Date", "window_id"])
    predictions = predictions.drop_duplicates(subset="Date", keep="last")
    predictions = predictions.sort_values("Date").reset_index(drop=True)

    logger.info(
        f"Walk-forward complete: {window_id} windows, "
        f"{len(predictions)} OOS predictions "
        f"({predictions['Date'].min().date()} to {predictions['Date'].max().date()})"
    )

    # Regime distribution
    dist = predictions["regime_label"].value_counts().sort_index()
    for regime, count in dist.items():
        pct = count / len(predictions) * 100
        name = {0: "Calm", 1: "Moderate", 2: "Turbulent"}.get(regime, f"R{regime}")
        logger.info(f"  {name}: {count} days ({pct:.1f}%)")

    return predictions


def _fit_predict_hmm(train_data, test_data, hmm_cfg):
    """Fit HMM on train, predict probabilities on test."""
    try:
        detector = HMMRegimeDetector(hmm_cfg)
        train_feature = train_data["market_volatility"].dropna().values
        test_feature = test_data["market_volatility"].dropna().values

        detector.fit(train_feature)
        probs = detector.predict_proba(test_feature)

        # Ensure 3 columns even if n_states > 3
        if probs.shape[1] != 3:
            probs = probs[:, :3]  # Already aggregated by predict_proba

        return probs
    except Exception as e:
        logger.warning(f"HMM failed: {e}, using uniform probs")
        return np.full((len(test_data), 3), 1/3)


def _fit_predict_garch(train_data, test_data, market_return, garch_cfg):
    """Fit GARCH on train returns, predict probabilities on test."""
    try:
        train_dates = train_data["Date"].values
        test_dates = test_data["Date"].values

        train_ret = market_return.loc[market_return.index.isin(train_dates)]
        test_ret = market_return.loc[market_return.index.isin(test_dates)]

        if len(train_ret) < 100:
            return np.full((len(test_data), 3), 1/3)

        detector = GARCHRegimeDetector(garch_cfg)
        detector.fit(train_ret)

        # Use the fitted model's conditional volatility approach on test data
        probs = detector.predict_proba(test_ret)

        return probs[:len(test_data)]
    except Exception as e:
        logger.warning(f"GARCH failed: {e}, using uniform probs")
        return np.full((len(test_data), 3), 1/3)


def _fit_predict_kmeans(train_data, test_data, kmeans_cfg):
    """Fit KMeans on train features, predict probabilities on test."""
    try:
        feature_cols = ["market_volatility", "market_volume"]
        available = [c for c in feature_cols if c in train_data.columns]

        if len(available) == 0:
            return np.full((len(test_data), 3), 1/3)

        train_X = train_data[available].dropna()
        test_X = test_data[available].dropna()

        if len(train_X) < 50 or len(test_X) == 0:
            return np.full((len(test_data), 3), 1/3)

        detector = KMeansRegimeDetector(kmeans_cfg)
        detector.fit(train_X)
        probs = detector.predict_proba(test_X)

        # Pad if test_X had fewer rows due to NaN removal
        if len(probs) < len(test_data):
            padded = np.full((len(test_data), 3), 1/3)
            padded[:len(probs)] = probs
            return padded

        return probs[:len(test_data)]
    except Exception as e:
        logger.warning(f"KMeans failed: {e}, using uniform probs")
        return np.full((len(test_data), 3), 1/3)


def _fit_predict_gmm(train_data, test_data, gmm_cfg):
    """Fit GMM on train features, predict probabilities on test."""
    try:
        feature_cols = ["market_volatility", "market_volume"]
        available = [c for c in feature_cols if c in train_data.columns]

        if len(available) == 0:
            return np.full((len(test_data), 3), 1/3)

        train_X = train_data[available].dropna()
        test_X = test_data[available].dropna()

        if len(train_X) < 50 or len(test_X) == 0:
            return np.full((len(test_data), 3), 1/3)

        detector = GMMRegimeDetector(gmm_cfg)
        detector.fit(train_X)
        probs = detector.predict_proba(test_X)

        if len(probs) < len(test_data):
            padded = np.full((len(test_data), 3), 1/3)
            padded[:len(probs)] = probs
            return padded

        return probs[:len(test_data)]
    except Exception as e:
        logger.warning(f"GMM failed: {e}, using uniform probs")
        return np.full((len(test_data), 3), 1/3)


def _fit_predict_markov_switching(train_data, test_data, market_return, ms_cfg):
    """Fit Markov-Switching on train returns, predict probabilities on test."""
    try:
        train_dates = train_data["Date"].values
        test_dates = test_data["Date"].values

        train_ret = market_return.loc[market_return.index.isin(train_dates)]
        test_ret = market_return.loc[market_return.index.isin(test_dates)]

        if len(train_ret) < 200:
            return np.full((len(test_data), 3), 1/3)

        detector = MarkovSwitchingRegimeDetector(ms_cfg)
        detector.fit(train_ret)

        if not detector.is_fitted_:
            return np.full((len(test_data), 3), 1/3)

        probs = detector.predict_proba(test_ret)
        return probs[:len(test_data)]
    except Exception as e:
        logger.warning(f"MarkovSwitching failed: {e}, using uniform probs")
        return np.full((len(test_data), 3), 1/3)


def step4_generate_signals(predictions, market_return):
    """Convert regime predictions to allocation signals."""
    logger.info("=" * 60)
    logger.info("STEP 4: Generating strategy signals")
    logger.info("=" * 60)

    # Build regime probs DataFrame indexed by Date
    regime_probs = predictions.set_index("Date")[
        ["prob_calm", "prob_moderate", "prob_turbulent"]
    ]
    regime_labels = predictions.set_index("Date")["regime_label"]

    # Align market returns
    common_dates = regime_probs.index.intersection(market_return.index)
    mkt_ret = market_return.loc[common_dates]

    signals = generate_all_signals(
        regime_probs=regime_probs.loc[common_dates],
        regime_labels=regime_labels.loc[common_dates],
        market_returns=mkt_ret,
    )

    logger.info(f"Signals generated: {list(signals.columns)}")
    for col in signals.columns:
        valid = signals[col].dropna()
        logger.info(f"  {col}: {len(valid)} valid days, mean={valid.mean():.3f}")

    return signals, mkt_ret, regime_probs, regime_labels


def step5_run_backtests(signals, market_return, spy_data, tlt_data, vix_data):
    """Run backtests for all strategies, trivial benchmarks, and standard benchmarks."""
    logger.info("=" * 60)
    logger.info("STEP 5: Running backtests")
    logger.info("=" * 60)

    # Prepare bond returns
    bond_returns = None
    if tlt_data is not None:
        if "Adj Close" in tlt_data.columns:
            bond_returns = tlt_data["Adj Close"].pct_change().dropna()
            bond_returns.name = "bond_return"

    config = load_config("backtest")
    results = run_all_backtests(signals, market_return, bond_returns, config)

    # Trivial signal benchmarks (to prove the ensemble adds value)
    cost_config = config["execution"]
    constraint_config = config["constraints"]

    # 200-day SMA crossover
    logger.info("Backtesting: sma_200 (trivial)")
    spy_prices = spy_data.set_index("Date")["Adj Close"]
    sma_signal = sma_crossover_signal(spy_prices, window=200)
    sma_signal = sma_signal.shift(1)  # execution lag
    results["sma_200"] = run_backtest(
        market_return, sma_signal, cost_config, constraint_config,
    )

    # VIX > 20 threshold
    if vix_data is not None and "VIX" in vix_data.columns:
        logger.info("Backtesting: vix_20 (trivial)")
        vix_series = vix_data.set_index("Date")["VIX"]
        vix_signal = vix_threshold_signal(vix_series, threshold=20.0)
        vix_signal = vix_signal.shift(1)  # execution lag
        results["vix_20"] = run_backtest(
            market_return, vix_signal, cost_config, constraint_config,
        )

    # Vol-managed (no regime detection)
    logger.info("Backtesting: vol_managed (trivial)")
    vm_signal = vol_managed_signal(market_return, vol_target=0.10, lookback=63)
    results["vol_managed"] = run_backtest(
        market_return, vm_signal, cost_config, constraint_config,
    )

    # Save individual backtest results
    for name, bt in results.items():
        bt.to_parquet(BACKTEST_DIR / f"{name}.parquet")

    logger.info(f"Backtests complete: {len(results)} strategies")
    for name, bt in results.items():
        cum = bt["cumulative_return"].iloc[-1]
        max_dd = bt["drawdown"].min()
        logger.info(f"  {name}: cum_return={cum:.4f}, max_dd={max_dd:.4f}")

    return results


def step6_evaluate(backtest_results, predictions, market_return):
    """Compute all performance metrics and save results."""
    logger.info("=" * 60)
    logger.info("STEP 6: Performance evaluation")
    logger.info("=" * 60)

    eval_cfg = load_config("evaluation")

    # Master performance table
    logger.info("Computing performance table with bootstrap CIs...")
    table = performance_table(
        backtest_results,
        risk_free_rate=load_config("backtest")["risk_free_rate"]["fallback"],
        include_ci=True,
        n_bootstrap=eval_cfg["bootstrap"]["n_samples"],
    )
    save_performance_table(table)

    print("\n" + "=" * 80)
    print("PERFORMANCE TABLE")
    print("=" * 80)
    print(format_performance_table(table))
    print("=" * 80 + "\n")

    # Regime-conditional performance
    regime_labels = predictions.set_index("Date")["regime_label"]
    regime_results = {}
    for name, bt in backtest_results.items():
        common = bt.index.intersection(regime_labels.index)
        if len(common) > 100:
            rp = performance_by_regime(bt["net_return"].loc[common], regime_labels.loc[common])
            regime_results[name] = rp

    if regime_results:
        # Save best strategy's regime performance
        best_strategy = table.index[0]
        if best_strategy in regime_results:
            regime_results[best_strategy].to_csv(RESULTS_DIR / "regime_performance.csv")
            logger.info(f"Regime performance for {best_strategy}:")
            print(regime_results[best_strategy].to_string())

    # Crisis performance
    crisis_results = {}
    for name, bt in backtest_results.items():
        cp = crisis_performance(bt["net_return"], eval_cfg["crisis_periods"])
        crisis_results[name] = cp

    if crisis_results:
        best_strategy = table.index[0]
        if best_strategy in crisis_results:
            crisis_results[best_strategy].to_csv(RESULTS_DIR / "crisis_performance.csv")
            print(f"\nCrisis performance ({best_strategy}):")
            print(crisis_results[best_strategy].to_string())

    # Cost sensitivity for each strategy (not benchmarks)
    logger.info("Running cost sensitivity analysis...")
    strategy_names = [n for n in backtest_results if n not in ["buy_and_hold", "sixty_forty", "risk_parity"]]
    sensitivity_results = {}
    breakeven_costs = {}

    for name in strategy_names:
        bt = backtest_results[name]
        # Reconstruct signal from position
        signal = bt["position"]
        sens = cost_sensitivity(
            market_return, signal,
            cost_levels_bps=[0, 2, 5, 10, 15, 20, 30, 50],
        )
        sensitivity_results[name] = sens
        be = find_breakeven_cost(sens, metric="sharpe", threshold=0.0)
        breakeven_costs[name] = be

    # Save cost sensitivity
    all_sens = []
    for name, df in sensitivity_results.items():
        df = df.copy()
        df["strategy"] = name
        all_sens.append(df)
    if all_sens:
        pd.concat(all_sens).to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)

    print("\nBreakeven costs (bps where Sharpe drops to 0):")
    for name, be in breakeven_costs.items():
        print(f"  {name}: {be:.1f} bps")

    # Save Sharpe CIs (bootstrap + HAC-adjusted)
    ci_rows = []
    for name, bt in backtest_results.items():
        try:
            sharpe, lower, upper = bootstrap_sharpe_ci(bt["net_return"], n_bootstrap=1000)
            _, hac_se, hac_lower, hac_upper = hac_sharpe_se(bt["net_return"])
            ci_rows.append({
                "strategy": name, "sharpe": sharpe,
                "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                "hac_se": hac_se, "hac_ci_lower": hac_lower, "hac_ci_upper": hac_upper,
            })
        except Exception:
            pass
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / "sharpe_ci.csv", index=False)

    # Sharpe difference tests vs buy-and-hold
    logger.info("Running Sharpe difference tests vs buy-and-hold...")
    if "buy_and_hold" in backtest_results:
        bench_ret = backtest_results["buy_and_hold"]["net_return"]
        raw_pvalues = {}
        diff_rows = []
        for name, bt in backtest_results.items():
            if name == "buy_and_hold":
                continue
            diff, pval = sharpe_difference_test(bt["net_return"], bench_ret, n_bootstrap=5000)
            raw_pvalues[name] = pval
            diff_rows.append({"strategy": name, "sharpe_diff": diff, "p_value_raw": pval})

        # Bonferroni correction
        corrected = bonferroni_correction(raw_pvalues)
        for row in diff_rows:
            row["p_value_bonferroni"] = corrected[row["strategy"]]

        diff_df = pd.DataFrame(diff_rows)
        diff_df.to_csv(RESULTS_DIR / "sharpe_tests.csv", index=False)

        print("\nSharpe Difference Tests vs Buy-and-Hold:")
        for _, row in diff_df.iterrows():
            sig = "*" if row["p_value_bonferroni"] < 0.05 else ""
            print(f"  {row['strategy']}: diff={row['sharpe_diff']:+.3f}, "
                  f"p={row['p_value_raw']:.3f}, p_bonf={row['p_value_bonferroni']:.3f} {sig}")

    # Fama-French factor regression
    logger.info("Running Fama-French factor regressions...")
    try:
        ff_factors = download_ff_factors()
        ff_results = run_factor_regressions(backtest_results, ff_factors)
        ff_results.to_csv(RESULTS_DIR / "factor_regression.csv")

        print("\nFama-French Alpha (annualized):")
        for name in ff_results.index:
            alpha = ff_results.loc[name, "alpha_annual"]
            tstat = ff_results.loc[name, "alpha_tstat"]
            pval = ff_results.loc[name, "alpha_pvalue"]
            r2 = ff_results.loc[name, "r_squared"]
            sig = "*" if pval < 0.05 else ""
            print(f"  {name}: alpha={alpha:.2%}, t={tstat:.2f}, p={pval:.3f}, R2={r2:.3f} {sig}")
    except Exception as e:
        logger.warning(f"Factor regression failed: {e}")

    # === NEW ANALYSIS ===

    # Regime prediction accuracy
    logger.info("Regime prediction accuracy...")
    regime_labels_series = predictions.set_index("Date")["regime_label"]
    pred_acc = regime_prediction_accuracy(market_return, regime_labels_series)
    pred_acc.to_csv(RESULTS_DIR / "prediction_accuracy.csv", index=False)
    print("\nRegime Prediction Accuracy (realized vol by predicted regime):")
    print(pred_acc.to_string(index=False))

    # Regime lead time
    logger.info("Regime prediction lead time...")
    lead_time = regime_lead_time(market_return, regime_labels_series)
    if len(lead_time) > 0:
        lead_time.to_csv(RESULTS_DIR / "regime_lead_time.csv", index=False)
        median_lead = lead_time["lead_time_days"].dropna().median()
        print(f"\nMedian regime lead time: {median_lead:.0f} days before drawdown")

    # Signal decay / IC
    if "regime_momentum" in backtest_results:
        logger.info("Signal decay analysis...")
        bt_rm = backtest_results["regime_momentum"]
        ic_df = signal_decay_analysis(bt_rm["position"], market_return)
        ic_df.to_csv(RESULTS_DIR / "signal_decay.csv", index=False)
        print("\nSignal Decay (Information Coefficient by horizon):")
        print(ic_df.to_string(index=False))

    # Calendar effects
    if "regime_momentum" in backtest_results and "buy_and_hold" in backtest_results:
        logger.info("Calendar decomposition...")
        cal = calendar_decomposition(
            backtest_results["regime_momentum"]["net_return"],
            backtest_results["buy_and_hold"]["net_return"],
        )
        cal["monthly"].to_csv(RESULTS_DIR / "calendar_monthly.csv", index=False)
        cal["yearly"].to_csv(RESULTS_DIR / "calendar_yearly.csv", index=False)

    # Carry cost of defensiveness
    if "regime_momentum" in backtest_results:
        logger.info("Carry cost analysis...")
        carry = carry_cost_of_defensiveness(
            backtest_results["regime_momentum"]["position"],
            market_return,
        )
        print(f"\nCarry Cost of Defensiveness:")
        print(f"  Missed gains: {carry['total_missed_gains']:.2%}")
        print(f"  Avoided losses: {carry['total_avoided_losses']:.2%}")
        print(f"  Net value: {carry['net_value_of_defensiveness']:.2%}")
        print(f"  Avg position: {carry['avg_daily_position']:.1%}")
        pd.DataFrame([carry]).to_csv(RESULTS_DIR / "carry_cost.csv", index=False)

    # Drawdown duration analysis
    logger.info("Drawdown duration analysis...")
    dd_dur = drawdown_duration_analysis(backtest_results)
    dd_dur.to_csv(RESULTS_DIR / "drawdown_durations.csv")
    print("\nDrawdown Duration (trading days):")
    print(dd_dur[["max_drawdown", "max_dd_duration", "avg_dd_duration"]].to_string())

    # Crisis walkthrough (2008)
    if "regime_momentum" in backtest_results:
        logger.info("Crisis walkthrough (2008)...")
        walkthrough = crisis_walkthrough(
            predictions, backtest_results["regime_momentum"],
            market_return, start="2008-09-01", end="2008-12-31",
        )
        walkthrough.to_csv(RESULTS_DIR / "crisis_walkthrough_2008.csv", index=False)
        if len(walkthrough) > 0:
            print(f"\n2008 Crisis Walkthrough ({len(walkthrough)} days):")
            print(f"  Market: {walkthrough['cum_market'].iloc[-1] - 1:.1%}")
            print(f"  Strategy: {walkthrough['cum_strategy'].iloc[-1] - 1:.1%}")

    return table


def save_run_report(table, elapsed, predictions):
    """Save reproducibility report with timestamp, git hash, versions, and key metrics."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(get_path(".")),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    # Data checksum
    try:
        data_path = get_path("data/processed/equity_data.parquet")
        checksum = hashlib.md5(data_path.read_bytes()).hexdigest()
    except Exception:
        checksum = "unknown"

    lines = [
        f"Run Report",
        f"{'=' * 60}",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Git hash: {git_hash}",
        f"Python: {sys.version.split()[0]}",
        f"Elapsed: {elapsed:.1f}s",
        f"Data checksum (equity_data.parquet): {checksum}",
        f"",
        f"Key package versions:",
        f"  pandas={pd.__version__}, numpy={np.__version__}",
        f"",
        f"OOS predictions: {len(predictions)} days",
        f"  Date range: {predictions['Date'].min()} to {predictions['Date'].max()}",
        f"",
        f"Key Metrics:",
    ]

    for strategy in table.index:
        sharpe = table.loc[strategy, "Sharpe"]
        cagr = table.loc[strategy, "CAGR"]
        max_dd = table.loc[strategy, "Max_DD"]
        lines.append(f"  {strategy}: Sharpe={sharpe:.3f}, CAGR={cagr:.2%}, MaxDD={max_dd:.2%}")

    report = "\n".join(lines)
    report_path = RESULTS_DIR / "run_report.txt"
    report_path.write_text(report)
    logger.info(f"Run report saved to {report_path}")


def main():
    start_time = time.time()

    setup_results_dirs()

    # Step 1: Load data
    equity_data, spy_data, vix_data, tlt_data = step1_load_data()

    # Step 2: Compute features (SPY as benchmark, not equal-weight NASDAQ)
    df_ret, mood_df, market_return, vol_df = step2_compute_features(equity_data, spy_data)

    # Step 3: Walk-forward regime detection
    predictions = step3_walk_forward_regimes(mood_df, market_return)
    predictions.to_parquet(RESULTS_DIR / "regime_predictions.parquet")

    # Step 4: Generate signals
    signals, mkt_ret_aligned, regime_probs, regime_labels = step4_generate_signals(
        predictions, market_return,
    )
    signals.to_parquet(RESULTS_DIR / "strategy_signals.parquet")

    # Step 5: Backtests
    backtest_results = step5_run_backtests(signals, mkt_ret_aligned, spy_data, tlt_data, vix_data)

    # Step 6: Evaluate
    table = step6_evaluate(backtest_results, predictions, mkt_ret_aligned)

    elapsed = time.time() - start_time

    # Save run report for reproducibility
    save_run_report(table, elapsed, predictions)

    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

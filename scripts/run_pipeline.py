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
from pathlib import Path

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
from systematic_regime_trading.data.storage import load_parquet, load_parquet_subdir
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
from systematic_regime_trading.models.ensemble import EnsembleRegimeDetector
from systematic_regime_trading.signals.generator import generate_all_signals
from systematic_regime_trading.backtest.engine import run_backtest, run_all_backtests
from systematic_regime_trading.backtest.sensitivity import cost_sensitivity, find_breakeven_cost
from systematic_regime_trading.evaluation.metrics import compute_metrics
from systematic_regime_trading.evaluation.report import (
    performance_table, save_performance_table, format_performance_table,
)
from systematic_regime_trading.evaluation.regime_perf import (
    performance_by_regime, crisis_performance,
)
from systematic_regime_trading.evaluation.significance import bootstrap_sharpe_ci


RESULTS_DIR = get_path("results")
BACKTEST_DIR = RESULTS_DIR / "backtest_results"


def setup_results_dirs():
    """Create results directory structure."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)


def step1_load_data():
    """Load cleaned equity data and auxiliary data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    equity_data = load_parquet("equity_data")
    logger.info(f"Equity data: {len(equity_data):,} rows, "
                f"{equity_data['ticker'].nunique()} tickers")

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

    return equity_data, vix_data, tlt_data


def step2_compute_features(equity_data):
    """Compute market-level features from raw data."""
    logger.info("=" * 60)
    logger.info("STEP 2: Computing features")
    logger.info("=" * 60)

    features_cfg = load_config("features")

    # Daily returns
    logger.info("Computing daily returns...")
    df_ret = compute_daily_returns_unified(equity_data)

    # Market indicators
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
    mood_df = compute_mood_index(
        indicators_dict,
        window=corr_cfg["standardization_window"],
    )

    # Equal-weight market return for backtesting
    market_return = df_ret.groupby("Date")["simple_return"].mean()
    market_return.name = "market_return"

    logger.info(f"Features: {len(mood_df)} days, {mood_df.shape[1]} indicators")
    logger.info(f"Market return: {len(market_return)} days")

    return df_ret, mood_df, market_return, vol_df


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

        # --- Ensemble ---
        ensemble = EnsembleRegimeDetector(config=models_cfg["ensemble"])
        model_probs = {
            "hmm": hmm_probs,
            "garch": garch_probs,
            "kmeans": kmeans_probs,
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


def step5_run_backtests(signals, market_return, tlt_data):
    """Run backtests for all strategies + benchmarks."""
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

    # Save Sharpe CIs
    ci_rows = []
    for name, bt in backtest_results.items():
        try:
            sharpe, lower, upper = bootstrap_sharpe_ci(bt["net_return"], n_bootstrap=1000)
            ci_rows.append({"strategy": name, "sharpe": sharpe, "ci_lower": lower, "ci_upper": upper})
        except Exception:
            pass
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / "sharpe_ci.csv", index=False)

    return table


def main():
    start_time = time.time()

    setup_results_dirs()

    # Step 1: Load data
    equity_data, vix_data, tlt_data = step1_load_data()

    # Step 2: Compute features
    df_ret, mood_df, market_return, vol_df = step2_compute_features(equity_data)

    # Step 3: Walk-forward regime detection
    predictions = step3_walk_forward_regimes(mood_df, market_return)
    predictions.to_parquet(RESULTS_DIR / "regime_predictions.parquet")

    # Step 4: Generate signals
    signals, mkt_ret_aligned, regime_probs, regime_labels = step4_generate_signals(
        predictions, market_return,
    )
    signals.to_parquet(RESULTS_DIR / "strategy_signals.parquet")

    # Step 5: Backtests
    backtest_results = step5_run_backtests(signals, mkt_ret_aligned, tlt_data)

    # Step 6: Evaluate
    table = step6_evaluate(backtest_results, predictions, mkt_ret_aligned)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

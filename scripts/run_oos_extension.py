"""
Out-of-sample extension: 2021-2026.

Downloads new data, runs the SAME models forward without refitting,
and reports performance. This is the most important credibility test.

If the strategy works on data it has never seen, the edge is real.
If it doesn't, that's equally valuable to report honestly.

Usage:
    python scripts/run_oos_extension.py
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
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from systematic_regime_trading.utils.config import load_config, get_path
from systematic_regime_trading.data.loaders import download_spy, download_vix
from systematic_regime_trading.data.storage import save_parquet_subdir
from systematic_regime_trading.models.hmm import HMMRegimeDetector
from systematic_regime_trading.models.garch import GARCHRegimeDetector
from systematic_regime_trading.models.kmeans import KMeansRegimeDetector
from systematic_regime_trading.models.gmm import GMMRegimeDetector
from systematic_regime_trading.models.markov_switching import MarkovSwitchingRegimeDetector
from systematic_regime_trading.models.ensemble import EnsembleRegimeDetector
from systematic_regime_trading.features.volatility import (
    compute_daily_returns_unified,
    compute_market_volatility_index_unified,
)
from systematic_regime_trading.features.indicators import (
    compute_market_direction_unified,
    compute_market_volume_unified,
    compute_market_breadth_unified,
    compute_market_atr_unified,
    compute_mood_index,
)
from systematic_regime_trading.signals.generator import generate_all_signals
from systematic_regime_trading.backtest.engine import run_backtest
from systematic_regime_trading.backtest.benchmarks import buy_and_hold
from systematic_regime_trading.backtest.trivial_signals import sma_crossover_signal, vol_managed_signal
from systematic_regime_trading.evaluation.metrics import compute_metrics
from systematic_regime_trading.evaluation.report import performance_table

RESULTS_DIR = get_path("results") / "oos_extension"


def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    oos_start = "2021-01-01"
    oos_end = "2026-03-25"

    models_cfg = load_config("models")

    # === Step 1: Download OOS data ===
    logger.info("Downloading OOS data (2021-2026)...")

    spy_oos = download_spy(oos_start, oos_end)
    spy_returns = spy_oos.set_index("Date")["spy_return"].dropna()
    spy_prices = spy_oos.set_index("Date")["Adj Close"]

    logger.info(f"SPY OOS: {len(spy_returns)} days ({spy_returns.index[0].date()} to {spy_returns.index[-1].date()})")

    # We need the equity universe to compute cross-sectional features
    # Download a subset of liquid NASDAQ stocks for features only
    from systematic_regime_trading.data.loaders import download_equity_data
    from systematic_regime_trading.data.storage import load_parquet

    # Load cached universe tickers
    universe = pd.read_csv(get_path("data/cache/nasdaq_constituents.csv"))
    tickers = universe["ticker"].tolist()

    logger.info(f"Downloading {len(tickers)} tickers for OOS period...")
    equity_oos = download_equity_data(tickers, oos_start, oos_end)

    # === Step 2: Compute features for OOS period ===
    logger.info("Computing OOS features...")
    df_ret = compute_daily_returns_unified(equity_oos)
    vol_df = compute_market_volatility_index_unified(df_ret).set_index("Date")
    dir_df = compute_market_direction_unified(df_ret)
    vol_obj_df = compute_market_volume_unified(df_ret)
    brd_df = compute_market_breadth_unified(df_ret)
    atr_df = compute_market_atr_unified(df_ret)

    indicators_dict = {
        "market_volatility": vol_df,
        "market_direction": dir_df,
        "market_volume": vol_obj_df,
        "market_breadth": brd_df,
        "market_atr": atr_df,
    }
    mood_df = compute_mood_index(indicators_dict, window=252)

    feature_df = mood_df.copy()
    feature_df["Date"] = feature_df.index
    feature_df = feature_df.dropna(subset=["market_volatility"]).reset_index(drop=True)

    logger.info(f"OOS features: {len(feature_df)} days")

    # === Step 3: Load the FULL training data to fit models on ===
    # Train on ALL data up to 2020, then predict on 2021+
    logger.info("Loading training data (2000-2020)...")
    train_equity = load_parquet("equity_data")
    train_ret = compute_daily_returns_unified(train_equity)
    train_vol = compute_market_volatility_index_unified(train_ret).set_index("Date")
    train_dir = compute_market_direction_unified(train_ret)
    train_vol_obj = compute_market_volume_unified(train_ret)
    train_brd = compute_market_breadth_unified(train_ret)
    train_atr = compute_market_atr_unified(train_ret)

    train_indicators = {
        "market_volatility": train_vol,
        "market_direction": train_dir,
        "market_volume": train_vol_obj,
        "market_breadth": train_brd,
        "market_atr": train_atr,
    }
    train_mood = compute_mood_index(train_indicators, window=252)
    train_feature = train_mood.dropna(subset=["market_volatility"])

    # SPY training returns for GARCH and Markov-Switching
    from systematic_regime_trading.data.storage import load_parquet_subdir
    spy_train = load_parquet_subdir("spy", "auxiliary")
    spy_train_ret = spy_train.set_index("Date")["spy_return"].dropna()

    # === Step 4: Fit models on full training data, predict OOS ===
    logger.info("Fitting models on 2000-2020, predicting 2021+...")

    n_test = len(feature_df)

    # HMM
    try:
        hmm = HMMRegimeDetector(models_cfg["hmm"])
        hmm.fit(train_feature["market_volatility"].values)
        hmm_probs = hmm.predict_proba(feature_df["market_volatility"].values)
    except Exception as e:
        logger.warning(f"HMM failed: {e}")
        hmm_probs = np.full((n_test, 3), 1/3)

    # GARCH
    try:
        garch = GARCHRegimeDetector(models_cfg["garch"])
        garch.fit(spy_train_ret)
        garch_probs = garch.predict_proba(spy_returns)
        garch_probs = garch_probs[:n_test]
    except Exception as e:
        logger.warning(f"GARCH failed: {e}")
        garch_probs = np.full((n_test, 3), 1/3)

    # KMeans
    try:
        feature_cols = ["market_volatility", "market_volume"]
        avail = [c for c in feature_cols if c in train_feature.columns]
        kmeans = KMeansRegimeDetector(models_cfg["kmeans"])
        kmeans.fit(train_feature[avail].dropna())
        km_probs = kmeans.predict_proba(feature_df[avail].dropna())
        if len(km_probs) < n_test:
            padded = np.full((n_test, 3), 1/3)
            padded[:len(km_probs)] = km_probs
            km_probs = padded
        else:
            km_probs = km_probs[:n_test]
    except Exception as e:
        logger.warning(f"KMeans failed: {e}")
        km_probs = np.full((n_test, 3), 1/3)

    # GMM
    try:
        gmm = GMMRegimeDetector(models_cfg.get("gmm", {}))
        gmm.fit(train_feature[avail].dropna())
        gmm_probs = gmm.predict_proba(feature_df[avail].dropna())
        if len(gmm_probs) < n_test:
            padded = np.full((n_test, 3), 1/3)
            padded[:len(gmm_probs)] = gmm_probs
            gmm_probs = padded
        else:
            gmm_probs = gmm_probs[:n_test]
    except Exception as e:
        logger.warning(f"GMM failed: {e}")
        gmm_probs = np.full((n_test, 3), 1/3)

    # Markov-Switching
    try:
        ms = MarkovSwitchingRegimeDetector(models_cfg.get("markov_switching", {}))
        ms.fit(spy_train_ret)
        ms_probs = ms.predict_proba(spy_returns)
        ms_probs = ms_probs[:n_test]
    except Exception as e:
        logger.warning(f"MarkovSwitching failed: {e}")
        ms_probs = np.full((n_test, 3), 1/3)

    # Ensemble
    ensemble = EnsembleRegimeDetector(
        weights={"hmm": 0.25, "garch": 0.25, "kmeans": 0.15, "gmm": 0.20, "markov_switching": 0.15},
        config=models_cfg["ensemble"],
    )
    combined = ensemble.combine({
        "hmm": hmm_probs, "garch": garch_probs, "kmeans": km_probs,
        "gmm": gmm_probs, "markov_switching": ms_probs,
    })
    labels = np.argmax(combined, axis=1)

    predictions = pd.DataFrame({
        "Date": feature_df["Date"].values,
        "regime_label": labels,
        "prob_calm": combined[:, 0],
        "prob_moderate": combined[:, 1],
        "prob_turbulent": combined[:, 2],
    })
    predictions.to_parquet(RESULTS_DIR / "oos_predictions.parquet")

    dist = predictions["regime_label"].value_counts().sort_index()
    for regime, count in dist.items():
        name = {0: "Calm", 1: "Moderate", 2: "Turbulent"}.get(regime, f"R{regime}")
        logger.info(f"  {name}: {count} days ({count/len(predictions)*100:.1f}%)")

    # === Step 5: Generate signals and backtest ===
    logger.info("Generating OOS signals and running backtests...")
    regime_probs = predictions.set_index("Date")[["prob_calm", "prob_moderate", "prob_turbulent"]]
    regime_labels = predictions.set_index("Date")["regime_label"]

    common = regime_probs.index.intersection(spy_returns.index)
    mkt_ret = spy_returns.loc[common]

    signals = generate_all_signals(
        regime_probs.loc[common], regime_labels.loc[common], mkt_ret,
    )

    config = load_config("backtest")
    cost_config = config["execution"]
    constraint_config = config["constraints"]

    results = {}
    for col in signals.columns:
        bt = run_backtest(mkt_ret, signals[col], cost_config, constraint_config)
        results[col] = bt

    results["buy_and_hold"] = buy_and_hold(mkt_ret, cost_config)

    # Trivial benchmarks
    sma_sig = sma_crossover_signal(spy_prices.reindex(common), window=200).shift(1)
    results["sma_200"] = run_backtest(mkt_ret, sma_sig, cost_config, constraint_config)

    vm_sig = vol_managed_signal(mkt_ret, vol_target=0.10, lookback=63)
    results["vol_managed"] = run_backtest(mkt_ret, vm_sig, cost_config, constraint_config)

    # === Step 6: Evaluate ===
    table = performance_table(results, include_ci=False)

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE PERFORMANCE (2021-2026)")
    print("=" * 80)
    from systematic_regime_trading.evaluation.report import format_performance_table
    print(format_performance_table(table))
    print("=" * 80)

    table.to_csv(RESULTS_DIR / "oos_performance_table.csv")

    # Save backtests
    for name, bt in results.items():
        bt.to_parquet(RESULTS_DIR / f"{name}.parquet")

    elapsed = time.time() - start_time
    logger.info(f"OOS extension complete in {elapsed:.1f}s")
    print(f"\nOOS results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

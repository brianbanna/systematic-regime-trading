"""
Out-of-sample extension: 2021-2026.

Two modes:
1. FROZEN: Same models trained on 2000-2020, applied without refitting (v1)
2. ROLLING: Continue walk-forward quarterly into 2021-2026 (v2, recommended)

Rolling recalibration is how real trading systems work. The walk-forward
loop keeps expanding the training window as new data arrives.

Usage:
    python scripts/run_oos_extension.py          # rolling (default)
    python scripts/run_oos_extension.py --frozen  # frozen models
"""

import pandas as pd
import numpy as np
import logging
import time
import warnings
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from systematic_regime_trading.utils.config import load_config, get_path
from systematic_regime_trading.data.loaders import download_spy, download_equity_data, download_vix
from systematic_regime_trading.data.storage import load_parquet, load_parquet_subdir
from systematic_regime_trading.features.volatility import (
    compute_daily_returns_unified, compute_market_volatility_index_unified,
)
from systematic_regime_trading.features.indicators import (
    compute_market_direction_unified, compute_market_volume_unified,
    compute_market_breadth_unified, compute_market_atr_unified,
    compute_mood_index,
)
from systematic_regime_trading.models.hmm import HMMRegimeDetector
from systematic_regime_trading.models.garch import GARCHRegimeDetector
from systematic_regime_trading.models.kmeans import KMeansRegimeDetector
from systematic_regime_trading.models.gmm import GMMRegimeDetector
from systematic_regime_trading.models.markov_switching import MarkovSwitchingRegimeDetector
from systematic_regime_trading.models.ensemble import EnsembleRegimeDetector
from systematic_regime_trading.signals.generator import generate_all_signals
from systematic_regime_trading.backtest.engine import run_backtest
from systematic_regime_trading.backtest.benchmarks import buy_and_hold
from systematic_regime_trading.backtest.trivial_signals import sma_crossover_signal, vol_managed_signal
from systematic_regime_trading.evaluation.report import performance_table, format_performance_table

RESULTS_DIR = get_path("results") / "oos_extension"


def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_mode = "--frozen" in sys.argv
    mode_label = "FROZEN" if frozen_mode else "ROLLING RECALIBRATION"

    logger.info(f"OOS Extension Mode: {mode_label}")
    logger.info("=" * 60)

    models_cfg = load_config("models")
    oos_start = "2021-01-01"
    oos_end = "2026-03-25"

    # Download OOS data
    logger.info("Downloading OOS data...")
    spy_oos = download_spy(oos_start, oos_end)
    spy_returns = spy_oos.set_index("Date")["spy_return"].dropna()
    spy_prices = spy_oos.set_index("Date")["Adj Close"]

    universe = pd.read_csv(get_path("data/cache/nasdaq_constituents.csv"))
    tickers = universe["ticker"].tolist()
    logger.info(f"Downloading {len(tickers)} tickers for OOS...")
    equity_oos = download_equity_data(tickers, oos_start, oos_end)

    # Compute OOS features
    logger.info("Computing OOS features...")
    df_ret_oos = compute_daily_returns_unified(equity_oos)
    vol_oos = compute_market_volatility_index_unified(df_ret_oos).set_index("Date")
    dir_oos = compute_market_direction_unified(df_ret_oos)
    vol_obj_oos = compute_market_volume_unified(df_ret_oos)
    brd_oos = compute_market_breadth_unified(df_ret_oos)
    atr_oos = compute_market_atr_unified(df_ret_oos)

    oos_indicators = {
        "market_volatility": vol_oos,
        "market_direction": dir_oos,
        "market_volume": vol_obj_oos,
        "market_breadth": brd_oos,
        "market_atr": atr_oos,
    }

    # Use training-period standardization parameters for consistency
    try:
        feature_stats = pd.read_parquet(get_path("results/feature_standardization.parquet"))
        logger.info("Using training-period standardization parameters")
        # Apply training mean/std to OOS data instead of fresh rolling z-score
        mood_oos = compute_mood_index(oos_indicators, window=252)
        # Override z-scores with training-calibrated values
        for col in mood_oos.columns:
            if col in feature_stats.columns:
                train_mean = feature_stats.loc["mean", col]
                train_std = feature_stats.loc["std", col]
                if train_std > 0:
                    mood_oos[col] = (mood_oos[col] * mood_oos[col].rolling(252).std().iloc[-1] + mood_oos[col].rolling(252).mean().iloc[-1] - train_mean) / train_std
    except Exception:
        logger.warning("No training stats found, using fresh standardization")
        mood_oos = compute_mood_index(oos_indicators, window=252)

    feature_oos = mood_oos.dropna(subset=["market_volatility"]).copy()
    feature_oos["Date"] = feature_oos.index
    feature_oos = feature_oos.reset_index(drop=True)
    logger.info(f"OOS features: {len(feature_oos)} days")

    if frozen_mode:
        predictions = _run_frozen(feature_oos, spy_returns, models_cfg)
    else:
        predictions = _run_rolling(feature_oos, spy_returns, models_cfg)

    predictions.to_parquet(RESULTS_DIR / "oos_predictions.parquet")

    # Generate signals and backtest
    logger.info("Generating OOS signals and backtesting...")
    regime_probs = predictions.set_index("Date")[["prob_calm", "prob_moderate", "prob_turbulent"]]
    regime_labels = predictions.set_index("Date")["regime_label"]
    common = regime_probs.index.intersection(spy_returns.index)
    mkt_ret = spy_returns.loc[common]

    signals = generate_all_signals(
        regime_probs.loc[common], regime_labels.loc[common], mkt_ret,
    )

    config = load_config("backtest")
    results = {}
    for col in signals.columns:
        results[col] = run_backtest(mkt_ret, signals[col], config["execution"], config["constraints"])

    results["buy_and_hold"] = buy_and_hold(mkt_ret, config["execution"])

    sma_sig = sma_crossover_signal(spy_prices.reindex(common), 200).shift(1)
    results["sma_200"] = run_backtest(mkt_ret, sma_sig, config["execution"], config["constraints"])

    vm_sig = vol_managed_signal(mkt_ret, vol_target=0.10, lookback=63)
    results["vol_managed"] = run_backtest(mkt_ret, vm_sig, config["execution"], config["constraints"])

    # Save results
    for name, bt in results.items():
        bt.to_parquet(RESULTS_DIR / f"{name}.parquet")

    table = performance_table(results, include_ci=False)
    table.to_csv(RESULTS_DIR / "oos_performance_table.csv")

    print(f"\n{'=' * 80}")
    print(f"OUT-OF-SAMPLE ({mode_label}) 2021-2026")
    print(f"{'=' * 80}")
    print(format_performance_table(table))
    print(f"{'=' * 80}")

    elapsed = time.time() - start_time
    logger.info(f"OOS extension complete in {elapsed:.1f}s")


def _run_frozen(feature_oos, spy_returns, models_cfg):
    """Train on ALL 2000-2020 data, predict OOS without refitting."""
    logger.info("Loading full training data (2000-2020)...")
    train_equity = load_parquet("equity_data")
    train_ret = compute_daily_returns_unified(train_equity)
    train_vol = compute_market_volatility_index_unified(train_ret).set_index("Date")
    train_dir = compute_market_direction_unified(train_ret)
    train_vol_obj = compute_market_volume_unified(train_ret)
    train_brd = compute_market_breadth_unified(train_ret)
    train_atr = compute_market_atr_unified(train_ret)

    train_mood = compute_mood_index({
        "market_volatility": train_vol, "market_direction": train_dir,
        "market_volume": train_vol_obj, "market_breadth": train_brd,
        "market_atr": train_atr,
    }, window=252)
    train_feature = train_mood.dropna(subset=["market_volatility"])

    spy_train = load_parquet_subdir("spy", "auxiliary")
    spy_train_ret = spy_train.set_index("Date")["spy_return"].dropna()

    n_test = len(feature_oos)

    # Fit models on full training data
    logger.info("Fitting models on 2000-2020...")
    hmm = HMMRegimeDetector(models_cfg["hmm"])
    hmm.fit(train_feature["market_volatility"].values)
    hmm_probs = hmm.predict_proba(feature_oos["market_volatility"].values)[:n_test]

    garch = GARCHRegimeDetector(models_cfg["garch"])
    garch.fit(spy_train_ret)
    garch_probs = garch.predict_proba(spy_returns)[:n_test]

    avail = ["market_volatility", "market_volume"]
    avail = [c for c in avail if c in train_feature.columns and c in feature_oos.columns]
    kmeans = KMeansRegimeDetector(models_cfg["kmeans"])
    kmeans.fit(train_feature[avail].dropna())
    km_probs = kmeans.predict_proba(feature_oos[avail].dropna())

    gmm = GMMRegimeDetector(models_cfg.get("gmm", {}))
    gmm.fit(train_feature[avail].dropna())
    gmm_probs = gmm.predict_proba(feature_oos[avail].dropna())

    ms = MarkovSwitchingRegimeDetector(models_cfg.get("markov_switching", {}))
    ms.fit(spy_train_ret)
    ms_probs = ms.predict_proba(spy_returns)[:n_test]

    # Pad to n_test
    for name, arr in [("km", km_probs), ("gmm", gmm_probs)]:
        if len(arr) < n_test:
            padded = np.full((n_test, 3), 1/3)
            padded[:len(arr)] = arr
            if name == "km": km_probs = padded
            else: gmm_probs = padded

    ensemble = EnsembleRegimeDetector(
        weights={"hmm": 0.25, "garch": 0.25, "kmeans": 0.15, "gmm": 0.20, "markov_switching": 0.15},
        config=models_cfg["ensemble"],
    )
    combined = ensemble.combine({
        "hmm": hmm_probs[:n_test], "garch": garch_probs[:n_test],
        "kmeans": km_probs[:n_test], "gmm": gmm_probs[:n_test],
        "markov_switching": ms_probs[:n_test],
    })

    return pd.DataFrame({
        "Date": feature_oos["Date"].values,
        "regime_label": np.argmax(combined, axis=1),
        "prob_calm": combined[:, 0],
        "prob_moderate": combined[:, 1],
        "prob_turbulent": combined[:, 2],
    })


def _run_rolling(feature_oos, spy_returns, models_cfg):
    """Continue walk-forward with quarterly recalibration into OOS period."""
    logger.info("Loading full training data for rolling recalibration...")

    # Load training features
    train_equity = load_parquet("equity_data")
    train_ret = compute_daily_returns_unified(train_equity)
    train_vol = compute_market_volatility_index_unified(train_ret).set_index("Date")
    train_dir = compute_market_direction_unified(train_ret)
    train_vol_obj = compute_market_volume_unified(train_ret)
    train_brd = compute_market_breadth_unified(train_ret)
    train_atr = compute_market_atr_unified(train_ret)
    train_mood = compute_mood_index({
        "market_volatility": train_vol, "market_direction": train_dir,
        "market_volume": train_vol_obj, "market_breadth": train_brd,
        "market_atr": train_atr,
    }, window=252)
    train_feature = train_mood.dropna(subset=["market_volatility"]).copy()
    train_feature["Date"] = train_feature.index
    train_feature = train_feature.reset_index(drop=True)

    # SPY training returns
    spy_train = load_parquet_subdir("spy", "auxiliary")
    spy_train_ret = spy_train.set_index("Date")["spy_return"].dropna()

    # Combine training + OOS features
    combined_features = pd.concat([train_feature, feature_oos], ignore_index=True)
    combined_spy = pd.concat([spy_train_ret, spy_returns])
    combined_spy = combined_spy[~combined_spy.index.duplicated(keep="first")].sort_index()

    # Walk-forward from end of training into OOS
    wf_cfg = models_cfg["walk_forward"]
    step_days = wf_cfg["step_days"]

    n_train_end = len(train_feature)
    n_total = len(combined_features)

    all_predictions = []
    window_id = 0
    train_end_idx = n_train_end  # Start where in-sample ended

    avail = ["market_volatility", "market_volume"]

    while train_end_idx < n_total:
        test_end_idx = min(train_end_idx + step_days, n_total)
        train_data = combined_features.iloc[:train_end_idx]
        test_data = combined_features.iloc[train_end_idx:test_end_idx]

        if len(test_data) == 0:
            break

        if window_id % 4 == 0:
            logger.info(f"OOS Window {window_id}: train {len(train_data)} days, "
                       f"test {len(test_data)} days")

        n_test = len(test_data)

        # HMM
        try:
            hmm = HMMRegimeDetector(models_cfg["hmm"])
            hmm.fit(train_data["market_volatility"].dropna().values)
            hmm_p = hmm.predict_proba(test_data["market_volatility"].dropna().values)[:n_test]
        except Exception:
            hmm_p = np.full((n_test, 3), 1/3)

        # GARCH
        try:
            train_dates = train_data["Date"].values
            train_spy = combined_spy.loc[combined_spy.index.isin(train_dates)]
            garch = GARCHRegimeDetector(models_cfg["garch"])
            garch.fit(train_spy)
            test_dates = test_data["Date"].values
            test_spy = combined_spy.loc[combined_spy.index.isin(test_dates)]
            garch_p = garch.predict_proba(test_spy)[:n_test]
        except Exception:
            garch_p = np.full((n_test, 3), 1/3)

        # KMeans
        try:
            train_X = train_data[avail].dropna()
            test_X = test_data[avail].dropna()
            km = KMeansRegimeDetector(models_cfg["kmeans"])
            km.fit(train_X)
            km_p = km.predict_proba(test_X)
            if len(km_p) < n_test:
                padded = np.full((n_test, 3), 1/3)
                padded[:len(km_p)] = km_p
                km_p = padded
            else:
                km_p = km_p[:n_test]
        except Exception:
            km_p = np.full((n_test, 3), 1/3)

        # GMM
        try:
            gmm = GMMRegimeDetector(models_cfg.get("gmm", {}))
            gmm.fit(train_X)
            gmm_p = gmm.predict_proba(test_X)
            if len(gmm_p) < n_test:
                padded = np.full((n_test, 3), 1/3)
                padded[:len(gmm_p)] = gmm_p
                gmm_p = padded
            else:
                gmm_p = gmm_p[:n_test]
        except Exception:
            gmm_p = np.full((n_test, 3), 1/3)

        # Markov-Switching
        try:
            ms = MarkovSwitchingRegimeDetector(models_cfg.get("markov_switching", {}))
            ms.fit(train_spy)
            ms_p = ms.predict_proba(test_spy)[:n_test]
        except Exception:
            ms_p = np.full((n_test, 3), 1/3)

        # Ensemble
        ensemble = EnsembleRegimeDetector(
            weights={"hmm": 0.25, "garch": 0.25, "kmeans": 0.15, "gmm": 0.20, "markov_switching": 0.15},
            config=models_cfg["ensemble"],
        )
        combined_p = ensemble.combine({
            "hmm": hmm_p, "garch": garch_p, "kmeans": km_p,
            "gmm": gmm_p, "markov_switching": ms_p,
        })

        window_result = pd.DataFrame({
            "Date": test_data["Date"].values,
            "regime_label": np.argmax(combined_p, axis=1),
            "prob_calm": combined_p[:, 0],
            "prob_moderate": combined_p[:, 1],
            "prob_turbulent": combined_p[:, 2],
        })
        all_predictions.append(window_result)

        window_id += 1
        train_end_idx += step_days

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)

    logger.info(f"Rolling recalibration: {window_id} windows, {len(predictions)} OOS predictions")
    return predictions


if __name__ == "__main__":
    main()

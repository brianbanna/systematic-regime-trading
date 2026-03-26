"""
Commodities regime detection extension.

Downloads crude oil futures data, computes commodity-specific features,
runs HMM and GARCH regime detection, and analyzes cross-asset spillover
between equity and commodity regimes.

Demonstrates the framework's applicability to commodity markets,
directly relevant for energy/commodity trading desk applications.

Usage:
    python scripts/run_commodities.py
"""

import pandas as pd
import numpy as np
import logging
import warnings
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from systematic_regime_trading.utils.config import load_config, get_path
from systematic_regime_trading.data.storage import load_parquet_subdir
from systematic_regime_trading.models.hmm import HMMRegimeDetector
from systematic_regime_trading.models.garch import GARCHRegimeDetector
from systematic_regime_trading.models.ensemble import EnsembleRegimeDetector
from systematic_regime_trading.evaluation.metrics import compute_metrics

RESULTS_DIR = get_path("results") / "commodities"


def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_cfg = load_config("models")

    # === Step 1: Download crude oil data ===
    logger.info("Downloading crude oil futures (CL=F)...")
    import yfinance as yf

    cl = yf.download("CL=F", start="2000-01-01", end="2026-03-25",
                     auto_adjust=False, progress=False)

    cl_df = pd.DataFrame({
        "Date": cl.index,
        "Close": cl["Close"].values.flatten(),
        "Adj Close": cl["Adj Close"].values.flatten() if "Adj Close" in cl.columns else cl["Close"].values.flatten(),
        "Volume": cl["Volume"].values.flatten() if "Volume" in cl.columns else 0,
    })
    cl_df["Date"] = pd.to_datetime(cl_df["Date"])
    cl_df = cl_df.set_index("Date")
    cl_df["return"] = cl_df["Close"].pct_change()
    cl_df = cl_df.dropna(subset=["return"])

    logger.info(f"Crude oil: {len(cl_df)} days ({cl_df.index[0].date()} to {cl_df.index[-1].date()})")
    logger.info(f"  Price range: ${cl_df['Close'].min():.2f} to ${cl_df['Close'].max():.2f}")

    # === Step 2: Compute commodity features ===
    logger.info("Computing commodity features...")

    # Realized volatility at multiple horizons
    for window in [20, 60]:
        cl_df[f"realized_vol_{window}d"] = (
            cl_df["return"].rolling(window).std() * np.sqrt(252)
        )

    # Vol-of-vol
    cl_df["vol_of_vol"] = cl_df["realized_vol_20d"].rolling(60).std()

    # Return momentum
    cl_df["momentum_20d"] = cl_df["return"].rolling(20).mean() * 252

    cl_df = cl_df.dropna()
    logger.info(f"Features computed: {len(cl_df)} days")

    # === Step 3: Regime detection on crude oil ===
    logger.info("Running regime detection on crude oil...")

    # Split train/test
    split_date = "2015-01-01"
    train = cl_df.loc[:split_date]
    test = cl_df.loc[split_date:]

    logger.info(f"Train: {len(train)} days, Test: {len(test)} days")

    # HMM on realized vol
    hmm_cfg = {"n_states": 3, "n_iter": 1000, "covariance_type": "full", "random_state": 42}
    hmm = HMMRegimeDetector(hmm_cfg)
    hmm.fit(train["realized_vol_20d"].values)
    hmm_probs_test = hmm.predict_proba(test["realized_vol_20d"].values)
    hmm_labels_test = hmm.predict(test["realized_vol_20d"].values)

    # GARCH on returns
    garch = GARCHRegimeDetector(models_cfg["garch"])
    garch.fit(train["return"])
    garch_probs_test = garch.predict_proba(test["return"])
    garch_labels_test = garch.predict(test["return"])

    # Pad if needed
    n_test = len(test)
    if len(hmm_probs_test) < n_test:
        padded = np.full((n_test, 3), 1/3)
        padded[:len(hmm_probs_test)] = hmm_probs_test
        hmm_probs_test = padded
    if len(garch_probs_test) < n_test:
        padded = np.full((n_test, 3), 1/3)
        padded[:len(garch_probs_test)] = garch_probs_test
        garch_probs_test = padded

    # Ensemble (2 models)
    ensemble = EnsembleRegimeDetector(
        weights={"hmm": 0.5, "garch": 0.5},
        config=models_cfg["ensemble"],
    )
    combined = ensemble.combine({"hmm": hmm_probs_test[:n_test], "garch": garch_probs_test[:n_test]})
    ensemble_labels = np.argmax(combined, axis=1)

    # Save predictions
    commodity_pred = pd.DataFrame({
        "Date": test.index,
        "regime_label": ensemble_labels,
        "prob_calm": combined[:, 0],
        "prob_moderate": combined[:, 1],
        "prob_turbulent": combined[:, 2],
        "hmm_label": np.argmax(hmm_probs_test[:n_test], axis=1),
        "garch_label": np.argmax(garch_probs_test[:n_test], axis=1),
        "close_price": test["Close"].values,
        "realized_vol_20d": test["realized_vol_20d"].values,
    })
    commodity_pred.to_parquet(RESULTS_DIR / "crude_oil_regimes.parquet")

    # Regime distribution
    dist = commodity_pred["regime_label"].value_counts().sort_index()
    print("\nCrude Oil Regime Distribution:")
    for regime, count in dist.items():
        name = {0: "Calm", 1: "Moderate", 2: "Turbulent"}.get(regime, f"R{regime}")
        print(f"  {name}: {count} days ({count/len(commodity_pred)*100:.1f}%)")

    # Regime-conditional performance
    print("\nCrude Oil Returns by Regime:")
    for regime in sorted(commodity_pred["regime_label"].unique()):
        mask = commodity_pred["regime_label"] == regime
        regime_ret = test["return"][mask.values]
        name = {0: "Calm", 1: "Moderate", 2: "Turbulent"}.get(regime, f"R{regime}")
        print(f"  {name}: mean={regime_ret.mean()*252:.1%}, vol={regime_ret.std()*np.sqrt(252):.1%}")

    # === Step 4: Cross-asset spillover analysis ===
    logger.info("Cross-asset spillover analysis...")

    # Load equity VIX
    try:
        vix = load_parquet_subdir("vix", "auxiliary")
        vix = vix.set_index("Date")

        # Align dates
        common = cl_df.index.intersection(vix.index)
        if len(common) > 100:
            equity_vol = vix.loc[common, "VIX"]
            commodity_vol = cl_df.loc[common, "realized_vol_20d"] * 100  # scale to match VIX

            # Rolling correlation
            rolling_corr = equity_vol.rolling(60).corr(commodity_vol)

            # Lead-lag analysis
            lead_lags = {}
            for lag in range(-20, 21):
                if lag >= 0:
                    corr = equity_vol.iloc[lag:].reset_index(drop=True).corr(
                        commodity_vol.iloc[:len(equity_vol)-lag].reset_index(drop=True)
                    )
                else:
                    corr = commodity_vol.iloc[-lag:].reset_index(drop=True).corr(
                        equity_vol.iloc[:len(commodity_vol)+lag].reset_index(drop=True)
                    )
                lead_lags[lag] = corr

            spillover_df = pd.DataFrame({
                "lag_days": list(lead_lags.keys()),
                "correlation": list(lead_lags.values()),
            })
            spillover_df.to_csv(RESULTS_DIR / "cross_asset_spillover.csv", index=False)

            print(f"\nCross-Asset Spillover:")
            print(f"  Contemporaneous corr(VIX, CL vol): {lead_lags[0]:.3f}")
            print(f"  VIX leads CL by 5 days: {lead_lags[5]:.3f}")
            print(f"  CL leads VIX by 5 days: {lead_lags[-5]:.3f}")
            print(f"  Rolling 60d correlation mean: {rolling_corr.dropna().mean():.3f}")

    except Exception as e:
        logger.warning(f"Spillover analysis failed: {e}")

    # Save crude oil price/vol data for charts
    cl_df.to_parquet(RESULTS_DIR / "crude_oil_data.parquet")

    elapsed = time.time() - start_time
    logger.info(f"Commodities analysis complete in {elapsed:.1f}s")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

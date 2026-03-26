"""
Macro data downloaders.

Downloads credit spreads, yield curve, financial stress, and SKEW
from FRED and CBOE. These are the cross-asset signals that equity-only
features miss entirely.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_macro_features(
    start: str = "2000-01-01",
    end: str = "2026-12-31",
) -> pd.DataFrame:
    """
    Download all macro features and merge into a single DataFrame.

    Features:
    - hy_spread: High-yield credit spread (ICE BofA OAS)
    - yield_curve: 10Y minus 2Y Treasury spread
    - financial_stress: St. Louis Fed Financial Stress Index
    - skew: CBOE SKEW index (tail risk pricing)

    Returns:
        DataFrame indexed by Date with macro feature columns
    """
    features = {}

    # High-Yield Credit Spread
    try:
        hy = _download_fred_csv("BAMLH0A0HYM2", start, end)
        if hy is not None and len(hy) > 100:
            features["hy_spread"] = hy / 100  # Convert from percent to decimal
            logger.info(f"High-yield spread: {len(hy)} observations")
    except Exception as e:
        logger.warning(f"HY spread download failed: {e}")

    # Yield Curve (10Y - 2Y)
    try:
        t10y = _download_fred_csv("DGS10", start, end)
        t2y = _download_fred_csv("DGS2", start, end)
        if t10y is not None and t2y is not None:
            common = t10y.index.intersection(t2y.index)
            yc = t10y.loc[common] - t2y.loc[common]
            features["yield_curve"] = yc / 100
            logger.info(f"Yield curve: {len(yc)} observations")
    except Exception as e:
        logger.warning(f"Yield curve download failed: {e}")

    # Financial Stress Index
    try:
        stress = _download_fred_csv("STLFSI2", start, end)
        if stress is not None and len(stress) > 50:
            features["financial_stress"] = stress
            logger.info(f"Financial stress: {len(stress)} observations")
    except Exception as e:
        logger.warning(f"Financial stress download failed: {e}")

    # CBOE SKEW Index (from yfinance)
    try:
        import yfinance as yf
        skew = yf.download("^SKEW", start=start, end=end, progress=False)
        if len(skew) > 100:
            skew_series = skew["Close"]
            if hasattr(skew_series, 'columns'):
                skew_series = skew_series.iloc[:, 0]
            features["skew"] = skew_series
            logger.info(f"CBOE SKEW: {len(skew_series)} observations")
    except Exception as e:
        logger.warning(f"SKEW download failed: {e}")

    if not features:
        logger.error("No macro features downloaded successfully")
        return pd.DataFrame()

    # Merge all features on date
    result = pd.DataFrame(features)
    result.index = pd.to_datetime(result.index)
    result.index.name = "Date"

    # Forward-fill for non-trading days (macro data often has gaps)
    result = result.ffill().bfill()

    logger.info(f"Macro features: {len(result)} days, {list(result.columns)}")
    return result


def _download_fred_csv(series_id: str, start: str, end: str) -> pd.Series:
    """
    Download a FRED series via the public CSV endpoint (no API key needed).
    """
    import urllib.request
    import io

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
        f"&cosd={start}&coed={end}"
    )

    try:
        response = urllib.request.urlopen(url, timeout=30)
        content = response.read().decode("utf-8")

        df = pd.read_csv(io.StringIO(content), parse_dates=["DATE"], index_col="DATE")
        col = df.columns[0]

        # FRED uses "." for missing values
        series = pd.to_numeric(df[col], errors="coerce")
        series = series.dropna()
        series.index.name = "Date"

        return series

    except Exception as e:
        logger.warning(f"FRED CSV download failed for {series_id}: {e}")

        # Try fredapi as fallback
        try:
            import os
            from fredapi import Fred

            key = os.environ.get("FRED_API_KEY")
            if key:
                fred = Fred(api_key=key)
                series = fred.get_series(series_id, observation_start=start, observation_end=end)
                series = series.dropna()
                series.index.name = "Date"
                return series
        except Exception:
            pass

        return None


def compute_derived_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from raw macro data.

    Adds:
    - hy_spread_change: 5-day change in credit spread (acceleration)
    - yield_curve_inverted: binary flag (1 if inverted)
    - stress_momentum: 20-day change in financial stress
    """
    result = macro_df.copy()

    if "hy_spread" in result.columns:
        result["hy_spread_change_5d"] = result["hy_spread"].diff(5)
        result["hy_spread_zscore"] = (
            (result["hy_spread"] - result["hy_spread"].rolling(252).mean())
            / result["hy_spread"].rolling(252).std().clip(lower=1e-6)
        )

    if "yield_curve" in result.columns:
        result["yield_curve_inverted"] = (result["yield_curve"] < 0).astype(float)

    if "financial_stress" in result.columns:
        result["stress_momentum"] = result["financial_stress"].diff(20)

    if "skew" in result.columns:
        result["skew_zscore"] = (
            (result["skew"] - result["skew"].rolling(252).mean())
            / result["skew"].rolling(252).std().clip(lower=1e-6)
        )

    return result

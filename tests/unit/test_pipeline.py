"""Tests for data pipeline entry point."""

import pandas as pd
import numpy as np
import pytest


class TestCleanPipeline:
    def test_clean_pipeline_runs_end_to_end(self):
        """Test that clean_pipeline produces valid output from raw data."""
        from systematic_regime_trading.data.cleaning import (
            clean_unified_dataframe,
            fill_missing_data_unified,
        )

        rng = np.random.default_rng(42)
        tickers = ["AAPL", "GOOG", "MSFT"]
        dates = pd.bdate_range("2020-01-01", periods=100)
        rows = []

        for ticker in tickers:
            base = rng.uniform(50, 200)
            for date in dates:
                price = base * (1 + rng.normal(0, 0.02))
                rows.append({
                    "ticker": ticker, "Date": date,
                    "Open": price, "High": price * 1.01,
                    "Low": price * 0.99, "Close": price,
                    "Adj Close": price,
                    "Volume": int(rng.uniform(1e5, 1e7)),
                })

        raw_df = pd.DataFrame(rows)

        cleaned = clean_unified_dataframe(raw_df)
        result = fill_missing_data_unified(cleaned, max_missing_pct=0.10)

        assert result["ticker"].nunique() == 3
        assert len(result) == 300
        assert result["Close"].isna().sum() == 0

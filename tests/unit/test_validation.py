"""Tests for walk-forward validation module."""

import numpy as np
import pandas as pd
import pytest

from systematic_regime_trading.models.validation import walk_forward_train
from systematic_regime_trading.models.hmm import HMMRegimeDetector


@pytest.fixture
def long_synthetic_data():
    """Synthetic data long enough for walk-forward (2000 trading days)."""
    rng = np.random.default_rng(42)
    n = 2000
    dates = pd.bdate_range("2012-01-01", periods=n)

    # Generate volatility with regime structure
    vol = np.zeros(n)
    regime = 0
    for t in range(n):
        if rng.random() < 0.02:  # 2% chance of regime switch
            regime = rng.integers(0, 3)
        base = [0.01, 0.025, 0.05][regime]
        vol[t] = abs(rng.normal(base, base * 0.3))

    return pd.DataFrame({"Date": dates, "volatility": vol})


class TestWalkForwardTrain:
    def test_produces_out_of_sample_predictions(self, long_synthetic_data):
        wf_config = {
            "min_train_years": 3,
            "test_window_days": 252,
            "step_days": 126,
            "expanding": True,
        }
        hmm_config = {"n_states": 3, "n_iter": 50, "covariance_type": "full", "random_state": 42}

        result = walk_forward_train(
            data=long_synthetic_data,
            feature_col="volatility",
            model_class=HMMRegimeDetector,
            model_config=hmm_config,
            wf_config=wf_config,
        )

        assert "Date" in result.columns
        assert "regime_label" in result.columns
        assert "regime_prob_calm" in result.columns
        assert "regime_prob_moderate" in result.columns
        assert "regime_prob_turbulent" in result.columns

        # All predictions should be out-of-sample (after training period)
        min_train_end = long_synthetic_data["Date"].iloc[3 * 252 - 1]
        assert result["Date"].min() >= min_train_end

        # No duplicate dates
        assert result["Date"].is_unique

    def test_multiple_windows(self, long_synthetic_data):
        wf_config = {
            "min_train_years": 3,
            "test_window_days": 252,
            "step_days": 126,
            "expanding": True,
        }
        hmm_config = {"n_states": 3, "n_iter": 50, "covariance_type": "full", "random_state": 42}

        result = walk_forward_train(
            data=long_synthetic_data,
            feature_col="volatility",
            model_class=HMMRegimeDetector,
            model_config=hmm_config,
            wf_config=wf_config,
        )

        # Should have multiple windows
        assert result["window_id"].nunique() > 1

    def test_probabilities_valid(self, long_synthetic_data):
        wf_config = {
            "min_train_years": 3,
            "test_window_days": 252,
            "step_days": 252,
            "expanding": True,
        }
        hmm_config = {"n_states": 3, "n_iter": 50, "covariance_type": "full", "random_state": 42}

        result = walk_forward_train(
            data=long_synthetic_data,
            feature_col="volatility",
            model_class=HMMRegimeDetector,
            model_config=hmm_config,
            wf_config=wf_config,
        )

        prob_sum = (
            result["regime_prob_calm"]
            + result["regime_prob_moderate"]
            + result["regime_prob_turbulent"]
        )
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_insufficient_data_raises(self):
        short_data = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-01", periods=100),
            "volatility": np.random.default_rng(42).random(100),
        })
        wf_config = {
            "min_train_years": 5,
            "test_window_days": 252,
            "step_days": 63,
            "expanding": True,
        }
        with pytest.raises(ValueError, match="Insufficient data"):
            walk_forward_train(
                data=short_data,
                feature_col="volatility",
                model_class=HMMRegimeDetector,
                model_config={"n_states": 3},
                wf_config=wf_config,
            )

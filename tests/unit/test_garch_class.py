"""Tests for GARCHRegimeDetector class."""

import numpy as np
import pandas as pd
import pytest

from systematic_regime_trading.models.garch import GARCHRegimeDetector


@pytest.fixture
def garch_detector():
    config = {"p": 1, "q": 1, "distribution": "t", "return_scaling": 100}
    return GARCHRegimeDetector(config)


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(42)
    n = 500
    returns = np.zeros(n)
    vol = 0.01
    for t in range(1, n):
        vol = 0.0001 + 0.1 * returns[t - 1] ** 2 + 0.85 * vol
        returns[t] = rng.normal(0, np.sqrt(vol))
    return pd.Series(returns, index=pd.bdate_range("2018-01-01", periods=n))


class TestGARCHRegimeDetector:
    def test_fit_returns_self(self, garch_detector, synthetic_returns):
        result = garch_detector.fit(synthetic_returns)
        assert result is garch_detector
        assert garch_detector.is_fitted_

    def test_predict_returns_labels(self, garch_detector, synthetic_returns):
        garch_detector.fit(synthetic_returns)
        labels = garch_detector.predict()
        assert len(labels) == len(synthetic_returns)
        assert set(labels).issubset({0, 1, 2})

    def test_predict_proba_valid(self, garch_detector, synthetic_returns):
        garch_detector.fit(synthetic_returns)
        probs = garch_detector.predict_proba()
        assert probs.shape == (len(synthetic_returns), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        assert (probs >= 0).all()

    def test_expanding_quantiles_no_lookahead(self, garch_detector, synthetic_returns):
        """Verify that regime labels use only past data."""
        garch_detector.fit(synthetic_returns)
        labels = garch_detector.predict()
        # First observation should be moderate (default) since no history
        assert labels[0] == 1

    def test_conditional_volatility_positive(self, garch_detector, synthetic_returns):
        garch_detector.fit(synthetic_returns)
        vol = garch_detector.get_conditional_volatility()
        assert (vol > 0).all()

    def test_persistence_computation(self, garch_detector, synthetic_returns):
        garch_detector.fit(synthetic_returns)
        pers = garch_detector.get_persistence()
        assert "persistence" in pers
        assert pers["alpha"] > 0
        assert pers["beta"] > 0

    def test_bic_returns_float(self, garch_detector, synthetic_returns):
        garch_detector.fit(synthetic_returns)
        bic = garch_detector.bic()
        assert isinstance(bic, float)
        assert np.isfinite(bic)

    def test_unfitted_raises(self, garch_detector):
        with pytest.raises(RuntimeError, match="not fitted"):
            garch_detector.predict()

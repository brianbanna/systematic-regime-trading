"""Tests for GARCH model module."""

import pandas as pd
import numpy as np
import pytest

from systematic_regime_trading.models.garch import (
    fit_garch,
    extract_conditional_volatility,
    compute_persistence,
    get_model_summary,
    forecast_volatility,
)


@pytest.fixture
def synthetic_returns():
    """Generate synthetic returns with volatility clustering."""
    rng = np.random.default_rng(42)
    n = 1000
    returns = np.zeros(n)
    vol = 0.01

    for t in range(1, n):
        vol = 0.0001 + 0.1 * returns[t - 1] ** 2 + 0.85 * vol
        returns[t] = rng.normal(0, np.sqrt(vol))

    dates = pd.bdate_range("2016-01-01", periods=n)
    return pd.Series(returns, index=dates, name="returns")


class TestFitGarch:
    def test_fits_successfully(self, synthetic_returns):
        model, results = fit_garch(synthetic_returns)
        assert results is not None
        assert "alpha[1]" in results.params.index
        assert "beta[1]" in results.params.index

    def test_uses_student_t_by_default(self, synthetic_returns):
        model, results = fit_garch(synthetic_returns)
        # The model should use Student-t distribution
        assert "nu" in results.params.index  # degrees of freedom param


class TestConditionalVolatility:
    def test_positive_volatility(self, synthetic_returns):
        _, results = fit_garch(synthetic_returns)
        cond_vol = extract_conditional_volatility(results)
        assert (cond_vol > 0).all()
        assert len(cond_vol) == len(synthetic_returns)


class TestPersistence:
    def test_persistence_computation(self, synthetic_returns):
        _, results = fit_garch(synthetic_returns)
        pers = compute_persistence(results)

        assert "alpha" in pers
        assert "beta" in pers
        assert "persistence" in pers
        assert "half_life" in pers
        assert "is_stationary" in pers

        # Alpha and beta should be positive
        assert pers["alpha"] > 0
        assert pers["beta"] > 0

        # Persistence should be alpha + beta
        np.testing.assert_allclose(
            pers["persistence"], pers["alpha"] + pers["beta"], atol=1e-10,
        )


class TestModelSummary:
    def test_returns_dataframe(self, synthetic_returns):
        _, results = fit_garch(synthetic_returns)
        summary = get_model_summary(results)
        assert isinstance(summary, pd.DataFrame)
        assert "Coefficient" in summary.columns
        assert "p-value" in summary.columns


class TestForecast:
    def test_forecast_horizon(self, synthetic_returns):
        _, results = fit_garch(synthetic_returns)
        forecast = forecast_volatility(results, horizon=5)
        assert len(forecast) == 5
        assert "volatility_annual" in forecast.columns
        assert (forecast["volatility"] > 0).all()

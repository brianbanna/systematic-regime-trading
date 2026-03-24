"""Tests for model diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from systematic_regime_trading.models.diagnostics import (
    select_hmm_states,
    select_kmeans_clusters,
    compute_regime_stability,
    compute_diagnostics_per_window,
)


@pytest.fixture
def synthetic_vol():
    rng = np.random.default_rng(42)
    return pd.Series(np.abs(rng.normal(0.02, 0.01, 500)))


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(42)
    calm = rng.multivariate_normal([0.01, 100], [[0.001, 0], [0, 500]], 200)
    moderate = rng.multivariate_normal([0.03, 300], [[0.002, 0], [0, 1000]], 150)
    turbulent = rng.multivariate_normal([0.06, 600], [[0.005, 0], [0, 2000]], 100)
    return pd.DataFrame(
        np.vstack([calm, moderate, turbulent]),
        columns=["vol", "volume"],
    )


class TestSelectHMMStates:
    def test_returns_best_n_states(self, synthetic_vol):
        result = select_hmm_states(synthetic_vol, state_range=[2, 3])
        assert "best_n_states" in result
        assert result["best_n_states"] in [2, 3]
        assert "bic_scores" in result
        assert len(result["bic_scores"]) == 2


class TestSelectKMeansClusters:
    def test_returns_best_k(self, synthetic_features):
        result = select_kmeans_clusters(synthetic_features, k_range=[2, 3])
        assert "best_k" in result
        assert result["best_k"] in [2, 3]
        assert "silhouette_scores" in result


class TestRegimeStability:
    def test_stability_between_0_and_1(self):
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 0, 0] * 5)
        stability = compute_regime_stability(labels, window=5)
        valid = stability.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()


class TestDiagnosticsPerWindow:
    def test_computes_metrics(self):
        result = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=20),
            "regime_label": [0] * 10 + [1] * 5 + [2] * 5,
            "regime_prob_calm": [0.8] * 10 + [0.1] * 5 + [0.05] * 5,
            "regime_prob_moderate": [0.1] * 10 + [0.8] * 5 + [0.1] * 5,
            "regime_prob_turbulent": [0.1] * 10 + [0.1] * 5 + [0.85] * 5,
            "window_id": [0] * 10 + [1] * 10,
        })
        diag = compute_diagnostics_per_window(result)
        assert len(diag) == 2
        assert "pct_calm" in diag.columns
        assert "avg_confidence" in diag.columns
        assert "n_transitions" in diag.columns

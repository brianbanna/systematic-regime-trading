"""Tests for EnsembleRegimeDetector class."""

import numpy as np
import pandas as pd
import pytest

from systematic_regime_trading.models.ensemble import (
    EnsembleRegimeDetector,
    generate_weight_combinations,
)


@pytest.fixture
def sample_probs():
    """Three models with probability outputs for 10 time steps."""
    rng = np.random.default_rng(42)
    n = 10
    probs = {}
    for name in ["hmm", "garch", "kmeans"]:
        raw = rng.random((n, 3))
        raw = raw / raw.sum(axis=1, keepdims=True)
        probs[name] = raw
    return probs


class TestEnsembleRegimeDetector:
    def test_combine_returns_valid_probs(self, sample_probs):
        ensemble = EnsembleRegimeDetector()
        combined = ensemble.combine(sample_probs)
        assert combined.shape == (10, 3)
        np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-10)
        assert (combined >= 0).all()

    def test_predict_returns_labels(self, sample_probs):
        ensemble = EnsembleRegimeDetector()
        labels = ensemble.predict(sample_probs)
        assert len(labels) == 10
        assert set(labels).issubset({0, 1, 2})

    def test_custom_weights(self, sample_probs):
        ensemble = EnsembleRegimeDetector(
            weights={"hmm": 0.5, "garch": 0.3, "kmeans": 0.2},
        )
        combined = ensemble.combine(sample_probs)
        np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-10)

    def test_missing_model_renormalizes(self, sample_probs):
        """If only 2 of 3 models available, renormalize weights."""
        partial = {"hmm": sample_probs["hmm"], "garch": sample_probs["garch"]}
        ensemble = EnsembleRegimeDetector()
        combined = ensemble.combine(partial)
        assert combined.shape == (10, 3)
        np.testing.assert_allclose(combined.sum(axis=1), 1.0, atol=1e-10)

    def test_agreement(self):
        labels = {
            "hmm": np.array([0, 0, 1, 2, 2]),
            "garch": np.array([0, 1, 1, 2, 0]),
            "kmeans": np.array([0, 0, 1, 1, 2]),
        }
        ensemble = EnsembleRegimeDetector()
        agreement = ensemble.agreement(labels)
        assert len(agreement) == 5
        assert agreement.iloc[0] == 1.0  # All agree on 0
        assert agreement.iloc[2] == 1.0  # All agree on 1


class TestWeightCombinations:
    def test_all_sum_to_one(self):
        combos = generate_weight_combinations()
        for w in combos:
            assert abs(sum(w) - 1.0) < 1e-10

    def test_respects_bounds(self):
        combos = generate_weight_combinations(
            min_weight=0.2, max_weight=0.6, step=0.1,
        )
        for w in combos:
            for wi in w:
                assert wi >= 0.19  # float tolerance
                assert wi <= 0.61

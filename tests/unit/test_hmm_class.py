"""Tests for HMMRegimeDetector class."""

import numpy as np
import pandas as pd
import pytest

from systematic_regime_trading.models.hmm import HMMRegimeDetector


@pytest.fixture
def hmm_detector():
    config = {"n_states": 3, "n_iter": 100, "covariance_type": "full", "random_state": 42}
    return HMMRegimeDetector(config)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    # 3 distinct regimes
    calm = rng.normal(0.01, 0.002, 200)
    moderate = rng.normal(0.025, 0.005, 150)
    turbulent = rng.normal(0.05, 0.01, 100)
    return pd.Series(np.concatenate([calm, moderate, turbulent]))


class TestHMMRegimeDetector:
    def test_fit_returns_self(self, hmm_detector, synthetic_data):
        result = hmm_detector.fit(synthetic_data)
        assert result is hmm_detector
        assert hmm_detector.is_fitted_

    def test_predict_returns_correct_shape(self, hmm_detector, synthetic_data):
        hmm_detector.fit(synthetic_data)
        labels = hmm_detector.predict(synthetic_data)
        assert len(labels) == len(synthetic_data)
        assert set(labels).issubset({0, 1, 2})

    def test_predict_proba_returns_probabilities(self, hmm_detector, synthetic_data):
        hmm_detector.fit(synthetic_data)
        probs = hmm_detector.predict_proba(synthetic_data)
        assert probs.shape == (len(synthetic_data), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        assert (probs >= 0).all()

    def test_state_ordering_ascending(self, hmm_detector, synthetic_data):
        hmm_detector.fit(synthetic_data)
        means = hmm_detector.model_.means_.flatten()
        # Means should be in ascending order after fit
        assert all(means[i] <= means[i + 1] for i in range(len(means) - 1))

    def test_transition_matrix_valid(self, hmm_detector, synthetic_data):
        hmm_detector.fit(synthetic_data)
        trans = hmm_detector.get_transition_matrix()
        assert trans.shape == (3, 3)
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-10)

    def test_bic_returns_float(self, hmm_detector, synthetic_data):
        hmm_detector.fit(synthetic_data)
        bic = hmm_detector.bic(synthetic_data)
        assert isinstance(bic, float)
        assert np.isfinite(bic)

    def test_unfitted_raises(self, hmm_detector, synthetic_data):
        with pytest.raises(RuntimeError, match="not fitted"):
            hmm_detector.predict(synthetic_data)

    def test_5_state_remap(self):
        rng = np.random.default_rng(99)
        # Longer data with 5 distinct levels for stable 5-state fit
        data = pd.Series(np.concatenate([
            rng.normal(0.005, 0.001, 300),
            rng.normal(0.01, 0.002, 300),
            rng.normal(0.02, 0.003, 200),
            rng.normal(0.035, 0.005, 200),
            rng.normal(0.06, 0.008, 200),
        ]))
        config = {
            "n_states": 5, "n_iter": 200, "covariance_type": "diag",
            "random_state": 42,
            "remap_to_3": {"calm": [0, 1], "moderate": [2], "turbulent": [3, 4]},
        }
        detector = HMMRegimeDetector(config)
        detector.fit(data)
        labels = detector.predict(data)
        assert set(labels).issubset({0, 1, 2})

        probs = detector.predict_proba(data)
        assert probs.shape == (len(data), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

"""Tests for KMeansRegimeDetector class."""

import numpy as np
import pandas as pd
import pytest

from systematic_regime_trading.models.kmeans import KMeansRegimeDetector


@pytest.fixture
def kmeans_detector():
    config = {"n_clusters": 3, "random_state": 42}
    return KMeansRegimeDetector(config)


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(42)
    # 3 regimes with different feature distributions
    calm = rng.multivariate_normal([0.01, 100], [[0.001, 0], [0, 500]], 200)
    moderate = rng.multivariate_normal([0.03, 300], [[0.002, 0], [0, 1000]], 150)
    turbulent = rng.multivariate_normal([0.06, 600], [[0.005, 0], [0, 2000]], 100)
    return pd.DataFrame(
        np.vstack([calm, moderate, turbulent]),
        columns=["volatility", "volume"],
    )


class TestKMeansRegimeDetector:
    def test_fit_returns_self(self, kmeans_detector, synthetic_features):
        result = kmeans_detector.fit(synthetic_features)
        assert result is kmeans_detector
        assert kmeans_detector.is_fitted_

    def test_predict_returns_correct_shape(self, kmeans_detector, synthetic_features):
        kmeans_detector.fit(synthetic_features)
        labels = kmeans_detector.predict(synthetic_features)
        assert len(labels) == len(synthetic_features)
        assert set(labels).issubset({0, 1, 2})

    def test_predict_proba_valid(self, kmeans_detector, synthetic_features):
        kmeans_detector.fit(synthetic_features)
        probs = kmeans_detector.predict_proba(synthetic_features)
        assert probs.shape == (len(synthetic_features), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        assert (probs >= 0).all()

    def test_label_ordering(self, kmeans_detector, synthetic_features):
        """State 0 should have lower centroid mean than state 2."""
        kmeans_detector.fit(synthetic_features)
        centroids = kmeans_detector.get_centroids(original_scale=True)
        means = centroids.mean(axis=1)
        assert means[0] < means[2]

    def test_silhouette_positive(self, kmeans_detector, synthetic_features):
        kmeans_detector.fit(synthetic_features)
        sil = kmeans_detector.silhouette(synthetic_features)
        assert sil > 0

    def test_nan_input_raises(self, kmeans_detector):
        X = np.array([[1, 2], [np.nan, 3], [4, 5]])
        with pytest.raises(ValueError, match="NaN"):
            kmeans_detector.fit(X)

    def test_unfitted_raises(self, kmeans_detector, synthetic_features):
        with pytest.raises(RuntimeError, match="not fitted"):
            kmeans_detector.predict(synthetic_features)

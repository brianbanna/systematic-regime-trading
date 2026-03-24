"""
K-Means clustering for regime detection.

Labels ordered by mean volatility (ascending: 0=calm, 2=turbulent).
Pseudo-probabilities via inverse distance to cluster centers.

Key improvements over ADA:
- Class-based interface with .fit(), .predict(), .predict_proba()
- Inverse-distance pseudo-probabilities for proportional allocation
- Config-driven parameters
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List, Optional
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


class KMeansRegimeDetector:
    """
    K-Means clustering on market features.
    Labels ordered by mean volatility (ascending).
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config("models")["kmeans"]

        self.n_clusters = config.get("n_clusters", 3)
        self.random_state = config.get("random_state", 42)
        self.feature_config = config.get("features", {})

        self.pipeline_ = None
        self.cluster_order_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame) -> "KMeansRegimeDetector":
        """
        Fit scaler + KMeans on training data.

        Args:
            X: Feature DataFrame (rows=observations, cols=features)

        Returns:
            self
        """
        X_arr = self._to_array(X)
        self._validate_input(X_arr)

        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )),
        ])

        raw_labels = self.pipeline_.fit_predict(X_arr)

        # Order clusters by centroid magnitude (ascending = calm -> turbulent)
        kmeans = self.pipeline_.named_steps["kmeans"]
        scaler = self.pipeline_.named_steps["scaler"]

        # Get centroids in original scale
        centroids_scaled = kmeans.cluster_centers_
        centroids_original = scaler.inverse_transform(centroids_scaled)

        # Sort by mean of all features (proxy for volatility)
        centroid_means = centroids_original.mean(axis=1)
        self.cluster_order_ = np.argsort(centroid_means)

        self.is_fitted_ = True

        # Compute silhouette score for diagnostics
        if len(np.unique(raw_labels)) > 1:
            sil = silhouette_score(X_arr, raw_labels)
            logger.info(
                f"KMeans fitted: {self.n_clusters} clusters, "
                f"silhouette={sil:.3f}"
            )
        else:
            logger.warning("KMeans: all points assigned to single cluster")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Regime labels ordered by volatility (0=calm, 2=turbulent).

        Args:
            X: Feature DataFrame

        Returns:
            Array of regime labels
        """
        self._check_fitted()
        X_arr = self._to_array(X)
        raw_labels = self.pipeline_.predict(X_arr)
        return self._reorder_labels(raw_labels)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Pseudo-probabilities based on inverse distance to cluster centers.

        P(regime_k) = (1/d_k) / sum(1/d_j) for all clusters j.

        Returns:
            Array of shape (n_samples, n_clusters) with probabilities
        """
        self._check_fitted()
        X_arr = self._to_array(X)
        scaler = self.pipeline_.named_steps["scaler"]
        kmeans = self.pipeline_.named_steps["kmeans"]

        X_scaled = scaler.transform(X_arr)
        distances = kmeans.transform(X_scaled)  # distances to each centroid

        # Reorder columns to match sorted labels
        distances = distances[:, self.cluster_order_]

        # Inverse distance with floor to avoid division by zero
        inv_dist = 1.0 / np.maximum(distances, 1e-10)
        probs = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        return probs

    def get_centroids(self, original_scale: bool = True) -> np.ndarray:
        """Get cluster centroids, optionally in original feature scale."""
        self._check_fitted()
        kmeans = self.pipeline_.named_steps["kmeans"]
        centroids = kmeans.cluster_centers_

        if original_scale:
            scaler = self.pipeline_.named_steps["scaler"]
            centroids = scaler.inverse_transform(centroids)

        # Reorder to match label ordering
        return centroids[self.cluster_order_]

    def silhouette(self, X: pd.DataFrame) -> float:
        """Compute silhouette score on data."""
        self._check_fitted()
        X_arr = self._to_array(X)
        labels = self.predict(X)
        if len(np.unique(labels)) <= 1:
            return -1.0
        return silhouette_score(X_arr, labels)

    # --- Private helpers ---

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, pd.Series):
            return X.values.reshape(-1, 1)
        return X

    def _validate_input(self, X: np.ndarray):
        if np.isnan(X).any():
            raise ValueError(
                f"Input contains {np.isnan(X).sum()} NaN values. "
                "Clean data before clustering."
            )

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    def _reorder_labels(self, labels: np.ndarray) -> np.ndarray:
        """Map raw cluster IDs to sorted order."""
        reverse_map = {old: new for new, old in enumerate(self.cluster_order_)}
        return np.array([reverse_map[l] for l in labels])


# --- Backwards-compatible function wrappers ---

def clustering_pipeline(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    pipe: Pipeline = None,
    prediction_only: bool = False,
    train: bool = True,
) -> Tuple[np.ndarray, Pipeline]:
    """Build or use a Scaler -> KMeans pipeline."""
    if isinstance(X, np.ndarray):
        if np.isnan(X).any():
            raise ValueError(f"Input contains {np.isnan(X).sum()} NaN values.")
    else:
        if X.isnull().any().any():
            raise ValueError("Input contains NaN values.")

    if pipe is not None and prediction_only:
        labels = pipe.predict(X)
    elif pipe is not None and train:
        labels = pipe.fit_predict(X)
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state)),
        ])
        labels = pipe.fit_predict(X)

    return labels, pipe


def rearrange_labels_by_feature(
    df: pd.DataFrame, labels: np.ndarray, feature_name: str,
) -> np.ndarray:
    df_temp = df.copy()
    df_temp["cluster"] = labels
    cluster_means = df_temp.groupby("cluster")[feature_name].mean().sort_values()
    mapping = {old: new for new, old in enumerate(cluster_means.index)}
    return np.array([mapping[l] for l in labels])


def compute_transition_matrix(regimes: np.ndarray) -> np.ndarray:
    n_states = len(np.unique(regimes))
    counts = np.zeros((n_states, n_states))
    arr = np.array(regimes, dtype=int)
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, sums, where=sums != 0, out=np.zeros_like(counts))


def compute_avg_duration(labels: np.ndarray, regimes: list) -> dict:
    durations = {r: [] for r in regimes}
    arr = np.array(labels)
    current = arr[0]
    dur = 1
    for i in range(1, len(arr)):
        if arr[i] == current:
            dur += 1
        else:
            durations[current].append(dur)
            current = arr[i]
            dur = 1
    durations[current].append(dur)
    return {r: np.mean(d) if d else 0 for r, d in durations.items()}

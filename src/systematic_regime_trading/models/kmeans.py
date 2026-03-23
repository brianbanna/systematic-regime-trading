"""
K-Means clustering for volatility regime detection.

Core clustering pipeline, feature importance, label reordering,
transition analysis. Plotting functions removed (see visualization/).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple, List


def clustering_pipeline(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    pipe: Pipeline = None,
    prediction_only: bool = False,
    train: bool = True,
) -> Tuple[np.ndarray, Pipeline]:
    """
    Build or use a Scaler -> PCA -> KMeans pipeline.

    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
        pipe: Optional prebuilt pipeline
        prediction_only: If True, only predict with provided pipeline
        train: If True, fit the provided pipeline

    Returns:
        Tuple of (labels, fitted_pipeline)
    """
    if isinstance(X, np.ndarray):
        if np.isnan(X).any():
            raise ValueError(
                f"Input contains {np.isnan(X).sum()} NaN values. "
                "Please clean data before clustering."
            )
    else:
        if X.isnull().any().any():
            nan_cols = X.isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            raise ValueError(
                f"Input contains NaN values in columns: {nan_cols.to_dict()}. "
                "Please clean data before clustering."
            )

    if pipe is not None and prediction_only:
        labels = pipe.predict(X)
    elif pipe is not None and train:
        labels = pipe.fit_predict(X)
    else:
        n_features = X.shape[1]
        n_components = min(2, n_features)

        if n_features < 3:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state)),
            ])
        else:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state)),
            ])
        labels = pipe.fit_predict(X)

    return labels, pipe


def get_feature_importance(
    pipe: Pipeline, feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Calculate feature importance based on PCA component weights.

    Args:
        pipe: Fitted pipeline containing PCA step
        feature_names: List of original feature names

    Returns:
        List of (feature_name, importance) tuples sorted descending
    """
    pca = pipe.named_steps["pca"]
    importance = np.abs(pca.components_).sum(axis=0)
    importance = importance / importance.sum()

    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance


def rearrange_labels_by_feature(
    df: pd.DataFrame, labels: np.ndarray, feature_name: str
) -> np.ndarray:
    """
    Reorder cluster labels by ascending mean of target feature.

    Ensures State 0 = lowest feature value, State N = highest.

    Args:
        df: DataFrame containing the feature
        labels: Original cluster labels
        feature_name: Feature to sort by

    Returns:
        Relabeled array
    """
    df_temp = df.copy()
    df_temp["cluster"] = labels
    cluster_means = df_temp.groupby("cluster")[feature_name].mean().sort_values()
    label_mapping = {old: new for new, old in enumerate(cluster_means.index)}
    return np.array([label_mapping[label] for label in labels])


def compute_transition_matrix(regimes: np.ndarray) -> np.ndarray:
    """
    Compute transition probability matrix from regime sequence.

    Args:
        regimes: Array of regime labels (integers)

    Returns:
        Transition probability matrix (n_states x n_states)
    """
    n_states = len(np.unique(regimes))
    transition_counts = np.zeros((n_states, n_states))

    regime_array = np.array(regimes, dtype=int)

    for i in range(len(regime_array) - 1):
        transition_counts[regime_array[i], regime_array[i + 1]] += 1

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_counts, row_sums,
        where=row_sums != 0, out=np.zeros_like(transition_counts),
    )

    return transition_probs


def compute_avg_duration(
    labels: np.ndarray, regimes: list
) -> dict:
    """
    Compute average consecutive duration for each regime.

    Args:
        labels: Array of regime labels
        regimes: List of unique regime values

    Returns:
        Dict mapping regime -> average duration in periods
    """
    durations = {r: [] for r in regimes}
    labels_array = np.array(labels)
    current_regime = labels_array[0]
    current_duration = 1

    for i in range(1, len(labels_array)):
        if labels_array[i] == current_regime:
            current_duration += 1
        else:
            durations[current_regime].append(current_duration)
            current_regime = labels_array[i]
            current_duration = 1
    durations[current_regime].append(current_duration)

    return {r: np.mean(d) if d else 0 for r, d in durations.items()}

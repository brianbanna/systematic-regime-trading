"""
Model selection and diagnostics for regime detection models.

HMM: compare 2-5 states using BIC
GARCH: compare GARCH(1,1) vs GARCH(2,1), Normal vs Student-t
K-Means: silhouette score for k=2,3,4

Results stored per walk-forward window for diagnostic reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import warnings

from systematic_regime_trading.models.hmm import HMMRegimeDetector
from systematic_regime_trading.models.garch import GARCHRegimeDetector
from systematic_regime_trading.models.kmeans import KMeansRegimeDetector

logger = logging.getLogger(__name__)


def select_hmm_states(
    data: np.ndarray,
    state_range: List[int] = None,
    config_base: dict = None,
) -> Dict:
    """
    Compare HMM with different number of states using BIC.

    Args:
        data: Training data (1D array or Series)
        state_range: List of n_states to try (default: [2, 3, 4, 5])
        config_base: Base HMM config to override n_states

    Returns:
        Dict with 'best_n_states', 'bic_scores', 'all_results'
    """
    if state_range is None:
        state_range = [2, 3, 4, 5]

    if config_base is None:
        config_base = {
            "n_iter": 1000, "covariance_type": "full", "random_state": 42,
        }

    results = []
    for n in state_range:
        try:
            config = {**config_base, "n_states": n}
            model = HMMRegimeDetector(config)
            model.fit(data)
            bic = model.bic(data)
            ll = model.score(data)
            results.append({
                "n_states": n, "bic": bic, "log_likelihood": ll,
                "converged": model.model_.monitor_.converged,
            })
        except Exception as e:
            logger.warning(f"HMM with {n} states failed: {e}")
            results.append({"n_states": n, "bic": np.inf, "log_likelihood": -np.inf})

    results_df = pd.DataFrame(results)
    best_idx = results_df["bic"].idxmin()
    best_n = int(results_df.loc[best_idx, "n_states"])

    logger.info(
        f"HMM state selection: best={best_n} states "
        f"(BIC={results_df.loc[best_idx, 'bic']:.1f})"
    )

    return {
        "best_n_states": best_n,
        "bic_scores": dict(zip(results_df["n_states"], results_df["bic"])),
        "all_results": results_df,
    }


def select_garch_spec(
    returns: pd.Series,
    specs: List[dict] = None,
) -> Dict:
    """
    Compare GARCH specifications using BIC.

    Args:
        returns: Return series for fitting
        specs: List of GARCH spec dicts to try

    Returns:
        Dict with 'best_spec', 'bic_scores', 'all_results'
    """
    if specs is None:
        specs = [
            {"p": 1, "q": 1, "distribution": "normal", "label": "GARCH(1,1)-N"},
            {"p": 1, "q": 1, "distribution": "t", "label": "GARCH(1,1)-t"},
            {"p": 2, "q": 1, "distribution": "t", "label": "GARCH(2,1)-t"},
        ]

    results = []
    for spec in specs:
        label = spec.pop("label", f"p={spec.get('p')},q={spec.get('q')},d={spec.get('distribution')}")
        try:
            config = {
                "p": spec.get("p", 1),
                "q": spec.get("q", 1),
                "mean_model": "Constant",
                "vol_model": "GARCH",
                "distribution": spec.get("distribution", "t"),
                "return_scaling": 100,
            }
            model = GARCHRegimeDetector(config)
            model.fit(returns)
            bic = model.bic()
            persistence = model.get_persistence()
            results.append({
                "spec": label, "bic": bic,
                "persistence": persistence["persistence"],
                "is_stationary": persistence["is_stationary"],
            })
        except Exception as e:
            logger.warning(f"GARCH spec {label} failed: {e}")
            results.append({"spec": label, "bic": np.inf})

    results_df = pd.DataFrame(results)
    best_idx = results_df["bic"].idxmin()
    best_spec = results_df.loc[best_idx, "spec"]

    logger.info(f"GARCH spec selection: best={best_spec} (BIC={results_df.loc[best_idx, 'bic']:.1f})")

    return {
        "best_spec": best_spec,
        "bic_scores": dict(zip(results_df["spec"], results_df["bic"])),
        "all_results": results_df,
    }


def select_kmeans_clusters(
    X: np.ndarray,
    k_range: List[int] = None,
    config_base: dict = None,
) -> Dict:
    """
    Compare K-Means with different k using silhouette score.

    Args:
        X: Feature matrix
        k_range: List of k values to try (default: [2, 3, 4])
        config_base: Base K-Means config

    Returns:
        Dict with 'best_k', 'silhouette_scores', 'all_results'
    """
    if k_range is None:
        k_range = [2, 3, 4]

    if config_base is None:
        config_base = {"random_state": 42}

    results = []
    for k in k_range:
        try:
            config = {**config_base, "n_clusters": k}
            model = KMeansRegimeDetector(config)
            model.fit(X)
            sil = model.silhouette(X)
            results.append({"k": k, "silhouette": sil})
        except Exception as e:
            logger.warning(f"KMeans with k={k} failed: {e}")
            results.append({"k": k, "silhouette": -1.0})

    results_df = pd.DataFrame(results)
    best_idx = results_df["silhouette"].idxmax()
    best_k = int(results_df.loc[best_idx, "k"])

    logger.info(
        f"KMeans cluster selection: best k={best_k} "
        f"(silhouette={results_df.loc[best_idx, 'silhouette']:.3f})"
    )

    return {
        "best_k": best_k,
        "silhouette_scores": dict(zip(results_df["k"], results_df["silhouette"])),
        "all_results": results_df,
    }


def compute_regime_stability(labels: np.ndarray, window: int = 20) -> pd.Series:
    """
    Compute rolling regime stability (fraction of window in same regime).

    Args:
        labels: Array of regime labels
        window: Rolling window size

    Returns:
        Series of stability scores (0 to 1)
    """
    n = len(labels)
    stability = np.zeros(n)

    for t in range(window, n):
        window_labels = labels[t - window:t]
        _, counts = np.unique(window_labels, return_counts=True)
        stability[t] = counts.max() / window

    # Fill warmup period
    stability[:window] = np.nan

    return pd.Series(stability)


def compute_diagnostics_per_window(
    walk_forward_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute diagnostic metrics per walk-forward window.

    Args:
        walk_forward_results: Output from walk_forward_train

    Returns:
        DataFrame with per-window metrics
    """
    diagnostics = []

    for wid in walk_forward_results["window_id"].unique():
        window = walk_forward_results[walk_forward_results["window_id"] == wid]

        labels = window["regime_label"].values
        n = len(labels)

        # Regime distribution
        unique, counts = np.unique(labels, return_counts=True)
        regime_dist = dict(zip(unique, counts / n))

        # Number of regime transitions
        transitions = (np.diff(labels) != 0).sum()

        # Average confidence (max probability per observation)
        prob_cols = ["regime_prob_calm", "regime_prob_moderate", "regime_prob_turbulent"]
        if all(c in window.columns for c in prob_cols):
            max_probs = window[prob_cols].values.max(axis=1)
            avg_confidence = max_probs.mean()
        else:
            avg_confidence = np.nan

        diagnostics.append({
            "window_id": wid,
            "n_observations": n,
            "date_start": window["Date"].min(),
            "date_end": window["Date"].max(),
            "pct_calm": regime_dist.get(0, 0),
            "pct_moderate": regime_dist.get(1, 0),
            "pct_turbulent": regime_dist.get(2, 0),
            "n_transitions": transitions,
            "transition_rate": transitions / max(n - 1, 1),
            "avg_confidence": avg_confidence,
        })

    return pd.DataFrame(diagnostics)

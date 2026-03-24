"""
Ensemble methods for combining regime detection models.

Combines HMM, GARCH, and KMeans predictions using probability-weighted
voting (not just hard labels). Produces smoother signals with less turnover.

Key improvement over ADA: combines probabilities rather than hard labels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from systematic_regime_trading.utils.config import load_config

logger = logging.getLogger(__name__)


class EnsembleRegimeDetector:
    """
    Combines HMM, GARCH, and KMeans predictions.
    Uses probability-weighted voting (not just hard labels).
    """

    def __init__(
        self,
        weights: dict = None,
        config: dict = None,
    ):
        if config is None:
            config = load_config("models")["ensemble"]

        self.n_regimes = config.get("n_regimes", 3)
        self.method = config.get("method", "weighted_vote")

        if weights is None:
            # Equal weights by default
            self.weights = {"hmm": 1/3, "garch": 1/3, "kmeans": 1/3}
        else:
            self.weights = weights

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def combine(self, probs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted average of regime probabilities.

        Args:
            probs: Dict mapping model name to probability array
                   Each array has shape (n_samples, n_regimes)

        Returns:
            Combined probabilities of shape (n_samples, n_regimes)
        """
        available = {k: v for k, v in probs.items() if k in self.weights}

        if not available:
            raise ValueError(
                f"No matching models. Expected keys from {list(self.weights.keys())}, "
                f"got {list(probs.keys())}"
            )

        # Renormalize weights for available models only
        total_w = sum(self.weights[k] for k in available)
        norm_weights = {k: self.weights[k] / total_w for k in available}

        # Weighted sum
        first_key = next(iter(available))
        n_samples = available[first_key].shape[0]
        combined = np.zeros((n_samples, self.n_regimes))

        for model_name, prob_array in available.items():
            combined += norm_weights[model_name] * prob_array

        return combined

    def predict(self, probs: Dict[str, np.ndarray]) -> np.ndarray:
        """Hard labels from argmax of combined probabilities."""
        combined = self.combine(probs)
        return np.argmax(combined, axis=1)

    def predict_proba(self, probs: Dict[str, np.ndarray]) -> np.ndarray:
        """Combined probabilities (same as combine)."""
        return self.combine(probs)

    def agreement(self, labels: Dict[str, np.ndarray]) -> pd.Series:
        """
        Compute agreement rate between models over time.

        Returns Series of agreement fraction (0 to 1) at each time step.
        """
        model_names = list(labels.keys())
        if len(model_names) < 2:
            return pd.Series(1.0, index=range(len(next(iter(labels.values())))))

        n = len(labels[model_names[0]])
        n_models = len(model_names)
        agreement = np.zeros(n)

        for t in range(n):
            votes = [labels[m][t] for m in model_names]
            # Fraction of models that agree with the majority vote
            values, counts = np.unique(votes, return_counts=True)
            agreement[t] = counts.max() / n_models

        return pd.Series(agreement)


# --- Weight optimization ---

def generate_weight_combinations(
    n_models: int = 3,
    min_weight: float = 0.10,
    max_weight: float = 0.80,
    step: float = 0.05,
) -> List[Tuple[float, ...]]:
    """Generate valid weight combinations that sum to 1."""
    weights_range = np.arange(min_weight, max_weight + step, step)
    combinations = []

    if n_models == 3:
        for w1 in weights_range:
            for w2 in weights_range:
                w3 = 1.0 - w1 - w2
                if min_weight <= w3 <= max_weight:
                    combinations.append((round(w1, 2), round(w2, 2), round(w3, 2)))

    return combinations


def optimize_ensemble_weights(
    probs_dict: Dict[str, np.ndarray],
    eval_func,
    model_names: List[str] = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    Grid search over weight combinations for optimal ensemble.

    Args:
        probs_dict: Dict mapping model name to probability arrays
        eval_func: Function that takes (combined_probs, labels) and returns score dict
        model_names: Names of models (default: keys of probs_dict)
        config: Ensemble config from models.yaml

    Returns:
        DataFrame with weights and scores, sorted by composite_score
    """
    if config is None:
        config = load_config("models")["ensemble"]

    if model_names is None:
        model_names = list(probs_dict.keys())

    combos = generate_weight_combinations(
        n_models=len(model_names),
        min_weight=config.get("min_weight", 0.10),
        max_weight=config.get("max_weight", 0.80),
        step=config.get("weight_step", 0.05),
    )

    results = []
    for weights in combos:
        weight_dict = dict(zip(model_names, weights))
        ensemble = EnsembleRegimeDetector(weights=weight_dict, config=config)
        combined = ensemble.combine(probs_dict)
        labels = np.argmax(combined, axis=1)

        scores = eval_func(combined, labels)
        result = {f"w_{name}": w for name, w in zip(model_names, weights)}
        result.update(scores)
        results.append(result)

    results_df = pd.DataFrame(results)
    if "composite_score" in results_df.columns:
        results_df = results_df.sort_values("composite_score", ascending=False)

    return results_df


# --- Backwards-compatible function wrappers ---

def majority_vote_ensemble(labels_list: List[np.ndarray]) -> np.ndarray:
    stacked = np.column_stack(labels_list)
    result = []
    for row in stacked:
        values, counts = np.unique(row, return_counts=True)
        result.append(values[np.argmax(counts)])
    return np.array(result)


def weighted_vote_ensemble(
    labels_list: List[np.ndarray], weights: List[float],
) -> np.ndarray:
    all_labels = np.unique(np.concatenate(labels_list))
    n_points = len(labels_list[0])
    result = np.zeros(n_points, dtype=int)
    for t in range(n_points):
        label_weights = {}
        for labels, w in zip(labels_list, weights):
            label = labels[t]
            label_weights[label] = label_weights.get(label, 0) + w
        result[t] = max(label_weights, key=label_weights.get)
    return result


def ensemble_voting(
    labels_model1: np.ndarray, labels_model2: np.ndarray,
    weights: Tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    result = []
    for l1, l2 in zip(labels_model1, labels_model2):
        result.append(float(l1) if l1 == l2 else (l1 + l2) / 2.0)
    return np.array(result)


def remap_ensemble_to_3_states(
    labels_ensemble: np.ndarray, remap_dict: Dict = None,
) -> np.ndarray:
    if remap_dict is None:
        remap_dict = {0: 0, 0.5: 0, 1: 1, 1.5: 2, 2: 2}
    return np.array([remap_dict[l] for l in labels_ensemble])


def get_optimal_weights(results_df: pd.DataFrame, n_models: int = 3) -> tuple:
    best_row = results_df.iloc[0]
    return tuple(best_row[f"w{i}"] for i in range(n_models))

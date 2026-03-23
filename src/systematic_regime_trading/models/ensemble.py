"""
Ensemble methods for combining regime detection models.

Majority voting, weighted voting, weight optimization via grid search,
and 5-state to 3-state remapping.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def majority_vote_ensemble(
    labels_list: List[np.ndarray],
) -> np.ndarray:
    """
    Simple majority vote from multiple models.

    Args:
        labels_list: List of label arrays from different models

    Returns:
        Array of majority-vote labels
    """
    stacked = np.column_stack(labels_list)
    result = []
    for row in stacked:
        values, counts = np.unique(row, return_counts=True)
        result.append(values[np.argmax(counts)])
    return np.array(result)


def weighted_vote_ensemble(
    labels_list: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Weighted vote across models.

    Each model's prediction is weighted. The label with highest total
    weight wins for each time point.

    Args:
        labels_list: List of label arrays
        weights: Weight per model (should sum to 1)

    Returns:
        Array of weighted-vote labels
    """
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
    labels_model1: np.ndarray,
    labels_model2: np.ndarray,
    weights: Tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """
    Combine two model predictions with intermediate classes.

    When models agree, use agreed label. When they disagree, use the
    average as an intermediate class (e.g., 0.5, 1.5).

    Args:
        labels_model1: Predictions from first model
        labels_model2: Predictions from second model
        weights: Voting weights

    Returns:
        Array with possible intermediate classes (0, 0.5, 1, 1.5, 2)
    """
    ensemble_labels = []
    for label1, label2 in zip(labels_model1, labels_model2):
        if label1 == label2:
            ensemble_labels.append(float(label1))
        else:
            ensemble_labels.append((label1 + label2) / 2.0)
    return np.array(ensemble_labels)


def generate_weight_combinations(
    n_models: int = 3,
    min_weight: float = 0.10,
    max_weight: float = 0.80,
    step: float = 0.05,
) -> List[Tuple[float, ...]]:
    """
    Generate valid weight combinations that sum to 1.

    Args:
        n_models: Number of models
        min_weight: Minimum weight per model
        max_weight: Maximum weight per model
        step: Weight increment step

    Returns:
        List of weight tuples
    """
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
    labels_list: List[np.ndarray],
    eval_func,
    weight_combinations: List[Tuple[float, ...]],
) -> pd.DataFrame:
    """
    Grid search over weight combinations to find optimal ensemble.

    Args:
        labels_list: List of label arrays from individual models
        eval_func: Function that takes labels and returns a score dict
        weight_combinations: List of weight tuples to evaluate

    Returns:
        DataFrame with weights and evaluation scores, sorted by composite score
    """
    results = []

    for weights in weight_combinations:
        ensemble_labels = weighted_vote_ensemble(labels_list, list(weights))
        scores = eval_func(ensemble_labels)
        result = {f"w{i}": w for i, w in enumerate(weights)}
        result.update(scores)
        results.append(result)

    results_df = pd.DataFrame(results)

    if "composite_score" in results_df.columns:
        results_df = results_df.sort_values("composite_score", ascending=False)

    return results_df


def get_optimal_weights(results_df: pd.DataFrame, n_models: int = 3) -> tuple:
    """
    Extract optimal weights from optimization results.

    Args:
        results_df: DataFrame from optimize_ensemble_weights
        n_models: Number of models

    Returns:
        Tuple of optimal weights
    """
    best_row = results_df.iloc[0]
    return tuple(best_row[f"w{i}"] for i in range(n_models))


def remap_ensemble_to_3_states(
    labels_ensemble: np.ndarray, remap_dict: Dict = None
) -> np.ndarray:
    """
    Remap 5-state ensemble labels to 3-state labels.

    Default mapping:
    - 0 (Calm) -> 0, 0.5 (Calm-Moderate) -> 0
    - 1 (Moderate) -> 1
    - 1.5 (Moderate-Turbulent) -> 2, 2 (Turbulent) -> 2

    Args:
        labels_ensemble: Array of 5-state labels
        remap_dict: Optional custom mapping

    Returns:
        Array of 3-state labels
    """
    if remap_dict is None:
        remap_dict = {0: 0, 0.5: 0, 1: 1, 1.5: 2, 2: 2}

    return np.array([remap_dict[label] for label in labels_ensemble])

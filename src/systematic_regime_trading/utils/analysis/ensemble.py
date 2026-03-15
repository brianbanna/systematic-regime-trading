"""
Purpose: Ensemble methods for combining multiple regime detection approaches.
Part of Step 7 of the ADA project (Method Comparison and Ensemble Framework).
"""

import numpy as np
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm


def majority_vote_ensemble(garch_labels, hmm_labels, kmeans_labels):
    """
    Simple majority vote: each method gets 1 vote.
    
    Returns most common label per day.
    In case of tie, returns the first label.
    """
    stacked = np.column_stack([garch_labels, hmm_labels, kmeans_labels])
    ensemble_labels = mode(stacked, axis=1, keepdims=False)[0].flatten()
    
    return ensemble_labels


def weighted_vote_ensemble(garch_labels, hmm_labels, kmeans_labels, weights):
    """
    Weighted vote with predefined coefficients.
    
    Args:
        garch_labels: Array of GARCH regime labels
        hmm_labels: Array of HMM regime labels
        kmeans_labels: Array of K-means regime labels
        weights: dict with keys 'garch', 'hmm', 'kmeans' (sum to 1.0)
    
    Returns:
        Array of ensemble regime labels
    """
    n_samples = len(garch_labels)
    n_regimes = 3  # Assuming 3-state models (0, 1, 2)
    
    # Initialize vote matrix
    vote_matrix = np.zeros((n_samples, n_regimes))
    
    # Add weighted votes
    for i in range(n_samples):
        vote_matrix[i, garch_labels[i]] += weights['garch']
        vote_matrix[i, hmm_labels[i]] += weights['hmm']
        vote_matrix[i, kmeans_labels[i]] += weights['kmeans']
    
    # Select regime with highest weighted vote
    ensemble_labels = np.argmax(vote_matrix, axis=1)
    
    return ensemble_labels


def generate_weight_combinations(step=0.05, min_weight=0.1, max_weight=0.8):
    """
    Generate valid weight combinations for three methods.
    
    Constraints:
    - All weights in [min_weight, max_weight]
    - Sum to 1.0
    
    Args:
        step: Grid resolution for weight search
        min_weight: Minimum weight per method (prevents ignoring methods)
        max_weight: Maximum weight per method (prevents dominance)
    
    Returns:
        List of weight dictionaries
    """
    weights_list = []
    
    for w_garch in np.arange(min_weight, max_weight + step, step):
        for w_hmm in np.arange(min_weight, max_weight + step, step):
            w_kmeans = 1.0 - w_garch - w_hmm
            
            # Check if w_kmeans is valid
            if min_weight <= w_kmeans <= max_weight:
                weights_list.append({
                    'garch': round(w_garch, 3),
                    'hmm': round(w_hmm, 3),
                    'kmeans': round(w_kmeans, 3)
                })
    
    return weights_list


def optimize_ensemble_weights(
    garch_labels, hmm_labels, kmeans_labels,
    volatility, dates, features_df,
    step=0.05, verbose=True
):
    """
    Systematic optimization of ensemble weights using grid search.
    
    Tests all valid weight combinations and evaluates each on 3 core metrics.
    
    Args:
        garch_labels, hmm_labels, kmeans_labels: Regime labels from each method
        volatility: Volatility series for evaluation
        dates: Date index for evaluation
        features_df: Feature DataFrame for evaluation
        step: Grid resolution
        verbose: Show progress bar
    
    Returns:
        DataFrame with all tested combinations and their scores
    """
    from .method_comparison import (
        compute_volatility_separation,
        compute_crisis_alignment,
        compute_temporal_coherence,
        compute_internal_consistency
    )
    
    weight_combinations = generate_weight_combinations(step=step)
    
    if verbose:
        print(f"Testing {len(weight_combinations)} weight combinations...")
    
    results = []
    
    iterator = tqdm(weight_combinations) if verbose else weight_combinations
    
    for weights in iterator:
        # Generate ensemble
        ensemble_labels = weighted_vote_ensemble(
            garch_labels, hmm_labels, kmeans_labels, weights
        )
        
        # Compute 3 core metrics
        vol_sep = compute_volatility_separation(ensemble_labels, volatility)
        crisis = compute_crisis_alignment(ensemble_labels, dates)
        coherence = compute_temporal_coherence(ensemble_labels)
        consistency = compute_internal_consistency(ensemble_labels, features_df)
        
        # Composite score: average of 3 core metrics (excluding internal_consistency)
        composite_score = (vol_sep + crisis + coherence) / 3
        
        results.append({
            'w_garch': weights['garch'],
            'w_hmm': weights['hmm'],
            'w_kmeans': weights['kmeans'],
            'volatility_sep': vol_sep,
            'crisis_align': crisis,
            'temporal_coh': coherence,
            'internal_cons': consistency,
            'composite_score': composite_score
        })
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        best = results_df.loc[results_df['composite_score'].idxmax()]
        print(f"\nBest composite score: {best['composite_score']:.4f}")
        print(f"Optimal weights: GARCH={best['w_garch']:.3f}, "
              f"HMM={best['w_hmm']:.3f}, K-means={best['w_kmeans']:.3f}")
    
    return results_df


def get_optimal_weights(optimization_results, metric='composite_score'):
    """
    Extract optimal weights from optimization results.
    
    Args:
        optimization_results: DataFrame from optimize_ensemble_weights
        metric: Which metric to optimize for (default: composite_score)
    
    Returns:
        Dict with optimal weights
    """
    best_row = optimization_results.loc[optimization_results[metric].idxmax()]
    
    return {
        'garch': best_row['w_garch'],
        'hmm': best_row['w_hmm'],
        'kmeans': best_row['w_kmeans']
    }


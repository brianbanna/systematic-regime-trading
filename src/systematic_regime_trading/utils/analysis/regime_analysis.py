"""
Regime Analysis Utilities

Generic functions for analyzing regime detection results across all methods
(GARCH, HMM, K-means).

Functions:
- compute_transition_matrix: Calculate transition probabilities between regimes
- compute_regime_statistics: Statistical summary for each regime
- compute_persistence_metrics: Regime stability and average duration
- classify_regimes_by_quantiles: Convert continuous values to discrete regimes
"""

import numpy as np
import pandas as pd


def compute_transition_matrix(regimes):
    """
    Compute transition probability matrix for regime sequence.
    
    Works for any regime detection method (GARCH, HMM, K-means).
    
    Args:
        regimes: Array-like of regime labels (0, 1, 2)
        
    Returns:
        3x3 numpy array of transition probabilities
        
    Example:
        >>> regimes = [0, 0, 1, 2, 2, 1, 0]
        >>> trans_matrix = compute_transition_matrix(regimes)
    """
    regimes = np.array(regimes)
    n_regimes = 3
    trans_counts = np.zeros((n_regimes, n_regimes))
    
    # Count transitions
    for i in range(len(regimes) - 1):
        current = int(regimes[i])
        next_regime = int(regimes[i + 1])
        trans_counts[current, next_regime] += 1
    
    # Normalize to probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    trans_probs = trans_counts / row_sums
    
    return trans_probs


def compute_regime_statistics(feature, regimes, regime_labels=None):
    """
    Compute statistical summary for each regime.
    
    Args:
        feature: Array-like of feature values (e.g., volatility, returns)
        regimes: Array-like of regime labels (0, 1, 2)
        regime_labels: Optional dict mapping regime ids to names
        
    Returns:
        pandas DataFrame with regime statistics
        
    Example:
        >>> stats = compute_regime_statistics(volatility, garch_regimes)
    """
    feature = np.array(feature)
    regimes = np.array(regimes)
    
    if regime_labels is None:
        regime_labels = {0: 'Calm', 1: 'Moderate', 2: 'Turbulent'}
    
    stats_list = []
    for regime_id in range(3):
        mask = regimes == regime_id
        regime_data = feature[mask]
        
        stats_list.append({
            'Regime': regime_labels[regime_id],
            'Count': len(regime_data),
            'Percentage': len(regime_data) / len(regimes) * 100,
            'Mean': np.mean(regime_data),
            'Std': np.std(regime_data),
            'Min': np.min(regime_data),
            'Median': np.median(regime_data),
            'Max': np.max(regime_data)
        })
    
    return pd.DataFrame(stats_list)


def compute_persistence_metrics(transition_matrix):
    """
    Compute persistence probabilities and average durations from transition matrix.
    
    Args:
        transition_matrix: 3x3 numpy array of transition probabilities
        
    Returns:
        dict with persistence probabilities and average durations
        
    Example:
        >>> metrics = compute_persistence_metrics(trans_matrix)
        >>> print(metrics['persistence'])  # [0.85, 0.78, 0.92]
        >>> print(metrics['avg_duration'])  # [6.67, 4.55, 12.5]
    """
    persistence = np.diag(transition_matrix)
    
    # Average duration = 1 / (1 - persistence)
    avg_duration = np.array([
        1 / (1 - p) if p < 1.0 else np.inf 
        for p in persistence
    ])
    
    return {
        'persistence': persistence,
        'avg_duration': avg_duration,
        'mean_persistence': np.mean(persistence)
    }


def classify_regimes_by_quantiles(values, quantiles=(0.33, 0.67)):
    """
    Convert continuous values to discrete regime labels using quantile thresholds.
    
    Useful for GARCH conditional volatility or any continuous measure.
    
    Args:
        values: Array-like of continuous values
        quantiles: Tuple of (low_threshold, high_threshold) as percentiles
        
    Returns:
        dict with:
            - regimes: Array of regime labels (0=low, 1=medium, 2=high)
            - thresholds: Tuple of (q_low, q_high) threshold values
            - regime_labels: Dict mapping ids to names
            
    Example:
        >>> result = classify_regimes_by_quantiles(garch_cond_vol)
        >>> regimes = result['regimes']
        >>> thresholds = result['thresholds']
    """
    values = np.array(values)
    q_low = np.quantile(values, quantiles[0])
    q_high = np.quantile(values, quantiles[1])
    
    regimes = np.zeros(len(values), dtype=int)
    regimes[(values >= q_low) & (values < q_high)] = 1  # Moderate
    regimes[values >= q_high] = 2  # Turbulent
    
    regime_labels = {0: 'Calm', 1: 'Moderate', 2: 'Turbulent'}
    
    return {
        'regimes': regimes,
        'thresholds': (q_low, q_high),
        'regime_labels': regime_labels
    }


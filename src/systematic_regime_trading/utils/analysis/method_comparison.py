"""
Purpose: Evaluation metrics for comparing regime detection methods.
Part of Step 7 of the ADA project (Method Comparison and Ensemble Framework).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import f_oneway


def compute_pairwise_metrics(labels1, labels2):
    """
    Compute pairwise similarity metrics between two regime labelings.
    
    Returns dict with ARI and NMI scores.
    """
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    
    return {
        'ARI': ari,
        'NMI': nmi
    }


def compute_volatility_separation(regime_labels, volatility):
    """
    Measure how well regimes separate different volatility levels.
    
    Uses F-statistic from ANOVA to quantify separation strength.
    Higher values indicate better separation.
    
    Returns normalized score in [0, 1].
    """
    unique_regimes = np.unique(regime_labels)
    groups = [volatility[regime_labels == r] for r in unique_regimes]
    
    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return 0.0
    
    f_stat, p_value = f_oneway(*groups)
    
    # Normalize F-statistic to [0, 1] using sigmoid-like transformation
    score = 1 - np.exp(-f_stat / 100)
    
    return float(score)


def compute_crisis_alignment(regime_labels, dates, crisis_periods=None):
    """
    Measure how well regimes align with known crisis periods.
    
    Default crisis periods: 2008 financial crisis and 2020 COVID crash.
    
    Returns precision score: fraction of crisis days correctly identified.
    """
    if crisis_periods is None:
        crisis_periods = [
            ('2008-01-01', '2009-06-30'),  # 2008 crisis
            ('2020-02-15', '2020-05-31')   # COVID crash
        ]
    
    dates = pd.to_datetime(dates)
    
    # Identify high-volatility regime (regime with highest mean volatility)
    regime_means = []
    for r in np.unique(regime_labels):
        regime_means.append(np.mean(regime_labels == r))
    
    # Simple heuristic: regime 2 (highest label) typically = high volatility
    high_vol_regime = np.max(regime_labels)
    
    # Mark crisis days
    crisis_mask = np.zeros(len(dates), dtype=bool)
    for start, end in crisis_periods:
        crisis_mask |= (dates >= start) & (dates <= end)
    
    if crisis_mask.sum() == 0:
        return 0.5  # Neutral if no crisis days in data
    
    # Fraction of crisis days identified as high-volatility regime
    detected_crisis = (regime_labels == high_vol_regime) & crisis_mask
    precision = detected_crisis.sum() / crisis_mask.sum()
    
    return float(precision)


def compute_temporal_coherence(regime_labels):
    """
    Measure temporal stability of regime assignments.
    
    Lower transition rate indicates more coherent regime persistence.
    
    Returns coherence score in [0, 1].
    """
    transitions = np.sum(np.diff(regime_labels) != 0)
    max_transitions = len(regime_labels) - 1
    
    if max_transitions == 0:
        return 1.0
    
    # Coherence = 1 - transition_rate
    transition_rate = transitions / max_transitions
    coherence = 1 - transition_rate
    
    return float(coherence)


def compute_internal_consistency(regime_labels, features_df):
    """
    Measure within-regime homogeneity across all features.
    
    Uses ratio of within-cluster to total variance (lower is better).
    
    Returns consistency score in [0, 1].
    """
    features = features_df.values
    
    # Compute total variance
    total_var = np.var(features, axis=0).sum()
    
    if total_var == 0:
        return 0.0
    
    # Compute within-cluster variance
    within_var = 0.0
    for r in np.unique(regime_labels):
        mask = regime_labels == r
        if mask.sum() > 1:
            within_var += np.var(features[mask], axis=0).sum()
    
    # Score = 1 - (within_var / total_var)
    # Higher score means lower within-cluster variance (better)
    score = 1 - (within_var / total_var)
    
    return float(np.clip(score, 0, 1))


def evaluate_method(regime_labels, volatility, dates, features_df, method_name="Method"):
    """
    Comprehensive evaluation of a single regime detection method.
    
    Returns dict with 3 core metric scores and composite score.
    """
    vol_sep = compute_volatility_separation(regime_labels, volatility)
    crisis = compute_crisis_alignment(regime_labels, dates)
    coherence = compute_temporal_coherence(regime_labels)
    consistency = compute_internal_consistency(regime_labels, features_df)
    
    # Composite score: average of 3 core metrics (excluding internal_consistency)
    composite = (vol_sep + crisis + coherence) / 3
    
    return {
        'method': method_name,
        'volatility_separation': vol_sep,
        'crisis_alignment': crisis,
        'temporal_coherence': coherence,
        'internal_consistency': consistency,
        'composite_score': composite
    }


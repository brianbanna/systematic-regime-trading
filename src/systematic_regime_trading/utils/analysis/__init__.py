# Empty __init__.py file - required to make directory a Python package

from .regime_analysis import (
    compute_transition_matrix,
    compute_regime_statistics,
    compute_persistence_metrics,
    classify_regimes_by_quantiles,
)

from .method_comparison import (
    compute_pairwise_metrics,
    compute_volatility_separation,
    compute_crisis_alignment,
    compute_temporal_coherence,
    compute_internal_consistency,
    evaluate_method,
)

from .ensemble import (
    majority_vote_ensemble,
    weighted_vote_ensemble,
    generate_weight_combinations,
    optimize_ensemble_weights,
    get_optimal_weights,
)

__all__ = [
    "compute_transition_matrix",
    "compute_regime_statistics",
    "compute_persistence_metrics",
    "classify_regimes_by_quantiles",
    "compute_pairwise_metrics",
    "compute_volatility_separation",
    "compute_crisis_alignment",
    "compute_temporal_coherence",
    "compute_internal_consistency",
    "evaluate_method",
    "majority_vote_ensemble",
    "weighted_vote_ensemble",
    "generate_weight_combinations",
    "optimize_ensemble_weights",
    "get_optimal_weights",
]


"""
HMM package for Hidden Markov Model regime detection.
"""

from .hmm_model import (
    fit_hmm,
    predict_states,
    get_transition_matrix,
    compute_state_persistence,
    label_states_by_volatility,
    get_state_statistics,
)

from .hmm_visualization import (
    plot_model_selection,
    plot_emission_distributions,
)

__all__ = [
    # Model functions
    "fit_hmm",
    "predict_states",
    "get_transition_matrix",
    "compute_state_persistence",
    "label_states_by_volatility",
    "get_state_statistics",
    # Visualization functions
    "plot_model_selection",
    "plot_emission_distributions",
]
"""Tests for HMM model module."""

import pandas as pd
import numpy as np
import pytest

from systematic_regime_trading.models.hmm import (
    fit_hmm,
    predict_states,
    get_transition_matrix,
    compute_state_persistence,
    get_state_statistics,
    label_states_by_volatility,
    compute_log_likelihood,
    get_emission_parameters,
    compute_state_probabilities,
)


class TestFitHMM:
    def test_fits_and_returns_model_and_states(self, sample_volatility_series):
        model, states = fit_hmm(sample_volatility_series, n_states=3)
        assert model is not None
        assert len(states) == len(sample_volatility_series)
        assert set(states).issubset({0, 1, 2})

    def test_correct_number_of_states(self, sample_volatility_series):
        model, states = fit_hmm(sample_volatility_series, n_states=3)
        assert model.n_components == 3

    def test_reproducible_with_same_seed(self, sample_volatility_series):
        _, states1 = fit_hmm(sample_volatility_series, n_states=3, random_state=42)
        _, states2 = fit_hmm(sample_volatility_series, n_states=3, random_state=42)
        np.testing.assert_array_equal(states1, states2)


class TestPredictStates:
    def test_predict_same_length(self, sample_volatility_series):
        model, _ = fit_hmm(sample_volatility_series, n_states=3)
        predicted = predict_states(model, sample_volatility_series)
        assert len(predicted) == len(sample_volatility_series)


class TestTransitionMatrix:
    def test_rows_sum_to_one(self, sample_volatility_series):
        model, _ = fit_hmm(sample_volatility_series, n_states=3)
        trans = get_transition_matrix(model)
        row_sums = trans.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_shape_matches_states(self, sample_volatility_series):
        model, _ = fit_hmm(sample_volatility_series, n_states=3)
        trans = get_transition_matrix(model)
        assert trans.shape == (3, 3)


class TestStatePersistence:
    def test_persistence_between_0_and_1(self, sample_volatility_series):
        model, _ = fit_hmm(sample_volatility_series, n_states=3)
        persistence = compute_state_persistence(model)
        for state, metrics in persistence.items():
            assert 0 <= metrics["persistence_prob"] <= 1
            assert metrics["expected_duration"] >= 1


class TestLabelStates:
    def test_relabels_by_ascending_volatility(self, sample_volatility_series):
        model, states = fit_hmm(sample_volatility_series, n_states=3)
        relabeled, mapping, names = label_states_by_volatility(
            model, states, sample_volatility_series,
        )
        # State 0 should have lower mean vol than state 2
        mean_0 = sample_volatility_series[relabeled == 0].mean()
        mean_2 = sample_volatility_series[relabeled == 2].mean()
        assert mean_0 < mean_2


class TestStateProbabilities:
    def test_probabilities_sum_to_one(self, sample_volatility_series):
        model, _ = fit_hmm(sample_volatility_series, n_states=3)
        probs = compute_state_probabilities(model, sample_volatility_series)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_shape_matches_data(self, sample_volatility_series):
        model, _ = fit_hmm(sample_volatility_series, n_states=3)
        probs = compute_state_probabilities(model, sample_volatility_series)
        assert probs.shape == (len(sample_volatility_series), 3)

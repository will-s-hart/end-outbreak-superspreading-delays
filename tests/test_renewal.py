"""Tests for the delay convolutions, the design matrix and the ``R``-switch convention."""

from __future__ import annotations

import numpy as np
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import renewal

# --- delay_weighted_sum ------------------------------------------------------------------


def _explicit_weighted_sum(values, weights, first_lag, n_days):
    """The definition, written out as a double loop."""
    out = np.zeros(n_days)
    for t in range(n_days):
        for i, weight in enumerate(weights):
            source = t - (first_lag + i)
            if 0 <= source < len(values):
                out[t] += weight * values[source]
    return out


@pytest.mark.parametrize("first_lag", [0, 1, 2])
def test_delay_weighted_sum_matches_the_definition(first_lag):
    values = np.array([1.0, 0.0, 2.0, 3.0, 0.0, 1.0])
    weights = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(
        renewal.delay_weighted_sum(values, weights, first_lag=first_lag),
        _explicit_weighted_sum(values, weights, first_lag, values.size),
    )


def test_lag_one_excludes_the_current_day_and_lag_zero_includes_it():
    values = np.array([1.0, 5.0])
    weights = np.array([1.0])
    np.testing.assert_allclose(renewal.delay_weighted_sum(values, weights, first_lag=1), [0.0, 1.0])
    np.testing.assert_allclose(renewal.delay_weighted_sum(values, weights, first_lag=0), [1.0, 5.0])


def test_day_zero_has_no_force_of_infection_under_a_lag_one_delay():
    # The sole-index-case initial condition: nothing precedes day 0, so its renewal sum is
    # empty and the likelihood starts at day 1 (§3.1).
    counts = np.array([1.0, 0.0, 0.0])
    Lambda = renewal.delay_weighted_sum(counts, dd.serial_interval_weights(max_lag=10), first_lag=1)
    assert Lambda[0] == 0.0
    assert (Lambda[1:] > 0).all()


def test_delay_weighted_sum_can_project_past_the_end_of_the_series():
    values = np.array([1.0])
    weights = np.array([0.2, 0.5, 0.3])
    projected = renewal.delay_weighted_sum(values, weights, first_lag=1, n_days=5)
    np.testing.assert_allclose(projected, [0.0, 0.2, 0.5, 0.3, 0.0])


def test_delay_weighted_sum_truncates_when_n_days_is_short():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0])
    np.testing.assert_allclose(
        renewal.delay_weighted_sum(values, weights, first_lag=1, n_days=2), [0.0, 1.0]
    )


def test_delay_weighted_sum_rejects_a_negative_first_lag():
    with pytest.raises(ValueError, match="first_lag"):
        renewal.delay_weighted_sum(np.ones(3), np.ones(1), first_lag=-1)


# --- delay_design_matrix -----------------------------------------------------------------


@pytest.mark.parametrize("first_lag", [0, 1])
def test_design_matrix_reproduces_the_dense_convolution(first_lag):
    values = np.array([1.0, 0.0, 2.0, 3.0, 0.0, 1.0])
    weights = np.array([0.5, 0.3, 0.2])
    design = renewal.delay_design_matrix(values.size, weights, first_lag=first_lag)
    np.testing.assert_allclose(
        design @ values,
        renewal.delay_weighted_sum(values, weights, first_lag=first_lag),
    )


def test_restricting_the_columns_is_exact_when_the_other_days_are_empty():
    # This is the SSI case: Y_t is identically zero on days with no cases, so the latent
    # block only needs a column per non-empty cohort.
    values = np.array([1.0, 0.0, 2.0, 0.0, 0.0, 1.0])
    weights = dd.serial_interval_weights(max_lag=8)
    source_days = np.flatnonzero(values > 0)
    design = renewal.delay_design_matrix(values.size, weights, first_lag=1, source_days=source_days)
    assert design.shape == (values.size, source_days.size)
    np.testing.assert_allclose(
        design @ values[source_days],
        renewal.delay_weighted_sum(values, weights, first_lag=1),
    )


def test_design_matrix_entries_are_the_weights_at_the_right_lag():
    weights = np.array([0.5, 0.3, 0.2])
    design = renewal.delay_design_matrix(5, weights, first_lag=1)
    assert design[3, 2] == pytest.approx(weights[0])  # lag 1
    assert design[3, 1] == pytest.approx(weights[1])  # lag 2
    assert design[3, 0] == pytest.approx(weights[2])  # lag 3
    assert design[3, 3] == 0.0  # lag 0 is outside a lag-1 support
    assert design[3, 4] == 0.0  # the future never drives the past
    assert np.tril(design).sum() == pytest.approx(design.sum())


def test_design_matrix_rejects_empty_weights():
    with pytest.raises(ValueError, match="non-empty"):
        renewal.delay_design_matrix(4, np.array([]), first_lag=1)


# --- the R switch ------------------------------------------------------------------------


def test_switch_index_is_zero_before_the_switch_day_and_one_from_it_onwards():
    index = renewal.switch_index(6, 3)
    np.testing.assert_array_equal(index, [0, 0, 0, 1, 1, 1])


def test_reproduction_number_switches_on_the_switch_day_itself():
    R = renewal.reproduction_number_by_day(1.5, 0.4, n_days=5, switch_day=2)
    np.testing.assert_allclose(R, [1.5, 1.5, 0.4, 0.4, 0.4])


def test_a_switch_day_past_the_window_leaves_R_pre_everywhere():
    R = renewal.reproduction_number_by_day(1.5, 0.4, n_days=4, switch_day=99)
    np.testing.assert_allclose(R, 1.5)


def test_switch_index_agrees_with_the_outbreak_data_convention():
    # outbreak_data exposes the same convention as a property; they must not drift apart.
    from end_of_outbreak import outbreak_data

    data = outbreak_data.load_onset_data()
    np.testing.assert_array_equal(
        renewal.switch_index(data.n_days, data.ert_arrival_day),
        data.reproduction_number_period,
    )


# --- likelihood_days ---------------------------------------------------------------------


def test_likelihood_days_drops_day_zero_and_undriven_days():
    force_of_infection = np.array([0.0, 0.4, 0.0, 0.2])
    counts = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(renewal.likelihood_days(force_of_infection, counts), [1, 3])


def test_likelihood_days_raises_when_an_undriven_day_carries_cases():
    force_of_infection = np.array([0.0, 0.4, 0.0])
    counts = np.array([1, 1, 2])
    with pytest.raises(ValueError, match="zero force of infection"):
        renewal.likelihood_days(force_of_infection, counts)


def test_likelihood_days_ignores_day_zero_when_it_is_undriven():
    # Day 0 always has zero force of infection under a lag-1 delay and always carries the
    # index case; that is the initial condition, not an inconsistency.
    force_of_infection = np.array([0.0, 0.4])
    counts = np.array([1, 0])
    np.testing.assert_array_equal(renewal.likelihood_days(force_of_infection, counts), [1])


def test_likelihood_days_checks_the_lengths_match():
    with pytest.raises(ValueError, match="same length"):
        renewal.likelihood_days(np.zeros(3), np.zeros(4, dtype=np.int64))


def test_the_equateur_series_drives_every_inference_day():
    # With max_lag equal to the window, the index case contributes at every lag, so no day
    # of the real series is dropped. Worth pinning: it is why SSE-SO has 110 latents.
    from end_of_outbreak import outbreak_data

    data = outbreak_data.load_onset_data()
    w = dd.serial_interval_weights(max_lag=dd.DEFAULT_MAX_LAG)
    Lambda = renewal.delay_weighted_sum(data.onsets.astype(np.float64), w, first_lag=1)
    days = renewal.likelihood_days(Lambda, data.onsets)
    np.testing.assert_array_equal(days, np.arange(1, data.n_days))

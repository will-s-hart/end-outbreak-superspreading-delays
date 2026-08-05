"""Tests for the Cori discretisation, the lag-0 rules and the onset-anchored delay budget."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from end_of_outbreak import delay_distributions as dd

# --- GammaDelay --------------------------------------------------------------------------


def test_gamma_delay_moments_round_trip():
    delay = dd.GammaDelay(mean=15.3, sd=9.3)
    assert delay.variance == pytest.approx(86.49)
    frozen = delay.frozen()
    assert frozen.mean() == pytest.approx(15.3)
    assert frozen.std() == pytest.approx(9.3)


@pytest.mark.parametrize(("mean", "sd"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_gamma_delay_rejects_non_positive_parameters(mean, sd):
    with pytest.raises(ValueError):
        dd.GammaDelay(mean=mean, sd=sd)


# --- discretisation: normalisation, lag-0 rules and the tail convention ------------------


@pytest.mark.parametrize("first_lag", [0, 1])
def test_weights_are_a_proper_pmf(first_lag):
    weights = dd.discretise_gamma(dd.GENERIC_EVD_SERIAL_INTERVAL, max_lag=60, first_lag=first_lag)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_first_lag_one_folds_the_lag_zero_bin_into_lag_one():
    delay = dd.GammaDelay(mean=3.0, sd=3.0)  # shape 1: appreciable mass below one day
    with_zero = dd.discretise_gamma(delay, max_lag=40, first_lag=0)
    without_zero = dd.discretise_gamma(delay, max_lag=40, first_lag=1)

    assert with_zero.size == without_zero.size + 1
    assert with_zero[0] > 0  # there is genuinely something to fold
    assert without_zero[0] == pytest.approx(with_zero[0] + with_zero[1])
    np.testing.assert_allclose(without_zero[1:], with_zero[2:])


def test_named_constructors_use_the_documented_first_lags():
    delay = dd.GammaDelay(mean=4.0, sd=4.0)
    np.testing.assert_allclose(
        dd.serial_interval_weights(delay, max_lag=40),
        dd.discretise_gamma(delay, max_lag=40, first_lag=1),
    )
    np.testing.assert_allclose(
        dd.incubation_weights(delay, max_lag=40),
        dd.discretise_gamma(delay, max_lag=40, first_lag=1),
    )
    np.testing.assert_allclose(
        dd.tost_weights(delay, max_lag=40),
        dd.discretise_gamma(delay, max_lag=40, first_lag=0),
    )


def test_incubation_has_no_mass_at_lag_zero():
    # The causal-ordering requirement of the onset-anchored models: an infection on day t
    # cannot produce an onset on day t. Enforced by the array starting at lag 1, which the
    # first-lag constants record.
    assert dd.INCUBATION_FIRST_LAG == 1
    assert dd.SERIAL_INTERVAL_FIRST_LAG == 1
    assert dd.TOST_FIRST_LAG == 0


def test_tost_keeps_its_lag_zero_mass():
    # A case may transmit on its own onset day; with shape < 1 that bin carries a lot.
    tost = dd.tost_weights(dd.GammaDelay(mean=3.9, sd=4.57), max_lag=60)
    assert tost[0] > 0.1


def test_truncated_tail_mass_is_folded_into_the_final_bin():
    delay = dd.GENERIC_EVD_SERIAL_INTERVAL
    short = dd.serial_interval_weights(delay, max_lag=12)
    long = dd.serial_interval_weights(delay, max_lag=60)

    assert short.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(short[:-1], long[: short.size - 1])
    assert short[-1] == pytest.approx(1.0 - long[: short.size - 1].sum())
    assert short[-1] > long[short.size - 1]  # the final bin absorbs the whole tail


def test_discretisation_preserves_the_mean_and_slightly_inflates_the_variance():
    # The Cori triangular kernel is mean-preserving and adds ~1/6 to the variance.
    delay = dd.GENERIC_EVD_SERIAL_INTERVAL
    weights = dd.serial_interval_weights(delay, max_lag=120)
    mean, variance = dd.delay_moments(weights, first_lag=dd.SERIAL_INTERVAL_FIRST_LAG)
    assert mean == pytest.approx(delay.mean, abs=0.05)
    assert variance == pytest.approx(delay.variance + 1.0 / 6.0, rel=0.02)


def test_generic_serial_interval_reproduces_figure_s1():
    """External validation against Fig. S1 of Thompson et al. (2024) supplementary.

    That figure plots exactly this distribution — the generic EVD serial interval,
    discretised by the Cori et al. method. Its readable features are a very small first bar,
    a mode at lag 10 just above 0.05, and a tail that has all but vanished by lag 50.
    """
    w = dd.serial_interval_weights()

    assert w[0] == pytest.approx(0.0058, abs=5e-4)
    assert int(np.argmax(w)) + 1 == 10
    assert w.max() == pytest.approx(0.0513, abs=5e-4)
    assert w[49] < 0.002
    # Unimodal: strictly increasing up to the mode, strictly decreasing after it. The final
    # bin is excluded because it absorbs the truncated tail and so ticks back up.
    mode = int(np.argmax(w))
    assert (np.diff(w[: mode + 1]) > 0).all()
    assert (np.diff(w[mode:-1]) < 0).all()


def test_discretise_accepts_any_frozen_continuous_distribution():
    weights = dd.discretise(scipy.stats.lognorm(s=0.5, scale=10.0), max_lag=60)
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(("max_lag", "first_lag"), [(0, 1), (-1, 0)])
def test_discretise_rejects_a_too_small_max_lag(max_lag, first_lag):
    with pytest.raises(ValueError, match="max_lag"):
        dd.discretise_gamma(dd.GENERIC_EVD_SERIAL_INTERVAL, max_lag=max_lag, first_lag=first_lag)


def test_discretise_rejects_an_unsupported_first_lag():
    with pytest.raises(ValueError, match="first_lag"):
        dd.discretise_gamma(dd.GENERIC_EVD_SERIAL_INTERVAL, max_lag=30, first_lag=2)


# --- cumulative --------------------------------------------------------------------------


def test_cumulative_starts_at_zero_and_ends_at_one():
    w = dd.serial_interval_weights(max_lag=60)
    F = dd.cumulative(w)
    assert F.size == w.size + 1
    assert F[0] == 0.0
    assert F[-1] == pytest.approx(1.0)
    assert F[1] == pytest.approx(w[0])
    # (1 - F_0) = 1: a case appearing on day t still has all its transmission to come.
    assert 1.0 - F[0] == 1.0


# --- the variance budget -----------------------------------------------------------------


def test_default_budget_leaves_a_plausible_tost():
    tost = dd.check_delay_budget()
    assert tost.mean == pytest.approx(15.3 - 11.4)
    assert tost.variance == pytest.approx(86.49 - 65.61)
    assert 0 < tost.mean < 10
    assert 0 < tost.sd < 10


def test_budget_is_additive_by_construction():
    si = dd.GENERIC_EVD_SERIAL_INTERVAL
    inc = dd.WHO_ERT_INCUBATION_PERIOD
    tost = dd.check_delay_budget(serial_interval=si, incubation=inc)
    assert tost.mean + inc.mean == pytest.approx(si.mean)
    assert tost.variance + inc.variance == pytest.approx(si.variance)


def test_budget_rejects_a_negative_residual_variance():
    # The outbreak-specific serial interval (SD 6.08, var 36.97) cannot accommodate the
    # default incubation period (var 65.61). This is exactly why the generic serial interval
    # is the project default.
    with pytest.raises(ValueError, match="inadmissible"):
        dd.check_delay_budget(
            serial_interval=dd.OUTBREAK_SPECIFIC_SERIAL_INTERVAL,
            incubation=dd.WHO_ERT_INCUBATION_PERIOD,
        )


def test_budget_rejects_a_negative_residual_mean():
    with pytest.raises(ValueError, match="residual TOST mean"):
        dd.check_delay_budget(
            serial_interval=dd.GammaDelay(mean=10.0, sd=20.0),
            incubation=dd.GammaDelay(mean=12.0, sd=2.0),
        )


# --- convolution: the delays reproduce the serial interval -------------------------------


def test_convolution_is_supported_from_lag_one():
    f_tost = np.array([0.5, 0.5])  # lags 0, 1
    f_inc = np.array([0.5, 0.5])  # lags 1, 2
    conv = dd.convolve_onset_to_onset(f_tost, f_inc)
    # S + A takes values 1, 2, 3 with probabilities 1/4, 1/2, 1/4.
    np.testing.assert_allclose(conv, [0.25, 0.5, 0.25])
    mean, _ = dd.delay_moments(conv, first_lag=1)
    assert mean == pytest.approx(2.0)


def test_convolution_moments_are_additive():
    delays = dd.build_onset_anchored_delays(max_lag=120)
    tost_mean, tost_var = dd.delay_moments(delays.tost, first_lag=dd.TOST_FIRST_LAG)
    inc_mean, inc_var = dd.delay_moments(delays.incubation, first_lag=dd.INCUBATION_FIRST_LAG)
    conv_mean, conv_var = dd.delay_moments(delays.implied_serial_interval(), first_lag=1)

    assert conv_mean == pytest.approx(tost_mean + inc_mean)
    assert conv_var == pytest.approx(tost_var + inc_var)


def test_implied_serial_interval_matches_the_target():
    delays = dd.build_onset_anchored_delays()
    target_mean, target_var = dd.delay_moments(
        delays.serial_interval, first_lag=dd.SERIAL_INTERVAL_FIRST_LAG
    )
    implied_mean, implied_var = dd.delay_moments(delays.implied_serial_interval(), first_lag=1)

    assert implied_mean == pytest.approx(target_mean, abs=0.1)
    # Convolving two discretised delays smooths twice rather than once, so the implied
    # interval carries roughly one extra 1/6 d^2 of kernel variance.
    assert implied_var == pytest.approx(target_var + 1.0 / 6.0, rel=0.02)
    assert delays.serial_interval_discrepancy() < 0.02


def test_build_raises_when_the_convolution_check_fails():
    with pytest.raises(ValueError, match="total variation"):
        dd.build_onset_anchored_delays(tolerance=0.0)


def test_build_can_skip_the_convolution_check():
    delays = dd.build_onset_anchored_delays(tolerance=None)
    assert delays.tost.size == dd.DEFAULT_MAX_LAG + 1
    assert delays.incubation.size == dd.DEFAULT_MAX_LAG


def test_build_propagates_an_inadmissible_budget():
    with pytest.raises(ValueError, match="inadmissible"):
        dd.build_onset_anchored_delays(
            serial_interval=dd.OUTBREAK_SPECIFIC_SERIAL_INTERVAL,
            incubation=dd.WHO_ERT_INCUBATION_PERIOD,
        )


# --- total variation ---------------------------------------------------------------------


def test_total_variation_distance_pads_to_the_longer_pmf():
    assert dd.total_variation_distance(np.array([1.0]), np.array([1.0, 0.0])) == 0.0
    assert dd.total_variation_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        1.0
    )

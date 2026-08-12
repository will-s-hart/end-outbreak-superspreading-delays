"""Tests for the filtering approximation to the real-time estimand.

Three things need pinning:

1. **It is exact where nothing is approximated.** DLO and SSE have no latent state, so no
   filter runs and the route must reproduce the closed form to floating point. That makes them
   the control in the estimator comparison: their gap against refitting is entirely the
   parameter conditioning.
2. **It agrees with smoothing on the last day.** At the end of the window there is no future
   data, so the filtering law *is* the smoothing law. At fixed parameters the filtered curve
   and the MCMC curve must therefore meet there, within Monte-Carlo error — the one day on
   which the approximation is not an approximation.
3. **Thinning keeps the chain layout.** The standard error beside the curve is the spread
   across chains, so thinning across the flattened draw axis would silently corrupt it.
"""

from __future__ import annotations

import numpy as np
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import filtered_risk, fitting, reporting
from end_of_outbreak import risk_of_additional_cases as rac

COUNTS = np.array([1, 0, 2, 1, 0, 3, 0, 1, 0, 0, 1, 0])
SWITCH_DAY = 5
R_PRE, R_POST, K = 1.4, 0.6, 0.5

SHORT_DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)


def _fixed_state(n_draws: int = 1, *, n_chains: int = 1) -> rac.PosteriorState:
    """A posterior state at one repeated parameter draw, and no latents of its own."""
    return rac.PosteriorState(
        R_pre=np.full(n_draws, R_PRE),
        R_post=np.full(n_draws, R_POST),
        k=np.full(n_draws, K),
        sampled_infectivity=None,
        n_chains=n_chains,
    )


# --- 1. exact without latents ---------------------------------------------------------------


@pytest.mark.parametrize("model", ["dlo", "sse"])
def test_the_filtered_route_is_the_closed_form_when_there_is_no_latent_state(model):
    result = filtered_risk.risk_by_filtering(
        model,
        _fixed_state(3),
        counts=COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
    )
    expected = rac.log_probability_of_no_further_cases(
        model,
        counts=COUNTS,
        serial_interval=SHORT_DELAYS.serial_interval,
        R_pre=np.full(3, R_PRE),
        k=np.full(3, K),
    )
    np.testing.assert_array_equal(result.estimate.log_no_further_cases, expected)
    # No filter ran, so there is nothing per-day to diagnose and the result says so.
    assert result.diagnostics is None


@pytest.mark.parametrize("model", ["dlo", "sse"])
def test_a_latent_free_model_has_no_gap_between_cases_and_transmission(model):
    """RAC and RAT coincide under every infection-anchored model, by assumption."""
    result = filtered_risk.risk_by_filtering(
        model, _fixed_state(), counts=COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY
    )
    np.testing.assert_array_equal(
        result.estimate.log_no_further_cases, result.estimate.log_no_further_transmission
    )


# --- 2. filtering is smoothing on the last day ----------------------------------------------


@pytest.mark.parametrize("model", ["ssi", "sse_so", "ssi_so"])
def test_filtering_meets_smoothing_at_the_end_of_the_window(model):
    """No data follow the final day, so ``p(Y | data through T)`` is the full posterior.

    At fixed parameters the filter's day-``T`` snapshot and an MCMC fit to the whole record are
    therefore estimates of the same thing, and this is the one conditioning day on which the
    approximation costs nothing at all.
    """
    last = COUNTS.size - 1
    filtered = filtered_risk.risk_by_filtering(
        model,
        _fixed_state(),
        counts=COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        days=np.array([last]),
        n_particles=20_000,
        rng=np.random.default_rng(7),
    )
    idata = fitting.fit_model(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_PRE,
        R_post=R_POST,
        k=K,
        latent_parameterisation="marginalised_inverse_cdf",
        sampler=fitting.SamplerSettings(draws=3000, tune=1000, chains=4, seed=21),
    )
    state = rac.posterior_state(
        model,
        idata,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        latent_parameterisation="marginalised_inverse_cdf",
        fixed_R_pre=R_PRE,
        fixed_R_post=R_POST,
        fixed_k=K,
    )
    smoothed = rac.risk_log_probabilities(
        model,
        state,
        counts=COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        days=np.array([last]),
    )
    error, _ = smoothed.standard_errors()
    difference = abs(
        float(filtered.estimate.risk_of_additional_cases().risk[0])
        - float(smoothed.risk_of_additional_cases().risk[0])
    )
    assert difference < max(5.0 * float(error[0]), 2e-3)
    sustained_error = smoothed.sustained_transmission_standard_error()
    sustained_difference = abs(
        float(filtered.estimate.risk_of_sustained_transmission().risk[0])
        - float(smoothed.risk_of_sustained_transmission().risk[0])
    )
    assert sustained_difference < max(5.0 * float(sustained_error[0]), 2e-3)


@pytest.mark.parametrize("model", ["ssi", "sse_so", "ssi_so"])
def test_a_filter_reports_how_healthy_its_clouds_were(model):
    result = filtered_risk.risk_by_filtering(
        model,
        _fixed_state(2),
        counts=COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=200,
        rng=np.random.default_rng(3),
    )
    assert result.diagnostics is not None
    assert result.diagnostics.days.size == COUNTS.size
    assert np.all(result.diagnostics.min_effective_sample_size > 0.0)
    assert np.all(result.diagnostics.min_effective_sample_size <= 200.0 + 1e-9)
    assert np.all(
        result.diagnostics.min_effective_sample_size
        <= result.diagnostics.mean_effective_sample_size + 1e-9
    )


def test_the_two_risks_are_ordered_under_an_onset_model():
    result = filtered_risk.risk_by_filtering(
        "sse_so",
        _fixed_state(2),
        counts=COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=200,
        rng=np.random.default_rng(5),
    )
    cases = result.estimate.risk_of_additional_cases().risk
    transmission = result.estimate.risk_of_additional_transmission().risk
    np.testing.assert_array_less(transmission, cases + 1e-12)


# --- 3. thinning ----------------------------------------------------------------------------


def test_thinning_keeps_the_chains_intact_and_in_order():
    """Thinning the flattened axis would mix chains and corrupt the standard error."""
    n_chains, per_chain = 4, 50
    total = n_chains * per_chain
    state = rac.PosteriorState(
        R_pre=np.arange(total, dtype=float),
        R_post=np.arange(total, dtype=float),
        k=np.full(total, K),
        sampled_infectivity=np.tile(np.arange(total, dtype=float)[:, None], (1, COUNTS.size)),
        true_counts=np.tile(np.arange(total, dtype=float)[:, None], (1, COUNTS.size)),
        n_chains=n_chains,
    )
    thinned = filtered_risk.thin_draws(state, 20)
    assert thinned.n_chains == n_chains
    assert thinned.n_draws == 20
    kept = thinned.R_pre.reshape(n_chains, -1)
    for chain in range(n_chains):
        assert np.all(kept[chain] // per_chain == chain)  # each block stays in its own chain
    assert np.all(np.diff(thinned.R_pre) > 0)  # and in order
    assert thinned.true_counts is not None
    np.testing.assert_array_equal(thinned.true_counts[:, 0], thinned.R_pre)


def test_thinning_upwards_is_a_no_op():
    state = _fixed_state(8, n_chains=2)
    assert filtered_risk.thin_draws(state, 100) is state


def test_thinning_below_the_chain_count_is_refused():
    with pytest.raises(ValueError, match="cannot thin"):
        filtered_risk.thin_draws(_fixed_state(8, n_chains=4), 2)


def test_delayed_reporting_is_refused_by_the_single_fit_route():
    with pytest.raises(NotImplementedError, match="historical reported-count snapshot"):
        filtered_risk.risk_by_filtering(
            "cori",
            _fixed_state(),
            counts=COUNTS,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            reporting_model=reporting.ReportingModel(
                probability=0.8, delay=dd.GammaDelay(mean=2.0, sd=1.0)
            ),
        )


def test_dlo_filtered_rac_remains_explicitly_unsupported_under_under_reporting():
    with pytest.raises(NotImplementedError, match="future force-of-infection profile"):
        filtered_risk.risk_by_filtering(
            "dlo",
            _fixed_state(),
            counts=COUNTS,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            reporting_model=reporting.ReportingModel(probability=0.8),
        )


def test_sse_filtered_rac_uses_the_negative_binomial_count_adaptation():
    result = filtered_risk.risk_by_filtering(
        "sse",
        _fixed_state(2),
        counts=COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        days=np.array([COUNTS.size - 1]),
        n_particles=256,
        reporting_model=reporting.ReportingModel(probability=0.8),
        rng=np.random.default_rng(18),
    )
    assert np.isfinite(result.estimate.log_no_further_cases).all()
    assert result.diagnostics is not None

"""Tests for the real-time estimator: one fit per conditioning day.

Four things need pinning, and they are in increasing order of how expensive the failure would
be to notice:

1. **Truncating the series is the only change.** A day-``t`` estimate is the existing closed
   form applied to ``counts[:t + 1]``, so the driver must reproduce a fit done by hand at the
   same seed, exactly.
2. **Every truncation is buildable.** A hundred and ten windows per model is where a builder
   edge case hides — an early window whose latent block is entirely marginalised, say — and it
   would only surface in the middle of a two-hour pipeline run.
3. **Nothing falls between the two latent blocks.** A latent that is neither sampled nor
   integrated out reads as zero, and the risk comes out plausible and too low.
4. **The fixed-``k`` naive identity.** DLO and SSE are latent-free and their likelihood
   factorises across the ``R`` switch, so with ``k`` held fixed the posterior of ``R_pre``
   stops moving once the window covers the pre-switch days — and their refit curve must then
   agree with one built from a single fit to the whole record.
"""

from __future__ import annotations

import numpy as np
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import fitting, outbreak_data, pymc_models, refit_risk
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.model_specifications import LogNormalPrior, specification_of

COUNTS = np.array([1, 0, 2, 1, 0, 3, 0, 1, 0, 0, 1, 0, 0, 0])
SWITCH_DAY = 6
K = 0.6
PRIOR = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)

SHORT_DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)

FAST = fitting.SamplerSettings(draws=200, tune=200, chains=2, seed=3)


@pytest.fixture(scope="module")
def real_series():
    return outbreak_data.load_onset_data(), dd.build_onset_anchored_delays()


# --- the conditioning days ------------------------------------------------------------------


def test_the_curve_starts_at_day_one_because_day_zero_carries_no_likelihood():
    days = refit_risk.conditioning_days(COUNTS.size)
    assert days[0] == 1
    assert days[-1] == COUNTS.size - 1


def test_a_first_day_outside_the_window_is_refused():
    with pytest.raises(ValueError, match="initial condition"):
        refit_risk.conditioning_days(COUNTS.size, first_day=0)
    with pytest.raises(ValueError, match="first_day"):
        refit_risk.conditioning_days(COUNTS.size, first_day=COUNTS.size)


# --- 1. the driver is the closed form on a truncated series ---------------------------------


@pytest.mark.parametrize("model", ["sse", "ssi", "sse_so", "ssi_so"])
def test_refitting_reproduces_a_fit_done_by_hand_at_the_same_seed(model):
    """Day ``t`` is `fit(counts[:t+1])` then the closed form at day ``t``, and nothing else."""
    day = 9
    parameterisation = "marginalised_inverse_cdf" if specification_of(model).has_latents else None
    result = refit_risk.risk_by_refitting(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        latent_parameterisation=parameterisation,
        days=np.array([day]),
        sampler=FAST,
        seed_for_day=lambda _: 11,
    )

    window = COUNTS[: day + 1]
    idata = fitting.fit_model(
        model,
        window,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        latent_parameterisation=parameterisation,
        sampler=refit_risk.replace_seed(FAST, 11),
    )
    state = rac.posterior_state(
        model,
        idata,
        window,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        latent_parameterisation=parameterisation,
        fixed_k=K,
    )
    expected = rac.risk_log_probabilities(
        model,
        state,
        counts=window,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        days=np.array([day]),
    )
    np.testing.assert_allclose(result.estimate.log_no_further_cases, expected.log_no_further_cases)
    np.testing.assert_allclose(
        result.estimate.log_no_further_transmission, expected.log_no_further_transmission
    )


def test_the_state_summed_over_a_window_ignores_the_days_after_it():
    """Λ(t), W(t) and M(t) sum only over ``u ≤ t``, so truncation changes conditioning alone."""
    day = 7
    whole = rac.pooled_remaining_weight(COUNTS.astype(float), SHORT_DELAYS.serial_interval)
    truncated = rac.pooled_remaining_weight(
        COUNTS[: day + 1].astype(float), SHORT_DELAYS.serial_interval
    )
    assert truncated[day] == pytest.approx(whole[day])


def test_the_final_day_reuses_the_fit_it_is_handed():
    """The whole-record fit the pipeline already holds serves the last conditioning day."""
    idata = fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        sampler=FAST,
    )
    last = COUNTS.size - 1
    result = refit_risk.risk_by_refitting(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        days=np.array([last]),
        sampler=FAST,
        # A seed that would give different draws, so reuse is what is being detected.
        seed_for_day=lambda _: 987,
        final_day_fit=idata,
    )
    state = rac.posterior_state(
        "sse", idata, COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY, fixed_k=K
    )
    expected = rac.risk_log_probabilities(
        "sse",
        state,
        counts=COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        days=np.array([last]),
    )
    np.testing.assert_allclose(result.estimate.log_no_further_cases, expected.log_no_further_cases)


# --- 2 and 3. every window builds, and every latent is accounted for -------------------------


@pytest.mark.parametrize("model", ["dlo", "sse", "ssi", "sse_so", "ssi_so"])
def test_every_truncation_of_the_real_series_builds(model, real_series):
    """The bulk failure mode: one window in a hundred and ten that the builder refuses.

    Building is what breaks, not sampling, so this covers every day without any MCMC. The
    early windows are the interesting ones: before any case can be reached, a latent block can
    be entirely marginalised and the sampler sees none of it.
    """
    data, delays = real_series
    parameterisation = "marginalised_inverse_cdf" if specification_of(model).has_latents else None
    fully_marginalised = 0
    for day in refit_risk.conditioning_days(data.onsets.size):
        window = data.onsets[: day + 1]
        built = pymc_models.build_model(
            model,
            window,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=1.0,
            R_post=0.5,
            k=K,
            latent_parameterisation=parameterisation,
        )
        variable = specification_of(model).latent_variable
        if variable is not None and variable not in {rv.name for rv in built.free_RVs}:
            fully_marginalised += 1
    if model == "sse_so":
        assert fully_marginalised > 0  # early windows reach no case at all


@pytest.mark.parametrize("model", ["ssi", "sse_so", "ssi_so"])
def test_no_latent_falls_between_the_sampled_and_unsampled_blocks(model, real_series):
    """Every latent carrying mass is either sampled or integrated out, on every window.

    Reading a missing latent as zero would understate the risk while looking entirely
    plausible, so `posterior_state` refuses; this checks the two blocks really do cover the
    whole thing rather than relying on that refusal never firing.
    """
    data, delays = real_series
    draws = np.array([1.0])
    for day in (1, 2, 5, 20, 57, 58, 59, 90, data.onsets.size - 1):
        window = data.onsets[: day + 1]
        structure = pymc_models.latent_block_structure(
            model, window, delays=delays, switch_day=data.ert_arrival_day
        )
        unsampled = rac.unsampled_latent_conditional(
            model,
            window,
            delays=delays,
            switch_day=data.ert_arrival_day,
            latent_parameterisation="marginalised_inverse_cdf",
            R_pre=draws,
            R_post=draws,
            k=draws,
        )
        built = pymc_models.build_model(
            model,
            window,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=1.0,
            R_post=1.0,
            k=K,
            latent_parameterisation="marginalised_inverse_cdf",
        )
        dimension = pymc_models.latent_dimension(model)
        sampled = (
            pymc_models.model_days(built, dimension)
            if dimension in built.coords
            else np.empty(0, dtype=np.int64)
        )
        carries_mass = structure.days[structure.scale > 0.0]
        missing = np.setdiff1d(carries_mass, np.union1d(sampled, unsampled.days))
        assert missing.size == 0, f"{model} day {day} loses latents on {missing.tolist()}"


def test_sse_so_keeps_a_boundary_latent_wherever_the_window_ends():
    """Its infections are in the retained pipeline but reach no fitted onset, so it is a prior."""
    for day in (5, 9, COUNTS.size - 1):
        window = COUNTS[: day + 1]
        unsampled = rac.unsampled_latent_conditional(
            "sse_so",
            window,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            latent_parameterisation="marginalised_inverse_cdf",
            R_pre=np.array([1.0]),
            R_post=np.array([1.0]),
            k=np.array([K]),
        )
        assert day in unsampled.days
        # Nothing constrains it, so its conditional is its prior: rate k, not k + c_u.
        assert unsampled.rate[0, list(unsampled.days).index(day)] == pytest.approx(K)


# --- 4. the fixed-k naive identity ------------------------------------------------------------


@pytest.mark.parametrize("model", ["dlo", "sse"])
def test_a_latent_free_fixed_k_curve_stops_moving_after_the_switch(model):
    """With ``k`` fixed the likelihood factorises across the switch, so ``R_pre`` freezes.

    Days ``1..switch_day - 1`` carry ``R_pre`` and the rest carry ``R_post``, so once the
    window covers the pre-switch days no further datum reaches ``R_pre`` — and DLO and SSE
    retain only the observed counts. Their real-time curve therefore coincides with one built
    from a single fit to the whole record, which is the cheapest available check that
    refitting is doing what it claims.
    """
    days = np.array([SWITCH_DAY, COUNTS.size - 1])
    settings = fitting.SamplerSettings(draws=4000, tune=1000, chains=4, seed=17)
    refit = refit_risk.risk_by_refitting(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        days=days,
        sampler=settings,
        seed_for_day=lambda day: 40 + day,
    )
    whole_record = fitting.fit_model(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        sampler=fitting.SamplerSettings(draws=4000, tune=1000, chains=4, seed=99),
    )
    state = rac.posterior_state(
        model, whole_record, COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY, fixed_k=K
    )
    reference = rac.risk_log_probabilities(
        model, state, counts=COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY, days=days
    )

    refit_curve = refit.estimate.risk_of_additional_cases().risk
    reference_curve = reference.risk_of_additional_cases().risk
    refit_error, _ = refit.estimate.standard_errors()
    reference_error, _ = reference.standard_errors()
    # Two independent estimates, so their difference carries √2 standard errors.
    scale = np.sqrt(2.0) * np.maximum(refit_error, reference_error)
    np.testing.assert_array_less(np.abs(refit_curve - reference_curve), 4.0 * scale)

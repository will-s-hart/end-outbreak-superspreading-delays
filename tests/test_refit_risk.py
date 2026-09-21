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

from dataclasses import replace

import numpy as np
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import fitting, outbreak_data, pymc_models, refit_risk, reporting
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


def test_sparse_historical_snapshots_infer_days_and_allow_corrections():
    early = COUNTS[:5].copy()
    late = COUNTS[:9].copy()
    late[2] -= 1  # a corrected history need not dominate the earlier snapshot
    resolved = refit_risk._resolve_snapshots(
        (early, late), days=None, reporting_model=reporting.ReportingModel(probability=0.8)
    )
    assert [day for day, _ in resolved] == [4, 8]
    np.testing.assert_array_equal(resolved[0][1], early)
    np.testing.assert_array_equal(resolved[1][1], late)


def test_snapshot_lengths_must_increase_and_replace_the_days_argument():
    snapshots = (COUNTS[:5], COUNTS[:9])
    with pytest.raises(ValueError, match="days cannot be supplied"):
        refit_risk._resolve_snapshots(
            snapshots,
            days=np.array([4, 8]),
            reporting_model=reporting.COMPLETE_REPORTING,
        )
    with pytest.raises(ValueError, match="increase strictly"):
        refit_risk._resolve_snapshots(
            (COUNTS[:5], COUNTS[:5]),
            days=None,
            reporting_model=reporting.COMPLETE_REPORTING,
        )


def test_a_delayed_reporting_curve_requires_historical_snapshots():
    delayed = reporting.ReportingModel(probability=0.8, delay=dd.GammaDelay(mean=2.0, sd=1.0))
    with pytest.raises(ValueError, match="requires a tuple"):
        refit_risk._resolve_snapshots(COUNTS, days=None, reporting_model=delayed)
    resolved = refit_risk._resolve_snapshots(
        (COUNTS[:5], COUNTS[:9]), days=None, reporting_model=delayed
    )
    assert [day for day, _ in resolved] == [4, 8]


def test_snapshot_tuple_drives_the_output_days():
    early = COUNTS[:4].copy()
    late = COUNTS[:7].copy()
    late[2] -= 1
    result = refit_risk.risk_by_refitting(
        "cori",
        (early, late),
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        reporting_model=reporting.ReportingModel(probability=0.8),
        sampler=fitting.SamplerSettings(draws=30, tune=30, chains=2, seed=8),
        seed_for_day=lambda day: day,
    )
    np.testing.assert_array_equal(result.estimate.days, np.array([3, 6]))


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
    assert result.estimate.log_no_sustained_transmission is not None
    assert expected.log_no_sustained_transmission is not None
    np.testing.assert_allclose(
        result.estimate.log_no_sustained_transmission,
        expected.log_no_sustained_transmission,
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

    corrected = COUNTS.copy()
    corrected[-1] += 1
    with pytest.raises(ValueError, match="reported counts"):
        refit_risk.risk_by_refitting(
            "sse",
            corrected,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            R_pre=PRIOR,
            R_post=PRIOR,
            k=K,
            days=np.array([last]),
            sampler=FAST,
            final_day_fit=idata,
        )

    wrong_model = idata.copy(deep=True)
    wrong_model.attrs[fitting.MODEL_ATTRIBUTE] = "dlo"
    with pytest.raises(ValueError, match="records model"):
        refit_risk._validate_reused_fit(
            wrong_model,
            specification=specification_of("sse"),
            snapshot=COUNTS,
            day=last,
            switch_day=SWITCH_DAY,
            reporting_model=reporting.COMPLETE_REPORTING,
        )

    wrong_reporting = idata.copy(deep=True)
    wrong_reporting.attrs[fitting.REPORTING_PROBABILITY_ATTRIBUTE] = 0.8
    with pytest.raises(ValueError, match="reporting assumption"):
        refit_risk._validate_reused_fit(
            wrong_reporting,
            specification=specification_of("sse"),
            snapshot=COUNTS,
            day=last,
            switch_day=SWITCH_DAY,
            reporting_model=reporting.COMPLETE_REPORTING,
        )

    wrong_day = idata.copy(deep=True)
    wrong_day.attrs[fitting.AS_OF_DAY_ATTRIBUTE] = last - 1
    with pytest.raises(ValueError, match="as-of day"):
        refit_risk._validate_reused_fit(
            wrong_day,
            specification=specification_of("sse"),
            snapshot=COUNTS,
            day=last,
            switch_day=SWITCH_DAY,
            reporting_model=reporting.COMPLETE_REPORTING,
        )

    # A fit run under a different switch convention is a different model, and reusing it would
    # put that model's posterior on this curve's last conditioning day.
    wrong_switch = idata.copy(deep=True)
    wrong_switch.attrs[fitting.SWITCH_DAY_ATTRIBUTE] = SWITCH_DAY + 1
    with pytest.raises(ValueError, match="switched R on day"):
        refit_risk._validate_reused_fit(
            wrong_switch,
            specification=specification_of("sse"),
            snapshot=COUNTS,
            day=last,
            switch_day=SWITCH_DAY,
            reporting_model=reporting.COMPLETE_REPORTING,
        )


# --- a fit with nothing free in it ----------------------------------------------------------


def test_fixing_every_parameter_of_a_latent_free_model_gives_a_point_mass_posterior():
    """SSE with ``R`` and ``k`` fixed has no free variable, and PyMC refuses such a model.

    Its posterior is nonetheless well defined — a point mass at what was fixed — so `fit_model`
    stands one up instead. Every fixed value has to come back off the fit, because none of them
    is anywhere in the draws.
    """
    idata = fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=0.95,
        R_post=0.95,
        k=K,
        sampler=FAST,
    )
    assert not idata.posterior.data_vars
    assert idata.posterior.sizes == {"chain": FAST.chains, "draw": 1}
    assert fitting.fitted_reproduction_numbers(idata) == (0.95, 0.95)
    assert fitting.fitted_dispersion(idata) == K
    assert fitting.fitted_switch_day(idata) == SWITCH_DAY
    # The observations travel too, or the fit cannot be reused as a final-day fit.
    assert pymc_models.OBSERVED_VARIABLE in idata.observed_data


def test_an_estimated_parameter_is_recorded_as_having_no_fixed_value():
    idata = fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=0.5,
        k=K,
        sampler=FAST,
    )
    assert fitting.fitted_reproduction_numbers(idata) == (None, 0.5)


def test_a_point_mass_fit_survives_the_round_trip_to_netcdf(tmp_path):
    idata = fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=0.95,
        R_post=0.95,
        k=K,
        sampler=FAST,
    )
    reloaded = fitting.load_fit(fitting.save_fit(idata, tmp_path / "point_mass.nc"))
    assert fitting.fitted_reproduction_numbers(reloaded) == (0.95, 0.95)
    assert reloaded.posterior.sizes == {"chain": FAST.chains, "draw": 1}


def test_saving_compresses_without_changing_a_single_value(tmp_path):
    """Fits are committed and large, so they are gzipped. Lossless is the whole requirement.

    Checked array by array rather than by file size: a silent dtype or fill-value change on the
    way through the encoder would be invisible in the size and fatal in the results. That the
    filter is actually on is asserted separately, by reading the encoding back --- a fit this
    small comes out *larger* compressed, because gzip's per-chunk overhead swamps arrays of a
    few hundred draws. The saving is on the real thing: 113 MB to 39 MB on a sweep fit.
    """
    idata = fitting.fit_model(
        "ssi",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        latent_parameterisation="marginalised_inverse_cdf",
        sampler=FAST,
    )
    path = fitting.save_fit(idata, tmp_path / "compressed.nc")
    reloaded = fitting.load_fit(path)

    assert sorted(reloaded.groups) == sorted(idata.groups)
    for group in idata.groups:
        if group == "/":
            continue
        was, now = idata[group].to_dataset(), reloaded[group].to_dataset()
        assert set(now.data_vars) == set(was.data_vars), group
        for name, variable in was.data_vars.items():
            np.testing.assert_array_equal(np.asarray(now[name]), np.asarray(variable))
            assert now[name].dtype == variable.dtype, name
            assert now[name].dims == variable.dims, name

    for group in reloaded.groups:
        if group == "/":
            continue
        for name, variable in reloaded[group].to_dataset().data_vars.items():
            # Written as `compression="gzip"`, reported on the way back as h5netcdf's
            # `zlib`/`complevel` pair; they are the same HDF5 filter.
            assert variable.encoding.get("zlib") is True, f"{group}/{name}"


def test_thinning_keeps_every_nth_draw_of_every_group(tmp_path):
    """Storage only: the draws kept must be the draws sampled, not a resampling of them."""
    idata = fitting.fit_model(
        "ssi",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        latent_parameterisation="marginalised_inverse_cdf",
        sampler=FAST,
    )
    thinned = fitting.thin_draws(idata, 3)
    for group in ("posterior", "sample_stats"):
        was, now = idata[group].to_dataset(), thinned[group].to_dataset()
        assert now.sizes["draw"] == -(-was.sizes["draw"] // 3), group
        for name, variable in was.data_vars.items():
            np.testing.assert_array_equal(np.asarray(now[name]), np.asarray(variable)[:, ::3, ...])
    # `observed_data` has no draw dimension and must come through untouched.
    np.testing.assert_array_equal(
        np.asarray(thinned["observed_data"].to_dataset()["incidence"]),
        np.asarray(idata["observed_data"].to_dataset()["incidence"]),
    )
    assert thinned.attrs == idata.attrs


def test_thinning_by_one_is_the_identity(tmp_path):
    """The default, and the path every unthinned analysis takes."""
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
    assert fitting.thin_draws(idata, 1) is idata


def test_the_sampler_settings_read_thin_from_the_config_block():
    """Absent from every analysis but the reporting sweeps, so the default has to be 1."""
    block = {"draws": 10, "tune": 10, "chains": 2, "target_accept": 0.9, "seed": 1}
    assert fitting.SamplerSettings.from_config(block).thin == 1
    assert fitting.SamplerSettings.from_config({**block, "thin": 5}).thin == 5


def test_only_an_unthinned_fit_may_be_reused_as_the_last_conditioning_day():
    """The coupling that keeps a storage decision out of the curve.

    Every other day of a curve is fit at full draws. Reusing a thinned archive for the last one
    would give that single point a larger Monte-Carlo error than its neighbours, and
    ``rac.mcselate`` is a maximum over the tail, so it could end up setting a quoted number.
    """
    assert fitting.SamplerSettings(thin=1).reuses_final_day_fit
    assert not fitting.SamplerSettings(thin=5).reuses_final_day_fit


def test_the_free_variables_survive_saving(tmp_path):
    """Bridge sampling walks the built model's free variables and demands each by name.

    ``Y_uniform`` looks redundant beside the ``Y`` recovered from it, and dropping it halves the
    file --- but it makes the fit unusable for `model_evidence`, which is why the saved fit is
    compressed rather than thinned. This pins that, since the failure is remote from the change.
    """
    idata = fitting.fit_model(
        "ssi",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        latent_parameterisation="marginalised_inverse_cdf",
        sampler=FAST,
    )
    reloaded = fitting.load_fit(fitting.save_fit(idata, tmp_path / "free.nc"))
    assert "Y_uniform" in reloaded.posterior.data_vars
    assert "Y" in reloaded.posterior.data_vars


def test_a_fit_that_sampled_nothing_passes_the_acceptance_gate():
    """``no_switch_fixed_R`` fixes every parameter, so SSE's posterior is empty and R-hat nan.

    That nan is not a failure: there is no sampling to diagnose. Treating it as one stopped the
    analysis's first cluster run outright, reporting 110 of 110 conditioning days as failing a
    criterion none of them could ever meet. The two readings of a non-finite R-hat are opposite
    and are told apart by ``n_sampled_variables``, so both are pinned here.
    """
    nothing_sampled = refit_risk.DayDiagnostics(
        day=7,
        divergences=0,
        max_r_hat=float("nan"),
        min_ess_bulk=float("nan"),
        seconds=0.0,
        n_sampled_variables=0,
    )
    assert not nothing_sampled.is_suspect(max_r_hat=1.02, divergence_fraction=0.01, n_draws=100)

    # Sampled something, yet no R-hat could be computed: exactly what the gate is for.
    undiagnosable = replace(nothing_sampled, n_sampled_variables=3)
    assert undiagnosable.is_suspect(max_r_hat=1.02, divergence_fraction=0.01, n_draws=100)

    # Divergences still count against a fit that sampled nothing, if one ever managed them.
    diverged = replace(nothing_sampled, divergences=50)
    assert diverged.is_suspect(max_r_hat=1.02, divergence_fraction=0.01, n_draws=100)


def test_the_fixed_R_curve_reports_no_suspect_days_end_to_end():
    """The gate as `analysis_driver` applies it, over a real fixed-`R` run rather than a stub."""
    result = refit_risk.risk_by_refitting(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=0.95,
        R_post=0.95,
        k=K,
        days=np.array([COUNTS.size - 3, COUNTS.size - 1]),
        sampler=FAST,
    )
    assert all(day.n_sampled_variables == 0 for day in result.diagnostics)
    assert result.suspect_days(max_r_hat=1.02, divergence_fraction=0.01) == []


def test_a_fixed_R_curve_is_deterministic_and_carries_no_monte_carlo_error():
    """The whole point of the fixed-``R`` variant: nothing is sampled, so nothing is noisy.

    It also pins the gap this closed. `risk_by_refitting` used to pass the fit no fixed ``R``
    at all, so `posterior_state` could not find one and raised — the fixed-``R`` analyses were
    unreachable from the refit path even though the calculators supported them.
    """
    days = np.array([COUNTS.size - 3, COUNTS.size - 1])
    result = refit_risk.risk_by_refitting(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=0.95,
        R_post=0.95,
        k=K,
        days=days,
        sampler=FAST,
    )
    np.testing.assert_array_equal(result.estimate.days, days)
    np.testing.assert_array_equal(result.estimate.standard_errors()[0], 0.0)

    # The same numbers, from the closed form applied to each window by hand.
    for position, day in enumerate(days):
        window = COUNTS[: day + 1]
        state = rac.posterior_state(
            "sse",
            fitting.fit_model(
                "sse",
                window,
                delays=SHORT_DELAYS,
                switch_day=SWITCH_DAY,
                R_pre=0.95,
                R_post=0.95,
                k=K,
                sampler=FAST,
            ),
            window,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            fixed_R_pre=0.95,
            fixed_R_post=0.95,
            fixed_k=K,
        )
        expected = rac.risk_log_probabilities(
            "sse",
            state,
            counts=window,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            days=np.array([day]),
        )
        np.testing.assert_allclose(
            result.estimate.log_no_further_cases[:, position : position + 1],
            expected.log_no_further_cases,
        )


def test_a_fit_that_sampled_nothing_reports_no_convergence_rather_than_raising():
    """`az.rhat` raises on an empty posterior, so the diagnostics need their own answer.

    ``NaN`` rather than a flattering 1.0: there was no sampling, so there is nothing to report,
    and ``NaN`` compares false against every threshold the acceptance gate applies.
    """
    idata = fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=0.95,
        R_post=0.95,
        k=K,
        sampler=FAST,
    )
    diagnostics = refit_risk.summarise_fit(idata, day=COUNTS.size - 1, seconds=0.0)
    assert np.isnan(diagnostics.max_r_hat)
    assert np.isnan(diagnostics.min_ess_bulk)
    assert diagnostics.divergences == 0


def test_a_switch_day_past_the_window_leaves_R_post_reaching_nothing():
    """How the no-switchpoint variants work: one R governs the whole window, exactly.

    ``R_post`` is still declared, so a single estimated ``R`` needs no new builder — but it
    must couple to nothing, or it would not be a single ``R`` at all.
    """
    structure = pymc_models.latent_block_structure(
        "ssi_so", COUNTS, delays=SHORT_DELAYS, switch_day=COUNTS.size
    )
    np.testing.assert_array_equal(structure.post_coupling, 0.0)
    assert structure.pre_coupling.sum() > 0.0


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


def _estimate(values: np.ndarray, *, day: int, n_chains: int) -> rac.DailyRiskEstimate:
    """A one-day estimate whose log-probabilities are ``values``, one row per draw."""
    column = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return rac.DailyRiskEstimate(
        days=np.array([day], dtype=np.int64),
        log_no_further_cases=column,
        log_no_further_transmission=column,
        n_chains=n_chains,
    )


def test_a_day_with_nothing_left_to_sample_joins_a_sampled_curve():
    """An early day can have its whole latent block marginalised away, leaving a point mass.

    `no_switch_fixed_R` fixes every parameter, so on day 1 of SSI-SO there is nothing left to
    sample at all and the fit collapses to one draw per chain — while later days sample
    `Y_uniform` and carry 8000. Refusing the mixture stopped the analysis. Such a day is
    *exact*, not under-sampled, so repeating its value is the honest way to line the arrays up.
    """
    exact = _estimate(np.full(4, -0.5), day=1, n_chains=4)
    sampled = _estimate(np.linspace(-0.9, -0.1, 8000), day=2, n_chains=4)
    curve = rac.DailyRiskEstimate.concatenate([exact, sampled])

    assert curve.n_draws == 8000
    np.testing.assert_array_equal(curve.days, [1, 2])
    # Repetition must not move the posterior mean of the exact day.
    np.testing.assert_allclose(curve.risk_of_additional_cases().risk[0], 1.0 - np.exp(-0.5))
    np.testing.assert_allclose(
        curve.risk_of_additional_cases().risk[1], 1.0 - np.exp(sampled.log_no_further_cases).mean()
    )
    # And an exact day has no Monte-Carlo error, before or after being repeated.
    assert curve.standard_errors()[0][0] == 0.0


def test_a_short_part_that_actually_varies_is_still_refused():
    """The guard's real job: a day fitted at different sampler settings is not repeatable."""
    varying = _estimate(np.array([-0.4, -0.6, -0.5, -0.5]), day=1, n_chains=4)
    sampled = _estimate(np.linspace(-0.9, -0.1, 8000), day=2, n_chains=4)
    with pytest.raises(ValueError, match="disagree on the draw count"):
        rac.DailyRiskEstimate.concatenate([varying, sampled])

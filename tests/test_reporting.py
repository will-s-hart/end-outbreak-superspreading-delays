"""Tests for the reporting layer: under-reporting, reporting delays, and the risk it drives.

Ordered by how invisible the mistake would be.

1. **The reduction identity.** Hold the latent unreported counts at zero and every model's
   partial-reporting joint density must equal its complete-reporting one plus
   ``Σ_t c_t log π_t`` — exactly, for every model and every ``π``. Nothing else here would
   catch a reporting layer that had quietly perturbed a model's own likelihood, and that is
   the failure that would sail through review, because the curves would still look like curves.
2. **Agreement with the totals formulation.** The unreported-count factorisation and the old
   totals-plus-binomial form are the same model for both Poisson and negative-binomial counts.
3. **The as-of day.** ``end-of-outbreak-vbd`` conditions on the record strictly *before* its
   calculation day and derives the as-of day from the length of the series; this project's
   windows *include* their conditioning day. Porting the rule rather than the intent would
   shift every reporting probability by one day and nothing would look wrong.
4. **Likelihood against simulation**, on histories short enough to enumerate.
5. Structure, refusals, and the risk path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy.stats

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import (
    fitting,
    outbreak_data,
    particle_filter,
    pymc_models,
    refit_risk,
    reporting,
)
from end_of_outbreak import latent_parameterisations as lp
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.model_specifications import LogNormalPrior, TransmissionParameters

COUNTS = np.array([2, 1, 4, 3, 1, 2, 2, 1, 3, 1])
"""Every day positive, so that the completely reported fit samples the same latent block the
under-reported one does and the two joint densities are comparable term by term."""

SWITCH_DAY = 4
R_PRE, R_POST, K = 1.4, 0.6, 0.5
MODELS = ("dlo", "sse", "ssi", "cori", "sse_so", "ssi_so", "cori_so")

DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=4.0, sd=3.0),
    incubation=dd.GammaDelay(mean=2.5, sd=2.0),
    max_lag=COUNTS.size,
    tolerance=None,
)


def _build(model, *, probability=1.0, delay=None, counts=COUNTS, as_of_day=None, **overrides):
    settings: dict[str, Any] = {
        "delays": DELAYS,
        "switch_day": SWITCH_DAY,
        "R_pre": R_PRE,
        "R_post": R_POST,
        "k": None if model.startswith("cori") else K,
        # Not `marginalised_inverse_cdf`: the completely reported fit would integrate out the
        # boundary latent that the under-reported one samples, and the two blocks would then
        # have different lengths.
        "latent_parameterisation": None if model.startswith("cori") else "inverse_cdf",
        "reporting_model": reporting.ReportingModel(probability=probability, delay=delay),
        "as_of_day": as_of_day,
    }
    settings.update(overrides)
    return pymc_models.build_model(model, counts, **settings)


def _matched_points(complete, partial, *, counts=COUNTS, seed=0):
    """A random point of the shared free variables, in both models' value coordinates."""
    rng = np.random.default_rng(seed)
    complete_point = complete.initial_point()
    partial_point = partial.initial_point()
    for variable in complete.free_RVs:
        name = complete.rvs_to_values[variable].name
        drawn = rng.normal(size=np.shape(complete_point[name]))
        complete_point[name] = drawn
        partial_point[name] = drawn
    partial_point[reporting.UNREPORTED_INCIDENCE_VARIABLE] = np.zeros(
        counts.size - 1, dtype=np.int64
    )
    return complete_point, partial_point


# --- 1. the reduction identity -------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("probability", [0.9, 0.6, 0.35])
def test_holding_unreported_counts_at_zero_recovers_the_complete_likelihood(model, probability):
    """``π = 1`` is not among these: it builds no latent block at all, which is the next test."""
    complete = _build(model)
    partial = _build(model, probability=probability)
    complete_point, partial_point = _matched_points(complete, partial)

    days = np.asarray(complete.coords[pymc_models.LIKELIHOOD_DAY_DIMENSION], dtype=np.int64)
    expected = float(complete.compile_logp()(complete_point)) + float(
        (COUNTS[days] * np.log(probability)).sum()
    )
    assert float(partial.compile_logp()(partial_point)) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("model", MODELS)
def test_every_model_matches_totals_plus_binomial_at_nonzero_unreported_counts(model):
    probability = 0.6
    unreported = np.arange(COUNTS.size - 1, dtype=np.int64) % 3
    totals = COUNTS.copy()
    totals[1:] += unreported
    complete = _build(model, counts=totals)
    partial = _build(model, probability=probability)
    complete_point, partial_point = _matched_points(complete, partial, counts=COUNTS, seed=7)
    partial_point[reporting.UNREPORTED_INCIDENCE_VARIABLE] = unreported

    days = np.asarray(complete.coords[pymc_models.LIKELIHOOD_DAY_DIMENSION], dtype=np.int64)
    old_joint = float(complete.compile_logp()(complete_point)) + float(
        scipy.stats.binom.logpmf(COUNTS[days], totals[days], probability).sum()
    )
    assert float(partial.compile_logp()(partial_point)) == pytest.approx(old_joint, abs=1e-8)


@pytest.mark.parametrize("model", MODELS)
def test_complete_reporting_builds_the_model_it_always_built(model):
    """The default and an explicit complete `ReportingModel` are the same graph."""
    default = pymc_models.build_model(
        model,
        COUNTS,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_PRE,
        R_post=R_POST,
        k=None if model.startswith("cori") else K,
        latent_parameterisation=None if model.startswith("cori") else "inverse_cdf",
    )
    explicit = _build(model)
    point = default.initial_point()
    assert [v.name for v in default.free_RVs] == [v.name for v in explicit.free_RVs]
    assert float(default.compile_logp()(point)) == float(explicit.compile_logp()(point))
    assert reporting.TRUE_INCIDENCE_VARIABLE not in {v.name for v in default.free_RVs}
    assert reporting.UNREPORTED_INCIDENCE_VARIABLE not in {v.name for v in default.free_RVs}


# --- 2. agreement with totals plus binomial -------------------------------------------------


@pytest.mark.parametrize("mean", [0.4, 3.2, 11.0])
@pytest.mark.parametrize("probability", [0.25, 0.6, 1.0])
def test_poisson_thinning_is_the_same_density_as_totals_plus_binomial(mean, probability):
    """``Poisson(D; μ)·Binom(c; D, π) == Poisson(c; πμ)·Poisson(D − c; (1 − π)μ)``.

    This is the independent Poisson split used for every conditionally Poisson model.
    """
    for reported in range(0, 6):
        for unreported in range(0, 6):
            total = reported + unreported
            totals_form = scipy.stats.poisson.logpmf(total, mean) + scipy.stats.binom.logpmf(
                reported, total, probability
            )
            split_form = scipy.stats.poisson.logpmf(
                reported, probability * mean
            ) + scipy.stats.poisson.logpmf(unreported, (1.0 - probability) * mean)
            assert totals_form == pytest.approx(split_form, abs=1e-10)


@pytest.mark.parametrize("mean, alpha", [(0.4, 0.2), (3.2, 1.1), (11.0, 7.0)])
@pytest.mark.parametrize("probability", [0.0, 0.25, 0.6, 1.0])
def test_negative_binomial_thinning_is_the_same_density_as_totals_plus_binomial(
    mean, alpha, probability
):
    """The reported marginal and conditional hidden NB multiply to the old joint density."""
    for reported in range(0, 6):
        for unreported in range(0, 6):
            total = reported + unreported
            totals_form = scipy.stats.nbinom.logpmf(
                total, alpha, alpha / (alpha + mean)
            ) + scipy.stats.binom.logpmf(reported, total, probability)
            reported_mean = probability * mean
            reported_form = scipy.stats.nbinom.logpmf(
                reported,
                alpha,
                alpha / (alpha + reported_mean),
            )
            hidden_alpha = alpha + reported
            denominator = alpha + probability * mean
            hidden_mean = (1.0 - probability) * mean * hidden_alpha / denominator
            hidden_form = scipy.stats.nbinom.logpmf(
                unreported,
                hidden_alpha,
                hidden_alpha / (hidden_alpha + hidden_mean),
            )
            assert totals_form == pytest.approx(reported_form + hidden_form, abs=1e-10)


# --- 3. the as-of day, and the one-day offset ----------------------------------------------


def test_without_a_delay_every_day_carries_the_same_reporting_probability():
    model = reporting.ReportingModel(probability=0.7)
    probability = model.probability_by_day(10, as_of_day=9)
    assert probability == pytest.approx(np.full(10, 0.7))


def test_the_conditioning_day_itself_has_had_a_zero_length_delay_not_a_negative_one():
    """The as-of day is the last day *included* in the window, so lag 0 is available on it.

    This is the offset that must not be inherited from ``end-of-outbreak-vbd``, whose windows
    stop strictly before their calculation day. Deriving the as-of day from ``len(counts) - 1``
    means different things in the two projects, so it is an argument here rather than a rule.
    """
    delay = dd.GammaDelay(mean=3.0, sd=2.0)
    model = reporting.ReportingModel(probability=1.0, delay=delay, max_lag=20)
    probability = model.probability_by_day(10, as_of_day=9)
    lag_zero_mass = dd.discretise_gamma(delay, max_lag=20, first_lag=0)[0]
    assert probability[9] == pytest.approx(lag_zero_mass)
    assert probability[9] > 0.0


def test_a_delay_truncates_recent_onset_days_and_leaves_old_ones_alone():
    model = reporting.ReportingModel(
        probability=0.9, delay=dd.GammaDelay(mean=3.0, sd=2.0), max_lag=20
    )
    probability = model.probability_by_day(30, as_of_day=29)
    assert np.all(np.diff(probability) <= 1e-12)  # non-increasing towards the as-of day
    assert probability[0] == pytest.approx(0.9)  # the oldest day has had every chance
    assert probability[-1] < 0.9


def test_the_as_of_day_moves_the_truncation_with_it():
    model = reporting.ReportingModel(
        probability=1.0, delay=dd.GammaDelay(mean=3.0, sd=2.0), max_lag=20
    )
    early = model.probability_by_day(30, as_of_day=10)
    late = model.probability_by_day(30, as_of_day=20)
    assert early[10] == pytest.approx(late[20])
    assert np.all(early[11:] == 0.0)  # onsets after the as-of day cannot have been reported


def test_a_refit_window_takes_its_conditioning_day_as_the_as_of_day():
    """The window is ``counts[:t + 1]``, so its last day is ``t`` and its length is ``t + 1``."""
    delay = dd.GammaDelay(mean=3.0, sd=2.0)
    model = reporting.ReportingModel(probability=1.0, delay=delay, max_lag=20)
    for t in (4, 7, 9):
        window = COUNTS[: t + 1]
        assert model.probability_by_day(window.size, as_of_day=t)[t] == pytest.approx(
            model.probability_by_day(COUNTS.size, as_of_day=t)[t]
        )


@pytest.mark.parametrize("probability", [-0.1, 1.5, np.nan])
def test_an_impossible_reporting_probability_is_refused(probability):
    with pytest.raises(ValueError, match="reporting probability"):
        reporting.ReportingModel(probability=probability)


def test_zero_reporting_probability_is_valid_until_a_reported_case_makes_it_impossible():
    model = reporting.ReportingModel(probability=0.0)
    assert np.all(model.probability_by_day(4, as_of_day=3) == 0.0)
    with pytest.raises(ValueError, match="reporting probability of zero"):
        _build("cori", probability=0.0)

    unreported_only = COUNTS.copy()
    unreported_only[1:] = 0
    built = _build("cori", probability=0.0, counts=unreported_only)
    assert np.isfinite(float(built.compile_logp()(built.initial_point())))


def test_a_reporting_block_round_trips_through_the_config_form():
    block = {"probability": 0.6, "delay": {"mean": 4.0, "sd": 2.0}}
    model = reporting.ReportingModel.from_config(block)
    assert model.probability == 0.6
    assert model.delay == dd.GammaDelay(mean=4.0, sd=2.0)
    assert not model.is_complete
    assert reporting.ReportingModel.from_config(None).is_complete


# --- 4. likelihood against simulation ------------------------------------------------------


def test_the_reported_likelihood_matches_thinned_simulation_on_a_short_history():
    """Marginalise the totals by enumeration and compare with simulate-then-thin frequencies.

    Cori is the model to do this on: it has no latent block, so what is being checked is the
    reporting layer itself rather than a Gamma block, and the marginal over the totals is a
    finite sum. The model is rebuilt once per *reported* series — the enumerated totals are a
    value of its latent, not a different model — which is what keeps this cheap enough to run
    with the unit tests.
    """
    probability = 0.5
    serial_interval = np.array([0.6, 0.4])
    delays = dd.OnsetAnchoredDelays(
        serial_interval=serial_interval,
        tost=DELAYS.tost,
        incubation=DELAYS.incubation,
        tost_delay=DELAYS.tost_delay,
        incubation_delay=DELAYS.incubation_delay,
    )
    reporting_model = reporting.ReportingModel(probability=probability)
    rng = np.random.default_rng(20260811)
    index_case, n_days, cap = 1, 3, 30

    n_replicates = 400_000
    true_counts = np.zeros((n_replicates, n_days), dtype=np.int64)
    true_counts[:, 0] = index_case
    for day in (1, 2):
        lags = min(day, serial_interval.size)
        force = true_counts[:, day - lags : day] @ serial_interval[:lags][::-1]
        true_counts[:, day] = rng.poisson(R_PRE * force)
    reported = reporting.thin(true_counts, reporting=reporting_model, as_of_day=n_days - 1, rng=rng)

    def model_probability(first, second):
        """``p(c_1 = first, c_2 = second)`` with the true counts summed out."""
        built = pymc_models.build_model(
            "cori",
            np.array([index_case, first, second]),
            delays=delays,
            switch_day=n_days,  # R_pre throughout
            R_pre=R_PRE,
            R_post=R_POST,
            reporting_model=reporting_model,
        )
        logp = built.compile_logp()
        point = built.initial_point()
        return sum(
            float(
                np.exp(
                    logp(
                        {
                            **point,
                            reporting.UNREPORTED_INCIDENCE_VARIABLE: np.array(
                                [a - first, b - second]
                            ),
                        }
                    )
                )
            )
            for a in range(first, cap)
            for b in range(second, cap)
        )

    outcomes = [(a, b) for a in range(3) for b in range(3)]
    analytic = np.array([model_probability(a, b) for a, b in outcomes])
    empirical = np.array(
        [float(((reported[:, 1] == a) & (reported[:, 2] == b)).mean()) for a, b in outcomes]
    )
    # The 3x3 grid does not exhaust the support, so the two need only agree cell by cell —
    # and on how much mass the grid holds.
    assert np.max(np.abs(analytic - empirical)) < 5e-3
    assert analytic.sum() == pytest.approx(float(empirical.sum()), abs=5e-3)


def test_thinning_leaves_the_index_case_alone_and_never_exceeds_the_truth():
    rng = np.random.default_rng(3)
    true_counts = np.tile(np.array([3, 4, 5, 6]), (500, 1))
    reported = reporting.thin(
        true_counts,
        reporting=reporting.ReportingModel(probability=0.5),
        as_of_day=3,
        rng=rng,
    )
    assert np.all(reported[:, 0] == 3)  # day 0 is the fixed index case
    assert np.all(reported <= true_counts)
    assert reported[:, 1:].mean() == pytest.approx(2.5, rel=0.05)


# --- 5. the latent block, the refusals, and the risk path ----------------------------------


@pytest.mark.parametrize("model", ["ssi", "sse_so", "ssi_so"])
def test_incomplete_reporting_samples_every_latent_and_marginalises_none(model):
    partial = _build(model, probability=0.6, latent_parameterisation="marginalised_inverse_cdf")
    dimension = pymc_models.latent_dimension(model)
    expected = COUNTS.size - 1 if model == "sse_so" else COUNTS.size
    assert len(partial.coords[dimension]) == expected
    assert not [name for name in partial.named_vars if name.endswith("_marginalised")]


def test_a_latent_whose_imputed_cohort_is_empty_is_pinned_at_zero():
    """``Gamma(k·0, k)`` is a point mass at zero, and a random scale must not lose that.

    Checked on the guarded quantile itself, because that is where the guard lives: an unguarded
    ``pm.icdf`` at a zero shape returns ``nan``, which would poison the whole latent vector
    rather than the one day that has no cases.
    """
    uniform = pt.as_tensor_variable(np.linspace(0.05, 0.95, 6))
    scale = pt.as_tensor_variable(np.array([0.0, 2.0, 0.0, 1.0, 5.0, 0.0]))
    guarded = np.asarray(
        lp.gamma_from_uniform(uniform, k=K, scale=scale, guard_zero_scale=True).eval()
    )
    assert np.all(np.isfinite(guarded))
    assert np.all(guarded[[0, 2, 5]] == 0.0)
    assert np.all(guarded[[1, 3, 4]] > 0.0)
    # The values where the cohort is non-empty must be exactly what the unguarded form gives.
    unguarded = np.asarray(
        lp.gamma_from_uniform(
            pt.as_tensor_variable(np.linspace(0.05, 0.95, 6)[[1, 3, 4]]),
            k=K,
            scale=pt.as_tensor_variable(np.array([2.0, 1.0, 5.0])),
        ).eval()
    )
    assert guarded[[1, 3, 4]] == pytest.approx(unguarded)


@pytest.mark.parametrize("model", ["ssi", "ssi_so", "sse_so"])
def test_the_joint_density_stays_finite_when_a_day_is_imputed_as_empty(model):
    counts = np.array([2, 0, 0, 1, 0, 2, 0, 0, 1, 0])
    partial = _build(model, probability=0.6, counts=counts)
    point = partial.initial_point()
    point[reporting.UNREPORTED_INCIDENCE_VARIABLE] = np.zeros(counts.size - 1, dtype=np.int64)
    assert np.isfinite(float(partial.compile_logp()(point))) or np.isneginf(
        float(partial.compile_logp()(point))
    )
    assert not np.isnan(float(partial.compile_logp()(point)))


@pytest.mark.parametrize("parameterisation", ["centred", "unit_mean", "marginalised"])
def test_incomplete_reporting_refuses_a_scale_dependent_free_variable(parameterisation):
    with pytest.raises(ValueError, match="inverse-CDF"):
        _build("ssi_so", probability=0.6, latent_parameterisation=parameterisation)


@pytest.mark.parametrize("probability", [0.3, 0.6, 0.95])
def test_the_initial_point_never_starts_a_chain_at_an_impossible_state(probability):
    """The initial unreported counts are non-negative and have a finite joint density."""
    partial = _build("sse_so", probability=probability)
    point = partial.initial_point()
    assert np.all(point[reporting.UNREPORTED_INCIDENCE_VARIABLE] >= 0)
    assert np.isfinite(float(partial.compile_logp()(point)))


def test_pymc_assigns_coordinate_wise_metropolis_to_the_unreported_counts():
    from pymc.sampling.mcmc import assign_step_methods

    built = _build("cori", probability=0.6)
    with built:
        explicit, selected = assign_step_methods(built)
        assert explicit == []
        assert [variable.name for variable in selected[pm.Metropolis]] == [
            reporting.UNREPORTED_INCIDENCE_VARIABLE
        ]
        step = pm.Metropolis(vars=selected[pm.Metropolis], model=built)
    assert step.elemwise_update


def test_a_day_whose_effective_reporting_probability_is_zero_cannot_carry_cases():
    with pytest.raises(ValueError, match="reporting probability of zero"):
        _build(
            "cori",
            probability=1.0,
            delay=dd.GammaDelay(mean=8.0, sd=1.0),
            as_of_day=0,
        )


@pytest.mark.parametrize("model", ["sse_so", "ssi_so"])
def test_the_risk_path_reads_the_true_counts_rather_than_the_reported_ones(model):
    """Identical true-count draws must reproduce exactly what a plain series would give."""
    n_draws = 5

    def estimate(counts):
        return rac.onset_event_probabilities(
            model,
            counts=counts,
            delays=DELAYS,
            switch_day=SWITCH_DAY,
            R_pre=np.full(n_draws, R_PRE),
            R_post=np.full(n_draws, R_POST),
            k=np.full(n_draws, K),
            latent=np.tile(np.linspace(0.1, 0.9, COUNTS.size), (n_draws, 1)),
        )

    plain = estimate(COUNTS)
    by_draw = estimate(np.tile(COUNTS.astype(np.float64), (n_draws, 1)))
    assert np.allclose(plain.log_no_further_cases, by_draw.log_no_further_cases)
    assert plain.log_no_sustained_transmission is not None
    assert by_draw.log_no_sustained_transmission is not None
    assert np.allclose(plain.log_no_sustained_transmission, by_draw.log_no_sustained_transmission)


def test_more_unreported_cases_can_only_raise_the_risk():
    """Two draws differing only in an imputed late case, on the model the figure reports."""
    n_draws = 2
    counts = np.tile(COUNTS.astype(np.float64), (n_draws, 1))
    counts[1, -1] += 1.0
    estimate = rac.onset_event_probabilities(
        "sse_so",
        counts=counts,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=np.full(n_draws, R_PRE),
        R_post=np.full(n_draws, R_POST),
        k=np.full(n_draws, K),
        latent=np.tile(np.linspace(0.1, 0.9, COUNTS.size), (n_draws, 1)),
    )
    assert np.all(estimate.log_no_further_cases[1] <= estimate.log_no_further_cases[0] + 1e-12)


@pytest.mark.parametrize("model", ["sse_so", "ssi_so"])
def test_every_truncation_of_the_real_series_builds_under_under_reporting(model):
    """It is building that breaks, not sampling — so build every window without sampling one."""
    data = outbreak_data.load_onset_data()
    reporting_model = reporting.ReportingModel(probability=0.6)
    for day in refit_risk.conditioning_days(data.onsets.size)[::17]:
        built = pymc_models.build_model(
            model,
            data.onsets[: int(day) + 1],
            delays=dd.build_onset_anchored_delays(max_lag=data.onsets.size),
            switch_day=data.ert_arrival_day,
            R_pre=R_PRE,
            R_post=R_POST,
            k=K,
            latent_parameterisation="marginalised_inverse_cdf",
            reporting_model=reporting_model,
            as_of_day=int(day),
        )
        assert np.isfinite(float(built.compile_logp()(built.initial_point())))


def _short_fit(**overrides: Any):
    settings: dict[str, Any] = {
        "delays": DELAYS,
        "switch_day": SWITCH_DAY,
        "R_pre": LogNormalPrior(median=1.0, sigma_log=0.5),
        "R_post": LogNormalPrior(median=1.0, sigma_log=0.5),
        "sampler": fitting.SamplerSettings(draws=20, tune=20, chains=1, seed=1),
    }
    settings.update(overrides)
    return fitting.fit_model("cori", COUNTS, **settings)


def test_the_fit_records_the_reporting_assumption_it_was_run_under():
    delay = dd.GammaDelay(mean=3.0, sd=2.0)
    idata = _short_fit(
        reporting_model=reporting.ReportingModel(probability=0.6, delay=delay, max_lag=COUNTS.size),
        as_of_day=COUNTS.size - 1,
    )
    recovered = fitting.fitted_reporting(idata)
    assert recovered.probability == pytest.approx(0.6)
    assert recovered.delay == delay
    assert recovered.max_lag == COUNTS.size
    assert fitting.fitted_as_of_day(idata, COUNTS) == COUNTS.size - 1
    np.testing.assert_array_equal(fitting.fitted_reported_counts(idata), COUNTS)
    assert reporting.TRUE_INCIDENCE_VARIABLE in idata.posterior
    assert reporting.UNREPORTED_INCIDENCE_VARIABLE in idata.posterior


def test_a_fit_with_no_reporting_assumption_reads_back_as_completely_reported():
    assert fitting.fitted_reporting(_short_fit()).is_complete


def test_explicit_complete_reporting_preserves_seeded_draws():
    default = _short_fit()
    explicit = _short_fit(reporting_model=reporting.COMPLETE_REPORTING)
    assert set(default.posterior.data_vars) == set(explicit.posterior.data_vars)
    for name in default.posterior.data_vars:
        np.testing.assert_array_equal(default.posterior[name], explicit.posterior[name])


def test_the_unreported_count_block_keeps_moving_with_ordinary_metropolis():
    """The failure this guards against is silent, and only appears past ~20 latent days.

    Scalar support makes PyMC update the vector as a batch of coordinates. This guards that
    load-bearing declaration on a block long enough to expose accidental blocked proposals.
    """
    rng = np.random.default_rng(0)
    counts = np.concatenate([[3], rng.poisson(0.7, size=45)]).astype(np.int64)
    delays = dd.build_onset_anchored_delays(max_lag=counts.size, tolerance=None)
    idata = fitting.fit_model(
        "cori",
        counts,
        delays=delays,
        switch_day=counts.size // 2,
        R_pre=LogNormalPrior(median=1.0, sigma_log=0.5),
        R_post=LogNormalPrior(median=1.0, sigma_log=0.5),
        reporting_model=reporting.ReportingModel(probability=0.6),
        as_of_day=counts.size - 1,
        sampler=fitting.SamplerSettings(draws=250, tune=250, chains=1, seed=4),
    )
    totals = idata.posterior[reporting.TRUE_INCIDENCE_VARIABLE].values.reshape(-1, counts.size - 1)
    assert np.all(totals >= counts[1:])
    assert (totals.std(axis=0) > 0).mean() > 0.8
    assert totals.mean(axis=0).sum() > counts[1:].sum()


# --- the particle filter -------------------------------------------------------------------


def test_negative_binomial_particle_split_uses_the_exact_marginal_and_conditional():
    mean, alpha, probability, reported = 3.2, 0.7, 0.4, 2
    means = np.full(100_000, mean)
    dispersions = np.full(means.shape, alpha)
    log_density = particle_filter._count_log_density(
        reported, mean=np.array([probability * mean]), dispersion=np.array([alpha])
    )
    expected_log_density = scipy.stats.nbinom.logpmf(
        reported, alpha, alpha / (alpha + probability * mean)
    )
    assert log_density[0] == pytest.approx(expected_log_density)

    totals = particle_filter._adapted_counts(
        reported,
        means,
        probability,
        np.random.default_rng(13),
        dispersion=dispersions,
    )
    hidden_alpha = alpha + reported
    hidden_mean = (1.0 - probability) * mean * hidden_alpha / (alpha + probability * mean)
    assert float((totals - reported).mean()) == pytest.approx(hidden_mean, rel=0.02)
    assert float((totals - reported).var()) == pytest.approx(
        hidden_mean + hidden_mean**2 / hidden_alpha, rel=0.04
    )


@pytest.mark.parametrize("model", MODELS)
def test_a_complete_reporting_model_filters_exactly_as_it_did_before(model):
    """Passing complete reporting explicitly must not perturb the draw sequence."""
    default = particle_filter.filter_model(
        model,
        COUNTS,
        TransmissionParameters(
            R_pre=R_PRE, R_post=R_POST, k=None if model.startswith("cori") else K
        ),
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=64,
        rng=np.random.default_rng(5),
    )
    explicit = particle_filter.filter_model(
        model,
        COUNTS,
        TransmissionParameters(
            R_pre=R_PRE, R_post=R_POST, k=None if model.startswith("cori") else K
        ),
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=64,
        reporting_model=reporting.COMPLETE_REPORTING,
        rng=np.random.default_rng(5),
    )
    assert default.log_evidence == explicit.log_evidence
    assert default.true_count_paths is None and explicit.true_count_paths is None


@pytest.mark.parametrize("model", MODELS)
def test_the_filter_imputes_counts_that_dominate_the_reported_ones(model):
    result = particle_filter.filter_model(
        model,
        COUNTS,
        TransmissionParameters(
            R_pre=R_PRE, R_post=R_POST, k=None if model.startswith("cori") else K
        ),
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=256,
        reporting_model=reporting.ReportingModel(probability=0.6),
        rng=np.random.default_rng(11),
    )
    assert result.true_count_paths is not None
    assert np.all(result.true_count_paths >= COUNTS[None, :])
    assert np.isfinite(result.log_evidence)
    # Recovering more cases than were reported is the whole point of the layer.
    assert result.true_count_paths.sum(axis=1).mean() > COUNTS.sum()


@pytest.mark.parametrize("model", MODELS)
def test_the_filter_survives_a_day_that_a_prior_proposal_would_have_killed(model):
    """The adapted proposal, not the imputation, is what makes the filter usable.

    Drawing ``D_t`` from the model and reweighting by ``Binomial(c_t; D_t, π_t)`` targets the
    same law but gives every particle zero weight as soon as a day carries more cases than the
    cloud happens to propose — which is exactly what a low reporting probability makes likely,
    and what killed the quick route on the real series. Conditioning on the reported cases and
    drawing only the unreported ones cannot fail that way, at any ``π``, so a handful of
    particles is enough here.
    """
    busy = np.array([1, 0, 0, 9, 0, 0, 0, 0])
    result = particle_filter.filter_model(
        model,
        busy,
        TransmissionParameters(
            R_pre=R_PRE, R_post=R_POST, k=None if model.startswith("cori") else K
        ),
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=32,
        reporting_model=reporting.ReportingModel(probability=0.2),
        rng=np.random.default_rng(3),
    )
    assert np.isfinite(result.log_evidence)
    assert result.true_count_paths is not None
    assert np.all(result.true_count_paths >= busy[None, :])


def test_the_filter_reproduces_the_reported_series_when_everything_is_reported_by_construction():
    """At ``π`` just below 1 the binomial pins the totals to the reported counts."""
    result = particle_filter.filter_model(
        "cori",
        COUNTS,
        TransmissionParameters(R_pre=R_PRE, R_post=R_POST, k=None),
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=512,
        reporting_model=reporting.ReportingModel(probability=1.0 - 1e-9),
        rng=np.random.default_rng(2),
    )
    assert result.true_count_paths is not None
    assert np.all(result.true_count_paths == COUNTS[None, :])

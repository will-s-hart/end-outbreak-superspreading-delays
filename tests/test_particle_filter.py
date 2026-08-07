"""Tests for the bootstrap particle filter (Stage 4).

The filter has to earn trust before it can be used to validate anything else, so it is checked
in two stages, exactly as §6.4 prescribes:

1. **Against the exact answer, where there is one.** DLO, SSE and Cori have no latent state, so
   the filter is not an approximation at all: its SMC log-evidence must equal the PyMC
   likelihood to floating-point precision, at any particle count. That pins the recursion, the
   ``R``-switch indexing and the observation densities before any latent is involved.
2. **Against an independent marginalisation, where there is not.** For SSI the evidence is
   compared with a naive Monte-Carlo integral over the latent priors — a different estimator
   built from the conditional independence of the infectivities given the counts.

The degeneracy diagnostic is tested too, because it is what the Stage-4b go/no-go decision
rests on: ``n_distinct`` must actually fall as one looks further into the past.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special
import scipy.stats

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import outbreak_data, particle_filter, pymc_models
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.model_specifications import TransmissionParameters

COUNTS = np.array([1, 0, 2, 1, 0, 3, 0, 1])
SWITCH_DAY = 4
PARAMETERS = TransmissionParameters(R_pre=1.3, R_post=0.5, k=0.6)

DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)
SERIAL_INTERVAL = DELAYS.serial_interval


def _parameters_for(model):
    k = None if model == "cori" else PARAMETERS.k
    return TransmissionParameters(R_pre=PARAMETERS.R_pre, R_post=PARAMETERS.R_post, k=k)


def _pymc_observation_logp(model, counts=COUNTS, parameters=None):
    """``log p(I_{1:T} | θ)`` from the built model, for the models where it is closed form."""
    parameters = _parameters_for(model) if parameters is None else parameters
    built = pymc_models.build_naive_model(
        model,
        counts,
        serial_interval=SERIAL_INTERVAL,
        switch_day=SWITCH_DAY,
        R_pre=parameters.R_pre,
        R_post=parameters.R_post,
        k=parameters.k,
    )
    return pymc_models.compile_observation_logp(built)()


def _filter(model, *, n_particles=1000, seed=0, counts=COUNTS, parameters=None, **kwargs):
    return particle_filter.filter_naive(
        model,
        counts,
        _parameters_for(model) if parameters is None else parameters,
        serial_interval=SERIAL_INTERVAL,
        switch_day=SWITCH_DAY,
        n_particles=n_particles,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


# --- the latent-free models: the filter is exact --------------------------------------------


@pytest.mark.parametrize("model", ["dlo", "sse", "cori"])
def test_the_filter_reproduces_the_closed_form_likelihood(model):
    """No latent state means no approximation: this is an equality, not an agreement."""
    result = _filter(model, n_particles=5)
    assert result.log_evidence == pytest.approx(_pymc_observation_logp(model), abs=1e-10)


@pytest.mark.parametrize("model", ["dlo", "sse", "cori"])
def test_the_particle_count_is_irrelevant_without_latents(model):
    few = _filter(model, n_particles=1, seed=1)
    many = _filter(model, n_particles=500, seed=2)
    assert few.log_evidence == pytest.approx(many.log_evidence, abs=1e-10)
    assert few.latent_paths is None


def test_the_increments_are_the_daily_conditional_likelihoods():
    """``log p(I_{1:t} | θ)`` day by day, which is what the Stage-4b PMMH sampler consumes."""
    result = _filter("sse", n_particles=3)
    for position, day in enumerate(result.days):
        prefix = _pymc_observation_logp("sse", counts=COUNTS[: int(day) + 1])
        assert result.cumulative_log_evidence[position] == pytest.approx(prefix, abs=1e-10)


def test_the_filtering_weight_is_the_analytic_lambda_when_there_are_no_latents():
    """Λ(t) is what RAC needs from the retained state; without latents it is a known constant."""
    result = _filter("dlo", n_particles=4)
    Lambda = rac.pooled_remaining_weight(COUNTS.astype(float), SERIAL_INTERVAL)
    for day in range(COUNTS.size):
        np.testing.assert_allclose(result.filtering_remaining_weight[day], Lambda[day])


# --- SSI: the filter against an independent marginalisation ----------------------------------


def _monte_carlo_log_evidence(counts, parameters, n_samples=400_000, seed=11):
    """``log p(I_{1:T} | θ)`` by integrating the latent priors directly.

    The infectivity priors are conditionally independent given the counts, so sampling
    ``Y_u ~ Gamma(k I_u, k)`` and averaging the Poisson likelihood is unbiased — the same
    identity the simulation tests use, and machinery the filter shares nothing with.
    """
    rng = np.random.default_rng(seed)
    cohorts = np.flatnonzero(counts > 0)
    Y = np.zeros((n_samples, counts.size))
    Y[:, cohorts] = rng.gamma(
        parameters.k * counts[cohorts], 1.0 / parameters.k, size=(n_samples, cohorts.size)
    )
    design = np.zeros((counts.size, counts.size))
    for t in range(counts.size):
        for u in range(t):
            lag = t - u
            if 1 <= lag <= SERIAL_INTERVAL.size:
                design[t, u] = SERIAL_INTERVAL[lag - 1]
    R_by_day = np.where(np.arange(counts.size) >= SWITCH_DAY, parameters.R_post, parameters.R_pre)
    mean_incidence = R_by_day * (Y @ design.T)
    days = np.arange(1, counts.size)
    log_terms = scipy.stats.poisson.logpmf(counts[days], mean_incidence[:, days]).sum(axis=1)
    return float(scipy.special.logsumexp(log_terms) - np.log(n_samples))


def test_ssi_evidence_matches_a_naive_marginalisation_of_the_latent_priors():
    result = _filter("ssi", n_particles=20_000, seed=3)
    reference = _monte_carlo_log_evidence(COUNTS, PARAMETERS)
    assert result.log_evidence == pytest.approx(reference, abs=0.05)


def test_large_k_collapses_the_ssi_filter_onto_cori():
    """``Y_t → I_t`` as ``k → ∞``, so the filter's evidence must approach the Poisson one."""
    almost_poisson = TransmissionParameters(R_pre=1.3, R_post=0.5, k=1e6)
    result = _filter("ssi", n_particles=2000, seed=4, parameters=almost_poisson)
    assert result.log_evidence == pytest.approx(_pymc_observation_logp("cori"), abs=0.01)


def test_the_smoothed_paths_vanish_exactly_where_there_are_no_cases():
    result = _filter("ssi", n_particles=200, seed=5)
    assert result.latent_paths is not None
    assert np.all(result.latent_paths[:, COUNTS == 0] == 0.0)
    assert np.all(result.latent_paths[:, COUNTS > 0] > 0.0)


def test_the_smoothed_paths_carry_the_information_in_the_later_data():
    """Smoothing is not prior sampling: days followed by cases are pulled upwards.

    Day 0 is followed by three case-carrying days within one serial interval, so its
    infectivity is dragged above its prior mean of ``I_0 = 1``. That is the whole content of
    the §5.6 distinction between a filtering and a smoothed state, in one number.
    """
    result = _filter("ssi", n_particles=20_000, seed=6)
    assert result.latent_paths is not None
    assert result.latent_paths[:, 0].mean() > 1.2
    # The filtering state at day 0 has seen nothing yet, so it is still the prior.
    assert result.filtering_remaining_weight[0].mean() == pytest.approx(1.0, abs=0.05)


def test_filtering_snapshots_retain_importance_weights_when_resampling_is_skipped():
    """Stored snapshots are equal-weight draws even when the live filter stays weighted."""
    counts = np.array([1, 2])
    k = 0.5
    R = 1.2
    serial_interval = np.array([0.5, 0.5])
    result = particle_filter.filter_naive(
        "ssi",
        counts,
        TransmissionParameters(R_pre=R, R_post=R, k=k),
        serial_interval=serial_interval,
        switch_day=1,
        n_particles=50_000,
        resample_threshold=0.0,
        rng=np.random.default_rng(61),
    )
    # D_1 updates Y_0 from Gamma(k, k) to Gamma(k + D_1, k + R w_1).
    expected_Y_0 = (k + counts[1]) / (k + R * serial_interval[0])
    expected_Lambda = serial_interval[1] * expected_Y_0 + counts[1]
    assert result.filtering_remaining_weight[1].mean() == pytest.approx(expected_Lambda, abs=0.05)


# --- degeneracy ------------------------------------------------------------------------------


def test_path_degeneracy_shows_up_in_the_distinct_ancestor_count():
    """The diagnostic Stage 4b's go/no-go rests on: distinct ancestors fall into the past."""
    data = outbreak_data.load_onset_data()
    delays = dd.build_onset_anchored_delays()
    result = particle_filter.filter_naive(
        "ssi",
        data.onsets,
        TransmissionParameters(R_pre=2.6, R_post=0.6, k=0.18),
        serial_interval=delays.serial_interval,
        switch_day=data.ert_arrival_day,
        n_particles=2000,
        rng=np.random.default_rng(7),
    )
    # Ancestry thins monotonically-ish into the past; on this series it is mild — a few hundred
    # distinct day-0 ancestors out of 2000 — which is why Stage 4b is worth attempting at all.
    assert result.n_distinct[0] < result.n_distinct[-1]
    assert result.n_distinct[0] < 0.5 * result.n_particles
    assert (result.effective_sample_size > 0).all()


def test_adaptive_resampling_leaves_more_ancestors_alive_than_resampling_every_step():
    always = _filter("ssi", n_particles=2000, seed=8, resample_threshold=1.0)
    adaptive = _filter("ssi", n_particles=2000, seed=8, resample_threshold=0.5)
    assert adaptive.resampled.sum() <= always.resampled.sum()
    assert adaptive.n_distinct[0] >= always.n_distinct[0]


# --- resampling ------------------------------------------------------------------------------


def test_systematic_resampling_reproduces_the_weights_in_expectation():
    weights = np.array([0.5, 0.25, 0.2, 0.04, 0.01])
    rng = np.random.default_rng(9)
    draws = np.stack(
        [
            np.bincount(particle_filter.systematic_resample(weights, rng), minlength=weights.size)
            for _ in range(4000)
        ]
    )
    np.testing.assert_allclose(draws.mean(axis=0) / weights.size, weights, atol=0.01)


def test_systematic_resampling_always_keeps_a_dominant_particle():
    weights = np.array([0.98, 0.01, 0.01])
    indices = particle_filter.systematic_resample(weights, np.random.default_rng(10))
    assert (indices == 0).sum() >= int(0.9 * weights.size)


# --- validation ------------------------------------------------------------------------------


def test_an_impossible_observation_is_reported_rather_than_silently_zeroed():
    """A case on a day nothing can reach is a contradiction, not a likelihood of zero."""
    counts = np.array([1, 0, 0, 1])
    with pytest.raises(ValueError, match="zero probability"):
        particle_filter.filter_naive(
            "cori",
            counts,
            TransmissionParameters(R_pre=1.0, R_post=1.0),
            serial_interval=np.array([1.0]),
            switch_day=2,
            n_particles=10,
            rng=np.random.default_rng(12),
        )


def test_the_naive_filter_redirects_onset_models_to_their_filter():
    with pytest.raises(ValueError, match="filter_onset_anchored"):
        _filter("sse_so")


def test_the_series_must_start_with_the_index_case():
    with pytest.raises(ValueError, match="starting with a case"):
        _filter("cori", counts=np.array([0, 1, 2]))

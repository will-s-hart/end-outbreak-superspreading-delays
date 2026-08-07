"""Stage-8 tests: onset simulation, reset-state RAC/RAT and particle filtering."""

from __future__ import annotations

import numpy as np
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import forward_simulation, particle_filter, pymc_models, renewal
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.model_specifications import TransmissionParameters

COUNTS = np.array([1, 0, 2, 1, 0, 3, 0, 1])
SWITCH_DAY = 4
K = 0.6
PARAMETERS = TransmissionParameters(R_pre=1.3, R_post=0.5, k=K)
DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)


def _latent_path(model: str, rng: np.random.Generator) -> np.ndarray | None:
    if model == "cori_so":
        return None
    if model == "sse_so":
        scale = renewal.delay_weighted_sum(
            COUNTS.astype(float), DELAYS.tost, first_lag=dd.TOST_FIRST_LAG
        )
        return rng.gamma(K * scale, 1.0 / K)
    path = np.zeros(COUNTS.size)
    path[COUNTS > 0] = rng.gamma(K * COUNTS[COUNTS > 0], 1.0 / K)
    return path


@pytest.mark.parametrize("model", ["cori_so", "sse_so", "ssi_so"])
def test_natural_and_convenient_onset_simulators_agree(model):
    parameters = TransmissionParameters(R_pre=1.1, R_post=0.7, k=None if model == "cori_so" else K)
    natural = forward_simulation.simulate_onset_anchored(
        model,
        parameters,
        delays=DELAYS,
        n_days=6,
        switch_day=3,
        n_replicates=50_000,
        form="natural",
        rng=np.random.default_rng(1),
    )
    convenient = forward_simulation.simulate_onset_anchored(
        model,
        parameters,
        delays=DELAYS,
        n_days=6,
        switch_day=3,
        n_replicates=50_000,
        form="convenient",
        rng=np.random.default_rng(2),
    )
    assert natural.infections is not None
    assert convenient.infections is None
    np.testing.assert_allclose(
        natural.onsets.mean(axis=0), convenient.onsets.mean(axis=0), atol=0.025
    )
    np.testing.assert_allclose(natural.onsets.var(axis=0), convenient.onsets.var(axis=0), atol=0.1)


@pytest.mark.parametrize(
    ("onset_model", "naive_model"),
    [("cori_so", "cori"), ("sse_so", "sse"), ("ssi_so", "ssi")],
)
def test_one_day_incubation_reduces_the_generators_to_the_naive_models(onset_model, naive_model):
    delays = dd.OnsetAnchoredDelays(
        serial_interval=np.array([0.7, 0.3]),
        tost=np.array([0.7, 0.3]),
        incubation=np.array([1.0]),
        tost_delay=DELAYS.tost_delay,
        incubation_delay=DELAYS.incubation_delay,
    )
    parameters = TransmissionParameters(
        R_pre=1.1, R_post=0.6, k=None if onset_model == "cori_so" else K
    )
    onset = forward_simulation.simulate_onset_anchored_convenient(
        onset_model,
        parameters,
        delays=delays,
        n_days=6,
        switch_day=3,
        n_replicates=60_000,
        rng=np.random.default_rng(3),
    )
    naive = forward_simulation.simulate_naive(
        naive_model,
        parameters,
        serial_interval=delays.serial_interval,
        n_days=6,
        switch_day=4,
        n_replicates=60_000,
        rng=np.random.default_rng(4),
    )
    np.testing.assert_allclose(onset.onsets.mean(axis=0), naive.counts.mean(axis=0), atol=0.025)
    np.testing.assert_allclose(onset.onsets.var(axis=0), naive.counts.var(axis=0), atol=0.3)


@pytest.mark.parametrize("model", ["cori_so", "sse_so", "ssi_so"])
def test_onset_zero_probabilities_match_explicit_reset_simulation(model):
    rng = np.random.default_rng(5)
    latent = _latent_path(model, rng)
    parameters = TransmissionParameters(
        R_pre=PARAMETERS.R_pre,
        R_post=PARAMETERS.R_post,
        k=None if model == "cori_so" else K,
    )
    analytic = rac.onset_event_probabilities(
        model,
        counts=COUNTS,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=parameters.R_pre,
        R_post=parameters.R_post,
        k=parameters.k,
        latent=None if latent is None else latent[None, :],
        days=np.array([3, 7]),
    )
    simulated = rac.simulated_onset_risk_curves(
        model,
        counts=COUNTS,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        parameters=parameters,
        latent=latent,
        days=np.array([3, 7]),
        n_replicates=40_000,
        rng=rng,
    )
    np.testing.assert_allclose(
        analytic.risk_of_additional_cases().risk, simulated.rac.risk, atol=0.009
    )
    np.testing.assert_allclose(
        analytic.risk_of_additional_transmission().risk, simulated.rat.risk, atol=0.009
    )


def test_resetting_R_to_zero_leaves_only_the_incubation_pipeline():
    latent = _latent_path("ssi_so", np.random.default_rng(6))
    assert latent is not None
    probabilities = rac.onset_event_probabilities(
        "ssi_so",
        counts=COUNTS,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PARAMETERS.R_pre,
        R_post=PARAMETERS.R_post,
        k=K,
        latent=latent[None, :],
        reset_R=0.0,
    )
    tost = renewal.delay_design_matrix(
        COUNTS.size,
        DELAYS.tost,
        first_lag=dd.TOST_FIRST_LAG,
        source_days=np.arange(COUNTS.size),
    )
    R = renewal.reproduction_number_by_day(
        PARAMETERS.R_pre, PARAMETERS.R_post, n_days=COUNTS.size, switch_day=SWITCH_DAY
    )
    E = R * (latent @ tost.T)
    pipeline = rac.incubation_pipeline_mean(E, DELAYS.incubation)
    np.testing.assert_allclose(probabilities.log_no_further_cases[0], -pipeline)
    np.testing.assert_allclose(probabilities.log_no_further_transmission, 0.0)


def test_rac_is_never_below_rat():
    latent = _latent_path("sse_so", np.random.default_rng(7))
    assert latent is not None
    probabilities = rac.onset_event_probabilities(
        "sse_so",
        counts=COUNTS,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=np.array([0.8, 1.3, 2.0]),
        R_post=np.array([0.4, 0.5, 0.9]),
        k=np.array([0.2, 0.6, 1.0]),
        latent=np.tile(latent, (3, 1)),
    )
    assert np.all(
        probabilities.risk_of_additional_cases().risk
        >= probabilities.risk_of_additional_transmission().risk
    )


def test_cori_so_filter_is_exact_and_carries_the_pipeline_state():
    parameters = TransmissionParameters(R_pre=PARAMETERS.R_pre, R_post=PARAMETERS.R_post)
    filtered = particle_filter.filter_onset_anchored(
        "cori_so",
        COUNTS,
        parameters,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        n_particles=7,
        rng=np.random.default_rng(8),
    )
    built = pymc_models.build_onset_anchored_model(
        "cori_so",
        COUNTS,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=parameters.R_pre,
        R_post=parameters.R_post,
    )
    assert filtered.log_evidence == pytest.approx(
        pymc_models.compile_observation_logp(built)(), abs=1e-10
    )
    assert filtered.expected_infection_paths is not None
    assert filtered.filtering_pipeline_mean is not None
    assert filtered.latent_paths is None


def test_one_day_incubation_reduces_the_ssi_so_filter_to_ssi_exactly():
    delays = dd.OnsetAnchoredDelays(
        serial_interval=DELAYS.tost,
        tost=DELAYS.tost,
        incubation=np.array([1.0]),
        tost_delay=DELAYS.tost_delay,
        incubation_delay=DELAYS.incubation_delay,
    )
    onset = particle_filter.filter_onset_anchored(
        "ssi_so",
        COUNTS,
        PARAMETERS,
        delays=delays,
        switch_day=SWITCH_DAY,
        n_particles=5000,
        rng=np.random.default_rng(9),
    )
    naive = particle_filter.filter_naive(
        "ssi",
        COUNTS,
        PARAMETERS,
        serial_interval=delays.serial_interval,
        switch_day=SWITCH_DAY + 1,
        n_particles=5000,
        rng=np.random.default_rng(9),
    )
    assert onset.log_evidence == pytest.approx(naive.log_evidence, abs=1e-10)

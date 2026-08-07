"""Fast regression tests for the independent PMMH validation route."""

from __future__ import annotations

import numpy as np
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import particle_mcmc
from end_of_outbreak.model_specifications import LogNormalPrior, TransmissionParameters

COUNTS = np.array([1, 1, 0, 1, 0, 0])
DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)
PRIOR = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.25, probability=0.025)


def test_log_evidence_variance_is_zero_for_an_exact_latent_free_filter():
    measured = particle_mcmc.estimate_log_evidence_variance(
        "sse",
        COUNTS,
        TransmissionParameters(R_pre=1.2, R_post=0.6, k=0.5),
        delays=DELAYS,
        switch_day=3,
        n_particles=7,
        repeats=5,
        rng=np.random.default_rng(1),
    )
    assert measured.variance == pytest.approx(0.0, abs=1e-15)


def test_pmmh_returns_reproducible_natural_scale_draws_with_fixed_k():
    first = particle_mcmc.sample(
        "sse",
        COUNTS,
        delays=DELAYS,
        switch_day=3,
        R_pre_prior=PRIOR,
        R_post_prior=PRIOR,
        k=0.5,
        n_particles=3,
        draws=200,
        tune=100,
        rng=np.random.default_rng(2),
    )
    again = particle_mcmc.sample(
        "sse",
        COUNTS,
        delays=DELAYS,
        switch_day=3,
        R_pre_prior=PRIOR,
        R_post_prior=PRIOR,
        k=0.5,
        n_particles=3,
        draws=200,
        tune=100,
        rng=np.random.default_rng(2),
    )
    np.testing.assert_array_equal(first.R_pre, again.R_pre)
    np.testing.assert_array_equal(first.R_post, again.R_post)
    assert first.k is not None
    np.testing.assert_array_equal(first.k, 0.5)
    assert np.all(first.R_pre > 0.0)
    assert np.all(first.R_post > 0.0)
    assert 0.05 < first.acceptance_rate < 0.95


def test_onset_pmmh_can_retain_filtering_pipeline_draws():
    result = particle_mcmc.sample(
        "ssi_so",
        COUNTS,
        delays=DELAYS,
        switch_day=3,
        R_pre_prior=PRIOR,
        R_post_prior=PRIOR,
        k=0.5,
        n_particles=100,
        draws=30,
        tune=20,
        store_filtering_state=True,
        rng=np.random.default_rng(3),
    )
    assert result.filtering_remaining_weight is not None
    assert result.filtering_pipeline_mean is not None
    assert result.filtering_remaining_weight.shape == (30, COUNTS.size)
    assert result.filtering_pipeline_mean.shape == (30, COUNTS.size)

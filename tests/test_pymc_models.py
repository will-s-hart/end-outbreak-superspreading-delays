"""Tests for the infection-anchored PyMC model builders.

Each likelihood is checked against a density written out by hand from the definitions in §3
of the implementation plan — for DLO, against the closed form printed in Supplementary
Analysis 1 of Thompson et al. (2024), whose negative-binomial variant *is* the DLO model.

The ``k → ∞`` collapse onto the Cori (Poisson) renewal model is checked here for DLO and SSE,
whose likelihoods are closed form. SSI's collapse involves marginalising its latent block and
so is checked in :mod:`tests.test_forward_simulation`, alongside the marginalisation
machinery it shares with the likelihood-versus-simulation tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.special
import scipy.stats

from end_of_outbreak import pymc_models, renewal
from end_of_outbreak.model_specifications import LogNormalPrior

# A short synthetic history that exercises everything: an index case, an empty day, a day
# whose count is driven by two earlier cohorts, and an R switch inside the window.
COUNTS = np.array([1, 0, 2, 1, 0, 3])
SERIAL_INTERVAL = np.array([0.5, 0.3, 0.2])
SWITCH_DAY = 3
R_PRE, R_POST, K = 1.4, 0.6, 0.5

LIKELIHOOD_DAYS = np.arange(1, COUNTS.size)


def _build(model, **overrides):
    counts = overrides.pop("counts", COUNTS)
    settings: dict[str, Any] = {
        "serial_interval": SERIAL_INTERVAL,
        "switch_day": SWITCH_DAY,
        "R_pre": R_PRE,
        "R_post": R_POST,
        "k": K,
        # The centred block is the Stage-2 baseline; the alternatives and the exact
        # marginalisation are exercised in tests/test_latent_parameterisations.py.
        "latent_parameterisation": "centred",
    }
    settings.update(overrides)
    return pymc_models.build_naive_model(model, counts, **settings)


def _force_of_infection(counts=COUNTS, serial_interval=SERIAL_INTERVAL):
    return renewal.delay_weighted_sum(
        np.asarray(counts, dtype=np.float64), serial_interval, first_lag=1
    )


def _mean_incidence(counts=COUNTS, **overrides):
    R_pre = overrides.get("R_pre", R_PRE)
    R_post = overrides.get("R_post", R_POST)
    switch_day = overrides.get("switch_day", SWITCH_DAY)
    counts = np.asarray(counts)
    R = renewal.reproduction_number_by_day(R_pre, R_post, n_days=counts.size, switch_day=switch_day)
    days = np.arange(1, counts.size)
    return (
        R[days]
        * _force_of_infection(counts, overrides.get("serial_interval", SERIAL_INTERVAL))[days]
    )


# --- DLO: the closed form of Thompson et al., Supplementary Analysis 1 ---------------------


def test_dlo_reproduces_the_thompson_supplementary_analysis_1_likelihood():
    """The external validation target of Stage 2.

    Supplementary Analysis 1 of Thompson et al. (2024) writes the likelihood for ``R`` under a
    negative-binomial number of cases each day as

        L(R) ∝ Π_t [ Γ(k + I_t) / (I_t! Γ(k)) ]
                 · ( R Σ_s I_{t-s} w_s / (k + R Σ_s I_{t-s} w_s) )^{I_t}
                 · ( k / (k + R Σ_s I_{t-s} w_s) )^k ,

    with a dispersion parameter that is **constant across days** — that is DLO's defining
    feature, not SSE's. It is transcribed here term by term from the supplement rather than
    routed through ``scipy.stats.nbinom``, so that the check is genuinely independent.
    """
    mean_incidence = _mean_incidence()
    observed = COUNTS[LIKELIHOOD_DAYS]
    thompson = (
        scipy.special.gammaln(K + observed)
        - scipy.special.gammaln(observed + 1.0)
        - scipy.special.gammaln(K)
        + observed * np.log(mean_incidence / (K + mean_incidence))
        + K * np.log(K / (K + mean_incidence))
    ).sum()

    assert pymc_models.joint_logp(_build("dlo")) == pytest.approx(thompson)


def test_dlo_dispersion_does_not_scale_with_the_force_of_infection():
    # The whole of §5.4: DLO's negative binomial carries the *same* number of successes k on
    # every day regardless of Λ_t, which is what makes k a day-level rather than an
    # individual-level quantity — and what makes DLO and SSE genuinely different models.
    mean_incidence = _mean_incidence()
    constant_dispersion = scipy.stats.nbinom.logpmf(
        COUNTS[LIKELIHOOD_DAYS], n=K, p=K / (K + mean_incidence)
    ).sum()
    assert pymc_models.joint_logp(_build("dlo")) == pytest.approx(constant_dispersion)
    assert pymc_models.joint_logp(_build("sse")) != pytest.approx(constant_dispersion)


# --- SSE and Cori -------------------------------------------------------------------------


def test_sse_is_negative_binomial_with_dispersion_proportional_to_the_force_of_infection():
    Lambda = _force_of_infection()[LIKELIHOOD_DAYS]
    R = renewal.reproduction_number_by_day(
        R_PRE, R_POST, n_days=COUNTS.size, switch_day=SWITCH_DAY
    )[LIKELIHOOD_DAYS]
    # p = k / (k + R) is free of Λ: the NB-closure property that pools independent infectors.
    expected = scipy.stats.nbinom.logpmf(COUNTS[LIKELIHOOD_DAYS], n=K * Lambda, p=K / (K + R)).sum()
    assert pymc_models.joint_logp(_build("sse")) == pytest.approx(expected)


def test_cori_is_the_poisson_renewal_likelihood():
    expected = scipy.stats.poisson.logpmf(COUNTS[LIKELIHOOD_DAYS], mu=_mean_incidence()).sum()
    assert pymc_models.joint_logp(_build("cori", k=None)) == pytest.approx(expected)


# --- SSI --------------------------------------------------------------------------------


def test_ssi_joint_density_is_the_gamma_prior_times_the_poisson_likelihood():
    model = _build("ssi")
    cohort_days = np.flatnonzero(COUNTS > 0)
    np.testing.assert_array_equal(
        pymc_models.model_days(model, pymc_models.COHORT_DAY_DIMENSION), cohort_days
    )

    rng = np.random.default_rng(0)
    Y = rng.gamma(K * COUNTS[cohort_days], 1.0 / K)
    design = renewal.delay_design_matrix(
        COUNTS.size, SERIAL_INTERVAL, first_lag=1, source_days=cohort_days
    )
    R = renewal.reproduction_number_by_day(R_PRE, R_POST, n_days=COUNTS.size, switch_day=SWITCH_DAY)
    expected = (
        scipy.stats.gamma.logpdf(Y, a=K * COUNTS[cohort_days], scale=1.0 / K).sum()
        + scipy.stats.poisson.logpmf(
            COUNTS[LIKELIHOOD_DAYS], mu=R[LIKELIHOOD_DAYS] * (design @ Y)[LIKELIHOOD_DAYS]
        ).sum()
    )
    assert pymc_models.joint_logp(model, Y=Y) == pytest.approx(expected)


def test_ssi_carries_one_latent_per_non_empty_cohort_including_day_zero():
    model = _build("ssi")
    cohort_days = pymc_models.model_days(model, pymc_models.COHORT_DAY_DIMENSION)
    assert cohort_days[0] == 0  # the index case's own infectivity drives the early days
    assert cohort_days.size == int((COUNTS > 0).sum())


def test_model_days_rejects_a_coordinate_the_model_does_not_have():
    with pytest.raises(KeyError, match="no coordinate"):
        pymc_models.model_days(_build("sse"), "not_a_dimension")


def test_the_equateur_series_gives_ssi_thirty_one_latents():
    # Pinned because the count is quoted throughout the plan and AGENTS.md. Under the exact
    # marginalisation of Stage 3 it drops to 30; see tests/test_latent_parameterisations.py.
    from end_of_outbreak import delay_distributions as dd
    from end_of_outbreak import outbreak_data

    data = outbreak_data.load_onset_data()
    model = pymc_models.build_naive_model(
        "ssi",
        data.onsets,
        serial_interval=dd.serial_interval_weights(),
        switch_day=data.ert_arrival_day,
        R_pre=1.0,
        R_post=0.5,
        k=0.18,
        latent_parameterisation="centred",
    )
    assert pymc_models.model_days(model, pymc_models.COHORT_DAY_DIMENSION).size == 31
    np.testing.assert_array_equal(
        pymc_models.model_days(model, pymc_models.LIKELIHOOD_DAY_DIMENSION),
        np.arange(1, data.n_days),
    )


# --- the R switch -------------------------------------------------------------------------


def test_only_days_from_the_switch_day_onwards_feel_R_post():
    unchanged = pymc_models.joint_logp(_build("sse", switch_day=COUNTS.size))
    # Moving R_post cannot matter when the switch day lies past the end of the window.
    moved = pymc_models.joint_logp(_build("sse", switch_day=COUNTS.size, R_post=9.9))
    assert unchanged == pytest.approx(moved)

    # ... but it does matter as soon as the switch falls inside it.
    inside = pymc_models.joint_logp(_build("sse", switch_day=SWITCH_DAY, R_post=9.9))
    assert inside != pytest.approx(unchanged)


def test_a_constant_R_makes_the_switch_day_irrelevant():
    early = pymc_models.joint_logp(_build("dlo", R_pre=0.9, R_post=0.9, switch_day=1))
    late = pymc_models.joint_logp(_build("dlo", R_pre=0.9, R_post=0.9, switch_day=5))
    assert early == pytest.approx(late)


# --- the k → ∞ limit ----------------------------------------------------------------------


@pytest.mark.parametrize("model", ["dlo", "sse"])
def test_large_k_collapses_the_closed_form_models_onto_cori(model):
    cori = pymc_models.joint_logp(_build("cori", k=None))
    assert pymc_models.joint_logp(_build(model, k=1e8)) == pytest.approx(cori, abs=1e-5)


@pytest.mark.parametrize("model", ["dlo", "sse"])
def test_the_collapse_is_a_limit_not_an_identity(model):
    cori = pymc_models.joint_logp(_build("cori", k=None))
    coarse = pymc_models.joint_logp(_build(model, k=0.5))
    finer = pymc_models.joint_logp(_build(model, k=50.0))
    assert abs(finer - cori) < abs(coarse - cori)


# --- priors -------------------------------------------------------------------------------


def test_priors_add_their_own_log_density_to_the_joint():
    R_prior = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)
    k_prior = LogNormalPrior.from_median_and_quantile(median=0.18, quantile=0.09, probability=0.025)
    model = _build("sse", R_pre=R_prior, R_post=R_prior, k=k_prior)
    with_priors = pymc_models.joint_logp(model, R_pre=R_PRE, R_post=R_POST, k=K)

    expected = (
        pymc_models.joint_logp(_build("sse"))
        + R_prior.frozen().logpdf(R_PRE)
        + R_prior.frozen().logpdf(R_POST)
        + k_prior.frozen().logpdf(K)
    )
    assert with_priors == pytest.approx(expected)


def test_the_joint_logp_helper_names_the_variables_it_wants():
    R_prior = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)
    model = _build("sse", R_pre=R_prior)
    with pytest.raises(ValueError, match="no values supplied"):
        pymc_models.joint_logp(model)
    with pytest.raises(ValueError, match="no free variables named"):
        pymc_models.joint_logp(model, R_pre=1.0, nonsense=2.0)


# --- validation ---------------------------------------------------------------------------


def test_cori_refuses_a_dispersion_parameter():
    with pytest.raises(ValueError, match="k → ∞ limit"):
        _build("cori")


@pytest.mark.parametrize("model", ["dlo", "sse", "ssi"])
def test_the_overdispersed_models_require_a_dispersion_parameter(model):
    with pytest.raises(ValueError, match="needs a dispersion parameter"):
        _build(model, k=None)


def test_day_zero_must_carry_the_index_case():
    with pytest.raises(ValueError, match="day 0 must carry at least one case"):
        _build("sse", counts=np.array([0, 1, 2]))


def test_counts_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _build("sse", counts=np.array([1, -1, 2]))


def test_the_serial_interval_must_be_normalised():
    with pytest.raises(ValueError, match="normalised probability vector"):
        _build("sse", serial_interval=np.array([0.5, 0.3]))


def test_a_fixed_parameter_must_be_positive():
    with pytest.raises(ValueError, match="R_pre must be positive"):
        _build("sse", R_pre=0.0)


def test_the_naive_builder_refuses_an_onset_anchored_model():
    with pytest.raises(ValueError, match="use build_onset_anchored_model"):
        _build("sse_so")


def test_a_latent_model_will_not_pick_a_parameterisation_for_you():
    # The Stage-3 choice is recorded in config/config.yaml; nothing may default silently.
    with pytest.raises(ValueError, match="latent_parameterisation must be given explicitly"):
        _build("ssi", latent_parameterisation=None)

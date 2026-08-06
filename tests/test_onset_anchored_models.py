"""Tests for the onset-anchored PyMC builders: Cori-SO, SSE-SO and SSI-SO.

Each likelihood is checked against a density written out by hand from §3.5 of the
implementation plan. The strongest checks here are the **structural** ones of §4.4: with a
deterministic one-day incubation period the onset-anchored models must reduce exactly to their
infection-anchored counterparts, with the serial interval read off the TOST and the ``R`` switch
displaced by one day. That displacement is not a fudge — it is the §5.7 timing convention in
miniature, since the naive models date the intervention by the day a *case appears* and the
onset-anchored ones by the day a *transmission occurs*.

The builders live here rather than in Stage 8 because Stage 3 must benchmark the sampler on the
**actual** SSE-SO model on the real data; a stylised stub with well-conditioned shapes would
benchmark the wrong problem entirely.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.stats

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import outbreak_data, pymc_models, renewal

COUNTS = np.array([1, 0, 2, 1, 0, 3, 0, 1])
SWITCH_DAY = 4
R_PRE, R_POST, K = 1.4, 0.6, 0.5

DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=4.0, sd=3.0),
    incubation=dd.GammaDelay(mean=2.5, sd=2.0),
    max_lag=COUNTS.size,
    tolerance=None,
)


def _build(model, *, counts=COUNTS, delays=DELAYS, **overrides):
    settings: dict[str, Any] = {
        "delays": delays,
        "switch_day": SWITCH_DAY,
        "R_pre": R_PRE,
        "R_post": R_POST,
        "k": K,
        "latent_parameterisation": "centred",
    }
    settings.update(overrides)
    if model == "cori_so":
        settings["k"] = None
        settings.pop("latent_parameterisation")
    return pymc_models.build_onset_anchored_model(model, counts, **settings)


# --- a reference implementation, transcribed from §3.5 ------------------------------------


def _reference_logp(model, *, counts=COUNTS, delays=DELAYS, latents=None, switch_day=SWITCH_DAY):
    """``log p(D, latents)``, written out directly from the model definitions."""
    n_days = counts.size
    transmission_days = np.arange(n_days - 1)
    R = renewal.reproduction_number_by_day(R_PRE, R_POST, n_days=n_days, switch_day=switch_day)[
        transmission_days
    ]
    incubation_design = renewal.delay_design_matrix(
        n_days, delays.incubation, first_lag=1, source_days=transmission_days
    )
    tost_sum = renewal.delay_weighted_sum(counts.astype(np.float64), delays.tost, first_lag=0)[
        transmission_days
    ]

    if model == "cori_so":
        expected_infections = R * tost_sum
        prior = 0.0
    elif model == "sse_so":
        expected_infections = R * latents
        prior = scipy.stats.gamma.logpdf(latents, a=K * tost_sum, scale=1.0 / K).sum()
    else:
        cohort_days = np.flatnonzero(counts > 0)
        tost_design = renewal.delay_design_matrix(
            n_days, delays.tost, first_lag=0, source_days=cohort_days
        )[transmission_days]
        expected_infections = R * (tost_design @ latents)
        prior = scipy.stats.gamma.logpdf(latents, a=K * counts[cohort_days], scale=1.0 / K).sum()

    onsets = incubation_design @ expected_infections
    days = np.arange(1, n_days)
    return prior + scipy.stats.poisson.logpmf(counts[days], mu=onsets[days]).sum()


def _prior_draw(model, rng, *, counts=COUNTS, delays=DELAYS):
    if model == "sse_so":
        scale = renewal.delay_weighted_sum(counts.astype(np.float64), delays.tost, first_lag=0)[
            : counts.size - 1
        ]
    else:
        scale = counts[counts > 0].astype(np.float64)
    return rng.gamma(K * scale, 1.0 / K)


# --- the likelihoods ----------------------------------------------------------------------


def test_cori_so_is_the_poisson_onset_renewal_likelihood():
    assert pymc_models.joint_logp(_build("cori_so")) == pytest.approx(_reference_logp("cori_so"))


@pytest.mark.parametrize(("model", "latent"), [("sse_so", "lambda_tilde"), ("ssi_so", "Y")])
def test_the_onset_anchored_joint_densities_match_the_written_definitions(model, latent):
    rng = np.random.default_rng(0)
    latents = _prior_draw(model, rng)
    built = _build(model)
    assert pymc_models.joint_logp(built, **{latent: latents}) == pytest.approx(
        _reference_logp(model, latents=latents)
    )


def test_sse_so_latents_are_indexed_by_transmission_day_not_onset_day():
    # E_t is the expected number of infections on day t; the final day of the window carries
    # no latent because its infections could only produce onsets after the window closes.
    built = _build("sse_so")
    transmission_days = pymc_models.model_days(built, pymc_models.TRANSMISSION_DAY_DIMENSION)
    np.testing.assert_array_equal(transmission_days, np.arange(COUNTS.size - 1))


def test_ssi_so_latents_are_indexed_by_onset_cohort():
    built = _build("ssi_so")
    np.testing.assert_array_equal(
        pymc_models.model_days(built, pymc_models.COHORT_DAY_DIMENSION),
        np.flatnonzero(COUNTS > 0),
    )


def test_the_incubation_period_may_not_place_mass_at_lag_zero():
    # Enforced structurally: f_inc is stored from lag 1, so E_t can never drive D_t. Without
    # that the recursion would not be well ordered, since f_tost may place mass at lag 0.
    assert dd.INCUBATION_FIRST_LAG == 1
    built = _build("cori_so")
    days = pymc_models.model_days(built, pymc_models.LIKELIHOOD_DAY_DIMENSION)
    assert days[0] >= 1


# --- §4.4: the deterministic one-day incubation limit --------------------------------------


def _one_day_incubation_delays():
    """Delays whose incubation period is exactly one day, so onsets track infections.

    The equivalent serial interval is ``w_s = f_tost,{s-1}``. Since ``w`` is stored from lag 1
    and ``f_tost`` from lag 0, that is the very same array — no shifting or renormalising.
    """
    return dd.OnsetAnchoredDelays(
        serial_interval=DELAYS.tost,
        tost=DELAYS.tost,
        incubation=np.array([1.0]),
        tost_delay=DELAYS.tost_delay,
        incubation_delay=DELAYS.incubation_delay,
    )


def test_a_one_day_incubation_reduces_cori_so_to_cori():
    """The §4.4 structural check, in its cleanest form.

    With ``f_inc = δ_1`` an infection on day ``t`` becomes an onset on day ``t + 1``, so
    ``D_j ~ Poisson(R_{j-1} Σ_s f_tost,s D_{j-1-s})``. That is the Cori renewal model with
    ``w_s = f_tost,{s-1}`` — and with ``R`` switching one day *later*, which is exactly the
    §5.7 convention difference between dating the intervention by the transmission and dating
    it by the case it produces.
    """
    delays = _one_day_incubation_delays()
    onset_anchored = pymc_models.joint_logp(
        pymc_models.build_onset_anchored_model(
            "cori_so",
            COUNTS,
            delays=delays,
            switch_day=SWITCH_DAY,
            R_pre=R_PRE,
            R_post=R_POST,
        )
    )
    naive = pymc_models.joint_logp(
        pymc_models.build_naive_model(
            "cori",
            COUNTS,
            serial_interval=delays.serial_interval,
            switch_day=SWITCH_DAY + 1,
            R_pre=R_PRE,
            R_post=R_POST,
        )
    )
    assert onset_anchored == pytest.approx(naive)


def test_a_one_day_incubation_reduces_ssi_so_to_ssi():
    """The same reduction for SSI-SO, which keeps its latent block unchanged throughout."""
    delays = _one_day_incubation_delays()
    rng = np.random.default_rng(3)
    Y = rng.gamma(K * COUNTS[COUNTS > 0], 1.0 / K)

    onset_anchored = pymc_models.joint_logp(
        pymc_models.build_onset_anchored_model(
            "ssi_so",
            COUNTS,
            delays=delays,
            switch_day=SWITCH_DAY,
            R_pre=R_PRE,
            R_post=R_POST,
            k=K,
            latent_parameterisation="centred",
        ),
        Y=Y,
    )
    naive = pymc_models.joint_logp(
        pymc_models.build_naive_model(
            "ssi",
            COUNTS,
            serial_interval=delays.serial_interval,
            switch_day=SWITCH_DAY + 1,
            R_pre=R_PRE,
            R_post=R_POST,
            k=K,
            latent_parameterisation="centred",
        ),
        Y=Y,
    )
    assert onset_anchored == pytest.approx(naive)


def test_a_one_day_incubation_reduces_sse_so_to_sse():
    """SSE-SO needs its latents integrated out first, which the case-free tail makes exact.

    On a history whose only case is the index case, every SSE-SO latent reaches nothing but
    zero-count days, so the ``marginalised`` parameterisation replaces the whole block with its
    exact closed form and the comparison with SSE's negative-binomial likelihood is analytic.
    """
    counts = np.array([1, 0, 0, 0, 0, 0])
    delays = _one_day_incubation_delays()
    onset_anchored = pymc_models.joint_logp(
        pymc_models.build_onset_anchored_model(
            "sse_so",
            counts,
            delays=delays,
            switch_day=SWITCH_DAY,
            R_pre=R_PRE,
            R_post=R_POST,
            k=K,
            latent_parameterisation="marginalised",
        )
    )
    naive = pymc_models.joint_logp(
        pymc_models.build_naive_model(
            "sse",
            counts,
            serial_interval=delays.serial_interval,
            switch_day=SWITCH_DAY + 1,
            R_pre=R_PRE,
            R_post=R_POST,
            k=K,
        )
    )
    assert onset_anchored == pytest.approx(naive)


# --- the k → ∞ limit -----------------------------------------------------------------------


def _marginal_logp(model, latent, k, *, n_draws, rng):
    """Monte-Carlo ``log p(D)``, integrating the latent block out under its prior."""
    built = pymc_models.build_onset_anchored_model(
        model,
        COUNTS,
        delays=DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_PRE,
        R_post=R_POST,
        k=k,
        latent_parameterisation="centred",
    )
    logp = pymc_models.compile_joint_logp(built)
    scale = (
        renewal.delay_weighted_sum(COUNTS.astype(np.float64), DELAYS.tost, first_lag=0)[
            : COUNTS.size - 1
        ]
        if model == "sse_so"
        else COUNTS[COUNTS > 0].astype(np.float64)
    )
    draws = rng.gamma(k * scale, 1.0 / k, size=(n_draws, scale.size))
    prior = scipy.stats.gamma.logpdf(draws, a=k * scale, scale=1.0 / k).sum(axis=1)
    likelihood = np.array([logp(**{latent: draws[i]}) for i in range(n_draws)])
    return float(np.log(np.mean(np.exp(likelihood - prior))))


@pytest.mark.parametrize(("model", "latent"), [("sse_so", "lambda_tilde"), ("ssi_so", "Y")])
def test_large_k_collapses_the_onset_anchored_models_onto_cori_so(model, latent):
    """The latents concentrate on their means as ``k → ∞``, leaving the Poisson limit.

    Checked at ``k = 200`` rather than at some enormous value: a Gamma log-density with shape
    of order ``10^5`` is a difference of large numbers and loses several decimal places, so
    pushing ``k`` higher degrades the comparison instead of sharpening it. At ``k = 200`` the
    latent standard deviation is already under 10% of the mean and few draws suffice.
    """
    rng = np.random.default_rng(5)
    cori_so = pymc_models.joint_logp(_build("cori_so"))
    assert _marginal_logp(model, latent, 200.0, n_draws=3000, rng=rng) == pytest.approx(
        cori_so, abs=0.01
    )


@pytest.mark.parametrize(("model", "latent"), [("sse_so", "lambda_tilde"), ("ssi_so", "Y")])
def test_the_onset_anchored_collapse_is_a_limit_not_an_identity(model, latent):
    rng = np.random.default_rng(6)
    cori_so = pymc_models.joint_logp(_build("cori_so"))
    coarse = _marginal_logp(model, latent, 0.5, n_draws=4000, rng=rng)
    assert abs(coarse - cori_so) > 0.1


# --- the real series -----------------------------------------------------------------------


def test_the_real_series_builds_and_keeps_every_inference_day():
    data = outbreak_data.load_onset_data()
    delays = dd.build_onset_anchored_delays()
    for model, k in (("cori_so", None), ("sse_so", 0.18), ("ssi_so", 0.18)):
        built = pymc_models.build_onset_anchored_model(
            model,
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=1.0,
            R_post=0.5,
            k=k,
            latent_parameterisation=None if k is None else "centred",
        )
        np.testing.assert_array_equal(
            pymc_models.model_days(built, pymc_models.LIKELIHOOD_DAY_DIMENSION),
            np.arange(1, data.n_days),
        )


def test_the_dispatcher_routes_on_anchoring():
    delays = dd.build_onset_anchored_delays(
        serial_interval=dd.GammaDelay(mean=4.0, sd=3.0),
        incubation=dd.GammaDelay(mean=2.5, sd=2.0),
        max_lag=COUNTS.size,
        tolerance=None,
    )
    common: dict[str, Any] = {
        "delays": delays,
        "switch_day": SWITCH_DAY,
        "R_pre": R_PRE,
        "R_post": R_POST,
        "k": K,
        "latent_parameterisation": "centred",
    }
    naive = pymc_models.build_model("ssi", COUNTS, **common)
    onset = pymc_models.build_model("ssi_so", COUNTS, **common)
    assert pymc_models.joint_logp(naive, Y=np.ones(COUNTS[COUNTS > 0].size)) != pytest.approx(
        pymc_models.joint_logp(onset, Y=np.ones(COUNTS[COUNTS > 0].size))
    )


# --- validation ------------------------------------------------------------------------------


def test_the_onset_anchored_builder_refuses_a_naive_model():
    with pytest.raises(ValueError, match="use build_naive_model"):
        _build("sse")


def test_cori_so_refuses_a_dispersion_parameter():
    with pytest.raises(ValueError, match="k → ∞ limit"):
        pymc_models.build_onset_anchored_model(
            "cori_so",
            COUNTS,
            delays=DELAYS,
            switch_day=SWITCH_DAY,
            R_pre=R_PRE,
            R_post=R_POST,
            k=K,
        )


def test_the_onset_anchored_models_need_a_parameterisation():
    with pytest.raises(ValueError, match="latent_parameterisation must be given explicitly"):
        _build("sse_so", latent_parameterisation=None)

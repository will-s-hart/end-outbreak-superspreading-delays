"""Tests for the marginal likelihoods (Stage 6).

The evidence *is* a normalising constant, so the failure mode that matters is not noise but a
silent constant offset — a Jacobian dropped, a transform applied backwards, a prior term counted
twice. Every such error produces a perfectly plausible number and a wrong Bayes factor. So the
tests are ordered by how invisible the mistake would be:

1. **The unconstrained target is the model's own density.** Pinned against
   :func:`end_of_outbreak.pymc_models.compile_joint_logp` on the natural scale plus an
   independently written Jacobian. Without this, everything below could agree with everything
   else and still be wrong by a constant.
2. **Bridge sampling against deterministic quadrature**, for DLO and SSE where ``k`` is fixed
   and the integral is two-dimensional. This is §6.5's own acceptance criterion, and it is the
   only check here that does not compare one Monte-Carlo estimator with another.
3. **The three estimators agree** on a short SSI history, judged against their reported standard
   errors rather than a round number.
4. **Marginalising latents exactly does not move the evidence.** ``marginalised_inverse_cdf``
   removes coordinates that ``inverse_cdf`` samples; the normalising constant is the same
   integral either way, so agreement is a check that the ``pm.Potential`` carries the whole of
   the removed factor.
5. **The probabilities.** Normalisation, invariance to a shared offset, and the hand-computable
   two-model case.
"""

from __future__ import annotations

import numpy as np
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import fitting, pymc_models
from end_of_outbreak import model_evidence as me
from end_of_outbreak.model_specifications import LogNormalPrior

COUNTS = np.array([1, 0, 2, 1, 0, 2, 0, 1])
SWITCH_DAY = 4
FIXED_K = 0.6

SHORT_DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)

R_PRE_PRIOR = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)
R_POST_PRIOR = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)

SAMPLER = fitting.SamplerSettings(draws=400, tune=600, chains=2, target_accept=0.9, seed=7)


def _build(model, *, k=FIXED_K, parameterisation="marginalised_inverse_cdf"):
    return pymc_models.build_model(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_PRE_PRIOR,
        R_post=R_POST_PRIOR,
        k=k,
        latent_parameterisation=parameterisation,
    )


def _fit(model, *, k=FIXED_K, parameterisation="marginalised_inverse_cdf", seed=None):
    settings = (
        SAMPLER
        if seed is None
        else fitting.SamplerSettings(
            draws=SAMPLER.draws,
            tune=SAMPLER.tune,
            chains=SAMPLER.chains,
            target_accept=SAMPLER.target_accept,
            seed=seed,
        )
    )
    return fitting.fit_model(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_PRE_PRIOR,
        R_post=R_POST_PRIOR,
        k=k,
        latent_parameterisation=parameterisation,
        sampler=settings,
    )


def _evidence(model, idata, **kwargs):
    kwargs.setdefault("k", FIXED_K)
    kwargs.setdefault("latent_parameterisation", "marginalised_inverse_cdf")
    return me.log_evidence(
        model,
        idata,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_PRE_PRIOR,
        R_post=R_POST_PRIOR,
        **kwargs,
    )


@pytest.fixture(scope="module")
def closed_form_fits():
    """DLO and SSE at fixed ``k``: no latents, so the whole integral is two-dimensional."""
    return {model: _fit(model) for model in ("dlo", "sse")}


@pytest.fixture(scope="module")
def ssi_fit():
    return _fit("ssi")


# ---------------------------------------------------------------------------------------
# 1. The unconstrained target is the built model's own density
# ---------------------------------------------------------------------------------------


def test_the_unconstrained_density_is_the_joint_plus_the_transform_jacobian():
    """The whole stage rests on this: an inverted transform shifts every evidence by a constant.

    ``R_pre`` and ``R_post`` are sampled as ``log R``, so the density in those coordinates is
    ``p(R) · R``. Writing that factor out by hand and comparing with
    :func:`pymc_models.compile_joint_logp` on the natural scale pins both the direction of the
    transform and the presence of the Jacobian, independently of anything else in this module.
    """
    built = _build("dlo")
    target = me.UnconstrainedTarget(built)
    joint = pymc_models.compile_joint_logp(built)

    R_pre, R_post = 1.7, 0.4
    unconstrained = np.array([[np.log(R_pre), np.log(R_post)]])
    natural = joint(R_pre=R_pre, R_post=R_post)
    jacobian = np.log(R_pre) + np.log(R_post)

    assert target.dimension == 2
    assert float(target.log_density(unconstrained)[0]) == pytest.approx(natural + jacobian)


def test_the_likelihood_term_is_the_density_less_the_prior():
    built = _build("sse")
    target = me.UnconstrainedTarget(built)
    observation = pymc_models.compile_observation_logp(built)

    R_pre, R_post = 2.1, 0.6
    point = np.array([[np.log(R_pre), np.log(R_post)]])
    assert float(target.log_likelihood(point)[0]) == pytest.approx(
        observation(R_pre=R_pre, R_post=R_post)
    )


def test_posterior_points_land_in_the_target_coordinates(closed_form_fits):
    """The draws and the density must live in the same space, or every weight is nonsense."""
    idata = closed_form_fits["dlo"]
    target = me.UnconstrainedTarget(_build("dlo"))
    points = target.posterior_points(idata)

    R_pre = np.asarray(idata.posterior.data_vars["R_pre"]).reshape(-1)
    assert points.shape == (R_pre.size, 2)
    # chain-major stacking, as posterior_state produces
    assert np.allclose(np.exp(points[:, 0]), R_pre)


def test_a_fit_missing_a_free_variable_is_refused(closed_form_fits):
    """Estimating ``k`` here but having fitted it fixed must not silently produce a number."""
    with pytest.raises(ValueError, match="no 'k' in its posterior"):
        _evidence(
            "dlo",
            closed_form_fits["dlo"],
            k=LogNormalPrior.from_median_and_quantile(
                median=0.18, quantile=0.09, probability=0.025
            ),
        )


# ---------------------------------------------------------------------------------------
# 2. Bridge sampling against direct numerical marginalisation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["dlo", "sse"])
def test_bridge_sampling_reproduces_deterministic_quadrature(model, closed_form_fits):
    """§6.5's acceptance criterion: the one check that is not sampler-against-sampler.

    At fixed ``k`` neither model has latents, so ``(log R_pre, log R_post)`` can be integrated on
    a Simpson grid. The quadrature shares only the integrand with bridge sampling — none of the
    proposal fitting, the weighting or the recursion.
    """
    idata = closed_form_fits[model]
    target = me.UnconstrainedTarget(_build(model))
    points = target.posterior_points(idata)
    exact = me.log_evidence_by_quadrature(
        target, centre=points.mean(axis=0), scale=points.std(axis=0)
    )
    estimate = _evidence(model, idata, rng=np.random.default_rng(11))

    # The box has to contain the mass before the number means anything.
    assert exact.log_boundary_ratio < -20.0
    assert abs(estimate.log_evidence - exact.log_evidence) < max(
        5.0 * estimate.standard_error, 0.01
    )


def test_quadrature_refuses_a_model_it_cannot_integrate(ssi_fit):
    """A tensor-product grid over 31 latent coordinates is not a check, it is a hang."""
    target = me.UnconstrainedTarget(_build("ssi"))
    with pytest.raises(ValueError, match="only tractable in a few dimensions"):
        me.log_evidence_by_quadrature(
            target,
            centre=np.zeros(target.dimension),
            scale=np.ones(target.dimension),
        )


# ---------------------------------------------------------------------------------------
# 3. The three estimators agree
# ---------------------------------------------------------------------------------------


def test_the_three_estimators_agree_on_a_latent_model(ssi_fit):
    """SSI has no quadrature to check against, so the estimators check each other.

    They are judged against their own reported standard errors: prior Monte Carlo on a short
    history is legitimate but wide, and demanding a fixed tolerance of it would be a test of the
    tolerance rather than of the estimator.
    """
    estimates = {
        estimator: _evidence(
            "ssi",
            ssi_fit,
            estimator=estimator,
            n_proposal_draws=40_000,
            rng=np.random.default_rng(3),
        )
        for estimator in me.ESTIMATORS
    }
    for estimate in estimates.values():
        assert np.isfinite(estimate.log_evidence)
        assert estimate.standard_error >= 0.0

    reference = estimates[me.BRIDGE_SAMPLING]
    for estimator, estimate in estimates.items():
        scale = np.hypot(estimate.standard_error, reference.standard_error)
        assert abs(estimate.log_evidence - reference.log_evidence) < max(4.0 * scale, 0.05), (
            f"{estimator} disagrees with bridge sampling by more than its own error bar"
        )


def test_an_unknown_estimator_is_refused(closed_form_fits):
    with pytest.raises(ValueError, match="unknown estimator"):
        _evidence("dlo", closed_form_fits["dlo"], estimator="harmonic_mean")


def test_a_latent_model_needs_its_parameterisation(ssi_fit):
    with pytest.raises(ValueError):
        _evidence("ssi", ssi_fit, latent_parameterisation=None)


# ---------------------------------------------------------------------------------------
# 4. Exact marginalisation of latents leaves the evidence alone
# ---------------------------------------------------------------------------------------


def test_marginalising_latents_exactly_does_not_move_the_evidence():
    """Different numbers of coordinates, same integral.

    ``marginalised_inverse_cdf`` integrates the uncoupled latents out in closed form and carries
    the result as a ``pm.Potential``; plain ``inverse_cdf`` samples them. If the potential were
    missing any part of the ``(1 + c_u/k)^{−k·scale_u}`` factor — a normalising constant, say —
    the two evidences would differ by exactly that, and nothing else in the suite would notice.
    """
    marginalised = _build("ssi", parameterisation="marginalised_inverse_cdf")
    full = _build("ssi", parameterisation="inverse_cdf")
    assert me.UnconstrainedTarget(marginalised).dimension < me.UnconstrainedTarget(full).dimension

    estimates = {}
    for name in ("marginalised_inverse_cdf", "inverse_cdf"):
        idata = _fit("ssi", parameterisation=name)
        estimates[name] = _evidence(
            "ssi",
            idata,
            latent_parameterisation=name,
            n_proposal_draws=20_000,
            rng=np.random.default_rng(5),
        )
    difference = (
        estimates["marginalised_inverse_cdf"].log_evidence - estimates["inverse_cdf"].log_evidence
    )
    scale = np.hypot(*(estimate.standard_error for estimate in estimates.values()))
    assert abs(difference) < max(4.0 * scale, 0.1)


# ---------------------------------------------------------------------------------------
# 5. Posterior model probabilities
# ---------------------------------------------------------------------------------------


def test_probabilities_normalise_and_follow_the_bayes_factor():
    probabilities = me.posterior_model_probabilities({"a": -100.0, "b": -101.0})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    # Uniform prior over models, so the odds are the Bayes factor.
    assert probabilities["a"] / probabilities["b"] == pytest.approx(np.exp(1.0))


def test_probabilities_are_invariant_to_a_shared_offset():
    base = {"dlo": -97.1, "sse": -106.8, "ssi": -84.8}
    shifted = {name: value - 5000.0 for name, value in base.items()}
    for name, probability in me.posterior_model_probabilities(base).items():
        assert me.posterior_model_probabilities(shifted)[name] == pytest.approx(probability)


def test_widely_separated_evidences_do_not_overflow():
    probabilities = me.posterior_model_probabilities({"near": -10.0, "far": -1000.0})
    assert probabilities["near"] == pytest.approx(1.0)
    assert probabilities["far"] == pytest.approx(0.0, abs=1e-300)


def test_the_key_order_is_preserved():
    names = ["ssi", "dlo", "sse"]
    probabilities = me.posterior_model_probabilities(dict.fromkeys(names, -1.0))
    assert list(probabilities) == names


def test_a_failed_estimate_is_not_silently_compared():
    with pytest.raises(ValueError, match="not be nan"):
        me.posterior_model_probabilities({"a": -1.0, "b": float("nan")})


# ---------------------------------------------------------------------------------------
# A model with nothing to integrate
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", me.ESTIMATORS)
def test_a_model_with_no_free_variables_has_the_likelihood_as_its_exact_evidence(estimator):
    """SSE with ``R`` and ``k`` both fixed: a point-mass posterior, so ``p(D) = p(D | θ)``.

    Checked against the negative-binomial likelihood written out from the renewal sums, not
    against the model's own compiled density, which is what the function returns.
    """
    import scipy.stats

    from end_of_outbreak import renewal

    R_pre, R_post = 0.9, 0.5
    evidence = me.log_evidence(
        "sse",
        None,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_pre,
        R_post=R_post,
        k=FIXED_K,
        estimator=estimator,
    )
    Lambda = renewal.delay_weighted_sum(
        COUNTS.astype(np.float64), SHORT_DELAYS.serial_interval, first_lag=1
    )
    days = renewal.likelihood_days(Lambda, COUNTS)
    R = renewal.reproduction_number_by_day(
        R_pre, R_post, n_days=COUNTS.size, switch_day=SWITCH_DAY
    )[days]
    size = FIXED_K * Lambda[days]
    expected = scipy.stats.nbinom.logpmf(COUNTS[days], size, FIXED_K / (FIXED_K + R)).sum()

    assert evidence.estimator == me.EXACT
    assert evidence.log_evidence == pytest.approx(float(expected), rel=1e-12)
    assert evidence.standard_error == 0.0
    assert evidence.n_draws == 0

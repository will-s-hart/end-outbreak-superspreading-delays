"""Tests for the Stage-3 latent strategies.

Three things need pinning:

1. **Every strategy targets the same distribution.** The reparameterised blocks are checked
   against the centred one by the exact change-of-variables identity, so a Jacobian slip or a
   mis-specified rate cannot pass.
2. **Marginalising the uncoupled latents is exact.** Checked twice — once by one-dimensional
   quadrature over a single removed latent, where the answer is exact to quadrature precision,
   and once by Monte-Carlo integration over a larger removed block.
3. **The negligible-latent threshold costs what §6.3 says it costs.** The induced change in the
   *observation* log-likelihood is measured and held against the bound
   ``max(R) · Σ_dropped scale``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.integrate
import scipy.special
import scipy.stats

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import latent_parameterisations as lp
from end_of_outbreak import outbreak_data, pymc_models, renewal

COUNTS = np.array([1, 0, 2, 1, 0, 3])
SERIAL_INTERVAL = np.array([0.5, 0.3, 0.2])
SWITCH_DAY = 3
R_PRE, R_POST, K = 1.4, 0.6, 0.5
COHORT_DAYS = np.flatnonzero(COUNTS > 0)
SCALE = COUNTS[COHORT_DAYS].astype(np.float64)


def _ssi(parameterisation, **overrides):
    settings: dict[str, Any] = {
        "serial_interval": SERIAL_INTERVAL,
        "switch_day": SWITCH_DAY,
        "R_pre": R_PRE,
        "R_post": R_POST,
        "k": K,
        "latent_parameterisation": parameterisation,
    }
    settings.update(overrides)
    return pymc_models.build_naive_model("ssi", COUNTS, **settings)


# --- the registry -------------------------------------------------------------------------


def test_every_registered_strategy_resolves_by_name():
    for name, parameterisation in lp.PARAMETERISATIONS.items():
        assert lp.parameterisation_of(name) is parameterisation
        assert lp.parameterisation_of(parameterisation) is parameterisation


def test_an_unknown_strategy_lists_the_known_ones():
    with pytest.raises(KeyError, match="unknown latent parameterisation"):
        lp.parameterisation_of("centered")  # American spelling, deliberately


def test_the_two_headings_of_section_6_3_are_kept_apart():
    # Mean-1 rescaling is an initialisation fix, not a reparameterisation: on the log scale it
    # is a pure translation, so the Gamma shape and hence the geometry are untouched.
    assert lp.UNIT_MEAN.changes_the_starting_point
    assert not lp.UNIT_MEAN.changes_the_space
    assert lp.INVERSE_CDF.changes_the_space
    assert not lp.INVERSE_CDF.changes_the_starting_point
    assert lp.MARGINALISED.changes_the_space
    assert not lp.CENTRED.changes_the_space
    assert not lp.CENTRED.changes_the_starting_point


def test_inverse_cdf_and_rescaling_do_not_compose():
    with pytest.raises(ValueError, match="no-op"):
        lp.LatentParameterisation(
            name="nonsense", reparameterisation="inverse_cdf", rescale_to_unit_mean=True
        )


def test_the_gamma_quantile_function_is_scale_equivariant():
    """Why the combination above is refused rather than benchmarked.

    ``scale · F⁻¹_{Gamma(α, α)}(u) = F⁻¹_{Gamma(α, k)}(u)`` for ``α = k · scale``, so under the
    inverse-CDF parameterisation rescaling composes with the ``u`` map as a constant factor and
    the sampled variable's target is untouched. There is nothing there to measure.
    """
    k, scale = 0.18, 2.5
    alpha = k * scale
    u = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(
        scale * scipy.stats.gamma.ppf(u, a=alpha, scale=1.0 / alpha),
        scipy.stats.gamma.ppf(u, a=alpha, scale=1.0 / k),
    )


# --- classification -------------------------------------------------------------------------


def test_classification_covers_every_position_exactly_once():
    scale = np.array([0.0, 0.5, 2.0, 0.001])
    couples = np.array([True, False, True, True])
    classification = lp.classify_latents(
        scale=scale,
        couples_to_observations=couples,
        parameterisation=lp.MARGINALISED,
        negligible_threshold=0.01,
    )
    assert classification.sampled.tolist() == [2]
    assert classification.marginalised.tolist() == [1]
    assert classification.dropped.tolist() == [3]
    assert classification.inactive.tolist() == [0]
    assert classification.n_latents == 4


def test_without_the_marginalisation_switch_nothing_is_integrated_out():
    classification = lp.classify_latents(
        scale=np.array([1.0, 2.0]),
        couples_to_observations=np.array([True, False]),
        parameterisation=lp.CENTRED,
    )
    assert classification.marginalised.size == 0
    assert classification.sampled.tolist() == [0, 1]


def test_a_zero_threshold_drops_nothing():
    classification = lp.classify_latents(
        scale=np.array([1e-12, 1.0]),
        couples_to_observations=np.ones(2, dtype=bool),
        parameterisation=lp.CENTRED,
        negligible_threshold=0.0,
    )
    assert classification.dropped.size == 0


def test_classification_rejects_a_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        lp.classify_latents(
            scale=np.ones(2),
            couples_to_observations=np.ones(2, dtype=bool),
            parameterisation=lp.CENTRED,
            negligible_threshold=-1.0,
        )


# --- every strategy targets the same distribution ---------------------------------------------


def test_mean_one_rescaling_is_the_same_law_up_to_its_jacobian():
    centred = pymc_models.compile_joint_logp(_ssi("centred"))
    rescaled = pymc_models.compile_joint_logp(_ssi("unit_mean"))
    rng = np.random.default_rng(0)
    for _ in range(4):
        unit = rng.gamma(K * SCALE, 1.0 / (K * SCALE))
        assert rescaled(Y_unit_mean=unit) == pytest.approx(
            centred(Y=SCALE * unit) + np.log(SCALE).sum()
        )


def test_the_inverse_cdf_block_is_the_same_law_in_uniform_coordinates():
    centred = pymc_models.compile_joint_logp(_ssi("centred"))
    inverted = pymc_models.compile_joint_logp(_ssi("inverse_cdf"))
    rng = np.random.default_rng(1)
    for _ in range(4):
        u = rng.uniform(size=SCALE.size)
        Y = scipy.stats.gamma.ppf(u, a=K * SCALE, scale=1.0 / K)
        prior = scipy.stats.gamma.logpdf(Y, a=K * SCALE, scale=1.0 / K).sum()
        # The Uniform prior contributes nothing, so all that is left is the likelihood.
        assert inverted(Y_uniform=u) == pytest.approx(centred(Y=Y) - prior)


def test_the_free_variable_is_named_for_the_strategy():
    assert lp.free_variable_name("Y", lp.CENTRED) == "Y"
    assert lp.free_variable_name("Y", lp.UNIT_MEAN) == "Y_unit_mean"
    assert lp.free_variable_name("Y", lp.INVERSE_CDF) == "Y_uniform"
    for parameterisation in lp.PARAMETERISATIONS.values():
        model = _ssi(parameterisation)
        assert lp.free_variable_name("Y", parameterisation) in {rv.name for rv in model.free_RVs}


# --- marginalising the uncoupled latents is exact -----------------------------------------------

# A history whose last case is on the second-to-last transmission day, so exactly one SSE-SO
# latent becomes uncoupled and the integral over it can be done by quadrature.
ONE_UNCOUPLED_COUNTS = np.array([1, 0, 2, 1, 0])
SO_DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=4.0, sd=3.0),
    incubation=dd.GammaDelay(mean=2.5, sd=2.0),
    max_lag=6,
    tolerance=None,
)


def _sse_so(counts, parameterisation, **overrides):
    settings: dict[str, Any] = {
        "delays": SO_DELAYS,
        "switch_day": SWITCH_DAY,
        "R_pre": R_PRE,
        "R_post": R_POST,
        "k": K,
        "latent_parameterisation": parameterisation,
    }
    settings.update(overrides)
    return pymc_models.build_onset_anchored_model("sse_so", counts, **settings)


def _tost_sum(counts):
    return renewal.delay_weighted_sum(counts.astype(np.float64), SO_DELAYS.tost, first_lag=0)[
        : counts.size - 1
    ]


def test_marginalising_one_latent_matches_quadrature():
    counts = ONE_UNCOUPLED_COUNTS
    full = _sse_so(counts, "centred")
    marginalised = _sse_so(counts, "marginalised")
    kept = pymc_models.model_days(marginalised, pymc_models.TRANSMISSION_DAY_DIMENSION)
    everything = pymc_models.model_days(full, pymc_models.TRANSMISSION_DAY_DIMENSION)
    removed = np.setdiff1d(everything, kept)
    assert removed.size == 1, "this history is meant to leave exactly one latent uncoupled"

    scale = _tost_sum(counts)
    rng = np.random.default_rng(2)
    kept_values = rng.gamma(K * scale[kept], 1.0 / K)
    evaluate_full = pymc_models.compile_joint_logp(full)

    def density(value):
        latents = np.empty(everything.size)
        latents[kept] = kept_values
        latents[removed] = value
        return np.exp(evaluate_full(lambda_tilde=latents))

    integral, error = scipy.integrate.quad(density, 0.0, np.inf, limit=200)
    assert error < 1e-5 * integral, "the quadrature has to be sharper than the claim it checks"
    evaluate_marginalised = pymc_models.compile_joint_logp(marginalised)
    assert evaluate_marginalised(lambda_tilde=kept_values) == pytest.approx(
        np.log(integral), abs=1e-5
    )


def test_marginalising_a_whole_block_matches_monte_carlo_integration():
    counts = np.array([1, 0, 2, 1, 0, 0, 0, 0, 0])
    full = _sse_so(counts, "centred")
    marginalised = _sse_so(counts, "marginalised")
    kept = pymc_models.model_days(marginalised, pymc_models.TRANSMISSION_DAY_DIMENSION)
    everything = pymc_models.model_days(full, pymc_models.TRANSMISSION_DAY_DIMENSION)
    removed = np.setdiff1d(everything, kept)
    assert removed.size > 1

    scale = _tost_sum(counts)
    rng = np.random.default_rng(3)
    kept_values = rng.gamma(K * scale[kept], 1.0 / K)
    evaluate_full = pymc_models.compile_joint_logp(full)

    n_draws = 4000
    latents = np.empty((n_draws, everything.size))
    latents[:, kept] = kept_values
    latents[:, removed] = rng.gamma(K * scale[removed], 1.0 / K, size=(n_draws, removed.size))
    prior = scipy.stats.gamma.logpdf(latents[:, removed], a=K * scale[removed], scale=1.0 / K).sum(
        axis=1
    )
    integrand = np.exp(
        np.array([evaluate_full(lambda_tilde=latents[i]) for i in range(n_draws)]) - prior
    )
    estimate = integrand.mean()
    standard_error = integrand.std(ddof=1) / np.sqrt(n_draws)

    evaluate_marginalised = pymc_models.compile_joint_logp(marginalised)
    assert evaluate_marginalised(lambda_tilde=kept_values) == pytest.approx(
        np.log(estimate), abs=5.0 * standard_error / estimate
    )


def test_the_conditional_posterior_of_a_marginalised_latent_is_the_conjugate_gamma():
    """The removed latents are recoverable, which is what the RAC reset state will need."""
    k, scale, coupling = 0.18, np.array([0.4, 2.0]), np.array([0.9, 1.7])
    shape, rate = lp.conditional_posterior(k=k, scale=scale, coupling=coupling)
    np.testing.assert_allclose(shape, k * scale)
    np.testing.assert_allclose(rate, k + coupling)

    # prior · exp(-c y) / normalising constant == the posterior density, exactly.
    y = np.array([0.13, 0.75])
    prior = scipy.stats.gamma.logpdf(y, a=k * scale, scale=1.0 / k)
    log_constant = k * scale * np.log(k / (k + coupling))
    np.testing.assert_allclose(
        prior - coupling * y - log_constant,
        scipy.stats.gamma.logpdf(y, a=shape, scale=1.0 / rate),
    )


# --- what marginalisation buys on the real series -------------------------------------------


@pytest.fixture(scope="module")
def real_series():
    data = outbreak_data.load_onset_data()
    delays = dd.build_onset_anchored_delays()
    tost_sum = renewal.delay_weighted_sum(data.onsets.astype(np.float64), delays.tost, first_lag=0)[
        : data.n_days - 1
    ]
    return data, delays, tost_sum


def _real_sse_so(real_series, parameterisation, **overrides):
    data, delays, _ = real_series
    settings: dict[str, Any] = {
        "delays": delays,
        "switch_day": data.ert_arrival_day,
        "R_pre": 1.0,
        "R_post": 0.5,
        "k": 0.18,
        "latent_parameterisation": parameterisation,
    }
    settings.update(overrides)
    return pymc_models.build_onset_anchored_model("sse_so", data.onsets, **settings)


def test_the_untouched_sse_so_block_is_as_pathological_as_the_plan_says(real_series):
    _, _, tost_sum = real_series
    model = _real_sse_so(real_series, "centred")
    days = pymc_models.model_days(model, pymc_models.TRANSMISSION_DAY_DIMENSION)
    shapes = 0.18 * tost_sum[days]

    # 110 latents, as §6.3 measured. The smallest shape is 2.67e-6 rather than the 2.2e-6
    # quoted there because that figure is day 110's, and day 110 carries no latent here: its
    # infections could only produce onsets after the window closes, so it is unidentified.
    assert days.size == 110
    np.testing.assert_array_equal(days, np.arange(110))
    assert shapes.max() == pytest.approx(0.383, abs=5e-4)
    assert shapes.min() == pytest.approx(2.67e-6, rel=0.02)
    assert [int((shapes < bound).sum()) for bound in (1e-2, 1e-3, 1e-4)] == [43, 31, 19]


def test_marginalisation_removes_exactly_the_pathological_tail(real_series):
    data, _, tost_sum = real_series
    model = _real_sse_so(real_series, "marginalised")
    days = pymc_models.model_days(model, pymc_models.TRANSMISSION_DAY_DIMENSION)

    # A latent is removable iff no day with a case follows it, so the split lands on the last
    # observed onset — 2 June 2018, day 58.
    last_case = int(np.max(np.flatnonzero(data.onsets > 0)))
    assert last_case == 58
    np.testing.assert_array_equal(days, np.arange(last_case))
    assert days.size == 58

    shapes = 0.18 * tost_sum[days]
    assert shapes.min() == pytest.approx(0.0244, abs=5e-4)
    # Four orders of magnitude of log-scale spread removed, at zero cost in accuracy.
    assert np.sqrt(scipy.special.polygamma(1, shapes.min())) < 50.0


def test_marginalisation_removes_the_last_cohort_from_the_individual_level_models(real_series):
    data, delays, _ = real_series
    for model, kwargs in (
        ("ssi", {"serial_interval": delays.serial_interval}),
        ("ssi_so", {"delays": delays}),
    ):
        build = (
            pymc_models.build_naive_model
            if model == "ssi"
            else pymc_models.build_onset_anchored_model
        )
        sizes = {
            name: pymc_models.model_days(
                build(
                    model,
                    data.onsets,
                    switch_day=data.ert_arrival_day,
                    R_pre=1.0,
                    R_post=0.5,
                    k=0.18,
                    latent_parameterisation=name,
                    **kwargs,
                ),
                pymc_models.COHORT_DAY_DIMENSION,
            ).size
            for name in ("centred", "marginalised")
        }
        # Only the final cohort is uncoupled here: every earlier one can still reach a case.
        assert sizes == {"centred": 31, "marginalised": 30}


# --- the negligible-latent threshold ----------------------------------------------------------


def _observation_logp_at_prior_mean(model, tost_sum):
    """The observation term alone, with every surviving latent held at its prior mean."""
    days = pymc_models.model_days(model, pymc_models.TRANSMISSION_DAY_DIMENSION)
    evaluate = pymc_models.compile_observation_logp(model)
    return evaluate(lambda_tilde=tost_sum[days])


@pytest.mark.parametrize("threshold", [1e-3, 1e-2, 3e-2])
def test_dropping_latents_costs_no_more_than_the_stated_bound(real_series, threshold):
    """§6.3's error bound, measured rather than assumed.

    Removing a latent removes ``R_t · reach_t · λ_t`` from ``Σ_j μ_j``, and every day it could
    have reached carries a zero count, so the observation log-likelihood can only rise — by at
    most ``max(R) · Σ_dropped λ_t``.
    """
    _, _, tost_sum = real_series
    exact = _observation_logp_at_prior_mean(_real_sse_so(real_series, "centred"), tost_sum)
    approximate = _observation_logp_at_prior_mean(
        _real_sse_so(real_series, "centred", negligible_latent_threshold=threshold), tost_sum
    )
    dropped = tost_sum[tost_sum < threshold].sum()
    bound = max(1.0, 0.5) * dropped

    assert 0.0 < approximate - exact <= bound
    assert dropped > 0.0, "the threshold is meant to drop something"


def test_tightening_the_threshold_shrinks_the_error(real_series):
    _, _, tost_sum = real_series
    exact = _observation_logp_at_prior_mean(_real_sse_so(real_series, "centred"), tost_sum)
    errors = [
        _observation_logp_at_prior_mean(
            _real_sse_so(real_series, "centred", negligible_latent_threshold=threshold),
            tost_sum,
        )
        - exact
        for threshold in (3e-2, 1e-2, 1e-3)
    ]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 0.01  # a tight threshold is indistinguishable from no threshold


def test_the_threshold_is_a_no_op_once_the_uncoupled_latents_are_marginalised(real_series):
    """The Stage-3 finding: with the exact marginalisation in place there is nothing left to drop.

    Every latent that survives marginalisation has ``λ_t >= 0.136`` on this series, so any
    threshold small enough to be defensible removes none of them.
    """
    _, _, tost_sum = real_series
    baseline = pymc_models.model_days(
        _real_sse_so(real_series, "marginalised"), pymc_models.TRANSMISSION_DAY_DIMENSION
    )
    assert tost_sum[baseline].min() > 0.1
    for threshold in (1e-3, 1e-2, 1e-1):
        days = pymc_models.model_days(
            _real_sse_so(real_series, "marginalised", negligible_latent_threshold=threshold),
            pymc_models.TRANSMISSION_DAY_DIMENSION,
        )
        np.testing.assert_array_equal(days, baseline)


def test_a_threshold_that_orphans_an_observed_case_is_refused():
    with pytest.raises(ValueError, match="no surviving latent to explain them"):
        _sse_so(ONE_UNCOUPLED_COUNTS, "centred", negligible_latent_threshold=10.0)


# --- initialisation ----------------------------------------------------------------------------


def test_the_geometric_mean_start_sits_at_the_typical_value_not_the_arithmetic_mean():
    scale = np.array([0.01, 1.0])
    values = lp.suggested_initial_values(
        "Y", k=0.18, sampled_scale=scale, parameterisation=lp.CENTRED_GEOMETRIC_INIT
    )["Y"]
    alpha = 0.18 * scale
    np.testing.assert_allclose(values, np.exp(scipy.special.digamma(alpha)) / 0.18)
    # PyMC would start at alpha/beta = scale. The geometric mean is below that at every shape
    # of interest, and the gap widens without limit as the shape shrinks -- at k = 0.18 the
    # start is already 60x too high for a unit scale, and 10^24 times too high at scale 0.01.
    assert values[1] < scale[1] / 50.0
    assert values[0] / scale[0] < values[1] / scale[1] / 1e6


def test_the_rescaled_start_is_expressed_in_the_rescaled_variable():
    scale = np.array([0.01, 1.0])
    values = lp.suggested_initial_values(
        "Y", k=0.18, sampled_scale=scale, parameterisation=lp.UNIT_MEAN_GEOMETRIC_INIT
    )
    assert set(values) == {"Y_unit_mean"}
    alpha = 0.18 * scale
    np.testing.assert_allclose(values["Y_unit_mean"], np.exp(scipy.special.digamma(alpha)) / alpha)


def test_the_geometric_mean_start_is_refused_when_it_underflows(real_series):
    """The hard numerical wall behind the Stage-3 conclusion.

    ``E[log Y] ≈ −1/α``, so a Gamma shape below about 1.4 × 10⁻³ has its entire typical set
    outside the range a double can represent: there is no number to start the chain at. The
    untouched SSE-SO block reaches 2.7 × 10⁻⁶, which is why no amount of initialisation or
    reparameterisation rescues it — the coordinates have to go.
    """
    _, _, tost_sum = real_series
    untouched = pymc_models.model_days(
        _real_sse_so(real_series, "centred_geometric_init"), pymc_models.TRANSMISSION_DAY_DIMENSION
    )
    with pytest.raises(ValueError, match="underflows to zero"):
        lp.suggested_initial_values(
            "lambda_tilde",
            k=0.18,
            sampled_scale=tost_sum[untouched],
            parameterisation=lp.CENTRED_GEOMETRIC_INIT,
        )

    # After marginalisation every surviving shape is representable, and the start is finite.
    marginalised = pymc_models.model_days(
        _real_sse_so(real_series, "marginalised"), pymc_models.TRANSMISSION_DAY_DIMENSION
    )
    values = lp.suggested_initial_values(
        "lambda_tilde",
        k=0.18,
        sampled_scale=tost_sum[marginalised],
        parameterisation=lp.CENTRED_GEOMETRIC_INIT,
    )["lambda_tilde"]
    assert (values > 0).all() and np.isfinite(values).all()


def test_the_underflow_threshold_is_where_the_arithmetic_says_it_is():
    # log(tiny) is about -708, and E[log Y] ~ -1/alpha, so the wall sits near alpha = 1.4e-3.
    assert pytest.approx(-708.4, abs=0.5) == lp.LOG_UNDERFLOW
    lp.suggested_initial_values(
        "Y", k=1.0, sampled_scale=np.array([2e-3]), parameterisation=lp.CENTRED_GEOMETRIC_INIT
    )
    with pytest.raises(ValueError, match="underflows to zero"):
        lp.suggested_initial_values(
            "Y", k=1.0, sampled_scale=np.array([1e-3]), parameterisation=lp.CENTRED_GEOMETRIC_INIT
        )


@pytest.mark.parametrize("parameterisation", ["centred", "unit_mean", "inverse_cdf"])
def test_strategies_without_an_explicit_start_supply_no_initial_values(parameterisation):
    assert (
        lp.suggested_initial_values(
            "Y",
            k=0.18,
            sampled_scale=np.ones(2),
            parameterisation=lp.parameterisation_of(parameterisation),
        )
        == {}
    )

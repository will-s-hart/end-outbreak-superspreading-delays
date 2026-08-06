"""Tests for the infection-anchored simulators, and their agreement with the likelihoods.

The Stage-2 deliverable is that the simulators and the PyMC builders describe the same
process. Two routes are used, because the models differ in what "the likelihood" means:

*Closed-form models (DLO, SSE, Cori).*
    Enumerate every short history on a small grid, evaluate the model's likelihood for each,
    and compare with the frequency the simulator produces. Direct and exact up to Monte-Carlo
    error.

*SSI.*
    The likelihood the simulator has to match is the **marginal** of ``I_{1:T}``, whereas the
    builder supplies the joint of ``(I, Y)``. They are bridged by the identity that makes the
    naive Monte-Carlo marginalisation valid: in

        p(I, Y) = Π_u Gamma(Y_u; k I_u, k) · Π_t Poisson(I_t; R_t Σ_s w_s Y_{t-s}),

    the infectivity priors are conditionally independent given ``I``, so

        p(I) = E_{Y ~ Π Gamma(k I_u, k)} [ Π_t Poisson(...) ].

    :func:`_poisson_factor` evaluates that inner product for a whole stack of ``Y`` draws at
    once, and :func:`test_the_poisson_factor_matches_the_pymc_joint_density` pins it against
    the built model before it is used, so the fast path is not trusted on its own.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest
import scipy.stats

from end_of_outbreak import forward_simulation, pymc_models, renewal
from end_of_outbreak.model_specifications import ModelSpecification, TransmissionParameters

# A three-day window with an R switch inside it. Short enough to enumerate, long enough that
# day 2 is driven by two different cohorts at two different lags.
SERIAL_INTERVAL = np.array([0.6, 0.4])
N_DAYS = 3
SWITCH_DAY = 2
SEED_COUNTS = np.array([1])
PARAMETERS = TransmissionParameters(R_pre=1.2, R_post=0.5, k=0.7)

ENUMERATED_COUNTS = range(4)
N_REPLICATES = 200_000
N_MARGINALISATION_DRAWS = 400_000


def _simulate(model, parameters=PARAMETERS, *, seed=20260805, **overrides):
    settings: dict[str, Any] = {
        "serial_interval": SERIAL_INTERVAL,
        "n_days": N_DAYS,
        "switch_day": SWITCH_DAY,
        "initial_counts": SEED_COUNTS,
        "n_replicates": N_REPLICATES,
    }
    settings.update(overrides)
    return forward_simulation.simulate_naive(
        model, parameters, rng=np.random.default_rng(seed), **settings
    )


def _histories():
    """Every history on the enumeration grid, as ``(counts, empirical-frequency mask)``."""
    return [
        np.concatenate((SEED_COUNTS, np.array(tail, dtype=np.int64)))
        for tail in itertools.product(ENUMERATED_COUNTS, repeat=N_DAYS - SEED_COUNTS.size)
    ]


def _empirical_probability(simulated, history):
    tail = simulated.counts[:, SEED_COUNTS.size :]
    return float((tail == history[SEED_COUNTS.size :]).all(axis=1).mean())


def _model_probability(model, history, parameters=PARAMETERS):
    built = pymc_models.build_naive_model(
        model,
        history,
        serial_interval=SERIAL_INTERVAL,
        switch_day=SWITCH_DAY,
        R_pre=parameters.R_pre,
        R_post=parameters.R_post,
        k=parameters.k,
    )
    return float(np.exp(pymc_models.joint_logp(built)))


# --- closed-form models: enumerate and compare -------------------------------------------


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        ("dlo", PARAMETERS),
        ("sse", PARAMETERS),
        ("cori", TransmissionParameters(R_pre=PARAMETERS.R_pre, R_post=PARAMETERS.R_post)),
    ],
)
def test_simulation_matches_the_likelihood_on_short_histories(model, parameters):
    simulated = _simulate(model, parameters)
    enumerated_mass = 0.0
    for history in _histories():
        expected = _model_probability(model, history, parameters)
        observed = _empirical_probability(simulated, history)
        standard_error = np.sqrt(expected * (1.0 - expected) / N_REPLICATES)
        assert observed == pytest.approx(expected, abs=5.0 * standard_error + 2e-4), (
            f"history {history.tolist()}: simulated {observed:.5f}, likelihood {expected:.5f}"
        )
        enumerated_mass += expected

    # The grid is not the whole sample space; whatever mass it does cover must also match.
    covered = float(
        np.isin(simulated.counts[:, SEED_COUNTS.size :], list(ENUMERATED_COUNTS)).all(axis=1).mean()
    )
    assert covered == pytest.approx(enumerated_mass, abs=5e-3)
    assert enumerated_mass > 0.7  # the grid has to be worth enumerating


# --- SSI: marginalise the latent block ----------------------------------------------------


def _cohort_days(history):
    return np.flatnonzero(np.asarray(history) > 0).astype(np.int64)


def _poisson_factor(Y, history, parameters=PARAMETERS):
    """``Π_t Poisson(I_t; R_t Σ_s w_s Y_{t-s})`` for a ``(n_draws, n_cohort)`` stack of ``Y``."""
    history = np.asarray(history, dtype=np.int64)
    cohort_days = _cohort_days(history)
    design = renewal.delay_design_matrix(
        history.size, SERIAL_INTERVAL, first_lag=1, source_days=cohort_days
    )
    days = renewal.likelihood_days(design.sum(axis=1), history)
    R = renewal.reproduction_number_by_day(
        parameters.R_pre, parameters.R_post, n_days=history.size, switch_day=SWITCH_DAY
    )
    mean_incidence = R[days] * (np.atleast_2d(Y) @ design[days].T)
    return np.exp(scipy.stats.poisson.logpmf(history[days], mu=mean_incidence).sum(axis=1))


def _ssi_marginal_probability(history, rng, parameters=PARAMETERS):
    """Naive Monte-Carlo estimate of ``p(I_{1:T})``, with its standard error."""
    history = np.asarray(history, dtype=np.int64)
    cohort_days = _cohort_days(history)
    k = parameters.k
    assert k is not None
    Y = rng.gamma(
        k * history[cohort_days],
        1.0 / k,
        size=(N_MARGINALISATION_DRAWS, cohort_days.size),
    )
    factor = _poisson_factor(Y, history, parameters)
    return float(factor.mean()), float(factor.std(ddof=1) / np.sqrt(factor.size))


def test_the_poisson_factor_matches_the_pymc_joint_density():
    """Pin the fast marginalisation path against the built model before relying on it."""
    rng = np.random.default_rng(11)
    k = PARAMETERS.k
    assert k is not None
    for history in ([1, 2, 1], [1, 0, 3], [1, 1, 0]):
        history = np.asarray(history, dtype=np.int64)
        cohort_days = _cohort_days(history)
        built = pymc_models.build_naive_model(
            "ssi",
            history,
            serial_interval=SERIAL_INTERVAL,
            switch_day=SWITCH_DAY,
            R_pre=PARAMETERS.R_pre,
            R_post=PARAMETERS.R_post,
            k=k,
            latent_parameterisation="centred",
        )
        logp = pymc_models.compile_joint_logp(built)
        for _ in range(5):
            Y = rng.gamma(k * history[cohort_days], 1.0 / k)
            prior = scipy.stats.gamma.logpdf(Y, a=k * history[cohort_days], scale=1.0 / k).sum()
            assert float(np.exp(logp(Y=Y) - prior)) == pytest.approx(
                float(_poisson_factor(Y, history)[0])
            )


def test_ssi_simulation_matches_its_marginal_likelihood():
    simulated = _simulate("ssi")
    rng = np.random.default_rng(7)
    enumerated_mass = 0.0
    for history in _histories():
        expected, marginalisation_error = _ssi_marginal_probability(history, rng)
        observed = _empirical_probability(simulated, history)
        simulation_error = np.sqrt(expected * (1.0 - expected) / N_REPLICATES)
        tolerance = 5.0 * (simulation_error + marginalisation_error) + 2e-4
        assert observed == pytest.approx(expected, abs=tolerance), (
            f"history {history.tolist()}: simulated {observed:.5f}, marginal {expected:.5f}"
        )
        enumerated_mass += expected
    assert enumerated_mass > 0.7


def test_ssi_and_sse_share_a_mean_but_not_a_one_step_distribution():
    """The two mechanisms are calibrated to the same offspring mean, and still differ.

    From a cohort of ``n`` cases and a one-day-ahead weight ``w_1``, SSE pools the infectors
    into ``NB(mean = R w_1 n, disp = k w_1 n)`` while SSI's latent infectivity gives
    ``NB(mean = R w_1 n, disp = k n)``. Same mean, different dispersion — the sharpest
    available statement that mechanism matters, and a strong check on both simulators.
    """
    cohort = 3
    w_1 = SERIAL_INTERVAL[0]
    k = PARAMETERS.k
    assert k is not None
    mean = PARAMETERS.R_pre * w_1 * cohort

    for model, dispersion in (("sse", k * w_1 * cohort), ("ssi", k * cohort)):
        simulated = _simulate(model, n_days=2, switch_day=2, initial_counts=np.array([cohort]))
        counts = simulated.counts[:, 1]
        expected = scipy.stats.nbinom.pmf(
            np.arange(9), n=dispersion, p=dispersion / (dispersion + mean)
        )
        observed = np.bincount(counts, minlength=9)[:9] / N_REPLICATES
        standard_error = np.sqrt(expected * (1.0 - expected) / N_REPLICATES)
        for count, (seen, target, error) in enumerate(
            zip(observed, expected, standard_error, strict=True)
        ):
            assert seen == pytest.approx(target, abs=5.0 * error + 2e-4), (
                f"{model}: P(I_1 = {count}) simulated {seen:.5f}, analytic {target:.5f}"
            )


# --- the k → ∞ limit ----------------------------------------------------------------------


@pytest.mark.parametrize("model", ["dlo", "sse", "ssi"])
def test_large_k_collapses_the_simulators_onto_cori(model):
    huge_k = TransmissionParameters(R_pre=PARAMETERS.R_pre, R_post=PARAMETERS.R_post, k=1e7)
    overdispersed = _simulate(model, huge_k, n_days=6, seed=3)
    poisson = _simulate(
        "cori",
        TransmissionParameters(R_pre=PARAMETERS.R_pre, R_post=PARAMETERS.R_post),
        n_days=6,
        seed=3,
    )
    np.testing.assert_allclose(
        overdispersed.counts.mean(axis=0), poisson.counts.mean(axis=0), atol=0.02
    )
    np.testing.assert_allclose(
        overdispersed.counts.var(axis=0), poisson.counts.var(axis=0), atol=0.05
    )


def test_small_k_is_far_more_overdispersed_than_the_poisson_limit():
    # The limit above is a limit, not an identity: at k = 0.18 the variance is much larger.
    overdispersed = _simulate(
        "sse",
        TransmissionParameters(R_pre=PARAMETERS.R_pre, R_post=PARAMETERS.R_post, k=0.18),
        n_days=6,
        seed=3,
    )
    poisson = _simulate(
        "cori",
        TransmissionParameters(R_pre=PARAMETERS.R_pre, R_post=PARAMETERS.R_post),
        n_days=6,
        seed=3,
    )
    assert overdispersed.total_cases.var() > 3.0 * poisson.total_cases.var()


# --- the SSI latent block ------------------------------------------------------------------


def test_ssi_infectivity_is_positive_exactly_where_there_are_cases():
    simulated = _simulate("ssi", n_days=6)
    assert simulated.infectivity is not None
    np.testing.assert_array_equal(simulated.infectivity > 0.0, simulated.counts > 0)


def test_ssi_infectivity_has_mean_one_per_case():
    cohort = 4
    k = PARAMETERS.k
    assert k is not None
    simulated = _simulate("ssi", n_days=2, initial_counts=np.array([cohort]))
    assert simulated.infectivity is not None
    seeded = simulated.infectivity[:, 0]
    assert seeded.mean() == pytest.approx(cohort, rel=0.02)
    assert seeded.var() == pytest.approx(cohort / k, rel=0.05)


def test_a_supplied_seed_infectivity_is_used_as_given():
    """The RAC checks of §6.4 condition on a *posterior* latent path, not a redrawn prior one."""
    supplied = np.array([3.5])
    simulated = _simulate(
        "ssi", n_days=2, initial_counts=np.array([1]), initial_infectivity=supplied
    )
    assert simulated.infectivity is not None
    np.testing.assert_array_equal(simulated.infectivity[:, 0], 3.5)
    # Day 1 is then Poisson(R_pre · w_1 · 3.5) exactly, which the counts must reflect.
    expected = PARAMETERS.R_pre * SERIAL_INTERVAL[0] * supplied[0]
    assert simulated.counts[:, 1].mean() == pytest.approx(expected, rel=0.02)


def test_a_seed_infectivity_may_differ_between_replicates():
    """One row per replicate is how a whole posterior block of paths is pushed through at once."""
    supplied = np.linspace(0.5, 2.5, N_REPLICATES)[:, None]
    simulated = _simulate(
        "ssi", n_days=2, initial_counts=np.array([1]), initial_infectivity=supplied
    )
    assert simulated.infectivity is not None
    np.testing.assert_allclose(simulated.infectivity[:, 0], supplied[:, 0])


def test_a_seed_infectivity_must_vanish_where_there_are_no_cases():
    with pytest.raises(ValueError, match="vanish on days with no cases"):
        _simulate(
            "ssi",
            n_days=3,
            initial_counts=np.array([1, 0]),
            initial_infectivity=np.array([1.0, 0.5]),
            n_replicates=5,
        )


def test_only_the_latent_models_take_a_seed_infectivity():
    with pytest.raises(ValueError, match="no latent infectivity"):
        _simulate("sse", n_days=2, initial_infectivity=np.array([1.0]), n_replicates=5)


def test_the_other_models_carry_no_infectivity():
    for model in ("dlo", "sse", "cori"):
        parameters = (
            TransmissionParameters(R_pre=1.0, R_post=1.0)
            if model == "cori"
            else TransmissionParameters(R_pre=1.0, R_post=1.0, k=0.5)
        )
        assert _simulate(model, parameters, n_replicates=10).infectivity is None


# --- structural behaviour -------------------------------------------------------------------


def test_the_switch_day_changes_R_on_the_day_itself():
    # A one-day serial interval makes the expected counts read straight off the parameters:
    # day 1 uses R_pre, day 2 uses R_post.
    parameters = TransmissionParameters(R_pre=2.0, R_post=0.25)
    simulated = _simulate(
        "cori",
        parameters,
        serial_interval=np.array([1.0]),
        n_days=3,
        switch_day=2,
        initial_counts=np.array([10]),
    )
    assert simulated.counts[:, 1].mean() == pytest.approx(20.0, rel=0.01)
    assert simulated.counts[:, 2].mean() == pytest.approx(5.0, rel=0.02)


def test_extinction_is_absorbing():
    simulated = _simulate(
        "sse",
        TransmissionParameters(R_pre=0.3, R_post=0.3, k=0.5),
        n_days=12,
        n_replicates=2000,
        seed=5,
    )
    window = SERIAL_INTERVAL.size
    for day in range(window, simulated.n_days):
        extinct = simulated.counts[:, day - window : day].sum(axis=1) == 0
        assert (simulated.counts[extinct, day:] == 0).all()
    assert extinct.any()  # the parameters really do drive some replicates to extinction


def test_the_seeded_prefix_is_returned_unchanged():
    seed = np.array([2, 0, 3])
    simulated = _simulate("sse", n_days=6, initial_counts=seed, n_replicates=50)
    np.testing.assert_array_equal(simulated.counts[:, : seed.size], np.tile(seed, (50, 1)))


def test_simulation_is_reproducible_from_its_generator():
    first = _simulate("ssi", n_replicates=500, seed=42)
    again = _simulate("ssi", n_replicates=500, seed=42)
    different = _simulate("ssi", n_replicates=500, seed=43)
    np.testing.assert_array_equal(first.counts, again.counts)
    assert not np.array_equal(first.counts, different.counts)


def test_reported_shapes_and_totals():
    simulated = _simulate("dlo", n_replicates=17, n_days=5)
    assert simulated.n_replicates == 17
    assert simulated.n_days == 5
    np.testing.assert_array_equal(simulated.total_cases, simulated.counts.sum(axis=1))


# --- validation -----------------------------------------------------------------------------


def test_a_model_needing_k_says_so():
    with pytest.raises(ValueError, match="needs a dispersion parameter"):
        _simulate("ssi", TransmissionParameters(R_pre=1.0, R_post=1.0), n_replicates=5)


def test_the_window_must_hold_the_seeded_prefix():
    with pytest.raises(ValueError, match="shorter than initial_counts"):
        _simulate("sse", n_days=1, initial_counts=np.array([1, 0]), n_replicates=5)


def test_the_prefix_must_start_with_the_index_case():
    with pytest.raises(ValueError, match="at least one case"):
        _simulate("sse", initial_counts=np.array([0, 1]), n_replicates=5)


def test_at_least_one_replicate_is_required():
    with pytest.raises(ValueError, match="n_replicates"):
        _simulate("sse", n_replicates=0)


def test_the_serial_interval_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _simulate("sse", serial_interval=np.array([]), n_replicates=5)


def test_the_onset_anchored_simulators_are_not_here_yet():
    onset_model = ModelSpecification(
        name="ssi_so",
        label="SSI-SO",
        anchoring="onsets",
        overdispersion_level="individual",
        latent_variable="Y",
        description="placeholder for the Stage-8 simulator",
    )
    with pytest.raises(NotImplementedError, match="Stage 8"):
        _simulate(onset_model, n_replicates=5)

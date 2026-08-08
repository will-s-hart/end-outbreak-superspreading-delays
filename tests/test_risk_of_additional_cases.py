"""Tests for the RAC calculators.

Five things need pinning, in order of how much they would cost to get wrong:

1. **The closed forms are the right arithmetic.** Each is checked against forward simulation of
   the same reset state — the equality check of §6.4 — with the conditioning *matched*: for SSI
   the same latent path reaches the analytic calculator and the simulator, so a difference in
   what the two condition on cannot be mistaken for agreement.
2. **The risk is affine in the latent path**, with :func:`latent_risk_basis` as its gradient.
   That is what licenses integrating the unsampled latents out of the risk with the Gamma
   moment generating function instead of drawing them, so it is checked against the closed
   forms by the identity linearity gives: ``β_u = log P(0) − log P(e_u)``.
3. **The reset state is complete.** The default parameterisation integrates out the latents the
   data cannot resolve, and those are exactly the late days a late conditioning day needs. The
   exact correction is checked against averaging the reconstruction, which is unbiased for the
   same quantity, and the conditional it marginalises over is checked directly.
4. **§5.4's inequality.** DLO's risk is bounded by the Poisson limit on one side and by the
   same total ``Λ(t)`` arriving on a single day on the other; that convexity argument is the
   headline result, so it is measured rather than asserted.
5. **The external validation.** Thompson et al.'s eqs. (3)–(5), under their conventions, are a
   one-day shift of this project's machinery at ``k → ∞``. That identity is exact, so it is
   tested exactly, and their published Équateur ``R`` estimate is checked numerically.
"""

from __future__ import annotations

import datetime
from typing import Any

import arviz as az
import numpy as np
import pytest
import scipy.stats

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import fitting, outbreak_data, pymc_models
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.model_specifications import (
    LogNormalPrior,
    TransmissionParameters,
    specification_of,
)

COUNTS = np.array([1, 0, 2, 1, 0, 3, 0, 1])
SERIAL_INTERVAL = np.array([0.5, 0.3, 0.2])
SWITCH_DAY = 4
R_PRE, R_POST, K = 1.3, 0.5, 0.6

SHORT_DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)


@pytest.fixture(scope="module")
def real_series():
    data = outbreak_data.load_onset_data()
    return data, dd.build_onset_anchored_delays()


def _reference_pooled_weight(counts, w, t):
    """``Λ(t) = Σ_{u ≤ t} I_u (1 − F_{t−u})``, written out."""
    F = np.concatenate(([0.0], np.cumsum(w)))
    total = 0.0
    for u in range(t + 1):
        lag = t - u
        survival = 1.0 - (F[lag] if lag < F.size else 1.0)
        total += counts[u] * survival
    return total


# --- Λ(t), the pooled remaining-transmission weight of §5.2 --------------------------------


def test_survival_weights_start_at_one_and_run_out():
    survival = rac.survival_weights(SERIAL_INTERVAL)
    assert survival[0] == pytest.approx(1.0)  # F_0 = 0: the day itself has all of its
    assert survival[-1] == pytest.approx(0.0)  # transmission still to come
    assert np.all(np.diff(survival) <= 0.0)


def test_pooled_remaining_weight_matches_the_definition():
    Lambda = rac.pooled_remaining_weight(COUNTS.astype(float), SERIAL_INTERVAL)
    expected = [_reference_pooled_weight(COUNTS, SERIAL_INTERVAL, t) for t in range(COUNTS.size)]
    np.testing.assert_allclose(Lambda, expected)


def test_the_conditioning_day_enters_with_full_weight():
    """``F_0 = 0``, so a case appearing on day ``t`` counts in full towards Λ(t)."""
    counts = np.array([1, 0, 0, 5])
    Lambda = rac.pooled_remaining_weight(counts.astype(float), SERIAL_INTERVAL)
    assert Lambda[3] == pytest.approx(5.0 + 1.0 * (1.0 - SERIAL_INTERVAL.sum()))


def test_pooled_remaining_weight_carries_a_draw_axis():
    """A whole posterior block of latent paths reduces in one call."""
    rng = np.random.default_rng(0)
    paths = rng.gamma(2.0, 1.0, size=(7, COUNTS.size))
    stacked = rac.pooled_remaining_weight(paths, SERIAL_INTERVAL)
    for row in range(paths.shape[0]):
        np.testing.assert_allclose(
            stacked[row], rac.pooled_remaining_weight(paths[row], SERIAL_INTERVAL)
        )


def test_the_future_force_of_infection_sums_to_the_remaining_weight():
    """``Σ_{j > t} μ_j = Λ(t)``: the same total transmission, differently apportioned.

    This is the identity §5.4 turns on. SSE sees only the total; DLO sees the profile, and
    applies a fresh dispersion to each day of it.
    """
    days = np.arange(COUNTS.size)
    profile = rac.future_force_of_infection(COUNTS, SERIAL_INTERVAL, days=days)
    Lambda = rac.pooled_remaining_weight(COUNTS.astype(float), SERIAL_INTERVAL)
    np.testing.assert_allclose(profile.sum(axis=1), Lambda)


def test_the_future_force_of_infection_discards_the_realised_future():
    """``I_u := 0`` for ``u > t``: the profile is the counterfactual one, not the observed one."""
    t = 3
    profile = rac.future_force_of_infection(COUNTS, SERIAL_INTERVAL, days=np.array([t]))[0]
    retained = np.where(np.arange(COUNTS.size) <= t, COUNTS, 0)
    for lag, mu in enumerate(profile):
        day = t + 1 + lag
        expected = sum(
            SERIAL_INTERVAL[s - 1] * (retained[day - s] if 0 <= day - s < retained.size else 0)
            for s in range(1, SERIAL_INTERVAL.size + 1)
        )
        assert mu == pytest.approx(expected)


# --- the closed forms of §5.3 --------------------------------------------------------------


def test_sse_pools_the_remaining_offspring_into_one_negative_binomial():
    Lambda = rac.pooled_remaining_weight(COUNTS.astype(float), SERIAL_INTERVAL)
    log_probability = rac.log_probability_of_no_further_cases(
        "sse", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=K
    )
    np.testing.assert_allclose(log_probability[0], -K * Lambda * np.log1p(R_PRE / K), rtol=1e-12)


def test_dlo_is_a_product_over_the_future_days():
    days = np.arange(COUNTS.size)
    profile = rac.future_force_of_infection(COUNTS, SERIAL_INTERVAL, days=days)
    log_probability = rac.log_probability_of_no_further_cases(
        "dlo", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=K
    )
    expected = [-K * np.log1p(R_PRE * profile[t] / K).sum() for t in days]
    np.testing.assert_allclose(log_probability[0], expected, rtol=1e-12)


@pytest.mark.parametrize("model", ["dlo", "sse"])
def test_large_k_collapses_the_closed_forms_onto_the_poisson_limit(model):
    poisson = rac.log_probability_of_no_further_cases(
        "cori", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE
    )
    overdispersed = rac.log_probability_of_no_further_cases(
        model, counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=1e8
    )
    np.testing.assert_allclose(overdispersed, poisson, rtol=1e-5)
    at_k = rac.log_probability_of_no_further_cases(
        model, counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=K
    )
    assert (at_k > poisson + 1e-6).any()  # a limit, not an identity


def test_dlo_lies_between_the_poisson_limit_and_a_single_day_of_pooling():
    """§5.4's convexity argument, both bounds.

    ``φ(μ) = −k log(1 + Rμ/k)`` is convex with ``φ(0) = 0``, so it is superadditive and
    ``Σ_j φ(μ_j) ≤ φ(Λ)``; and ``φ(μ) ≥ −Rμ`` pointwise gives the other side. Spreading the
    same total across days therefore moves DLO towards the Poisson limit.
    """
    Lambda = rac.pooled_remaining_weight(COUNTS.astype(float), SERIAL_INTERVAL)
    dlo = rac.log_probability_of_no_further_cases(
        "dlo", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=K
    )[0]
    assert np.all(dlo <= -K * np.log1p(R_PRE * Lambda / K) + 1e-12)
    assert np.all(dlo >= -R_PRE * Lambda - 1e-12)


def test_on_the_equateur_series_dlo_sits_close_to_the_poisson_limit(real_series):
    """The headline of §5.4, measured: ``k = 0.18`` does far less work in DLO than in SSE.

    The future force of infection on this series is thin and spread over the whole serial
    interval, which is the regime in which DLO's day-level dispersion barely bites.
    """
    data, delays = real_series
    common: dict[str, Any] = {
        "counts": data.onsets,
        "serial_interval": delays.serial_interval,
        "R_pre": 2.6,
    }
    sse = rac.risk_curve("sse", k=0.18, **common).risk
    dlo = rac.risk_curve("dlo", k=0.18, **common).risk
    poisson = rac.risk_curve("cori", **common).risk
    assert np.all(sse <= dlo + 1e-9)
    assert np.all(dlo <= poisson + 1e-9)
    late = data.n_days - 21  # day 90, in the decision-relevant stretch
    assert dlo[late] > 3 * sse[late]
    assert dlo[late] == pytest.approx(poisson[late], rel=0.1)


# --- the equality check of §6.4: the closed forms against forward simulation ----------------


@pytest.mark.parametrize("model", ["dlo", "sse", "cori"])
def test_the_closed_forms_match_forward_simulation(model):
    """Reset at every day, simulate the counterfactual future, and count the extinctions."""
    k = None if model == "cori" else K
    analytic = rac.risk_curve(
        model, counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=k
    )
    simulated = rac.simulated_risk_curve(
        model,
        counts=COUNTS,
        serial_interval=SERIAL_INTERVAL,
        parameters=TransmissionParameters(R_pre=R_PRE, R_post=R_POST, k=k),
        n_replicates=40_000,
        rng=np.random.default_rng(1),
    )
    # Three standard errors of a binomial proportion at 40 000 replicates.
    np.testing.assert_allclose(simulated.risk, analytic.risk, atol=0.008)


def test_ssi_matches_forward_simulation_at_a_fixed_latent_path():
    """Matched conditioning: the same ``Y`` goes to the calculator and to the simulator.

    Redrawing the seeded infectivities inside the simulator would answer a different question
    — it would condition on the counts alone rather than on the retained posterior state — and
    the disagreement would look like a bug in the arithmetic.
    """
    rng = np.random.default_rng(2)
    Y = np.zeros(COUNTS.size)
    Y[COUNTS > 0] = rng.gamma(K * COUNTS[COUNTS > 0], 1.0 / K)
    analytic = rac.risk_curve(
        "ssi",
        counts=COUNTS,
        serial_interval=SERIAL_INTERVAL,
        R_pre=R_PRE,
        k=K,
        infectivity=Y[None, :],
    )
    simulated = rac.simulated_risk_curve(
        "ssi",
        counts=COUNTS,
        serial_interval=SERIAL_INTERVAL,
        parameters=TransmissionParameters(R_pre=R_PRE, R_post=R_POST, k=K),
        infectivity=Y,
        n_replicates=40_000,
        rng=rng,
    )
    np.testing.assert_allclose(simulated.risk, analytic.risk, atol=0.008)


def test_the_simulated_horizon_is_long_enough_to_be_exact():
    """Simulating ``len(w)`` days past ``t`` is exact: nothing the retained cases do lands later."""
    short = rac.simulated_risk_curve(
        "cori",
        counts=COUNTS,
        serial_interval=SERIAL_INTERVAL,
        parameters=TransmissionParameters(R_pre=R_PRE, R_post=R_PRE),
        days=np.array([COUNTS.size - 1]),
        n_replicates=20_000,
        rng=np.random.default_rng(3),
    )
    analytic = rac.risk_curve(
        "cori",
        counts=COUNTS,
        serial_interval=SERIAL_INTERVAL,
        R_pre=R_PRE,
        days=np.array([COUNTS.size - 1]),
    )
    assert short.risk[0] == pytest.approx(analytic.risk[0], abs=0.01)


# --- posterior averaging --------------------------------------------------------------------


def test_the_curve_averages_the_probability_not_the_risk():
    """RAC is a posterior probability, so the average happens inside ``1 − ·``."""
    R = np.array([0.5, 1.0, 2.0, 3.0])
    k = np.full(R.size, K)
    log_probability = rac.log_probability_of_no_further_cases(
        "sse", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R, k=k
    )
    curve = rac.risk_curve("sse", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R, k=k)
    np.testing.assert_allclose(curve.risk, 1.0 - np.exp(log_probability).mean(axis=0))
    assert curve.n_draws == R.size


def test_first_day_below_finds_where_the_curve_settles():
    curve = rac.RiskCurve(
        days=np.arange(6), risk=np.array([0.9, 0.2, 0.06, 0.04, 0.03, 0.02]), n_draws=1
    )
    assert curve.first_day_below(0.05) == 3
    assert curve.first_day_below(0.5) == 1
    assert curve.first_day_below(0.01) is None


def test_a_scalar_k_is_broadcast_over_the_draws():
    R = np.array([1.0, 2.0, 3.0])
    per_draw = rac.risk_curve(
        "sse", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R, k=np.full(R.size, K)
    )
    broadcast = rac.risk_curve("sse", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R, k=K)
    np.testing.assert_allclose(broadcast.risk, per_draw.risk)


# --- rebuilding the latents the fit integrated out ------------------------------------------


def _fake_posterior(sampled_days, sampled_values, *, R_pre, R_post, k=None):
    """An ``InferenceData`` shaped like a fit, so the reconstruction can be tested exactly."""
    posterior = {
        "R_pre": R_pre[None, :],
        "R_post": R_post[None, :],
        "Y": sampled_values[None, :, :],
    }
    if k is not None:
        posterior["k"] = k[None, :]
    return az.from_dict(
        {"posterior": posterior},
        coords={pymc_models.COHORT_DAY_DIMENSION: sampled_days},
        dims={"Y": [pymc_models.COHORT_DAY_DIMENSION]},
    )


def test_the_reconstruction_splices_the_sampled_and_rebuilt_blocks_onto_the_calendar():
    structure = pymc_models.latent_block_structure(
        "ssi", COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY
    )
    removed_days, _, _ = pymc_models.marginalised_latent_conditional(
        "ssi",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_PRE,
        R_post=R_POST,
        k=K,
        latent_parameterisation="marginalised_inverse_cdf",
    )
    sampled_days = np.setdiff1d(structure.days, removed_days)
    assert removed_days.size > 0  # the final cohort is uncoupled: no case follows it

    n_draws = 4000
    values = np.tile(np.arange(1.0, sampled_days.size + 1), (n_draws, 1))
    idata = _fake_posterior(
        sampled_days,
        values,
        R_pre=np.full(n_draws, R_PRE),
        R_post=np.full(n_draws, R_POST),
    )
    state = rac.posterior_state(
        "ssi",
        idata,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        latent_parameterisation="marginalised_inverse_cdf",
        fixed_k=K,
    )
    paths = rac.reconstruct_latent_paths("ssi", state, counts=COUNTS, rng=np.random.default_rng(4))
    np.testing.assert_allclose(paths[:, sampled_days], values)
    assert np.all(paths[:, COUNTS == 0] == 0.0)  # no cohort, no infectivity

    # The rebuilt latents follow the conjugate conditional Gamma(k·scale, k + c_u) exactly.
    coupling = structure.coupling(R_PRE, R_POST)[np.isin(structure.days, removed_days)]
    shape = K * structure.scale[np.isin(structure.days, removed_days)]
    rate = K + coupling
    rebuilt = paths[:, removed_days]
    np.testing.assert_allclose(rebuilt.mean(axis=0), shape / rate, rtol=0.05)
    np.testing.assert_allclose(rebuilt.var(axis=0), shape / rate**2, rtol=0.15)


def _log_probabilities_at(model, path, days, *, R_pre, R_post, k, delays):
    """``log P(no case)`` and ``log P(no transmission)`` at an explicit latent path."""
    if specification_of(model).anchoring == "onsets":
        estimate = rac.onset_event_probabilities(
            model,
            counts=COUNTS,
            delays=delays,
            switch_day=SWITCH_DAY,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent=path,
            days=days,
        )
        return estimate.log_no_further_cases, estimate.log_no_further_transmission
    log_probability = rac.log_probability_of_no_further_cases(
        model,
        counts=COUNTS,
        serial_interval=delays.serial_interval,
        R_pre=R_pre,
        k=k,
        infectivity=path,
        days=days,
    )
    return log_probability, log_probability


@pytest.mark.parametrize("model", ["ssi", "sse_so", "ssi_so"])
def test_the_risk_is_affine_in_the_latent_path_with_the_basis_as_its_gradient(model):
    """``log P = −(α + Σ_u β_u Y_u)``, and :func:`latent_risk_basis` is that ``β``.

    Checked by the identity linearity gives for free — ``β_u = log P(0) − log P(e_u)`` — so
    the closed forms remain the source of truth and the basis cannot drift away from them.
    This is what licenses integrating the unsampled latents out with the Gamma moment
    generating function rather than drawing them.
    """
    days = np.array([2, 4, COUNTS.size - 1])
    R_pre, R_post, k = np.array([R_PRE]), np.array([R_POST]), np.array([K])
    zero = np.zeros((1, COUNTS.size))
    arguments = {"R_pre": R_pre, "R_post": R_post, "k": k, "delays": SHORT_DELAYS}
    base_cases, base_transmission = _log_probabilities_at(model, zero, days, **arguments)

    basis = rac.latent_risk_basis(
        model, counts=COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY, days=days
    )
    per_case_exponent = k * np.log1p(R_pre / k)
    beta_cases = basis.weights(R_reset=R_pre, R_pre=R_pre, R_post=R_post)
    beta_transmission = basis.weights(
        R_reset=R_pre, R_pre=R_pre, R_post=R_post, per_case_exponent=per_case_exponent
    )
    for latent_day in range(COUNTS.size):
        unit = np.zeros((1, COUNTS.size))
        unit[0, latent_day] = 1.0
        cases, transmission = _log_probabilities_at(model, unit, days, **arguments)
        np.testing.assert_allclose(base_cases - cases, beta_cases[:, :, latent_day], atol=1e-12)
        np.testing.assert_allclose(
            base_transmission - transmission, beta_transmission[:, :, latent_day], atol=1e-12
        )


@pytest.mark.parametrize("model", ["ssi", "sse_so"])
def test_integrating_the_unsampled_latents_out_matches_drawing_them(model):
    """The exact correction is what averaging the reconstruction converges to.

    Drawing the removed latents and averaging ``exp(log P)`` is unbiased for the same quantity,
    so with enough replicates the two must meet — and the closed form gets there with no
    Monte-Carlo error of its own.
    """
    idata = fitting.fit_model(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025),
        R_post=LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025),
        k=K,
        latent_parameterisation="marginalised_inverse_cdf",
        sampler=fitting.SamplerSettings(draws=500, tune=500, chains=2, seed=13),
    )
    state = rac.posterior_state(
        model,
        idata,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        latent_parameterisation="marginalised_inverse_cdf",
        fixed_k=K,
    )
    assert state.unsampled is not None and state.unsampled.days.size > 0

    days = np.arange(1, COUNTS.size)
    exact = rac.risk_log_probabilities(
        model, state, counts=COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY, days=days
    ).risk_of_additional_cases()

    rng = np.random.default_rng(2)
    probabilities = []
    for _ in range(40):
        path = rac.reconstruct_latent_paths(model, state, counts=COUNTS, rng=rng)
        drawn, _ = _log_probabilities_at(
            model,
            path,
            days,
            R_pre=state.R_pre,
            R_post=state.R_post,
            k=state.k,
            delays=SHORT_DELAYS,
        )
        probabilities.append(np.exp(drawn).mean(axis=0))
    drawn_risk = 1.0 - np.mean(probabilities, axis=0)
    np.testing.assert_allclose(exact.risk, drawn_risk, atol=3e-3)


def test_the_conditional_broadcasts_over_draws_exactly_as_it_does_one_at_a_time():
    """The vectorised reconstruction is the same arithmetic as a loop, draw by draw."""
    R_pre = np.array([0.8, 1.3, 2.1])
    R_post = np.array([0.4, 0.5, 0.9])
    k = np.array([0.2, 0.6, 1.1])
    days, shape, rate = pymc_models.marginalised_latent_conditional(
        "ssi",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=R_pre,
        R_post=R_post,
        k=k,
        latent_parameterisation="marginalised",
    )
    assert shape.shape == (R_pre.size, days.size)
    for draw in range(R_pre.size):
        one_day, one_shape, one_rate = pymc_models.marginalised_latent_conditional(
            "ssi",
            COUNTS,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            R_pre=float(R_pre[draw]),
            R_post=float(R_post[draw]),
            k=float(k[draw]),
            latent_parameterisation="marginalised",
        )
        np.testing.assert_array_equal(one_day, days)
        np.testing.assert_allclose(shape[draw], one_shape)
        np.testing.assert_allclose(rate[draw], one_rate)


def test_a_fit_carries_everything_the_curve_needs():
    """End to end on a short history: fit, rebuild the state, and read off the curve."""
    prior = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)
    idata = fitting.fit_model(
        "ssi",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=prior,
        R_post=prior,
        k=K,
        latent_parameterisation="marginalised_inverse_cdf",
        sampler=fitting.SamplerSettings(draws=300, tune=300, chains=2, seed=5),
    )
    assert fitting.fitted_parameterisation(idata) == "marginalised_inverse_cdf"
    assert fitting.fitted_dispersion(idata) == pytest.approx(K)

    state = rac.posterior_state(
        "ssi",
        idata,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        latent_parameterisation="marginalised_inverse_cdf",
        fixed_k=fitting.fitted_dispersion(idata),
    )
    assert state.sampled_infectivity is not None
    assert state.sampled_infectivity.shape == (state.n_draws, COUNTS.size)
    # Every day with a case carries a latent, whether the sampler saw it or not.
    reconstructed = rac.reconstruct_latent_paths(
        "ssi", state, counts=COUNTS, rng=np.random.default_rng(6)
    )
    assert np.all(reconstructed[:, COUNTS > 0] > 0.0)

    curve = rac.risk_log_probabilities(
        "ssi", state, counts=COUNTS, delays=SHORT_DELAYS, switch_day=SWITCH_DAY
    ).risk_of_additional_cases()
    assert curve.days.size == COUNTS.size
    assert np.all((curve.risk >= 0.0) & (curve.risk <= 1.0))


# --- external validation: Thompson et al. (2024) --------------------------------------------


def test_their_reproduction_number_posterior_is_the_conjugate_update(real_series):
    data, delays = real_series
    shape, _ = rac.thompson_reproduction_number_posterior(
        data.onsets, delays.serial_interval, last_day=data.ert_arrival_day - 1
    )
    pre_ert = data.onsets[1 : data.ert_arrival_day]
    assert shape == pytest.approx(1.0 + pre_ert.sum())
    assert shape == pytest.approx(28.0)  # 27 cases after the index case, days 1–32


def test_their_equateur_estimate_matches_the_published_figure(real_series):
    """Fig. S3D of ``thompson_supp.pdf``, dashed line: a peak near 2.5 at a density near 0.8."""
    data, delays = real_series
    shape, rate = rac.thompson_reproduction_number_posterior(
        data.onsets, delays.serial_interval, last_day=data.ert_arrival_day - 1
    )
    posterior = scipy.stats.gamma(a=shape, scale=1.0 / rate)
    mode = (shape - 1.0) / rate
    assert mode == pytest.approx(2.5, abs=0.15)
    assert posterior.pdf(mode) == pytest.approx(0.8, abs=0.1)


def test_their_risk_is_this_projects_risk_shifted_by_one_day(real_series):
    """The conventions differ by exactly ``γ(t) = Λ(t − 1)`` — so the pipelines must agree.

    Theirs: probability of cases **on or after** day ``t``, given data strictly **before**
    day ``t``. Ours: probability of cases **after** day ``t``, given data **through** day
    ``t``. Under the Poisson limit and their Gamma posterior for ``R``, the two are the same
    number one day apart, and the Gamma integral of their eq. (5) is the analytic form of this
    project's posterior average.
    """
    data, delays = real_series
    counts, w = data.onsets, delays.serial_interval
    shape, rate = rac.thompson_reproduction_number_posterior(
        counts, w, last_day=data.ert_arrival_day - 1
    )
    theirs = rac.thompson_withdrawal_risk(counts, w, shape=shape, rate=rate)

    # theirs[j] is their day j + 1, which is our day j: the whole difference is one shift.
    Lambda = rac.pooled_remaining_weight(counts.astype(float), w)
    ours_analytic = 1.0 - (rate / (rate + Lambda)) ** shape
    np.testing.assert_allclose(ours_analytic[:-1], theirs, rtol=1e-12)

    # And the Monte-Carlo posterior average the pipeline actually computes agrees with it.
    rng = np.random.default_rng(7)
    R = rng.gamma(shape, 1.0 / rate, size=200_000)
    ours = rac.risk_curve("cori", counts=counts, serial_interval=w, R_pre=R)
    np.testing.assert_allclose(ours.risk, ours_analytic, atol=2e-3)


def test_the_replicated_curve_stays_high_until_late_june(real_series):
    """Fig. S3E, dashed line: still ~1 in mid-June, falling through July, non-zero at withdrawal."""
    data, delays = real_series
    shape, rate = rac.thompson_reproduction_number_posterior(
        data.onsets, delays.serial_interval, last_day=data.ert_arrival_day - 1
    )
    Lambda = rac.pooled_remaining_weight(data.onsets.astype(float), delays.serial_interval)
    curve = rac.RiskCurve(
        days=data.day_index, risk=1.0 - (rate / (rate + Lambda)) ** shape, n_draws=1
    )
    mid_june = outbreak_data.day_index_of(datetime.date(2018, 6, 17))
    assert curve.risk[mid_june] > 0.98
    assert curve.first_day_below(0.05) is not None
    assert data.date_of(curve.first_day_below(0.05)).month == 7
    assert curve.first_day_below(0.01) is None  # still above 1% when the ERT actually left
    assert 0.02 < curve.risk[-1] < 0.05


# --- validation ------------------------------------------------------------------------------


def test_ssi_will_not_guess_its_latent_paths():
    with pytest.raises(ValueError, match="latent infectivities"):
        rac.risk_curve("ssi", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=K)


def test_the_closed_form_models_refuse_a_latent_path():
    with pytest.raises(ValueError, match="no latent block"):
        rac.risk_curve(
            "sse",
            counts=COUNTS,
            serial_interval=SERIAL_INTERVAL,
            R_pre=R_PRE,
            k=K,
            infectivity=np.ones((1, COUNTS.size)),
        )


def test_the_poisson_limit_takes_no_dispersion():
    with pytest.raises(ValueError, match="k → ∞"):
        rac.risk_curve("cori", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=K)


def test_the_naive_calculator_redirects_onset_models_to_the_pipeline_calculator():
    with pytest.raises(ValueError, match="onset_event_probabilities"):
        rac.risk_curve("sse_so", counts=COUNTS, serial_interval=SERIAL_INTERVAL, R_pre=R_PRE, k=K)


def test_a_fixed_parameter_must_be_supplied_because_it_is_not_in_the_draws():
    idata = _fake_posterior(
        np.array([0]),
        np.ones((2, 1)),
        R_pre=np.ones(2),
        R_post=np.ones(2),
    )
    with pytest.raises(ValueError, match="fixed_k"):
        rac.posterior_state(
            "ssi",
            idata,
            COUNTS,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            latent_parameterisation="inverse_cdf",
        )

"""Bootstrap particle filters for the renewal models, at fixed parameters.

Two jobs, and they are separate (§6.4):

1. **Validate the RAC calculators.** The filter reconstructs the latent state from the data by
   a route that shares no machinery with the PyMC fits, so feeding its state into the
   closed forms of :mod:`end_of_outbreak.risk_of_additional_cases` is an independent check on
   the arithmetic. The conditioning has to match, which is the subtlety recorded below.
2. **Supply the SMC marginal likelihood** ``log p(I_{1:t} | θ)``. It falls out of the forward
   pass for free, and it is what the particle-MCMC check of Stage 4b consumes.

Why these models suit a particle filter unusually well
------------------------------------------------------
The latent transition is driven by *observed* data rather than by the previous latent state:
``Y_t | I_t ~ Gamma(k I_t, k)`` with ``I_t`` observed. So the filter is **fully adapted** —
propagate from the exact conditional, weight by the observation density — and there is no
proposal to tune. DLO, SSE and Cori have no latent state at all, so for them the filter is
exact rather than approximate: its log-evidence is the closed-form likelihood, evaluated by a
completely different code path, which is what makes it a usable regression test on the filter
itself before it is trusted on SSI.

Under incomplete reporting the counts join the state, and adaptation survives because Poisson
thinning splits the day exactly: ``c_t | μ_t ~ Poisson(π_t μ_t)`` independently of
``D_t − c_t | c_t, μ_t ~ Poisson((1 − π_t) μ_t)``. The filter therefore draws the *unreported*
cases and adds them to what was reported, rather than drawing the total and hoping it clears
the data — see :func:`_adapted_counts`. That keeps every model whose count law is Poisson.
DLO and SSE, whose count law is negative binomial because their dispersion is already
marginalised into it, admit the analogous adapted proposal but do not implement it, and refuse
rather than degenerate.

Filtering versus smoothing — do not blur them
---------------------------------------------
The forward pass delivers **filtering** states ``p(Y_{≤t} | I_{1:t})`` as well as, at the end,
draws from the joint **smoothing** law ``p(Y_{0:T} | I_{1:T})``. They answer different questions
and this module exposes both, named for what they are:

``filtering_remaining_weight``
    ``Λ(t)`` under the filtering distribution: a snapshot taken at day ``t`` from the ancestry
    as it stands then. ``Λ(t)`` is the whole of what RAC needs from the retained state for
    SSE/SSI (§5.2), so recording the scalar rather than the path keeps this cheap. This — with
    ``filtering_pipeline_mean`` for the onset models — is what
    :mod:`end_of_outbreak.filtered_risk` consumes.
``latent_paths``
    The ancestral paths at the end of the run. These are the ones to compare against an MCMC fit
    to the whole record — subject to path degeneracy, which ``n_distinct`` measures rather than
    hides.

Which one a check wants follows from what it conditions on. The arithmetic check fixes ``θ`` and
holds the closed forms against an MCMC state conditioned on the whole record, so it uses
``latent_paths``. :mod:`end_of_outbreak.filtered_risk` and PMMH both condition on filtering
states, so they estimate the same thing and their agreement is a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import NDArray

from end_of_outbreak import renewal, reporting
from end_of_outbreak.delay_distributions import TOST_FIRST_LAG, OnsetAnchoredDelays
from end_of_outbreak.model_specifications import (
    ModelSpecification,
    TransmissionParameters,
    specification_of,
)
from end_of_outbreak.risk_of_additional_cases import survival_weights, tost_survival_weights


@dataclass(frozen=True)
class ParticleFilterResult:
    """One forward pass: the marginal likelihood, the states, and how degenerate it got."""

    days: NDArray[np.int64]
    """Days carrying an observation, i.e. every day from 1 on (day 0 is the initial condition)."""

    log_evidence_increments: NDArray[np.float64]
    """``log p(I_t | I_{1:t−1}, θ)`` for each day in :attr:`days`."""

    effective_sample_size: NDArray[np.float64]
    """ESS of the normalised weights after each day's weighting, before any resampling."""

    resampled: NDArray[np.bool_]
    """Whether the particle set was resampled after each day."""

    n_distinct: NDArray[np.int64]
    """Distinct day-``u`` ancestors among the final particles: the path-degeneracy diagnostic.

    One entry per day of the window. It falls towards 1 as ``u`` recedes into the past, and how
    fast is exactly what decides whether the smoothing output is usable and whether the
    particle-MCMC check of §6.6 is feasible on the real series.
    """

    latent_paths: NDArray[np.float64] | None
    """``(n_particles, n_days)`` draws from the **joint smoothing** law; ``None`` if latent-free."""

    filtering_remaining_weight: NDArray[np.float64]
    """``(n_days, n_particles)``: ``Λ(t)`` under the **filtering** law at each day."""

    expected_infection_paths: NDArray[np.float64] | None
    """Smoothed ``E_t`` paths for onset models; ``None`` for infection-anchored models."""

    filtering_pipeline_mean: NDArray[np.float64] | None
    """Filtering incubation-pipeline mean for onset models; otherwise ``None``."""

    n_particles: int

    true_count_paths: NDArray[np.float64] | None = None
    """``(n_particles, n_days)`` smoothed true counts; ``None`` when reporting was complete.

    Under incomplete reporting the counts are themselves latent, so each particle carries its
    own history of them and the filter *draws* each day instead of reading it. Recorded for the
    same reason the fits record ``true_incidence``: it is the quantity the reporting assumption
    is there to recover.
    """

    @property
    def log_evidence(self) -> float:
        """``log p(I_{1:T} | θ)``: unbiased in ``exp``, and exact when the model is latent-free."""
        return float(self.log_evidence_increments.sum())

    @property
    def cumulative_log_evidence(self) -> NDArray[np.float64]:
        """``log p(I_{1:t} | θ)`` for each day in :attr:`days`."""
        return np.cumsum(self.log_evidence_increments)


def systematic_resample(
    weights: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.int64]:
    """Systematic resampling indices for normalised ``weights``.

    Lower variance than multinomial resampling at the same cost, and it keeps every particle
    whose weight exceeds ``1/N``, which matters here because a handful of particles carry most
    of the mass on the days with cases.
    """
    n_particles = weights.size
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard the last bin against accumulated rounding
    return np.searchsorted(cumulative, positions).astype(np.int64)


def filter_naive(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    parameters: TransmissionParameters,
    *,
    serial_interval: NDArray[np.float64],
    switch_day: int,
    n_particles: int = 1000,
    resample_threshold: float = 0.5,
    reporting_model: reporting.ReportingModel | None = None,
    as_of_day: int | None = None,
    rng: np.random.Generator | None = None,
) -> ParticleFilterResult:
    """Run the filter over ``dlo``, ``sse``, ``ssi`` or ``cori`` at fixed parameters.

    Parameters
    ----------
    model, counts, serial_interval, switch_day
        As for the model builders. Day 0 is the initial condition, so the weighting runs over
        days ``1, ..., T``.
    parameters
        The fixed ``(R_pre, R_post, k)`` to condition on.
    n_particles
        Particle count. Irrelevant for the latent-free models under complete reporting, whose
        answer is exact at any count.
    resample_threshold
        Resample when the effective sample size falls below this fraction of ``n_particles``.
        Adaptive resampling leaves more distinct ancestors alive, which is what the smoothing
        output depends on; the evidence estimate is unbiased either way.
    reporting_model, as_of_day
        The reporting assumption, and the day the series was observed on. Under incomplete
        reporting the true counts join the particle state: the day's *unreported* cases are
        drawn from ``Poisson((1 − π_t) μ_t)`` and added to the reported ones, and the particle
        is weighted by ``Poisson(c_t; π_t μ_t)``. The filter stays fully adapted throughout —
        in the counts by that thinning split, and in the latent Gamma, which is still drawn
        from its exact conditional given the day's now-imputed count. ``dlo`` and ``sse`` are
        refused, since their count law is negative binomial rather than Poisson.
    rng
        NumPy generator; a fresh default one is used if omitted.
    """
    specification = specification_of(model)
    if specification.anchoring != "infections":
        raise ValueError(
            f"model {specification.name!r} is onset-anchored; use filter_onset_anchored so "
            "the incubation pipeline is retained as part of the state"
        )
    counts = np.asarray(counts, dtype=np.int64)
    w = np.asarray(serial_interval, dtype=np.float64)
    if counts.ndim != 1 or counts.size < 2 or counts[0] < 1:
        raise ValueError("counts must be a series of at least two days starting with a case")
    if n_particles < 1:
        raise ValueError("n_particles must be at least one")
    if not 0.0 <= resample_threshold <= 1.0:
        raise ValueError("resample_threshold must lie in [0, 1]")
    rng = np.random.default_rng() if rng is None else rng

    n_days = counts.size
    reporting_model = _resolve_reporting(reporting_model)
    complete = reporting_model.is_complete
    probability = reporting_model.probability_by_day(
        n_days, as_of_day=n_days - 1 if as_of_day is None else as_of_day
    )
    if not complete and specification.has_dispersion and not specification.has_latents:
        # dlo and sse: the dispersion is marginalised into the count law, so the day's counts
        # are negative binomial and the Poisson thinning split does not apply. An adapted
        # proposal does exist — thinning a negative binomial leaves one — but nothing needs it,
        # and proposing from the prior instead would degenerate on any day carrying cases.
        raise NotImplementedError(
            f"model {specification.name!r} has a negative-binomial count law, so the filter's "
            "adapted Poisson proposal for the unreported cases does not apply. Use the fitted "
            "route ('refit_daily') for it under incomplete reporting"
        )
    k = parameters.require_k(specification) if specification.has_dispersion else None
    R_by_day = renewal.reproduction_number_by_day(
        parameters.R_pre, parameters.R_post, n_days=n_days, switch_day=switch_day
    )
    survival = np.zeros(n_days, dtype=np.float64)
    reach = survival_weights(w)[:n_days]
    survival[: reach.size] = reach

    gamma_latent = specification.latent_variable is not None
    # Whether there is *any* state to resample: under incomplete reporting the counts are
    # latent, so even the closed-form models carry a particle history.
    stochastic = gamma_latent or not complete
    driving = np.zeros((n_particles, n_days), dtype=np.float64)
    true_counts = None if complete else np.zeros((n_particles, n_days), dtype=np.float64)
    lineage = np.zeros((n_particles, n_days), dtype=np.int64)
    filtering_weight = np.zeros((n_days, n_particles), dtype=np.float64)

    driving[:, 0] = (
        rng.gamma(k * counts[0], 1.0 / k, size=n_particles)
        if gamma_latent and k is not None
        else float(counts[0])
    )
    if true_counts is not None:
        true_counts[:, 0] = float(counts[0])  # day 0 is the fixed index case
    lineage[:, 0] = np.arange(n_particles, dtype=np.int64)
    filtering_weight[0] = driving[:, 0] * survival[0]

    log_weights = np.full(n_particles, -np.log(n_particles))
    days = np.arange(1, n_days, dtype=np.int64)
    increments = np.zeros(days.size, dtype=np.float64)
    ess = np.zeros(days.size, dtype=np.float64)
    resampled = np.zeros(days.size, dtype=bool)
    day_counts = np.zeros(n_particles, dtype=np.float64)
    count_mean = np.zeros(n_particles, dtype=np.float64)

    for position, day in enumerate(days):
        lags = min(int(day), w.size)
        force_of_infection = driving[:, int(day) - lags : int(day)] @ w[:lags][::-1]
        if complete:
            log_density = _observation_log_density(
                specification,
                count=int(counts[day]),
                force_of_infection=force_of_infection,
                R=float(R_by_day[day]),
                k=k,
            )
        else:
            # Every model that reaches here is Poisson in the counts, so the weight is the
            # thinned Poisson at the reported count and the day's totals are drawn afterwards,
            # from whichever particles survive the resampling.
            count_mean = float(R_by_day[day]) * force_of_infection
            log_density = _poisson_log_density(
                int(counts[day]), count_mean * float(probability[day])
            )

        increment = float(scipy.special.logsumexp(log_weights + log_density))
        if not np.isfinite(increment):
            raise ValueError(
                f"every particle assigns zero probability to the {counts[day]} case(s) on day "
                f"{int(day)}; the parameters or the seeding cannot produce this series"
            )
        increments[position] = increment
        log_weights = log_weights + log_density - increment

        normalised = np.exp(log_weights)
        ess[position] = 1.0 / float((normalised**2).sum())
        if stochastic and ess[position] < resample_threshold * n_particles:
            indices = systematic_resample(normalised, rng)
            driving = driving[indices]
            lineage = lineage[indices]
            if true_counts is not None:
                true_counts = true_counts[indices]
                count_mean = count_mean[indices]
            log_weights = np.full(n_particles, -np.log(n_particles))
            resampled[position] = True

        if true_counts is not None:
            day_counts = _adapted_counts(int(counts[day]), count_mean, float(probability[day]), rng)
            true_counts[:, day] = day_counts
        # Y_t is conditionally independent of everything else given I_t, so it is drawn after
        # the weighting rather than proposed before it: the filter is fully adapted. Under
        # incomplete reporting I_t is the count this particle just imputed rather than the data.
        if not gamma_latent:
            driving[:, day] = float(counts[day]) if complete else day_counts
        elif complete:
            if counts[day] > 0:
                assert k is not None  # every latent model here carries a dispersion parameter
                driving[:, day] = rng.gamma(k * counts[day], 1.0 / k, size=n_particles)
            # else: Y_t | I_t = 0 is a point mass at zero, which the array already holds.
        else:
            assert k is not None
            driving[:, day] = _gamma_by_particle(rng, shape=k * day_counts, rate=k)
        lineage[:, day] = np.arange(n_particles, dtype=np.int64)
        current_weight = driving[:, : int(day) + 1] @ survival[: int(day) + 1][::-1]
        # Adaptive resampling may leave the continuing particle cloud weighted.  The public
        # filtering snapshots are consumed as equally weighted draws, so resample a *view* of
        # that cloud even when the continuing filter correctly keeps its importance weights.
        # This does not alter the likelihood estimator, the ancestry, or the smoother.
        snapshot_indices = (
            systematic_resample(np.exp(log_weights), rng)
            if stochastic
            else np.arange(n_particles, dtype=np.int64)
        )
        filtering_weight[day] = current_weight[snapshot_indices]

    # A final resample turns the weighted particle set into equally weighted smoothing draws,
    # unless the last step already did it.
    if stochastic and not (resampled.size > 0 and resampled[-1]):
        indices = systematic_resample(np.exp(log_weights), rng)
        driving = driving[indices]
        lineage = lineage[indices]
        if true_counts is not None:
            true_counts = true_counts[indices]

    return ParticleFilterResult(
        days=days,
        log_evidence_increments=increments,
        effective_sample_size=ess,
        resampled=resampled,
        n_distinct=np.array(
            [np.unique(lineage[:, day]).size for day in range(n_days)], dtype=np.int64
        ),
        latent_paths=driving if gamma_latent else None,
        true_count_paths=true_counts,
        filtering_remaining_weight=filtering_weight,
        expected_infection_paths=None,
        filtering_pipeline_mean=None,
        n_particles=n_particles,
    )


def filter_onset_anchored(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    n_particles: int = 1000,
    resample_threshold: float = 0.5,
    reporting_model: reporting.ReportingModel | None = None,
    as_of_day: int | None = None,
    rng: np.random.Generator | None = None,
) -> ParticleFilterResult:
    """Run the onset-process filter for Cori-SO, SSE-SO or SSI-SO.

    On day ``t`` the filter first weights by
    ``D_t ~ Poisson(Σ_a f_inc,a E_{t-a})`` and then draws the day-``t`` latent from the Gamma
    law driven by the now-observed onset. This ordering is valid because incubation has no
    lag-0 mass. The final day's latent is drawn too: it cannot affect the fitted likelihood,
    but it belongs to the reset state's incubation pipeline and is essential for RAC at the
    boundary of the observation window.

    Under incomplete reporting the onsets themselves join the state: the particle is weighted
    by ``Poisson(c_t; π_t μ_t)``, the day's unreported cases are drawn from
    ``Poisson((1 − π_t) μ_t)`` and added to the reported ones, and the day's latent is then
    drawn from the Gamma law that the resulting ``D_t`` implies. The ordering still works,
    because ``μ_t`` depends on the latents strictly before ``t`` while the day's latent scale
    depends on the counts up to and including it.
    """
    specification = specification_of(model)
    if specification.anchoring != "onsets":
        raise ValueError(f"model {specification.name!r} is infection-anchored; use filter_naive")
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size < 2 or counts[0] < 1:
        raise ValueError("counts must be a series of at least two days starting with a case")
    if n_particles < 1:
        raise ValueError("n_particles must be at least one")
    if not 0.0 <= resample_threshold <= 1.0:
        raise ValueError("resample_threshold must lie in [0, 1]")
    rng = np.random.default_rng() if rng is None else rng

    n_days = counts.size
    reporting_model = _resolve_reporting(reporting_model)
    complete = reporting_model.is_complete
    probability = reporting_model.probability_by_day(
        n_days, as_of_day=n_days - 1 if as_of_day is None else as_of_day
    )
    k = parameters.require_k(specification) if specification.has_dispersion else None
    R_by_day = renewal.reproduction_number_by_day(
        parameters.R_pre, parameters.R_post, n_days=n_days, switch_day=switch_day
    )
    observed_tost_sum = renewal.delay_weighted_sum(
        counts.astype(np.float64), delays.tost, first_lag=TOST_FIRST_LAG
    )
    tost_survival = np.zeros(n_days, dtype=np.float64)
    tost_reach = tost_survival_weights(delays.tost)[:n_days]
    tost_survival[: tost_reach.size] = tost_reach
    incubation_survival = np.zeros(n_days, dtype=np.float64)
    incubation_reach = survival_weights(delays.incubation)[:n_days]
    incubation_survival[: incubation_reach.size] = incubation_reach

    gamma_latent = specification.latent_variable is not None
    stochastic = gamma_latent or not complete
    driving = np.zeros((n_particles, n_days), dtype=np.float64)
    expected = np.zeros((n_particles, n_days), dtype=np.float64)
    onsets = np.tile(counts.astype(np.float64), (n_particles, 1))
    lineage = np.zeros((n_particles, n_days), dtype=np.int64)
    filtering_weight = np.zeros((n_days, n_particles), dtype=np.float64)
    filtering_pipeline = np.zeros((n_days, n_particles), dtype=np.float64)
    if not complete:
        onsets[:, 1:] = 0.0  # every day after the fixed index is imputed as the filter runs

    _draw_onset_filter_state(
        specification,
        day=0,
        count=np.full(n_particles, float(counts[0])),
        tost_sum=np.full(n_particles, float(observed_tost_sum[0])),
        driving=driving,
        expected=expected,
        R=float(R_by_day[0]),
        k=k,
        tost=delays.tost,
        rng=rng,
        scalar=complete,
    )
    lineage[:, 0] = np.arange(n_particles, dtype=np.int64)
    filtering_weight[0] = (
        driving[:, 0] * tost_survival[0]
        if specification.name == "ssi_so"
        else onsets[:, 0] * tost_survival[0]
    )
    filtering_pipeline[0] = expected[:, 0] * incubation_survival[0]

    log_weights = np.full(n_particles, -np.log(n_particles))
    days = np.arange(1, n_days, dtype=np.int64)
    increments = np.zeros(days.size, dtype=np.float64)
    ess = np.zeros(days.size, dtype=np.float64)
    resampled = np.zeros(days.size, dtype=bool)
    for position, day_value in enumerate(days):
        day = int(day_value)
        lags = min(day, delays.incubation.size)
        mean_onsets = expected[:, day - lags : day] @ delays.incubation[:lags][::-1]
        if complete:
            log_density = _poisson_log_density(int(counts[day]), mean_onsets)
        else:
            # The onsets are Poisson in every onset-anchored model, so thinning splits the day
            # exactly: weight by the reported part, draw the unreported part after resampling.
            log_density = _poisson_log_density(
                int(counts[day]), mean_onsets * float(probability[day])
            )
        increment = float(scipy.special.logsumexp(log_weights + log_density))
        if not np.isfinite(increment):
            raise ValueError(
                f"every particle assigns zero probability to the {counts[day]} case(s) on "
                f"day {day}; the parameters or the seeding cannot produce this series"
            )
        increments[position] = increment
        log_weights = log_weights + log_density - increment

        normalised = np.exp(log_weights)
        ess[position] = 1.0 / float((normalised**2).sum())
        if stochastic and ess[position] < resample_threshold * n_particles:
            indices = systematic_resample(normalised, rng)
            driving = driving[indices]
            expected = expected[indices]
            lineage = lineage[indices]
            if not complete:
                onsets = onsets[indices]
                mean_onsets = mean_onsets[indices]
            log_weights = np.full(n_particles, -np.log(n_particles))
            resampled[position] = True

        if complete:
            day_tost_sum = np.full(n_particles, float(observed_tost_sum[day]))
        else:
            onsets[:, day] = _adapted_counts(
                int(counts[day]), mean_onsets, float(probability[day]), rng
            )
            lags_tost = min(day + 1, delays.tost.size)
            day_tost_sum = onsets[:, day + 1 - lags_tost : day + 1] @ delays.tost[:lags_tost][::-1]
        _draw_onset_filter_state(
            specification,
            day=day,
            count=onsets[:, day],
            tost_sum=day_tost_sum,
            driving=driving,
            expected=expected,
            R=float(R_by_day[day]),
            k=k,
            tost=delays.tost,
            rng=rng,
            scalar=complete,
        )
        lineage[:, day] = np.arange(n_particles, dtype=np.int64)
        current_weight = (
            driving[:, : day + 1] @ tost_survival[: day + 1][::-1]
            if specification.name == "ssi_so"
            else onsets[:, : day + 1] @ tost_survival[: day + 1][::-1]
        )
        current_pipeline = expected[:, : day + 1] @ incubation_survival[: day + 1][::-1]
        snapshot_indices = (
            systematic_resample(np.exp(log_weights), rng)
            if stochastic
            else np.arange(n_particles, dtype=np.int64)
        )
        filtering_weight[day] = current_weight[snapshot_indices]
        filtering_pipeline[day] = current_pipeline[snapshot_indices]

    if stochastic and not (resampled.size > 0 and resampled[-1]):
        indices = systematic_resample(np.exp(log_weights), rng)
        driving = driving[indices]
        expected = expected[indices]
        lineage = lineage[indices]
        if not complete:
            onsets = onsets[indices]

    return ParticleFilterResult(
        days=days,
        log_evidence_increments=increments,
        effective_sample_size=ess,
        resampled=resampled,
        n_distinct=np.array(
            [np.unique(lineage[:, day]).size for day in range(n_days)], dtype=np.int64
        ),
        latent_paths=driving if gamma_latent else None,
        true_count_paths=None if complete else onsets,
        filtering_remaining_weight=filtering_weight,
        expected_infection_paths=expected,
        filtering_pipeline_mean=filtering_pipeline,
        n_particles=n_particles,
    )


def filter_model(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    n_particles: int = 1000,
    resample_threshold: float = 0.5,
    reporting_model: reporting.ReportingModel | None = None,
    as_of_day: int | None = None,
    rng: np.random.Generator | None = None,
) -> ParticleFilterResult:
    """Dispatch to the filter matching the model's anchoring convention."""
    specification = specification_of(model)
    if specification.anchoring == "onsets":
        return filter_onset_anchored(
            specification,
            counts,
            parameters,
            delays=delays,
            switch_day=switch_day,
            n_particles=n_particles,
            resample_threshold=resample_threshold,
            reporting_model=reporting_model,
            as_of_day=as_of_day,
            rng=rng,
        )
    return filter_naive(
        specification,
        counts,
        parameters,
        serial_interval=delays.serial_interval,
        switch_day=switch_day,
        n_particles=n_particles,
        resample_threshold=resample_threshold,
        reporting_model=reporting_model,
        as_of_day=as_of_day,
        rng=rng,
    )


def _draw_onset_filter_state(
    specification: ModelSpecification,
    *,
    day: int,
    count: NDArray[np.floating[Any]],
    tost_sum: NDArray[np.floating[Any]],
    driving: NDArray[np.float64],
    expected: NDArray[np.float64],
    R: float,
    k: float | None,
    tost: NDArray[np.float64],
    rng: np.random.Generator,
    scalar: bool,
) -> None:
    """Draw the latent attached to the day's onsets and set one column of ``E``.

    ``count`` and ``tost_sum`` are per particle. ``scalar`` says they are known to be constant
    across particles — which is the case whenever the onsets are data — and takes the scalar
    ``rng`` calls the completely reported filter has always made, so that its draws are
    unchanged.
    """
    n_particles = driving.shape[0]
    if specification.name == "cori_so":
        driving[:, day] = count
        expected[:, day] = R * tost_sum
        return
    assert k is not None
    if specification.name == "sse_so":
        if scalar:
            if tost_sum[0] > 0.0:
                driving[:, day] = rng.gamma(k * float(tost_sum[0]), 1.0 / k, size=n_particles)
        else:
            driving[:, day] = _gamma_by_particle(rng, shape=k * tost_sum, rate=k)
        expected[:, day] = R * driving[:, day]
        return
    if scalar:
        if count[0] > 0:
            driving[:, day] = rng.gamma(k * float(count[0]), 1.0 / k, size=n_particles)
    else:
        driving[:, day] = _gamma_by_particle(rng, shape=k * count, rate=k)
    lags = min(day + 1, tost.size)
    expected[:, day] = R * (driving[:, day + 1 - lags : day + 1] @ tost[:lags][::-1])


def _gamma_by_particle(
    rng: np.random.Generator, *, shape: NDArray[np.floating[Any]], rate: float
) -> NDArray[np.float64]:
    """``Gamma(shape, rate)`` per particle, with a zero shape giving its point mass at zero."""
    drawn = np.zeros(shape.shape, dtype=np.float64)
    positive = shape > 0.0
    if positive.any():
        drawn[positive] = rng.gamma(shape[positive], 1.0 / rate)
    return drawn


def _adapted_counts(
    reported: int,
    mean: NDArray[np.float64],
    probability: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw a day's true counts from their exact conditional given the reported count.

    Poisson thinning splits the day into two independent pieces,

    ``c_t | μ_t ~ Poisson(π_t μ_t)``  and  ``D_t − c_t | c_t, μ_t ~ Poisson((1 − π_t) μ_t)``,

    so the reported cases can be *conditioned on* rather than reproduced by luck. Proposing
    ``D_t`` from the model and reweighting by ``Binomial(c_t; D_t, π_t)`` targets the same law,
    but degenerates the moment a day carries more than a case or two: every particle can draw a
    total below what was reported, and then the whole cloud takes zero weight and there is no
    filter left. Drawing the *unreported* cases instead keeps the filter fully adapted in the
    counts, as it already is in the Gamma block.

    The matching importance weight is the first factor, ``Poisson(c_t; π_t μ_t)``. The caller
    evaluates it before resampling and calls this afterwards, so that surviving particles draw
    fresh counts rather than carrying duplicated ones.
    """
    unreported = rng.poisson(np.maximum(mean, 0.0) * (1.0 - probability))
    return reported + unreported.astype(np.float64)


def _resolve_reporting(
    reporting_model: reporting.ReportingModel | None,
) -> reporting.ReportingModel:
    """Complete reporting is the default, so callers need not pass one."""
    return reporting.COMPLETE_REPORTING if reporting_model is None else reporting_model


def _poisson_log_density(count: int, mean: NDArray[np.float64]) -> NDArray[np.float64]:
    """Poisson log density with the degenerate zero-mean case handled explicitly."""
    density = np.zeros(mean.size, dtype=np.float64)
    driven = mean > 0.0
    density[driven] = scipy.stats.poisson.logpmf(count, mean[driven])
    if count > 0:
        density[~driven] = -np.inf
    return density


def _observation_log_density(
    specification: ModelSpecification,
    *,
    count: int,
    force_of_infection: NDArray[np.float64],
    R: float,
    k: float | None,
) -> NDArray[np.float64]:
    """``log p(I_t | state, θ)`` for every particle.

    A particle whose force of infection is zero puts a point mass at zero on the day's count —
    the same degeneracy :func:`end_of_outbreak.renewal.likelihood_days` removes from the PyMC
    graphs, except that here it is a property of the particle rather than of the data, so it
    has to be handled particle by particle.
    """
    density = np.zeros(force_of_infection.size, dtype=np.float64)
    driven = force_of_infection > 0.0
    if not driven.any():
        return np.full(density.shape, -np.inf if count > 0 else 0.0)
    mean_incidence = R * force_of_infection[driven]

    if specification.name in ("cori", "ssi"):
        density[driven] = scipy.stats.poisson.logpmf(count, mean_incidence)
    else:
        assert k is not None  # dlo and sse both carry a dispersion parameter
        dispersion = (
            np.full(mean_incidence.shape, k)
            if specification.name == "dlo"
            else k * force_of_infection[driven]
        )
        density[driven] = scipy.stats.nbinom.logpmf(
            count, dispersion, dispersion / (dispersion + mean_incidence)
        )
    if count > 0:
        density[~driven] = -np.inf
    return density

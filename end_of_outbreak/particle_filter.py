"""Bootstrap particle filter for the infection-anchored models, at fixed parameters.

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

Filtering versus smoothing — do not blur them
---------------------------------------------
The forward pass delivers **filtering** states ``p(Y_{≤t} | I_{1:t})``, whereas RAC resets from
a **smoothed** state ``p(Y_{≤t} | I_{1:T})`` (§5.1, §5.6). Comparing a filtering RAC with a
smoothed RAC and calling agreement a pass would be wrong; the difference between them is the
§5.6 approximation, which §6.6 sets out to *measure*. This module therefore exposes both, named
for what they are:

``filtering_remaining_weight``
    ``Λ(t)`` under the filtering distribution: a snapshot taken at day ``t`` from the ancestry
    as it stands then. ``Λ(t)`` is the whole of what RAC needs from the retained state for
    SSE/SSI (§5.2), so recording the scalar rather than the path keeps this cheap.
``latent_paths``
    The ancestral paths at the end of the run, i.e. draws from the **joint smoothing**
    distribution ``p(Y_{0:T} | I_{1:T})``. These are the ones to compare against an MCMC fit —
    subject to path degeneracy, which ``n_distinct`` measures rather than hides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import NDArray

from end_of_outbreak import renewal
from end_of_outbreak.model_specifications import (
    ModelSpecification,
    TransmissionParameters,
    specification_of,
)
from end_of_outbreak.risk_of_additional_cases import survival_weights


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

    n_particles: int

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
        Particle count. Irrelevant for the latent-free models, whose answer is exact at any
        count.
    resample_threshold
        Resample when the effective sample size falls below this fraction of ``n_particles``.
        Adaptive resampling leaves more distinct ancestors alive, which is what the smoothing
        output depends on; the evidence estimate is unbiased either way.
    rng
        NumPy generator; a fresh default one is used if omitted.
    """
    specification = specification_of(model)
    if specification.anchoring != "infections":
        raise NotImplementedError(
            f"model {specification.name!r} is onset-anchored; its filter carries the incubation "
            "pipeline as part of the state and arrives in Stage 8"
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
    k = parameters.require_k(specification) if specification.has_dispersion else None
    R_by_day = renewal.reproduction_number_by_day(
        parameters.R_pre, parameters.R_post, n_days=n_days, switch_day=switch_day
    )
    survival = np.zeros(n_days, dtype=np.float64)
    reach = survival_weights(w)[:n_days]
    survival[: reach.size] = reach

    latent = specification.latent_variable is not None
    driving = np.zeros((n_particles, n_days), dtype=np.float64)
    lineage = np.zeros((n_particles, n_days), dtype=np.int64)
    filtering_weight = np.zeros((n_days, n_particles), dtype=np.float64)

    driving[:, 0] = (
        rng.gamma(k * counts[0], 1.0 / k, size=n_particles)
        if latent and k is not None
        else float(counts[0])
    )
    lineage[:, 0] = np.arange(n_particles, dtype=np.int64)
    filtering_weight[0] = driving[:, 0] * survival[0]

    log_weights = np.full(n_particles, -np.log(n_particles))
    days = np.arange(1, n_days, dtype=np.int64)
    increments = np.zeros(days.size, dtype=np.float64)
    ess = np.zeros(days.size, dtype=np.float64)
    resampled = np.zeros(days.size, dtype=bool)

    for position, day in enumerate(days):
        lags = min(int(day), w.size)
        force_of_infection = driving[:, int(day) - lags : int(day)] @ w[:lags][::-1]
        log_density = _observation_log_density(
            specification,
            count=int(counts[day]),
            force_of_infection=force_of_infection,
            R=float(R_by_day[day]),
            k=k,
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
        if latent and ess[position] < resample_threshold * n_particles:
            indices = systematic_resample(normalised, rng)
            driving = driving[indices]
            lineage = lineage[indices]
            log_weights = np.full(n_particles, -np.log(n_particles))
            resampled[position] = True

        # Y_t is conditionally independent of everything else given the observed I_t, so it is
        # drawn after the weighting rather than proposed before it: the filter is fully adapted.
        if not latent:
            driving[:, day] = float(counts[day])
        elif counts[day] > 0:
            assert k is not None  # every latent model here carries a dispersion parameter
            driving[:, day] = rng.gamma(k * counts[day], 1.0 / k, size=n_particles)
        # else: Y_t | I_t = 0 is a point mass at zero, which the array already holds.
        lineage[:, day] = np.arange(n_particles, dtype=np.int64)
        filtering_weight[day] = driving[:, : int(day) + 1] @ survival[: int(day) + 1][::-1]

    # A final resample turns the weighted particle set into equally weighted smoothing draws,
    # unless the last step already did it.
    if latent and not (resampled.size > 0 and resampled[-1]):
        indices = systematic_resample(np.exp(log_weights), rng)
        driving = driving[indices]
        lineage = lineage[indices]

    return ParticleFilterResult(
        days=days,
        log_evidence_increments=increments,
        effective_sample_size=ess,
        resampled=resampled,
        n_distinct=np.array(
            [np.unique(lineage[:, day]).size for day in range(n_days)], dtype=np.int64
        ),
        latent_paths=driving if latent else None,
        filtering_remaining_weight=filtering_weight,
        n_particles=n_particles,
    )


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

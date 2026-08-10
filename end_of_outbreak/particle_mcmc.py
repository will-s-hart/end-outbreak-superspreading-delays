"""Particle marginal Metropolis–Hastings over the transmission parameters (§6.6).

This module is deliberately outside every Snakemake dependency list. PMMH is an independent
validation route to the PyMC posterior, not a results path: the report analyses use NUTS.

Its parameter posterior is the whole-record one, and its states are *filtering* states, so its
RAC/RAT curve targets :mod:`end_of_outbreak.filtered_risk` — the comparison estimator, which
conditions the same way — and agreement there is a test. It is not comparable with the
headline curves, which refit per conditioning day.

The Markov chain runs in ``(log R_pre, log R_post, log k)``. A bootstrap particle filter
provides an unbiased likelihood estimate ``L_hat``; retaining that estimate on rejection gives
the exact-approximate PMMH kernel. The code evaluates the ratio in log space for stability but
does not attempt to debias ``log L_hat`` — unbiasedness of ``L_hat``, not of its logarithm, is
the validity argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats
from numpy.typing import NDArray

from end_of_outbreak import particle_filter
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import (
    LogNormalPrior,
    ModelSpecification,
    TransmissionParameters,
    specification_of,
)


@dataclass(frozen=True)
class ParticleMCMCResult:
    """Post-tuning PMMH draws and optional filtering-state draws."""

    R_pre: NDArray[np.float64]
    R_post: NDArray[np.float64]
    k: NDArray[np.float64] | None
    log_evidence: NDArray[np.float64]
    accepted: NDArray[np.bool_]
    proposal_scale: NDArray[np.float64]
    filtering_remaining_weight: NDArray[np.float64] | None = None
    filtering_pipeline_mean: NDArray[np.float64] | None = None

    @property
    def n_draws(self) -> int:
        return int(self.R_pre.size)

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted.mean())


@dataclass(frozen=True)
class LogEvidenceVariance:
    """The PMMH tuning diagnostic at one fixed parameter value."""

    estimates: NDArray[np.float64]

    @property
    def mean(self) -> float:
        return float(self.estimates.mean())

    @property
    def variance(self) -> float:
        return float(self.estimates.var(ddof=1))


def sample(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    R_pre_prior: LogNormalPrior,
    R_post_prior: LogNormalPrior,
    k: float | LogNormalPrior | None,
    n_particles: int,
    draws: int,
    tune: int,
    proposal_scale: float | NDArray[np.float64] = 0.15,
    initial_parameters: TransmissionParameters | None = None,
    target_accept: float = 0.234,
    adapt_interval: int = 50,
    store_filtering_state: bool = False,
    rng: np.random.Generator | None = None,
) -> ParticleMCMCResult:
    """Run a random-walk PMMH chain in unconstrained log coordinates.

    ``k`` is either a prior (estimated), a positive float (fixed), or ``None`` for a Poisson
    limit. During tuning a scalar multiplier on the proposal standard deviations is adapted in
    batches towards ``target_accept``; adaptation stops before any returned draw is collected.
    """
    specification = specification_of(model)
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size < 2 or counts[0] < 1:
        raise ValueError("counts must be a series of at least two days starting with a case")
    if n_particles < 1 or draws < 1 or tune < 0:
        raise ValueError("n_particles and draws must be positive, and tune non-negative")
    if not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must lie strictly between zero and one")
    if adapt_interval < 1:
        raise ValueError("adapt_interval must be positive")

    k_prior = k if isinstance(k, LogNormalPrior) else None
    fixed_k = float(k) if isinstance(k, float | int) else None
    if specification.has_dispersion and k is None:
        raise ValueError(f"model {specification.name!r} needs either a fixed k or a k prior")
    if not specification.has_dispersion and k is not None:
        raise ValueError(f"model {specification.name!r} is a Poisson limit and takes no k")

    priors = [R_pre_prior, R_post_prior] + ([] if k_prior is None else [k_prior])
    dimension = len(priors)
    scales = np.asarray(proposal_scale, dtype=np.float64)
    if scales.ndim == 0:
        scales = np.full(dimension, float(scales))
    if scales.shape != (dimension,) or (scales <= 0.0).any():
        raise ValueError(f"proposal_scale must be positive with shape {(dimension,)}")

    if initial_parameters is None:
        initial_parameters = TransmissionParameters(
            R_pre=R_pre_prior.median,
            R_post=R_post_prior.median,
            k=k_prior.median if k_prior is not None else fixed_k,
        )
    current = _log_coordinates(initial_parameters, estimate_k=k_prior is not None)
    rng = np.random.default_rng() if rng is None else rng
    current_parameters = _parameters_from_log(current, fixed_k=fixed_k)
    current_filter = _filter(
        specification,
        counts,
        current_parameters,
        delays=delays,
        switch_day=switch_day,
        n_particles=n_particles,
        rng=rng,
    )
    current_target = current_filter.log_evidence + _log_prior(current, priors)

    total_iterations = tune + draws
    samples = np.empty((draws, dimension), dtype=np.float64)
    log_evidence = np.empty(draws, dtype=np.float64)
    accepted = np.zeros(draws, dtype=bool)
    filtering_weight = (
        np.empty((draws, counts.size), dtype=np.float64) if store_filtering_state else None
    )
    filtering_pipeline = (
        np.empty((draws, counts.size), dtype=np.float64)
        if store_filtering_state and specification.anchoring == "onsets"
        else None
    )

    multiplier = 1.0
    batch_accepts = 0
    for iteration in range(total_iterations):
        proposed = current + rng.normal(scale=scales * multiplier)
        proposed_parameters = _parameters_from_log(proposed, fixed_k=fixed_k)
        try:
            proposed_filter = _filter(
                specification,
                counts,
                proposed_parameters,
                delays=delays,
                switch_day=switch_day,
                n_particles=n_particles,
                rng=rng,
            )
        except ValueError as error:
            # A finite particle set can assign zero likelihood to a possible but very remote
            # proposal. That is a valid zero likelihood estimate and therefore an ordinary
            # rejection, not a reason to abort the chain. Preserve all other validation errors.
            if "every particle assigns zero probability" not in str(error):
                raise
            proposed_filter = None
        proposed_target = (
            -np.inf
            if proposed_filter is None
            else proposed_filter.log_evidence + _log_prior(proposed, priors)
        )
        move = bool(np.log(rng.random()) < proposed_target - current_target)
        if move:
            assert proposed_filter is not None
            current = proposed
            current_parameters = proposed_parameters
            current_filter = proposed_filter
            current_target = proposed_target
            batch_accepts += 1

        if iteration < tune:
            if (iteration + 1) % adapt_interval == 0:
                rate = batch_accepts / adapt_interval
                step = min(0.5, 1.0 / np.sqrt((iteration + 1) / adapt_interval))
                multiplier *= float(np.exp(step * (rate - target_accept)))
                multiplier = float(np.clip(multiplier, 0.05, 20.0))
                batch_accepts = 0
            continue

        position = iteration - tune
        samples[position] = current
        log_evidence[position] = current_filter.log_evidence
        accepted[position] = move
        if filtering_weight is not None:
            indices = rng.integers(current_filter.n_particles, size=counts.size)
            filtering_weight[position] = current_filter.filtering_remaining_weight[
                np.arange(counts.size), indices
            ]
            if filtering_pipeline is not None:
                assert current_filter.filtering_pipeline_mean is not None
                filtering_pipeline[position] = current_filter.filtering_pipeline_mean[
                    np.arange(counts.size), indices
                ]

    natural = np.exp(samples)
    return ParticleMCMCResult(
        R_pre=natural[:, 0],
        R_post=natural[:, 1],
        k=natural[:, 2]
        if k_prior is not None
        else (None if fixed_k is None else np.full(draws, fixed_k, dtype=np.float64)),
        log_evidence=log_evidence,
        accepted=accepted,
        proposal_scale=scales * multiplier,
        filtering_remaining_weight=filtering_weight,
        filtering_pipeline_mean=filtering_pipeline,
    )


def estimate_log_evidence_variance(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    n_particles: int,
    repeats: int,
    rng: np.random.Generator | None = None,
) -> LogEvidenceVariance:
    """Repeat a fixed-parameter filter pass to measure ``Var(log L_hat)``."""
    if repeats < 2:
        raise ValueError("repeats must be at least two to estimate a variance")
    rng = np.random.default_rng() if rng is None else rng
    estimates = np.array(
        [
            _filter(
                specification_of(model),
                counts,
                parameters,
                delays=delays,
                switch_day=switch_day,
                n_particles=n_particles,
                rng=rng,
            ).log_evidence
            for _ in range(repeats)
        ],
        dtype=np.float64,
    )
    return LogEvidenceVariance(estimates=estimates)


def _filter(
    specification: ModelSpecification,
    counts: NDArray[np.int64],
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    n_particles: int,
    rng: np.random.Generator,
) -> particle_filter.ParticleFilterResult:
    return particle_filter.filter_model(
        specification,
        counts,
        parameters,
        delays=delays,
        switch_day=switch_day,
        n_particles=n_particles,
        rng=rng,
    )


def _log_coordinates(
    parameters: TransmissionParameters, *, estimate_k: bool
) -> NDArray[np.float64]:
    values = [parameters.R_pre, parameters.R_post]
    if estimate_k:
        if parameters.k is None:
            raise ValueError("initial_parameters needs k when k is estimated")
        values.append(parameters.k)
    return np.log(np.asarray(values, dtype=np.float64))


def _parameters_from_log(
    coordinates: NDArray[np.float64], *, fixed_k: float | None
) -> TransmissionParameters:
    values = np.exp(coordinates)
    return TransmissionParameters(
        R_pre=float(values[0]),
        R_post=float(values[1]),
        k=float(values[2]) if values.size == 3 else fixed_k,
    )


def _log_prior(coordinates: NDArray[np.float64], priors: list[LogNormalPrior]) -> float:
    """Prior density in log coordinates; the transformation Jacobian is already absorbed."""
    return float(
        sum(
            scipy.stats.norm.logpdf(value, loc=np.log(prior.median), scale=prior.sigma_log)
            for value, prior in zip(coordinates, priors, strict=True)
        )
    )


pmmh = sample
"""Descriptive alias matching the method name used in the implementation plan."""

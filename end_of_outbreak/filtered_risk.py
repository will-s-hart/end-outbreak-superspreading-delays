"""RAC, RAT and RST from one full-record fit and a filtering state — the fast approximation.

The gold standard refits per conditioning day (:mod:`end_of_outbreak.refit_risk`). This module
keeps the *parameters* from a single fit to the whole record and re-derives only the *latents*
from the data through day ``t``, with a bootstrap particle filter at each parameter draw. It is
what to reach for when prototyping an analysis, and it is the comparison that says how much the
refitting is buying.

It is an approximation in two distinct places, and they should not be conflated:

1. ``R_pre``, ``R_post`` and ``k`` are conditioned on all 111 days, not on the first ``t``.
2. Conditioning the latents on less data than the parameters ignores their posterior
   correlation, so the pair is not the day-``t`` joint posterior of either.

For **DLO and SSE there is no approximation of the second kind at all** — they have no latent
state, so no filter runs and the retained state is exactly the observed counts. Whatever gap
those two models show against the refit curve is entirely (1), which makes them a useful
control when reading the comparison.

Where the filtering state comes from
------------------------------------
:class:`~end_of_outbreak.particle_filter.ParticleFilterResult` already exposes precisely the two
summaries the closed forms need, and names them for what they are:
``filtering_remaining_weight`` is ``Λ(t)`` for SSI and ``W(t)`` for the onset models, and
``filtering_pipeline_mean`` is ``M(t)``. Both are equal-weight snapshots taken at day ``t``, so
averaging ``exp`` over particles is the conditional zero-probability at that draw.

This is also the route the particle-MCMC check targets. PMMH conditions on filtering latents
too, so its curve and this one estimate the *same* quantity and their agreement is a test — the
opposite of the old smoothed pipeline, against which a filtering curve could only ever be a
measurement of the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special
from numpy.typing import NDArray

from end_of_outbreak import branching_process, particle_filter, reporting
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import (
    ModelSpecification,
    TransmissionParameters,
    specification_of,
)

DEFAULT_PARTICLES = 500
DEFAULT_DRAWS = 1000


@dataclass(frozen=True)
class FilterDiagnostics:
    """How healthy the particle clouds were, per conditioning day, across the parameter draws.

    A filtering estimate is only as good as the cloud it is read off, and the cloud degrades
    where the data are most informative. The worst effective sample size over draws is what
    says whether more particles are needed.
    """

    days: NDArray[np.int64]
    min_effective_sample_size: NDArray[np.float64]
    mean_effective_sample_size: NDArray[np.float64]
    resample_fraction: NDArray[np.float64]


@dataclass(frozen=True)
class FilteredRiskResult:
    """The curve and, where a filter ran, the evidence that it was well conditioned."""

    estimate: rac.DailyRiskEstimate
    diagnostics: FilterDiagnostics | None
    """``None`` for DLO and SSE, which have no latent state and so run no filter at all."""


def thin_draws(state: rac.PosteriorState, n_draws: int) -> rac.PosteriorState:
    """Thin a posterior state to about ``n_draws``, evenly and **within** each chain.

    Thinning across the flattened draw axis would break the chain-major layout that
    :func:`~end_of_outbreak.risk_of_additional_cases.monte_carlo_standard_error` needs, so the
    stride is applied inside each chain and the chains are re-stacked in order.
    """
    total = state.n_draws
    if n_draws >= total:
        return state
    if n_draws < state.n_chains:
        raise ValueError(f"cannot thin {total} draws in {state.n_chains} chains to {n_draws}")
    per_chain = total // state.n_chains
    keep_per_chain = max(1, n_draws // state.n_chains)
    within = np.linspace(0, per_chain - 1, keep_per_chain).round().astype(np.int64)
    index = np.concatenate([chain * per_chain + within for chain in range(state.n_chains)])

    def take(values: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
        return None if values is None else values[index]

    unsampled = state.unsampled
    if unsampled is not None:
        unsampled = rac.UnsampledLatents(
            days=unsampled.days, shape=unsampled.shape[index], rate=unsampled.rate[index]
        )
    return rac.PosteriorState(
        R_pre=state.R_pre[index],
        R_post=state.R_post[index],
        k=take(state.k),
        sampled_infectivity=take(state.sampled_infectivity),
        unsampled=unsampled,
        n_chains=state.n_chains,
    )


def risk_by_filtering(
    model: str | ModelSpecification,
    state: rac.PosteriorState,
    *,
    counts: NDArray[np.int64],
    delays: OnsetAnchoredDelays,
    switch_day: int,
    days: NDArray[np.int64] | None = None,
    reset_R: float | None = None,
    n_particles: int = DEFAULT_PARTICLES,
    resample_threshold: float = 0.5,
    reporting_model: reporting.ReportingModel | None = None,
    as_of_day: int | None = None,
    rng: np.random.Generator | None = None,
) -> FilteredRiskResult:
    """RAC and RAT with the parameters from ``state`` and the latents filtered to each day.

    Parameters
    ----------
    state
        Draws from a fit to the **whole** record; thin it with :func:`thin_draws` first if the
        filter is to be run at fewer than every draw.
    days
        Conditioning days; every day of the window by default.
    n_particles
        Particles per parameter draw. The filter is fully adapted, so the latent proposal is
        exact and a few hundred particles suffice; the residual error is averaged over draws.
    reporting_model, as_of_day
        The reporting assumption the fit was run under. Under incomplete reporting the counts
        are latent too, so even the closed-form models acquire a state to filter.
    """
    specification = specification_of(model)
    counts = np.asarray(counts, dtype=np.int64)
    selected = (
        np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days, np.int64)
    )
    resolved_reporting = (
        reporting.COMPLETE_REPORTING if reporting_model is None else reporting_model
    )

    if not specification.has_latents and resolved_reporting.is_complete:
        # No latent state to filter: the retained state is the observed counts, so this is the
        # closed form at the full-record parameter draws and no particles are involved.
        if reset_R is not None:
            raise ValueError(f"model {specification.name!r} takes no reset_R")
        return FilteredRiskResult(
            estimate=rac.risk_log_probabilities(
                specification,
                state,
                counts=counts,
                delays=delays,
                switch_day=switch_day,
                days=selected,
            ),
            diagnostics=None,
        )
    if specification.name == "dlo" and not resolved_reporting.is_complete:
        raise NotImplementedError(
            "DLO under incomplete reporting has no filtered route: its closed form needs the "
            "whole future force-of-infection profile of each particle's imputed counts, not "
            "the scalar remaining weight the filter records. Use method 'refit_daily'"
        )

    rng = np.random.default_rng() if rng is None else rng
    n_draws = state.n_draws
    reset = state.R_pre if reset_R is None else np.full(n_draws, float(reset_R))
    dispersion = state.k
    if dispersion is None:  # cori / cori_so, reachable only under incomplete reporting
        per_case_exponent = reset
        extinction = branching_process.poisson_extinction_probability(reset)
    else:
        per_case_exponent = dispersion * np.log1p(reset / dispersion)
        extinction = branching_process.negative_binomial_extinction_probability(reset, dispersion)

    log_no_cases = np.empty((n_draws, selected.size), dtype=np.float64)
    log_no_transmission = np.empty((n_draws, selected.size), dtype=np.float64)
    log_no_sustained = np.empty((n_draws, selected.size), dtype=np.float64)
    effective_sample_size = np.empty((n_draws, selected.size), dtype=np.float64)
    resampled = np.zeros((n_draws, selected.size), dtype=np.float64)
    for draw in range(n_draws):
        result = particle_filter.filter_model(
            specification,
            counts,
            TransmissionParameters(
                R_pre=float(state.R_pre[draw]),
                R_post=float(state.R_post[draw]),
                k=None if dispersion is None else float(dispersion[draw]),
            ),
            delays=delays,
            switch_day=switch_day,
            n_particles=n_particles,
            resample_threshold=resample_threshold,
            reporting_model=resolved_reporting,
            as_of_day=as_of_day,
            rng=rng,
        )
        retained, pipeline = _retained_and_pipeline(
            specification,
            result,
            days=selected,
            reset=float(reset[draw]),
            per_case_exponent=float(per_case_exponent[draw]),
        )
        # Average over particles inside the draw: the particles approximate the day-t filtering
        # law, so this is E[exp(log P) | θ, data through t], and the draw axis is averaged later.
        log_no_cases[draw] = _log_mean_exp(-retained - pipeline)
        log_no_transmission[draw] = _log_mean_exp(
            -retained - pipeline * float(-np.expm1(-per_case_exponent[draw]))
        )
        remaining_weight = result.filtering_remaining_weight[selected]
        sustained_scale = (
            float(-np.log(extinction[draw]))
            if _pools_through_the_mixture(specification)
            else float(reset[draw] * (1.0 - extinction[draw]))
        )
        log_no_sustained[draw] = _log_mean_exp(
            -sustained_scale * remaining_weight - pipeline * float(1.0 - extinction[draw])
        )
        # The filter weights days 1 onwards, so day t sits at position t - 1. Day 0 is the
        # initial condition and carries no weighting step, hence a full-strength cloud.
        position = np.clip(selected - 1, 0, result.effective_sample_size.size - 1)
        effective_sample_size[draw] = np.where(
            selected == 0, float(n_particles), result.effective_sample_size[position]
        )
        resampled[draw] = np.where(selected == 0, 0.0, result.resampled[position])

    if np.any(log_no_transmission + 1e-12 < log_no_cases):
        raise AssertionError("RAC/RAT zero-probability ordering was violated")
    if np.any(log_no_sustained + 1e-12 < log_no_transmission):
        raise AssertionError("RAT/RST zero-probability ordering was violated")

    return FilteredRiskResult(
        estimate=rac.DailyRiskEstimate(
            days=selected,
            log_no_further_cases=log_no_cases,
            log_no_further_transmission=log_no_transmission,
            log_no_sustained_transmission=log_no_sustained,
            n_chains=state.n_chains,
        ),
        diagnostics=FilterDiagnostics(
            days=selected,
            min_effective_sample_size=effective_sample_size.min(axis=0),
            mean_effective_sample_size=effective_sample_size.mean(axis=0),
            resample_fraction=resampled.mean(axis=0),
        ),
    )


def _retained_and_pipeline(
    specification: ModelSpecification,
    result: particle_filter.ParticleFilterResult,
    *,
    days: NDArray[np.int64],
    reset: float,
    per_case_exponent: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``H(t)`` and ``M(t)`` per particle from a filter pass, as the closed forms define them."""
    weight = result.filtering_remaining_weight[days]
    scale = per_case_exponent if _pools_through_the_mixture(specification) else reset
    if specification.anchoring == "infections":
        # No incubation pipeline: an infection is a case, so the retained weight is the whole
        # of the reset state.
        return scale * weight, np.zeros_like(weight)
    if result.filtering_pipeline_mean is None:
        raise ValueError("the onset filter did not record a pipeline mean")
    return scale * weight, result.filtering_pipeline_mean[days]


def _pools_through_the_mixture(specification: ModelSpecification) -> bool:
    """Whether a model's retained cohorts transmit through one Gamma--Poisson mixture.

    SSE and SSE-SO put the overdispersion on the transmission *event*, so their retained
    exponent per unit of remaining weight is ``c = k log(1 + R/k)``. The individual-level models
    have already spent their dispersion in the latent block — given the infectivities the
    remaining offspring are Poisson — and the Poisson limits never had any, so for both the
    exponent is ``R`` itself.
    """
    return specification.overdispersion_level == "event"


def _log_mean_exp(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``log mean_particles exp(·)`` over the trailing particle axis of a ``(days, particles)``."""
    return scipy.special.logsumexp(log_values, axis=1) - np.log(log_values.shape[1])

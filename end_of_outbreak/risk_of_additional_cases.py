"""RAC, RAT and RST, and the reset state they are computed from.

RAC is the project's headline quantity, and it is a **real-time reset posterior predictive**:

    Fit the parameters *and the latents* to the record through day ``t`` alone. Then discard
    the realised trajectory after ``t``, reset ``R`` to ``R_pre`` from day ``t + 1`` onwards,
    and simulate a counterfactual future. RAC(t) is the posterior probability that this
    replicated future contains at least one further case.

So ``P(· | data up to day t)`` is the right reading, and every conditioning day gets its own
fit. ``R`` and the latents are correlated in the posterior, so re-using a full-record parameter
posterior alongside a filtered latent state is a different quantity, not a cheaper route to this
one; :mod:`end_of_outbreak.refit_risk` drives the fits and
:mod:`end_of_outbreak.filtered_risk` is the deliberately approximate comparison.

This module holds the estimand and nothing else: the closed forms, the exact marginalisation of
the latents the fit integrated out, and the posterior averaging. The curve container the figure
and report tiers consume lives in :mod:`end_of_outbreak.risk_curves`, so that editing a
presentation-facing definition does not invalidate every fit.

What the three naive models need
--------------------------------
"No further cases after day ``t``" is a statement about the *first* generation alone: if the
cases present by day ``t`` produce no offspring, there is nothing to produce a second
generation either. So each model's risk is a zero-probability of the pooled remaining
offspring, and the three differ in exactly the way §5.4 says they do:

``sse``, ``ssi``
    The history enters only through the scalar :func:`pooled_remaining_weight`
    ``Λ(t) = Σ_{u ≤ t} (driving_u)(1 − F_{t−u})`` — the observed counts for SSE, the latent
    infectivities for SSI. The remaining offspring pool into a single
    ``NB(mean = R Λ, disp = k Λ)`` or (given the latents) ``Poisson(R Λ)``.
``dlo``
    The history enters through the whole *profile* ``(μ_j)_{j > t}`` of the future force of
    infection (:func:`future_force_of_infection`), because a fresh, force-of-infection
    -independent dispersion ``k`` is applied on every future day.

Two inequalities follow, and they are what the tests pin the closed forms against. Writing
``φ(μ) = −k log(1 + Rμ/k)``, which is convex with ``φ(0) = 0`` and hence superadditive, and
noting ``−Rμ ≤ φ(μ)`` pointwise:

    ``−R Λ(t)  ≤  Σ_{j > t} φ(μ_j)  ≤  φ(Λ(t))``,

i.e. DLO's ``log P(no further cases)`` is bounded below by the Poisson limit's and above by
what the *same* total ``Λ(t)`` would give if it all arrived on one day. Spreading the total
across days dilutes the overdispersion, so on a series whose future force of infection is thin
and spread out — which is exactly the Équateur tail — DLO sits close to the Poisson limit while
SSE, whose dispersion ``k Λ`` scales with the same total, sits far from it. That gap is the
headline of §5.4: a ``k`` calibrated as an individual-level offspring dispersion does far less
work in DLO than in SSE/SSI. It is a property of the profile rather than a universal ordering —
concentrate the whole of ``Λ(t)`` into a day or two and the comparison with SSE reverses.

The latents the fit integrated out
---------------------------------
For SSI and the onset-anchored models the retained state includes the latent infectivities, and
the chosen ``marginalised_inverse_cdf`` parameterisation integrates some of them out exactly
rather than sampling them. Those latents are **not** in ``idata.posterior``, and they are
exactly the days from the last observed case onwards — the days a late conditioning day needs.
Reading the posterior array directly would silently truncate the state and understate RAC.

They do not have to be drawn back. In every model the retained state enters the risk only
through *linear* functionals of the latent path, so

    ``log P(no further event after t | θ, Y) = −(α(θ) + Σ_u β_u(θ) Y_u)``

with ``β_u ≥ 0`` (:func:`latent_risk_basis`). The removed latents are conditionally independent
``Gamma(A_u, B_u)`` given ``θ``, so the Gamma moment generating function integrates them out in
closed form,

    ``E[exp(−β_u Y_u) | θ] = (1 + β_u / B_u)^(−A_u)``,

which is what :func:`marginalised_risk_correction` evaluates and
:func:`risk_log_probabilities` adds to the closed form of the sampled block. That is a third
exact marginalisation, composing with the two the fit already performs: the likelihood
integrates the uncoupled latents out of the observation density, and this integrates the same
latents out of the risk. It is exact rather than Monte Carlo, so no random stream is involved.

:func:`reconstruct_latent_paths` still draws them, because the matched-conditioning
forward-simulation check needs an explicit path, but nothing on the results path calls it.

RAT
---
The risk of additional *transmission* coincides with RAC under all three naive models — that
identification is the conflation the project is about — and separates from it only under the
onset-anchored models. Their calculators live beside the incubation-pipeline reconstruction
below.

RST
---
The risk of sustained transmission is the probability that the reset branching process never
becomes extinct. It uses the smallest one-case extinction root from
:mod:`end_of_outbreak.branching_process`. SSE, SSI and their onset-anchored forms have individual
offspring families and therefore define RST; DLO's fresh day-level incidence dispersion does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from end_of_outbreak import branching_process, forward_simulation, pymc_models, renewal
from end_of_outbreak.delay_distributions import (
    SERIAL_INTERVAL_FIRST_LAG,
    TOST_FIRST_LAG,
    OnsetAnchoredDelays,
    survival_weights,
    tost_survival_weights,
)
from end_of_outbreak.latent_parameterisations import LatentParameterisation
from end_of_outbreak.model_specifications import (
    ModelSpecification,
    TransmissionParameters,
    specification_of,
)
from end_of_outbreak.risk_curves import RiskCurve

_DRAW_CHUNK_ELEMENTS = 2_000_000
"""Rough size of the temporary the DLO profile calculation is allowed to build, in floats."""


# ---------------------------------------------------------------------------------------
# What every calculator here returns
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyRiskEstimate:
    """Per-draw conditional zero-probabilities for RAC, RAT and RST, by conditioning day.

    Every route to the estimand produces this — refitting per day, the filtering
    approximation, and the particle checks — so they are directly comparable. The posterior
    average is taken in :meth:`risk_of_additional_cases`, because RAC(t) is a posterior
    *probability* and the average over draws belongs inside it.

    Under the three naive models RAC and RAT coincide by assumption, and the two arrays are
    equal; they separate only under the onset-anchored models, where the gap is the
    contribution of the latent incubation pipeline.
    """

    days: NDArray[np.int64]
    log_no_further_cases: NDArray[np.float64]
    """``(n_draws, n_days)``: ``log P(no onset after t | posterior draw)``."""

    log_no_further_transmission: NDArray[np.float64]
    """``(n_draws, n_days)``: ``log P(no transmission event after t | posterior draw)``."""

    log_no_sustained_transmission: NDArray[np.float64] | None = None
    """``log P(eventual extinction | draw)``; unavailable for non-branching DLO."""

    n_chains: int = 1
    """Chains the draws came from, stacked chain-major, which is what lets
    :func:`monte_carlo_standard_error` recover the between-chain spread."""

    @property
    def n_draws(self) -> int:
        return int(self.log_no_further_cases.shape[0])

    def risk_of_additional_cases(self) -> RiskCurve:
        return RiskCurve(
            days=self.days,
            risk=1.0 - np.exp(self.log_no_further_cases).mean(axis=0),
            n_draws=self.n_draws,
        )

    def risk_of_additional_transmission(self) -> RiskCurve:
        return RiskCurve(
            days=self.days,
            risk=1.0 - np.exp(self.log_no_further_transmission).mean(axis=0),
            n_draws=self.n_draws,
        )

    def risk_of_sustained_transmission(self) -> RiskCurve:
        if self.log_no_sustained_transmission is None:
            raise ValueError("risk of sustained transmission is unavailable for DLO")
        return RiskCurve(
            days=self.days,
            risk=1.0 - np.exp(self.log_no_sustained_transmission).mean(axis=0),
            n_draws=self.n_draws,
        )

    def sustained_transmission_standard_error(self) -> NDArray[np.float64]:
        if self.log_no_sustained_transmission is None:
            raise ValueError("risk of sustained transmission is unavailable for DLO")
        return monte_carlo_standard_error(
            self.log_no_sustained_transmission, n_chains=self.n_chains
        )

    def standard_errors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Monte-Carlo standard errors of RAC and RAT, from the between-chain spread.

        RST is optional, so its error has the separate
        :meth:`sustained_transmission_standard_error` accessor.
        """
        return (
            monte_carlo_standard_error(self.log_no_further_cases, n_chains=self.n_chains),
            monte_carlo_standard_error(self.log_no_further_transmission, n_chains=self.n_chains),
        )

    @classmethod
    def concatenate(cls, parts: list[DailyRiskEstimate]) -> DailyRiskEstimate:
        """Join per-day estimates into one curve. Every part must share a chain count."""
        if not parts:
            raise ValueError("no per-day estimates to concatenate")
        chains = {part.n_chains for part in parts}
        if len(chains) != 1:
            raise ValueError(f"the per-day estimates disagree on the chain count: {sorted(chains)}")
        draws = {part.n_draws for part in parts}
        if len(draws) != 1:
            raise ValueError(
                f"the per-day estimates disagree on the draw count: {sorted(draws)}. Every "
                "conditioning day must be fitted at the same sampler settings, or the curve "
                "would carry a different Monte-Carlo error on different days"
            )
        sustained = [part.log_no_sustained_transmission for part in parts]
        if any(values is None for values in sustained) and not all(
            values is None for values in sustained
        ):
            raise ValueError("the per-day estimates disagree on whether RST is available")
        return cls(
            days=np.concatenate([part.days for part in parts]),
            log_no_further_cases=np.concatenate(
                [part.log_no_further_cases for part in parts], axis=1
            ),
            log_no_further_transmission=np.concatenate(
                [part.log_no_further_transmission for part in parts], axis=1
            ),
            log_no_sustained_transmission=(
                None
                if sustained[0] is None
                else np.concatenate([values for values in sustained if values is not None], axis=1)
            ),
            n_chains=parts[0].n_chains,
        )


# ---------------------------------------------------------------------------------------
# The remaining-transmission weight and the future force of infection
# ---------------------------------------------------------------------------------------


def pooled_remaining_weight(
    driving: NDArray[np.float64],
    serial_interval: NDArray[np.float64],
    *,
    n_days: int | None = None,
) -> NDArray[np.float64]:
    """``Λ(t) = Σ_{u ≤ t} driving_u · (1 − F_{t−u})`` for every day (§5.2).

    Parameters
    ----------
    driving
        The series whose remaining transmission is being pooled: the observed counts for SSE
        and DLO, the latent infectivities ``Y`` for SSI. A leading axis is allowed and is
        carried through, so a whole posterior block of latent paths can be reduced at once.
    serial_interval
        Discrete serial interval ``w``, stored from lag 1.
    n_days
        Length of the returned curve; defaults to the length of ``driving``.

    Returns
    -------
    ``Λ`` with the same leading axes as ``driving`` and a trailing axis of length ``n_days``.
    """
    driving = np.asarray(driving, dtype=np.float64)
    if driving.ndim == 0:
        raise ValueError("driving must have at least one axis, indexed by day")
    length = driving.shape[-1] if n_days is None else n_days
    operator = renewal.delay_design_matrix(
        length,
        survival_weights(serial_interval),
        first_lag=0,
        source_days=np.arange(driving.shape[-1], dtype=np.int64),
    )
    return driving @ operator.T


def future_force_of_infection(
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    *,
    days: NDArray[np.int64],
) -> NDArray[np.float64]:
    """``μ_j = Σ_{s ≥ 1} w_s I_{j−s}`` for ``j > t``, with the realised future discarded.

    One row per conditioning day in ``days``, and one column per lag: row ``t`` holds
    ``μ_{t+1}, ..., μ_{t+len(w)}``. Setting ``I_u := 0`` for ``u > t`` is what makes this the
    force of infection *under the counterfactual in which no further case occurs*, which is
    exactly the event whose probability the DLO product evaluates. Beyond ``t + len(w)`` the
    serial interval can no longer reach the retained cases and ``μ_j`` is identically zero, so
    the infinite product of §5.3 is finite.
    """
    counts = np.asarray(counts, dtype=np.float64)
    w = np.asarray(serial_interval, dtype=np.float64)
    days = np.asarray(days, dtype=np.int64)
    profile = np.zeros((days.size, w.size), dtype=np.float64)
    day_index = np.arange(counts.size, dtype=np.int64)
    for row, t in enumerate(days):
        retained = np.where(day_index <= t, counts, 0.0)
        reachable = renewal.delay_weighted_sum(
            retained, w, first_lag=SERIAL_INTERVAL_FIRST_LAG, n_days=int(t) + 1 + w.size
        )
        profile[row] = reachable[int(t) + 1 :]
    return profile


# ---------------------------------------------------------------------------------------
# The closed forms of §5.3
# ---------------------------------------------------------------------------------------


def log_probability_of_no_further_cases(
    model: str | ModelSpecification,
    *,
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    R_pre: float | NDArray[np.float64],
    k: float | NDArray[np.float64] | None = None,
    infectivity: NDArray[np.float64] | None = None,
    days: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """``log P(no further cases after day t)`` for each parameter draw and each day.

    The three closed forms of §5.3, plus the Poisson limit:

    ======  =========================================================================
    Model   ``log P(no further cases after day t)``
    ======  =========================================================================
    sse     ``−k Λ(t) log(1 + R_pre / k)``
    ssi     ``−R_pre Λ_Y(t)``, with ``Λ_Y`` built from the latent infectivities
    dlo     ``−k Σ_{j > t} log(1 + R_pre μ_j / k)``
    cori    ``−R_pre Λ(t)`` — the ``k → ∞`` limit of both SSE and DLO
    ======  =========================================================================

    Parameters
    ----------
    model
        One of ``dlo``, ``sse``, ``ssi``, ``cori``. Use
        :func:`onset_event_probabilities` for an onset-anchored model; it also returns RAT.
    counts
        The observed series ``C_0, ..., C_T``, read as infections.
    serial_interval
        Discrete serial interval ``w``, from lag 1.
    R_pre
        The reproduction number the counterfactual future reverts to. A scalar or one value
        per posterior draw.
    k
        Dispersion, matching ``R_pre`` in shape. ``None`` only for ``cori``.
    infectivity
        ``(n_draws, n_days)`` complete latent path for SSI, as returned by
        :func:`reconstruct_latent_paths`. Rejected for the models that have no latents.
    days
        Conditioning days; defaults to every day of the window.

    Returns
    -------
    ``(n_draws, len(days))`` array of log-probabilities, non-positive throughout.
    """
    specification = specification_of(model)
    if specification.anchoring != "infections":
        raise ValueError(
            f"model {specification.name!r} is onset-anchored; use onset_event_probabilities, "
            "which takes the incubation/TOST delays and R_post needed to rebuild the reset "
            "pipeline"
        )
    counts = np.asarray(counts, dtype=np.int64)
    w = np.asarray(serial_interval, dtype=np.float64)
    days = np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days, np.int64)
    R = np.atleast_1d(np.asarray(R_pre, dtype=np.float64))
    if R.ndim != 1:
        raise ValueError("R_pre must be a scalar or a one-dimensional array of draws")
    dispersion = _dispersion_draws(specification, k, n_draws=R.size)

    if specification.latent_variable is None:
        if infectivity is not None:
            raise ValueError(
                f"model {specification.name!r} has no latent block, so its remaining-"
                "transmission weight is built from the observed counts; drop `infectivity`"
            )
        driving = counts.astype(np.float64)
    else:
        if infectivity is None:
            raise ValueError(
                f"model {specification.name!r} is driven by latent infectivities; pass the "
                "complete by-day paths from reconstruct_latent_paths"
            )
        driving = np.asarray(infectivity, dtype=np.float64)
        if driving.shape != (R.size, counts.size):
            raise ValueError(
                f"infectivity must have shape {(R.size, counts.size)}, got {driving.shape}"
            )

    if specification.name == "dlo":
        assert dispersion is not None  # DLO always carries a dispersion parameter
        return _dlo_log_probability(
            future_force_of_infection(counts, w, days=days), R=R, k=dispersion
        )

    Lambda = pooled_remaining_weight(driving, w)[..., days]
    if specification.name == "ssi":
        # Given the infectivities the remaining offspring are Poisson(R Λ_Y), so the dispersion
        # has already done its work in the latent block and does not reappear here.
        return -R[:, None] * Lambda
    if dispersion is None:  # cori, the k → ∞ limit
        return -R[:, None] * Lambda[None, :]
    # sse: the remaining offspring pool into NB(mean = R Λ, disp = k Λ) by NB closure.
    return -(dispersion * np.log1p(R / dispersion))[:, None] * Lambda[None, :]


def log_probability_of_no_sustained_transmission(
    model: str | ModelSpecification,
    *,
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    R_pre: float | NDArray[np.float64],
    k: float | NDArray[np.float64] | None = None,
    infectivity: NDArray[np.float64] | None = None,
    days: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """``log P(eventual extinction after day t)`` for an infection-anchored model.

    DLO is deliberately excluded: its fresh day-level negative-binomial draw does not assign an
    offspring family to each infected individual, so there is no Galton--Watson extinction root
    to evaluate.  For SSE the remaining-offspring pgf at the single-case root is ``q**Lambda``;
    for SSI it is Poisson conditional on the retained infectivities.
    """
    specification = specification_of(model)
    if specification.anchoring != "infections":
        raise ValueError(
            f"model {specification.name!r} is onset-anchored; use onset_event_probabilities"
        )
    if specification.name == "dlo":
        raise ValueError("DLO has no branching-process formula for sustained transmission")

    counts = np.asarray(counts, dtype=np.int64)
    selected = (
        np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days, np.int64)
    )
    R = _one_draw_axis(R_pre, "R_pre")
    dispersion = _dispersion_draws(specification, k, n_draws=R.size)

    if specification.latent_variable is None:
        if infectivity is not None:
            raise ValueError(
                f"model {specification.name!r} has no latent block; drop `infectivity`"
            )
        driving = counts.astype(np.float64)
    else:
        if infectivity is None:
            raise ValueError(
                f"model {specification.name!r} needs the complete retained infectivity path"
            )
        driving = np.asarray(infectivity, dtype=np.float64)
        if driving.shape != (R.size, counts.size):
            raise ValueError(
                f"infectivity must have shape {(R.size, counts.size)}, got {driving.shape}"
            )

    Lambda = pooled_remaining_weight(driving, serial_interval)[..., selected]
    if dispersion is None:
        q = branching_process.poisson_extinction_probability(R)
        return -(R * (1.0 - q))[:, None] * Lambda[None, :]

    q = branching_process.negative_binomial_extinction_probability(R, dispersion)
    if specification.name == "ssi":
        return -(R * (1.0 - q))[:, None] * Lambda
    return np.log(q)[:, None] * Lambda[None, :]


def _dispersion_draws(
    specification: ModelSpecification,
    k: float | NDArray[np.float64] | None,
    *,
    n_draws: int,
) -> NDArray[np.float64] | None:
    """``k`` as one value per draw, or ``None`` for the Poisson limit."""
    if not specification.has_dispersion:
        if k is not None:
            raise ValueError(f"model {specification.name!r} is the k → ∞ limit and takes no k")
        return None
    if k is None:
        raise ValueError(f"model {specification.name!r} needs a dispersion parameter k")
    dispersion = np.atleast_1d(np.asarray(k, dtype=np.float64))
    if dispersion.size == 1:
        dispersion = np.repeat(dispersion, n_draws)
    if dispersion.shape != (n_draws,):
        raise ValueError(f"k must hold one value per draw ({n_draws}), got {dispersion.shape}")
    return dispersion


def _dlo_log_probability(
    profile: NDArray[np.float64], *, R: NDArray[np.float64], k: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``−k Σ_{j > t} log(1 + R μ_j / k)`` over the future force-of-infection profile.

    Chunked over draws: the natural expression builds a
    ``(draws, conditioning days, lags)`` temporary, which is a gigabyte on the real series at
    4000 draws and a few megabytes once the draws are taken a slice at a time.
    """
    n_draws = R.size
    n_days, n_lags = profile.shape
    chunk = max(1, _DRAW_CHUNK_ELEMENTS // max(1, n_days * n_lags))
    result = np.empty((n_draws, n_days), dtype=np.float64)
    for start in range(0, n_draws, chunk):
        stop = min(start + chunk, n_draws)
        R_chunk = R[start:stop, None, None]
        k_chunk = k[start:stop, None, None]
        result[start:stop] = -(k_chunk * np.log1p(R_chunk * profile[None] / k_chunk)).sum(axis=2)
    return result


# ---------------------------------------------------------------------------------------
# Onset-anchored reset state: incubation pipeline, RAC and RAT (§5.1, §5.5)
# ---------------------------------------------------------------------------------------


def incubation_pipeline_mean(
    expected_infections: NDArray[np.float64], incubation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Mean number infected by day ``t`` whose onset lies after ``t``, for every ``t``.

    Given ``E_u``, independent Poisson splitting by incubation delay makes the retained
    pipeline Poisson with this mean:

    ``M(t) = Σ_{u≤t} E_u P(A > t-u)``.

    A leading posterior-draw axis is carried through unchanged.
    """
    return pooled_remaining_weight(expected_infections, incubation)


def remaining_tost_weight(
    driving: NDArray[np.float64], tost: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Driving weight assigned to transmission days strictly after ``t``.

    ``driving`` is the observed onset series for SSE-SO/Cori-SO and the complete latent ``Y``
    path for SSI-SO. TOST starts at lag zero, so this uses
    :func:`tost_survival_weights` rather than the lag-1 serial-interval survival function.
    """
    driving = np.asarray(driving, dtype=np.float64)
    if driving.ndim == 0:
        raise ValueError("driving must have at least one axis, indexed by day")
    operator = renewal.delay_design_matrix(
        driving.shape[-1],
        tost_survival_weights(tost),
        first_lag=TOST_FIRST_LAG,
        source_days=np.arange(driving.shape[-1], dtype=np.int64),
    )
    return driving @ operator.T


def onset_event_probabilities(
    model: str | ModelSpecification,
    *,
    counts: NDArray[np.int64],
    delays: OnsetAnchoredDelays,
    switch_day: int,
    R_pre: float | NDArray[np.float64],
    R_post: float | NDArray[np.float64],
    k: float | NDArray[np.float64] | None = None,
    latent: NDArray[np.float64] | None = None,
    reset_R: float | NDArray[np.float64] | None = None,
    days: NDArray[np.int64] | None = None,
    n_chains: int = 1,
) -> DailyRiskEstimate:
    """Conditional zero-probabilities defining RAC, RAT and RST for an onset model.

    The retained expected infections ``E_{u≤t}`` generate a Poisson incubation pipeline with
    mean ``M(t)``. A no-further-case path requires that whole pipeline to be empty, whereas a
    no-further-transmission path permits pipeline cases provided every one transmits zero
    times. The latter contribution is the offspring zero-probability averaged over the
    Poisson pipeline. This gives, conditional on one posterior draw,

    ``log P(no case) = -H(t) - M(t)``

    ``log P(no transmission) = -H(t) - M(t) [1 - exp(-c)]``

    where ``H(t)`` is the zero-transmission exponent of the retained onset cohorts and ``c``
    is the exponent for one future pipeline case. For Cori-SO/SSI-SO, ``H`` is respectively
    ``R W_D``/``R W_Y``; for SSE-SO it is
    ``k log(1 + R/k) W_D``. The same Gamma–Poisson mixture gives
    ``c = k log(1 + R/k)`` for both overdispersed models and ``c = R`` for Cori-SO.

    The posterior average is still Monte Carlo — these arrays hold one conditional probability
    per draw — but evaluating the conditional zero event analytically avoids an unnecessary
    second simulation layer and makes ``RAC >= RAT`` exact up to floating-point rounding.

    For RST, ``q`` is the smallest extinction fixed point for a single reset-regime case.
    Retained cohorts contribute ``W log(q)`` under SSE-SO or ``-R (1-q) W`` under
    SSI-SO/Cori-SO, while every future pipeline case contributes a factor ``q``. Normalised
    generation-time distributions therefore affect when descendants occur but not whether an
    indefinitely continuing constant-reset branching process eventually becomes extinct.
    """
    specification = specification_of(model)
    if specification.anchoring != "onsets":
        raise ValueError(
            f"model {specification.name!r} is infection-anchored; use risk_curve for RAC, "
            "under which RAC and RAT coincide"
        )
    counts = np.asarray(counts, dtype=np.int64)
    selected_days = (
        np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days, dtype=np.int64)
    )
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional onset series")
    if (selected_days < 0).any() or (selected_days >= counts.size).any():
        raise ValueError("days must index the observed onset series")

    R_pre_draws = _one_draw_axis(R_pre, "R_pre")
    n_draws = R_pre_draws.size
    R_post_draws = _broadcast_draws(R_post, "R_post", n_draws=n_draws)
    reset_draws = (
        R_pre_draws if reset_R is None else _broadcast_draws(reset_R, "reset_R", n_draws=n_draws)
    )
    dispersion = _dispersion_draws(specification, k, n_draws=n_draws)
    extinction = (
        branching_process.poisson_extinction_probability(reset_draws)
        if dispersion is None
        else branching_process.negative_binomial_extinction_probability(reset_draws, dispersion)
    )
    R_history = np.where(
        renewal.switch_index(counts.size, switch_day)[None, :] == 0,
        R_pre_draws[:, None],
        R_post_draws[:, None],
    )

    tost_operator = renewal.delay_design_matrix(
        counts.size,
        delays.tost,
        first_lag=TOST_FIRST_LAG,
        source_days=np.arange(counts.size, dtype=np.int64),
    )
    if specification.name == "cori_so":
        if latent is not None:
            raise ValueError("Cori-SO has no latent block; drop `latent`")
        force = counts.astype(np.float64) @ tost_operator.T
        expected_infections = R_history * force[None, :]
        retained_exponent = (
            reset_draws[:, None]
            * remaining_tost_weight(counts.astype(np.float64), delays.tost)[None, :]
        )
        retained_sustained_exponent = (reset_draws * (1.0 - extinction))[
            :, None
        ] * remaining_tost_weight(counts.astype(np.float64), delays.tost)[None, :]
        per_case_exponent = reset_draws
    else:
        if latent is None:
            raise ValueError(
                f"model {specification.name!r} needs the complete by-day latent paths from "
                "posterior_state"
            )
        latent_paths = np.asarray(latent, dtype=np.float64)
        if latent_paths.shape != (n_draws, counts.size):
            raise ValueError(
                f"latent must have shape {(n_draws, counts.size)}, got {latent_paths.shape}"
            )
        assert dispersion is not None
        per_case_exponent = dispersion * np.log1p(reset_draws / dispersion)
        if specification.name == "sse_so":
            expected_infections = R_history * latent_paths
            remaining_weight = remaining_tost_weight(counts.astype(np.float64), delays.tost)[
                None, :
            ]
            retained_exponent = per_case_exponent[:, None] * remaining_weight
            retained_sustained_exponent = -np.log(extinction)[:, None] * remaining_weight
        else:
            force = latent_paths @ tost_operator.T
            expected_infections = R_history * force
            remaining_weight = remaining_tost_weight(latent_paths, delays.tost)
            retained_exponent = reset_draws[:, None] * remaining_weight
            retained_sustained_exponent = (reset_draws * (1.0 - extinction))[
                :, None
            ] * remaining_weight

    pipeline = incubation_pipeline_mean(expected_infections, delays.incubation)
    log_no_cases = -retained_exponent - pipeline
    log_no_transmission = -retained_exponent - pipeline * (-np.expm1(-per_case_exponent))[:, None]
    log_no_sustained = -retained_sustained_exponent - pipeline * (1.0 - extinction)[:, None]
    if np.any(log_no_transmission + 1e-12 < log_no_cases):
        raise AssertionError("RAC/RAT zero-probability ordering was violated")
    if np.any(log_no_sustained + 1e-12 < log_no_transmission):
        raise AssertionError("RAT/RST zero-probability ordering was violated")
    return DailyRiskEstimate(
        days=selected_days,
        log_no_further_cases=log_no_cases[:, selected_days],
        log_no_further_transmission=log_no_transmission[:, selected_days],
        log_no_sustained_transmission=log_no_sustained[:, selected_days],
        n_chains=n_chains,
    )


def onset_log_probability_of_no_further_cases(
    model: str | ModelSpecification, **kwargs: Any
) -> NDArray[np.float64]:
    """Convenience view of :func:`onset_event_probabilities` for headline RAC."""
    return onset_event_probabilities(model, **kwargs).log_no_further_cases


def onset_log_probability_of_no_further_transmission(
    model: str | ModelSpecification, **kwargs: Any
) -> NDArray[np.float64]:
    """Convenience view of :func:`onset_event_probabilities` for supplementary RAT."""
    return onset_event_probabilities(model, **kwargs).log_no_further_transmission


def onset_log_probability_of_no_sustained_transmission(
    model: str | ModelSpecification, **kwargs: Any
) -> NDArray[np.float64]:
    """Convenience view of :func:`onset_event_probabilities` for RST."""
    sustained = onset_event_probabilities(model, **kwargs).log_no_sustained_transmission
    assert sustained is not None
    return sustained


def _one_draw_axis(value: float | NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """A scalar or one-dimensional posterior block, preserving a scalar as one draw."""
    draws = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if draws.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a one-dimensional array of draws")
    return draws


def _broadcast_draws(
    value: float | NDArray[np.float64], name: str, *, n_draws: int
) -> NDArray[np.float64]:
    """Broadcast a scalar parameter to the posterior draw count."""
    draws = _one_draw_axis(value, name)
    if draws.size == 1:
        draws = np.repeat(draws, n_draws)
    if draws.shape != (n_draws,):
        raise ValueError(f"{name} must hold one value per draw ({n_draws}), got {draws.shape}")
    return draws


# ---------------------------------------------------------------------------------------
# Posterior averaging
# ---------------------------------------------------------------------------------------


def risk_curve(
    model: str | ModelSpecification,
    *,
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    R_pre: float | NDArray[np.float64],
    k: float | NDArray[np.float64] | None = None,
    infectivity: NDArray[np.float64] | None = None,
    days: NDArray[np.int64] | None = None,
) -> RiskCurve:
    """RAC(t) over the conditioning days, averaged over the posterior draws supplied.

    Arguments are those of :func:`log_probability_of_no_further_cases`; every array carrying a
    draw axis must agree on its length, and that axis is averaged out here.
    """
    log_probability = log_probability_of_no_further_cases(
        model,
        counts=counts,
        serial_interval=serial_interval,
        R_pre=R_pre,
        k=k,
        infectivity=infectivity,
        days=days,
    )
    counts = np.asarray(counts, dtype=np.int64)
    day_index = np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days)
    return RiskCurve(
        days=np.asarray(day_index, dtype=np.int64),
        risk=1.0 - np.exp(log_probability).mean(axis=0),
        n_draws=int(log_probability.shape[0]),
    )


@dataclass(frozen=True)
class UnsampledLatents:
    """Latents the fit did not sample, and their exact conditional law given the parameters.

    Two kinds, and they are handled identically because their conditional laws have the same
    shape: those the ``marginalised`` parameterisations integrated out of the likelihood
    (:func:`end_of_outbreak.pymc_models.marginalised_latent_conditional`), and SSE-SO's
    boundary latent on the final day of the window, which cannot reach any fitted onset because
    incubation starts at lag 1 and therefore retains its prior exactly.
    """

    days: NDArray[np.int64]
    shape: NDArray[np.float64]
    """``(n_draws, n_unsampled)`` Gamma shape ``A_u``, one per parameter draw."""

    rate: NDArray[np.float64]
    """``(n_draws, n_unsampled)`` Gamma rate ``B_u``, one per parameter draw."""


@dataclass(frozen=True)
class PosteriorState:
    """The posterior draws a risk curve is built from: parameters, latents, and their law.

    The latent block is deliberately *not* completed here. ``sampled_infectivity`` carries the
    latents the sampler saw and zeros elsewhere, and :attr:`unsampled` carries the conditional
    law of the rest, which :func:`risk_log_probabilities` integrates out in closed form rather
    than drawing.
    """

    R_pre: NDArray[np.float64]
    R_post: NDArray[np.float64]
    k: NDArray[np.float64] | None
    sampled_infectivity: NDArray[np.float64] | None
    """``(n_draws, n_days)`` latent path, zero on every day the fit did not sample."""

    unsampled: UnsampledLatents | None = None
    n_chains: int = 1
    """Chains the draws came from. They are stacked chain-major, which is what lets
    :func:`monte_carlo_standard_error` recover the between-chain spread."""

    @property
    def n_draws(self) -> int:
        return int(self.R_pre.size)


def monte_carlo_standard_error(
    log_probability: NDArray[np.float64], *, n_chains: int
) -> NDArray[np.float64]:
    """Standard error of a posterior-averaged probability, from the spread across chains.

    RAC(t) is an average over draws, so it carries Monte-Carlo error, and any comparison of two
    RAC curves — two parameterisations, MCMC against a particle smoother — has to be judged
    against it rather than against a round number. Chains are independent, so the standard
    error of their mean is an estimate that needs no autocorrelation modelling.

    Parameters
    ----------
    log_probability
        ``(n_draws, n_days)`` per-draw log-probabilities, stacked **chain-major** — the layout
        :func:`posterior_state` produces.
    """
    if n_chains < 2:
        return np.full(log_probability.shape[1], np.nan)
    probability = np.exp(log_probability)
    per_chain = probability.reshape(n_chains, -1, probability.shape[1]).mean(axis=1)
    return per_chain.std(axis=0, ddof=1) / np.sqrt(n_chains)


def posterior_state(
    model: str | ModelSpecification,
    idata: Any,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    latent_parameterisation: str | LatentParameterisation | None = None,
    fixed_R_pre: float | None = None,
    fixed_R_post: float | None = None,
    fixed_k: float | None = None,
    negligible_latent_threshold: float = 0.0,
) -> PosteriorState:
    """Flatten a fit into draw-indexed arrays, with the law of the latents it did not sample.

    Parameters
    ----------
    model, counts, delays, switch_day, latent_parameterisation, negligible_latent_threshold
        As the fit was built; they determine which latents the sampler saw and which were
        removed, and the removed ones' conditional law.
    idata
        The ``InferenceData`` from the fit. Its ``posterior`` group supplies ``R_pre``,
        ``R_post``, the latent block, and ``k`` when ``k`` was estimated.
    fixed_R_pre, fixed_R_post, fixed_k
        Values for whichever parameters were **fixed** in the fit rather than given a prior: a
        fixed parameter is a constant in the graph, so it never appears in the draws. The
        fixed-``k`` analyses need ``fixed_k``; the fixed-``θ`` cross-checks of §6.4, which
        sample the latents alone, need all three. Each is ignored when the posterior carries
        the variable itself.
    """
    specification = specification_of(model)
    counts = np.asarray(counts, dtype=np.int64)
    posterior = idata.posterior
    n_draws = int(posterior.sizes["chain"] * posterior.sizes["draw"])
    R_pre = _parameter_draws(posterior, "R_pre", fixed=fixed_R_pre, n_draws=n_draws)
    R_post = _parameter_draws(posterior, "R_post", fixed=fixed_R_post, n_draws=n_draws)
    k = (
        _parameter_draws(posterior, "k", fixed=fixed_k, n_draws=n_draws)
        if specification.has_dispersion
        else None
    )

    sampled_infectivity = None
    unsampled = None
    if specification.has_latents:
        if latent_parameterisation is None:
            raise ValueError(
                f"model {specification.name!r} has a latent block, so the parameterisation it "
                "was fitted under must be given: it decides which latents the sampler saw"
            )
        assert k is not None  # every latent model in the project carries a dispersion
        sampled_infectivity, sampled_days = _sampled_latent_paths(
            specification, idata, counts, n_draws=n_draws
        )
        unsampled = unsampled_latent_conditional(
            specification,
            counts,
            delays=delays,
            switch_day=switch_day,
            latent_parameterisation=latent_parameterisation,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            negligible_latent_threshold=negligible_latent_threshold,
        )
        _check_block_accounted_for(
            specification,
            counts,
            delays=delays,
            switch_day=switch_day,
            sampled_days=sampled_days,
            unsampled_days=unsampled.days,
        )
    return PosteriorState(
        R_pre=R_pre,
        R_post=R_post,
        k=k,
        sampled_infectivity=sampled_infectivity,
        unsampled=unsampled,
        n_chains=int(posterior.sizes["chain"]),
    )


def risk_log_probabilities(
    model: str | ModelSpecification,
    state: PosteriorState,
    *,
    counts: NDArray[np.int64],
    delays: OnsetAnchoredDelays,
    switch_day: int,
    days: NDArray[np.int64] | None = None,
    reset_R: float | NDArray[np.float64] | None = None,
) -> DailyRiskEstimate:
    """RAC and RAT log zero-probabilities from a fit's draws — the one entry point.

    Two terms. The closed forms of §5.3 evaluated at the *sampled* latents, which are zero on
    every day the fit did not sample; and the exact correction of
    :func:`marginalised_risk_correction` for those unsampled days. Their sum is the conditional
    zero-probability with the unsampled latents integrated out rather than drawn, which is
    exact and carries no Monte-Carlo error of its own.
    """
    specification = specification_of(model)
    counts = np.asarray(counts, dtype=np.int64)
    selected = (
        np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days, np.int64)
    )

    if specification.anchoring == "onsets":
        estimate = onset_event_probabilities(
            specification,
            counts=counts,
            delays=delays,
            switch_day=switch_day,
            R_pre=state.R_pre,
            R_post=state.R_post,
            k=state.k,
            latent=state.sampled_infectivity,
            reset_R=reset_R,
            days=selected,
            n_chains=state.n_chains,
        )
    else:
        if reset_R is not None:
            raise ValueError(
                f"model {specification.name!r} resets to R_pre by construction and takes no "
                "reset_R; the argument exists for the onset-anchored pipeline checks"
            )
        log_probability = log_probability_of_no_further_cases(
            specification,
            counts=counts,
            serial_interval=delays.serial_interval,
            R_pre=state.R_pre,
            k=state.k,
            infectivity=state.sampled_infectivity,
            days=selected,
        )
        log_no_sustained = (
            None
            if specification.name == "dlo"
            else log_probability_of_no_sustained_transmission(
                specification,
                counts=counts,
                serial_interval=delays.serial_interval,
                R_pre=state.R_pre,
                k=state.k,
                infectivity=state.sampled_infectivity,
                days=selected,
            )
        )
        # RAC and RAT coincide under every infection-anchored model, by assumption: an
        # infection *is* a case there. That identification is the conflation being studied.
        estimate = DailyRiskEstimate(
            days=selected,
            log_no_further_cases=log_probability,
            log_no_further_transmission=log_probability,
            log_no_sustained_transmission=log_no_sustained,
            n_chains=state.n_chains,
        )

    if state.unsampled is None or state.unsampled.days.size == 0:
        _assert_risk_ordering(estimate)
        return estimate
    assert state.k is not None  # only the latent models have an unsampled block
    cases, transmission, sustained = marginalised_risk_correction(
        specification,
        state.unsampled,
        counts=counts,
        delays=delays,
        switch_day=switch_day,
        days=selected,
        R_pre=state.R_pre,
        R_post=state.R_post,
        k=state.k,
        reset_R=reset_R,
    )
    estimate = DailyRiskEstimate(
        days=estimate.days,
        log_no_further_cases=estimate.log_no_further_cases + cases,
        log_no_further_transmission=estimate.log_no_further_transmission + transmission,
        log_no_sustained_transmission=(
            None
            if estimate.log_no_sustained_transmission is None
            else estimate.log_no_sustained_transmission + sustained
        ),
        n_chains=estimate.n_chains,
    )
    _assert_risk_ordering(estimate)
    return estimate


def _assert_risk_ordering(estimate: DailyRiskEstimate) -> None:
    """Assert ``RST <= RAT <= RAC`` through the equivalent zero-event ordering."""
    if np.any(estimate.log_no_further_transmission + 1e-12 < estimate.log_no_further_cases):
        raise AssertionError("RAC/RAT zero-probability ordering was violated")
    if estimate.log_no_sustained_transmission is not None and np.any(
        estimate.log_no_sustained_transmission + 1e-12 < estimate.log_no_further_transmission
    ):
        raise AssertionError("RAT/RST zero-probability ordering was violated")


def _flatten_draws(posterior: Any, name: str) -> NDArray[np.float64]:
    """One posterior variable, with its chain and draw axes stacked into a leading draw axis."""
    if name not in posterior.data_vars:
        available = ", ".join(sorted(posterior.data_vars))
        raise ValueError(f"the posterior has no variable {name!r}; it holds {available}")
    variable = posterior.data_vars[name]
    return np.asarray(variable.stack(draw_index=("chain", "draw")).transpose("draw_index", ...))


def _parameter_draws(
    posterior: Any, name: str, *, fixed: float | None, n_draws: int
) -> NDArray[np.float64]:
    """Draws of a scalar parameter, or the constant it was held at, repeated per draw."""
    if name in posterior.data_vars:
        return _flatten_draws(posterior, name)
    if fixed is None:
        raise ValueError(
            f"the fit has no {name!r} in its posterior and no fixed value was supplied. A "
            "parameter that was fixed rather than estimated is a constant in the graph and "
            f"never appears in the draws, so pass fixed_{name} explicitly"
        )
    return np.full(n_draws, float(fixed), dtype=np.float64)


# ---------------------------------------------------------------------------------------
# The latents the fit did not sample, and integrating them out of the risk
# ---------------------------------------------------------------------------------------


def _sampled_latent_paths(
    specification: ModelSpecification, idata: Any, counts: NDArray[np.int64], *, n_draws: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The sampled latent block on the day axis, zero on every day the fit did not sample.

    The variable can be absent from the posterior altogether, and legitimately so: on an early
    conditioning day no latent reaches a day with a case, so the whole block is marginalised
    and the sampler sees none of it. :func:`posterior_state` checks that every latent is
    accounted for either here or in the unsampled block, which is the guard that matters.
    """
    assert specification.latent_variable is not None
    paths = np.zeros((n_draws, counts.size), dtype=np.float64)
    if specification.latent_variable not in idata.posterior.data_vars:
        return paths, np.empty(0, dtype=np.int64)
    sampled = _flatten_draws(idata.posterior, specification.latent_variable)
    if sampled.shape[0] != n_draws:
        raise ValueError(f"the latent block has {sampled.shape[0]} draws but the fit has {n_draws}")
    dimension = pymc_models.latent_dimension(specification)
    sampled_days = np.asarray(
        idata.posterior.data_vars[specification.latent_variable].coords[dimension].values,
        dtype=np.int64,
    )
    paths[:, sampled_days] = sampled
    return paths, sampled_days


def _check_block_accounted_for(
    specification: ModelSpecification,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    sampled_days: NDArray[np.int64],
    unsampled_days: NDArray[np.int64],
) -> None:
    """Every latent that carries mass must be either sampled or accounted for in closed form.

    The failure this rules out is the quiet one: a latent that is in neither block is read as
    zero, the risk comes out plausible, and it is too low. Latents whose scale is zero are
    point masses at zero and are correctly absent from both.
    """
    structure = pymc_models.latent_block_structure(
        specification, counts, delays=delays, switch_day=switch_day
    )
    carries_mass = structure.days[structure.scale > 0.0]
    missing = np.setdiff1d(carries_mass, np.union1d(sampled_days, unsampled_days))
    if missing.size > 0:
        raise ValueError(
            f"model {specification.name!r} has latents on days {missing.tolist()} that the fit "
            "neither sampled nor integrated out; reading them as zero would understate the risk"
        )


def unsampled_latent_conditional(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    latent_parameterisation: str | LatentParameterisation,
    R_pre: NDArray[np.float64],
    R_post: NDArray[np.float64],
    k: NDArray[np.float64],
    negligible_latent_threshold: float = 0.0,
) -> UnsampledLatents:
    """Every latent the fit left out of the posterior, with its exact conditional law.

    Two sources, in one place because the calculators must not have to remember either:

    - the latents the ``marginalised`` parameterisations integrated out of the likelihood,
      whose conditional is ``Gamma(k·scale_u, k + c_u)``; and
    - **SSE-SO's boundary latent** on the final day of the window. Its infections cannot reach
      a fitted onset, because incubation has no lag-0 mass, so it is absent from the graph
      altogether — but they *are* in the incubation pipeline retained on that day, and its
      conditional law is therefore its prior ``Gamma(k·λ_T, k)``.

    Given the parameters these are independent of the sampled block and of each other, which is
    what makes both the exact correction of :func:`marginalised_risk_correction` and the draws
    of :func:`reconstruct_latent_paths` valid.
    """
    specification = specification_of(model)
    if specification.latent_variable is None:
        raise ValueError(f"model {specification.name!r} has no latent block")
    counts = np.asarray(counts, dtype=np.int64)
    days, shape, rate = pymc_models.marginalised_latent_conditional(
        specification,
        counts,
        delays=delays,
        switch_day=switch_day,
        R_pre=R_pre,
        R_post=R_post,
        k=k,
        latent_parameterisation=latent_parameterisation,
        negligible_latent_threshold=negligible_latent_threshold,
    )
    shape = np.atleast_2d(np.asarray(shape, dtype=np.float64))
    rate = np.atleast_2d(np.asarray(rate, dtype=np.float64))

    if specification.name == "sse_so":
        final_day = counts.size - 1
        tost_sum = float(
            renewal.delay_weighted_sum(
                counts.astype(np.float64), delays.tost, first_lag=TOST_FIRST_LAG
            )[final_day]
        )
        if tost_sum > 0.0:
            days = np.concatenate((days, [final_day]))
            shape = np.concatenate((shape, (k * tost_sum)[:, None]), axis=1)
            rate = np.concatenate((rate, np.asarray(k, dtype=np.float64)[:, None]), axis=1)
    return UnsampledLatents(days=np.asarray(days, dtype=np.int64), shape=shape, rate=rate)


@dataclass(frozen=True)
class LatentRiskBasis:
    """The coefficients of the latent path in the risk exponent, with ``R`` factored out.

    Every model's conditional zero-probability is *affine* in the latent path,

    ``log P(no further case after t | θ, Y) = −(α(θ) + Σ_u β_u(θ) Y_u)``,

    and ``β`` depends on the draw only through the reproduction numbers. Two pieces, because
    RAT scales them differently: :attr:`retained` is the remaining transmission of the cohorts
    already retained on day ``t``, and :attr:`pre`/:attr:`post` are the incubation pipeline,
    split by the period the *transmission* falls in. So

    ``β_cases = R_reset·retained + R_pre·pre + R_post·post``

    ``β_transmission = R_reset·retained + (1 − e^{−c})(R_pre·pre + R_post·post)``,

    ``β_sustained = R_reset·(1 − q)·retained + (1 − q)(R_pre·pre + R_post·post)``,

    matching the three exponents :func:`onset_event_probabilities` writes out. The naive models
    have no pipeline, so ``pre`` and ``post`` vanish and the two coincide, as RAC and RAT do.
    """

    days: NDArray[np.int64]
    retained: NDArray[np.float64]
    """``(n_days, n_latent_days)``, indexed by conditioning day then by the latent's day."""

    pre: NDArray[np.float64]
    post: NDArray[np.float64]

    def weights(
        self,
        *,
        R_reset: NDArray[np.float64],
        R_pre: NDArray[np.float64],
        R_post: NDArray[np.float64],
        per_case_exponent: NDArray[np.float64] | None = None,
        pipeline_probability: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """``β`` per draw, per conditioning day, per latent day.

        ``per_case_exponent`` is ``c``; pass it for RAT. ``pipeline_probability`` is
        ``1 - q``; pass it for RST. Omit both for RAC.
        """
        if per_case_exponent is not None and pipeline_probability is not None:
            raise ValueError("choose either a RAT exponent or an RST pipeline probability")
        pipeline = R_pre[:, None, None] * self.pre + R_post[:, None, None] * self.post
        if per_case_exponent is not None:
            pipeline = pipeline * (-np.expm1(-per_case_exponent))[:, None, None]
        if pipeline_probability is not None:
            pipeline = pipeline * pipeline_probability[:, None, None]
        return R_reset[:, None, None] * self.retained + pipeline


def latent_risk_basis(
    model: str | ModelSpecification,
    *,
    counts: NDArray[np.int64],
    delays: OnsetAnchoredDelays,
    switch_day: int,
    days: NDArray[np.int64] | None = None,
    latent_days: NDArray[np.int64] | None = None,
) -> LatentRiskBasis:
    """Build :class:`LatentRiskBasis` from the same operators the closed forms use.

    ``latent_days`` restricts the columns, which is what keeps the arrays small when only the
    unsampled latents are wanted. The identity ``β_u = logP(0) − logP(e_u)`` holds by
    linearity and is what pins this against the closed forms in the tests, so the two cannot
    drift apart.
    """
    specification = specification_of(model)
    if specification.latent_variable is None:
        raise ValueError(f"model {specification.name!r} has no latent block")
    counts = np.asarray(counts, dtype=np.int64)
    n_days = counts.size
    selected = np.arange(n_days, dtype=np.int64) if days is None else np.asarray(days, np.int64)
    columns = (
        np.arange(n_days, dtype=np.int64)
        if latent_days is None
        else np.asarray(latent_days, np.int64)
    )
    every_day = np.arange(n_days, dtype=np.int64)
    zero = np.zeros((selected.size, columns.size), dtype=np.float64)

    if specification.anchoring == "infections":
        # ssi: log P = −R_pre Λ_Y(t), and Λ_Y is the survival-weighted latent path.
        survival = renewal.delay_design_matrix(
            n_days,
            survival_weights(delays.serial_interval),
            first_lag=0,
            source_days=every_day,
        )
        return LatentRiskBasis(
            days=selected, retained=survival[np.ix_(selected, columns)], pre=zero, post=zero
        )

    incubation_survival = renewal.delay_design_matrix(
        n_days, survival_weights(delays.incubation), first_lag=0, source_days=every_day
    )
    period = renewal.switch_index(n_days, switch_day)
    if specification.name == "sse_so":
        # The latent is the transmission day's own infectivity, so the pipeline weight is the
        # incubation survival and the retained cohorts are the observed onsets — no latents.
        pipeline = incubation_survival[selected][:, columns]
        in_pre = (period[columns] == 0).astype(np.float64)
        return LatentRiskBasis(
            days=selected, retained=zero, pre=pipeline * in_pre, post=pipeline * (1.0 - in_pre)
        )

    # ssi_so: the latent is an onset cohort's infectivity, so it reaches the pipeline through
    # the TOST hop first and the retained exponent through the TOST survival function.
    tost_operator = renewal.delay_design_matrix(
        n_days, delays.tost, first_lag=TOST_FIRST_LAG, source_days=every_day
    )
    tost_survival = renewal.delay_design_matrix(
        n_days, tost_survival_weights(delays.tost), first_lag=TOST_FIRST_LAG, source_days=every_day
    )
    reach = incubation_survival[selected]
    pre_reach = np.where(period[None, :] == 0, reach, 0.0)
    return LatentRiskBasis(
        days=selected,
        retained=tost_survival[np.ix_(selected, columns)],
        pre=(pre_reach @ tost_operator)[:, columns],
        post=((reach - pre_reach) @ tost_operator)[:, columns],
    )


def marginalised_risk_correction(
    model: str | ModelSpecification,
    unsampled: UnsampledLatents,
    *,
    counts: NDArray[np.int64],
    delays: OnsetAnchoredDelays,
    switch_day: int,
    days: NDArray[np.int64],
    R_pre: NDArray[np.float64],
    R_post: NDArray[np.float64],
    k: NDArray[np.float64],
    reset_R: float | NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``log E[exp(−β·Y_unsampled) | θ]`` for RAC, RAT and RST, exactly.

    The unsampled latents are conditionally independent ``Gamma(A_u, B_u)`` given the
    parameters, and the risk exponent is affine in them, so the Gamma moment generating
    function integrates them out term by term:

    ``E[exp(−β_u Y_u) | θ] = (1 + β_u / B_u)^(−A_u)``.

    This is the same conditional law :func:`reconstruct_latent_paths` would draw from, used
    exactly instead of by Monte Carlo — so the correction adds no sampling error of its own and
    needs no random stream.
    """
    specification = specification_of(model)
    basis = latent_risk_basis(
        specification,
        counts=counts,
        delays=delays,
        switch_day=switch_day,
        days=days,
        latent_days=unsampled.days,
    )
    n_draws = R_pre.size
    reset = R_pre if reset_R is None else _broadcast_draws(reset_R, "reset_R", n_draws=n_draws)
    per_case_exponent = k * np.log1p(reset / k)
    extinction = branching_process.negative_binomial_extinction_probability(reset, k)

    rate = unsampled.rate[:, None, :]
    shape = unsampled.shape[:, None, :]
    cases = -(
        shape * np.log1p(basis.weights(R_reset=reset, R_pre=R_pre, R_post=R_post) / rate)
    ).sum(axis=2)
    transmission = -(
        shape
        * np.log1p(
            basis.weights(
                R_reset=reset, R_pre=R_pre, R_post=R_post, per_case_exponent=per_case_exponent
            )
            / rate
        )
    ).sum(axis=2)
    sustained = -(
        shape
        * np.log1p(
            basis.weights(
                R_reset=reset * (1.0 - extinction),
                R_pre=R_pre,
                R_post=R_post,
                pipeline_probability=1.0 - extinction,
            )
            / rate
        )
    ).sum(axis=2)
    return cases, transmission, sustained


def reconstruct_latent_paths(
    model: str | ModelSpecification,
    state: PosteriorState,
    *,
    counts: NDArray[np.int64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """A complete latent path per posterior draw, indexed by day — for validation only.

    Nothing on the results path calls this. :func:`risk_log_probabilities` integrates the
    unsampled latents out of the risk in closed form instead, which is exact where this is
    Monte Carlo. What still needs an explicit path is the matched-conditioning check of §6.4,
    where the *same* ``Y`` has to reach the analytic calculator and the forward simulator; a
    distribution cannot be handed to a simulator.

    Returns
    -------
    ``(n_draws, n_days)`` array, zero on days that carry no latent at all — for the
    individual-level models, the days with no cases, whose infectivity is identically zero.
    """
    specification = specification_of(model)
    if specification.latent_variable is None:
        raise ValueError(f"model {specification.name!r} has no latent block to rebuild")
    if state.sampled_infectivity is None:
        raise ValueError(f"the state carries no latent block for model {specification.name!r}")
    rng = np.random.default_rng() if rng is None else rng
    paths = np.array(state.sampled_infectivity, dtype=np.float64, copy=True)
    if paths.shape[1] != np.asarray(counts).size:
        raise ValueError("the sampled latent block and the counts disagree on the window length")
    if state.unsampled is not None and state.unsampled.days.size > 0:
        paths[:, state.unsampled.days] = rng.gamma(
            state.unsampled.shape, 1.0 / state.unsampled.rate
        )
    return paths


# ---------------------------------------------------------------------------------------
# The same quantity by forward simulation — the equality check of §6.4
# ---------------------------------------------------------------------------------------


def simulated_risk_curve(
    model: str | ModelSpecification,
    *,
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    parameters: TransmissionParameters,
    infectivity: NDArray[np.float64] | None = None,
    days: NDArray[np.int64] | None = None,
    n_replicates: int = 20_000,
    rng: np.random.Generator | None = None,
) -> RiskCurve:
    """RAC(t) estimated by forward simulation from the reset state, at one parameter draw.

    The Monte-Carlo counterpart of :func:`risk_curve`, and the regression test that the closed
    forms of §5.3 are the right arithmetic: replay the observed prefix ``counts[: t + 1]``,
    reset ``R`` to ``parameters.R_pre`` from day ``t + 1``, and count the replicates in which
    no further case appears.

    Simulating ``len(w)`` days past the conditioning day is **exact**, not a truncation: any
    offspring of the retained cases appears within the support of the serial interval, and a
    case appearing later would need a further case to descend from.

    Parameters
    ----------
    parameters
        ``R_post`` is ignored — the reset makes the whole counterfactual future run at
        ``R_pre`` by construction.
    infectivity
        For SSI, the latent path to condition on, of length ``n_days``. Supplying one makes
        this the matched-conditioning check of §6.4: the same ``Y`` reaches the analytic
        calculator and the simulator. Omit it to have the simulator draw the seeded
        infectivities from their exact conditional ``Gamma(k I_u, k)``.
    """
    specification = specification_of(model)
    counts = np.asarray(counts, dtype=np.int64)
    w = np.asarray(serial_interval, dtype=np.float64)
    days = np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days, np.int64)
    rng = np.random.default_rng() if rng is None else rng
    # The reset runs the whole counterfactual future at R_pre, so R_post never applies.
    reset = TransmissionParameters(R_pre=parameters.R_pre, R_post=parameters.R_pre, k=parameters.k)

    risk = np.empty(days.size, dtype=np.float64)
    for position, t in enumerate(days):
        prefix = counts[: int(t) + 1]
        seeded_infectivity = None if infectivity is None else np.asarray(infectivity)[: int(t) + 1]
        simulation = forward_simulation.simulate_naive(
            specification,
            reset,
            serial_interval=w,
            n_days=int(t) + 1 + w.size,
            switch_day=0,
            initial_counts=prefix,
            initial_infectivity=seeded_infectivity,
            n_replicates=n_replicates,
            rng=rng,
        )
        further = simulation.counts[:, int(t) + 1 :].sum(axis=1)
        risk[position] = float((further > 0).mean())
    return RiskCurve(days=days, risk=risk, n_draws=n_replicates)


@dataclass(frozen=True)
class OnsetRiskCurves:
    """Matched onset-anchored RAC and RAT curves."""

    rac: RiskCurve
    rat: RiskCurve


def simulated_onset_risk_curves(
    model: str | ModelSpecification,
    *,
    counts: NDArray[np.int64],
    delays: OnsetAnchoredDelays,
    switch_day: int,
    parameters: TransmissionParameters,
    latent: NDArray[np.float64] | None = None,
    reset_R: float | None = None,
    days: NDArray[np.int64] | None = None,
    n_replicates: int = 20_000,
    rng: np.random.Generator | None = None,
) -> OnsetRiskCurves:
    """Forward-simulation check of onset RAC/RAT at one fixed retained state.

    The incubation pipeline is constructed explicitly by Poisson splitting of every retained
    ``E_u`` into onset days after ``t``. Future transmission is then simulated day by day from
    that pipeline. Descendants of the first new infection need not be generated: once that
    infection occurs both RAT and RAC are already true, while before it occurs the only future
    onsets are exactly the retained pipeline just constructed.
    """
    specification = specification_of(model)
    if specification.anchoring != "onsets":
        raise ValueError(f"model {specification.name!r} is infection-anchored")
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least one")
    counts = np.asarray(counts, dtype=np.int64)
    selected_days = (
        np.arange(counts.size, dtype=np.int64) if days is None else np.asarray(days, dtype=np.int64)
    )
    rng = np.random.default_rng() if rng is None else rng
    k = parameters.require_k(specification) if specification.has_dispersion else None
    reset = parameters.R_pre if reset_R is None else float(reset_R)
    if reset < 0.0:
        raise ValueError("reset_R must be non-negative")

    retained_latent = None if latent is None else np.asarray(latent, dtype=np.float64)
    if specification.has_latents:
        if retained_latent is None or retained_latent.shape != counts.shape:
            raise ValueError(
                f"{specification.name} needs one complete retained latent path of shape "
                f"{counts.shape}"
            )
    elif retained_latent is not None:
        raise ValueError(f"{specification.name} has no latent block")

    R_history = renewal.reproduction_number_by_day(
        parameters.R_pre,
        parameters.R_post,
        n_days=counts.size,
        switch_day=switch_day,
    )
    tost_operator = renewal.delay_design_matrix(
        counts.size,
        delays.tost,
        first_lag=TOST_FIRST_LAG,
        source_days=np.arange(counts.size, dtype=np.int64),
    )
    if specification.name == "sse_so":
        assert retained_latent is not None
        expected_history = R_history * retained_latent
    elif specification.name == "ssi_so":
        assert retained_latent is not None
        expected_history = R_history * (retained_latent @ tost_operator.T)
    else:
        expected_history = R_history * (counts.astype(np.float64) @ tost_operator.T)

    rac_values = np.empty(selected_days.size, dtype=np.float64)
    rat_values = np.empty(selected_days.size, dtype=np.float64)
    for position, day_value in enumerate(selected_days):
        t = int(day_value)
        horizon = t + 1 + delays.incubation.size + delays.tost.size
        simulated_onsets = np.zeros((n_replicates, horizon), dtype=np.int64)
        simulated_onsets[:, : t + 1] = counts[: t + 1]

        # Independent Poisson thinning builds the complete residual incubation pipeline.
        for infection_day in range(t + 1):
            for offset, probability in enumerate(delays.incubation, start=1):
                onset_day = infection_day + offset
                if onset_day <= t:
                    continue
                simulated_onsets[:, onset_day] += rng.poisson(
                    expected_history[infection_day] * probability,
                    size=n_replicates,
                )
        pipeline_case = simulated_onsets[:, t + 1 :].sum(axis=1) > 0

        future_latent = (
            None
            if not specification.has_latents
            else np.zeros((n_replicates, horizon), dtype=np.float64)
        )
        if future_latent is not None:
            assert retained_latent is not None
            future_latent[:, : t + 1] = retained_latent[: t + 1]

        transmitted = np.zeros(n_replicates, dtype=bool)
        for transmission_day in range(t + 1, horizon):
            lags = min(transmission_day + 1, delays.tost.size)
            if specification.name in ("sse_so", "cori_so"):
                scale = (
                    simulated_onsets[:, transmission_day + 1 - lags : transmission_day + 1]
                    @ delays.tost[:lags][::-1]
                )
                if specification.name == "sse_so":
                    assert future_latent is not None and k is not None
                    positive = scale > 0.0
                    future_latent[positive, transmission_day] = rng.gamma(
                        k * scale[positive], 1.0 / k, size=int(positive.sum())
                    )
                    force = future_latent[:, transmission_day]
                else:
                    force = scale
            else:
                assert future_latent is not None and k is not None
                new_onsets = simulated_onsets[:, transmission_day]
                positive = new_onsets > 0
                future_latent[positive, transmission_day] = rng.gamma(
                    k * new_onsets[positive], 1.0 / k, size=int(positive.sum())
                )
                force = (
                    future_latent[:, transmission_day + 1 - lags : transmission_day + 1]
                    @ delays.tost[:lags][::-1]
                )
            transmitted |= rng.poisson(reset * force) > 0

        rat_values[position] = float(transmitted.mean())
        rac_values[position] = float((pipeline_case | transmitted).mean())

    return OnsetRiskCurves(
        rac=RiskCurve(days=selected_days, risk=rac_values, n_draws=n_replicates),
        rat=RiskCurve(days=selected_days, risk=rat_values, n_draws=n_replicates),
    )


# ---------------------------------------------------------------------------------------
# External validation: Thompson et al. (2024) under their own conventions
# ---------------------------------------------------------------------------------------


def thompson_reproduction_number_posterior(
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    *,
    last_day: int,
) -> tuple[float, float]:
    """``(shape, rate)`` of the Gamma posterior for ``R`` of Thompson et al., eq. (3).

    Their estimate of ``R`` in the absence of the ERT uses the incidence data up to the day
    **before** the ERT arrives, conditions on the index case as an importation, and takes a
    flat prior on ``R``. The Poisson renewal likelihood is then conjugate:

        ``shape = 1 + Σ_{t=1}^{last_day} I_t``,  ``rate = Σ_{t=1}^{last_day} Σ_s w_s I_{t−s}``.

    Parameters
    ----------
    last_day
        Final day of the estimation window, inclusive — day 32 for the Équateur series, the
        day before the ERT arrived on day 33.
    """
    counts = np.asarray(counts, dtype=np.int64)
    w = np.asarray(serial_interval, dtype=np.float64)
    if not 1 <= last_day < counts.size:
        raise ValueError(f"last_day must lie in 1..{counts.size - 1}, got {last_day}")
    window = slice(1, last_day + 1)
    force_of_infection = renewal.delay_weighted_sum(
        counts.astype(np.float64), w, first_lag=SERIAL_INTERVAL_FIRST_LAG
    )
    return 1.0 + float(counts[window].sum()), float(force_of_infection[window].sum())


def thompson_withdrawal_risk(
    counts: NDArray[np.int64],
    serial_interval: NDArray[np.float64],
    *,
    shape: float,
    rate: float,
    days: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """The risk of withdrawing the ERT on day ``t``, exactly as Thompson et al. define it.

    Their eq. (5), with ``R`` marginalised analytically over the Gamma posterior of eq. (3):

        ``risk(t) = 1 − (rate / (rate + γ(t)))^shape``,
        ``γ(t) = Σ_{u < t} I_u (1 − F_{t−u−1})``.

    Two conventions differ from this project's RAC (§5.1): the risk is of cases **on or after**
    day ``t``, and it conditions on data strictly **before** day ``t``. Both amount to the same
    one-day shift, ``γ(t) = Λ(t − 1)``, so their curve at day ``t`` is this project's curve at
    day ``t − 1`` under the Poisson limit and their ``R``. That identity is what makes this an
    external validation of the whole RAC pipeline rather than a re-implementation of it.
    """
    counts = np.asarray(counts, dtype=np.int64)
    days = np.arange(1, counts.size, dtype=np.int64) if days is None else np.asarray(days, np.int64)
    if (days < 1).any():
        raise ValueError("their risk is defined for days 1 onwards: γ(t) looks back from t − 1")
    Lambda = pooled_remaining_weight(counts.astype(np.float64), serial_interval)
    gamma = Lambda[days - 1]
    return 1.0 - (rate / (rate + gamma)) ** shape

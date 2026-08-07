"""The risk of additional cases (RAC), and the reset state it is computed from.

RAC is the project's headline quantity. It is a **retrospective reset posterior predictive**
(§5.1 of the implementation plan), not a filtering probability, and it is best read as a
procedure:

    Fit the parameters *and the latents* to the complete record, days 0–110. Then, for each
    day ``t``: retain the inferred state attached to the history through day ``t``, discard
    the realised trajectory after ``t``, reset ``R`` to ``R_pre`` from day ``t + 1`` onwards,
    and simulate a counterfactual future. RAC(t) is the posterior probability that this
    replicated future contains at least one further case.

Two consequences the shorthand ``P(· | data up to day t)`` hides: the parameters come from all
111 days by design, and for the latent models the retained state is **smoothed** — informed by
data after day ``t`` — because one fit serves every day of the curve (§5.6).

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

Rebuilding the latents first
----------------------------
For SSI and the onset-anchored models the retained state includes the latent
infectivities, and the chosen ``marginalised_inverse_cdf`` parameterisation integrates some of
them out exactly rather than sampling them. Those latents are **not** in ``idata.posterior``,
and on the real series they are exactly the days from the last observed case onwards — the days
a late conditioning day needs. :func:`reconstruct_latent_paths` rebuilds them from the closed
form of :func:`end_of_outbreak.pymc_models.marginalised_latent_conditional`, once per posterior
draw rather than once per conditioning day, and every calculator here consumes the complete
by-day path it returns. Reading the posterior array directly would silently truncate the state
and understate RAC.

RAT
---
The risk of additional *transmission* coincides with RAC under all three naive models — that
identification is the conflation the project is about — and separates from it only under the
onset-anchored models. Their calculators live beside the incubation-pipeline reconstruction
below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from end_of_outbreak import forward_simulation, pymc_models, renewal
from end_of_outbreak.delay_distributions import (
    SERIAL_INTERVAL_FIRST_LAG,
    TOST_FIRST_LAG,
    OnsetAnchoredDelays,
    cumulative,
)
from end_of_outbreak.latent_parameterisations import LatentParameterisation
from end_of_outbreak.model_specifications import (
    ModelSpecification,
    TransmissionParameters,
    specification_of,
)

_DRAW_CHUNK_ELEMENTS = 2_000_000
"""Rough size of the temporary the DLO profile calculation is allowed to build, in floats."""


# ---------------------------------------------------------------------------------------
# The remaining-transmission weight and the future force of infection
# ---------------------------------------------------------------------------------------


def survival_weights(serial_interval: NDArray[np.float64]) -> NDArray[np.float64]:
    """``1 − F_r`` for ``r = 0, 1, ...``: the chance a case's transmission is still to come.

    ``F_r = Σ_{s ≤ r} w_s`` with ``F_0 = 0``, so ``survival[0] = 1``: a case appearing on the
    conditioning day itself still has *all* of its transmission ahead of it. Stored densely
    from lag 0, so it composes with :func:`end_of_outbreak.renewal.delay_design_matrix` under
    ``first_lag=0``.
    """
    return np.clip(1.0 - cumulative(np.asarray(serial_interval, dtype=np.float64)), 0.0, 1.0)


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


def tost_survival_weights(tost: NDArray[np.float64]) -> NDArray[np.float64]:
    """Probability that a cohort's transmission occurs *after* each attained age.

    TOST is stored from lag zero, so a cohort observed on the conditioning day has already had
    its lag-0 transmission opportunity. Consequently the first entry is ``1 - f_tost[0]``, in
    contrast to :func:`survival_weights` for a lag-1 delay, whose first entry is one.
    """
    weights = np.asarray(tost, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("tost must be a non-empty one-dimensional array")
    return np.clip(1.0 - np.cumsum(weights), 0.0, 1.0)


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


@dataclass(frozen=True)
class OnsetEventProbabilities:
    """Per-draw zero-probabilities for onset-anchored RAC and RAT."""

    days: NDArray[np.int64]
    log_no_further_cases: NDArray[np.float64]
    """``log P(no onset after t | posterior draw)``."""

    log_no_further_transmission: NDArray[np.float64]
    """``log P(no transmission event after t | posterior draw)``."""

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
) -> OnsetEventProbabilities:
    """Conditional zero-probabilities defining RAC and RAT for an onset model.

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
            retained_exponent = (
                per_case_exponent[:, None]
                * remaining_tost_weight(counts.astype(np.float64), delays.tost)[None, :]
            )
        else:
            force = latent_paths @ tost_operator.T
            expected_infections = R_history * force
            retained_exponent = reset_draws[:, None] * remaining_tost_weight(
                latent_paths, delays.tost
            )

    pipeline = incubation_pipeline_mean(expected_infections, delays.incubation)
    log_no_cases = -retained_exponent - pipeline
    log_no_transmission = -retained_exponent - pipeline * (-np.expm1(-per_case_exponent))[:, None]
    if np.any(log_no_transmission + 1e-12 < log_no_cases):
        raise AssertionError("RAC/RAT zero-probability ordering was violated")
    return OnsetEventProbabilities(
        days=selected_days,
        log_no_further_cases=log_no_cases[:, selected_days],
        log_no_further_transmission=log_no_transmission[:, selected_days],
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


@dataclass(frozen=True)
class RiskCurve:
    """RAC over the conditioning days: one posterior-averaged probability per day.

    RAC(t) is a single number, not a distribution: it is the posterior *probability* of a
    further case, so the average over draws happens inside it (§5.1, and eq. (5) of Thompson
    et al. (2024), which does the same integral in closed form).
    """

    days: NDArray[np.int64]
    risk: NDArray[np.float64]
    """RAC(t): the probability of at least one further case after day ``t``."""

    n_draws: int

    @property
    def probability_of_no_further_cases(self) -> NDArray[np.float64]:
        """``1 − RAC(t)``, the quantity the closed forms of §5.3 actually evaluate."""
        return 1.0 - self.risk

    def first_day_below(self, threshold: float) -> int | None:
        """First conditioning day on which RAC falls below ``threshold`` and stays there.

        The decision-relevant summary of the curve: the report quantifies onset-anchoring by
        how far this date moves at the 0.05 and 0.01 thresholds (§5.7, Stage 9).
        """
        above = np.flatnonzero(self.risk >= threshold)
        settled = 0 if above.size == 0 else int(above[-1]) + 1
        return None if settled >= self.risk.size else int(self.days[settled])


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
class PosteriorState:
    """The posterior draws a RAC curve is built from, latents included and complete by day."""

    R_pre: NDArray[np.float64]
    R_post: NDArray[np.float64]
    k: NDArray[np.float64] | None
    infectivity: NDArray[np.float64] | None
    """``(n_draws, n_days)`` latent path, sampled and rebuilt parts spliced together."""

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
    rng: np.random.Generator | None = None,
) -> PosteriorState:
    """Flatten a fit into draw-indexed arrays, rebuilding the latents that were integrated out.

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
    rng
        Generator for the reconstruction draws.
    """
    specification = specification_of(model)
    posterior = idata.posterior
    n_draws = int(posterior.sizes["chain"] * posterior.sizes["draw"])
    R_pre = _parameter_draws(posterior, "R_pre", fixed=fixed_R_pre, n_draws=n_draws)
    R_post = _parameter_draws(posterior, "R_post", fixed=fixed_R_post, n_draws=n_draws)
    k = (
        _parameter_draws(posterior, "k", fixed=fixed_k, n_draws=n_draws)
        if specification.has_dispersion
        else None
    )

    infectivity = None
    if specification.has_latents:
        if latent_parameterisation is None:
            raise ValueError(
                f"model {specification.name!r} has a latent block, so the parameterisation it "
                "was fitted under must be given: it decides which latents need rebuilding"
            )
        assert k is not None  # every latent model in the project carries a dispersion
        infectivity = reconstruct_latent_paths(
            specification,
            idata,
            counts,
            delays=delays,
            switch_day=switch_day,
            latent_parameterisation=latent_parameterisation,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            negligible_latent_threshold=negligible_latent_threshold,
            rng=rng,
        )
    return PosteriorState(
        R_pre=R_pre,
        R_post=R_post,
        k=k,
        infectivity=infectivity,
        n_chains=int(posterior.sizes["chain"]),
    )


def risk_curve_from_posterior(
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
    days: NDArray[np.int64] | None = None,
    rng: np.random.Generator | None = None,
) -> RiskCurve:
    """RAC(t) straight from a fit: reconstruct the state, then average the closed form."""
    state = posterior_state(
        model,
        idata,
        counts,
        delays=delays,
        switch_day=switch_day,
        latent_parameterisation=latent_parameterisation,
        fixed_R_pre=fixed_R_pre,
        fixed_R_post=fixed_R_post,
        fixed_k=fixed_k,
        negligible_latent_threshold=negligible_latent_threshold,
        rng=rng,
    )
    return risk_curve(
        model,
        counts=counts,
        serial_interval=delays.serial_interval,
        R_pre=state.R_pre,
        k=state.k,
        infectivity=state.infectivity,
        days=days,
    )


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
# Rebuilding the latents the fit integrated out
# ---------------------------------------------------------------------------------------


def reconstruct_latent_paths(
    model: str | ModelSpecification,
    idata: Any,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    latent_parameterisation: str | LatentParameterisation,
    R_pre: NDArray[np.float64],
    R_post: NDArray[np.float64],
    k: NDArray[np.float64],
    negligible_latent_threshold: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """A complete latent path per posterior draw, indexed by day.

    The default parameterisation integrates out every latent the data constrain only through
    ``exp(−Σ_j μ_j)``, so ``idata.posterior`` holds only part of the block — and on the real
    series the missing part is exactly the run of days from the last observed case onwards,
    which is the range a late conditioning day needs. Those latents are conditionally
    independent of the sampled block and of each other given the parameters, so drawing each
    from its closed-form conditional ``Gamma(k · scale_u, k + c_u)`` gives *exact* draws from
    the full smoothed posterior (§5.1, §6.3).

    Done **once per posterior draw**, not once per conditioning day, so the
    one-fit-serves-every-day economy of §5.6 is untouched.

    Returns
    -------
    ``(n_draws, n_days)`` array, zero on days that carry no latent at all — for the
    individual-level models, the days with no cases, whose infectivity is identically zero.
    """
    specification = specification_of(model)
    if specification.latent_variable is None:
        raise ValueError(f"model {specification.name!r} has no latent block to rebuild")
    counts = np.asarray(counts, dtype=np.int64)
    rng = np.random.default_rng() if rng is None else rng

    sampled = _flatten_draws(idata.posterior, specification.latent_variable)
    n_draws = R_pre.size
    if sampled.shape[0] != n_draws:
        raise ValueError(f"the latent block has {sampled.shape[0]} draws but R_pre has {n_draws}")
    dimension = pymc_models.latent_dimension(specification)
    sampled_days = np.asarray(
        idata.posterior.data_vars[specification.latent_variable].coords[dimension].values,
        dtype=np.int64,
    )

    paths = np.zeros((n_draws, counts.size), dtype=np.float64)
    paths[:, sampled_days] = sampled

    rebuilt_days, shape, rate = pymc_models.marginalised_latent_conditional(
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
    if rebuilt_days.size > 0:
        paths[:, rebuilt_days] = rng.gamma(shape, 1.0 / rate)

    # SSE-SO's inference graph stops at transmission day T - 1: E_T cannot reach an observed
    # onset because f_inc starts at lag one. The reset state at conditioning day T nevertheless
    # needs E_T, whose infections are already in the incubation pipeline by the end of that
    # day. It is independent of the fitted likelihood and therefore retains its prior exactly.
    # Rebuild this predictive-boundary latent here so every returned path is genuinely complete
    # by calendar day, just as the marginalised in-window block is.
    if specification.name == "sse_so":
        final_day = counts.size - 1
        tost_sum = renewal.delay_weighted_sum(
            counts.astype(np.float64), delays.tost, first_lag=TOST_FIRST_LAG
        )[final_day]
        if tost_sum > 0.0:
            paths[:, final_day] = rng.gamma(k * tost_sum, 1.0 / k)
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

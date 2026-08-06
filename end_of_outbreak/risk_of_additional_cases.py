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
For SSI (and, from Stage 8, the onset-anchored models) the retained state includes the latent
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
onset-anchored models. The RAT calculators therefore arrive with the onset-anchored simulators
in Stage 8, alongside the incubation-pipeline reconstruction they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from end_of_outbreak import forward_simulation, pymc_models, renewal
from end_of_outbreak.delay_distributions import (
    SERIAL_INTERVAL_FIRST_LAG,
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
        One of ``dlo``, ``sse``, ``ssi``, ``cori``. The onset-anchored models have no closed
        form and arrive in Stage 8.
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
        raise NotImplementedError(
            f"model {specification.name!r} is onset-anchored: it has no closed-form RAC, and "
            "its Monte-Carlo calculators (with the incubation pipeline of §5.1) arrive in "
            "Stage 8"
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

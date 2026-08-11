"""Forward simulators for all renewal models.

One vectorised simulator covers DLO, SSE, SSI and the Cori (``k → ∞``) limit. It is the
generative counterpart of :mod:`end_of_outbreak.pymc_models`, and the pair is checked against
each other directly: simulate many replicates of a short history, and the empirical
distribution of ``I_{1:T}`` must match the likelihood the builder assigns to it.

The step for each model, with ``Λ_t`` the delay-weighted sum over earlier days:

``dlo``
    ``I_t ~ NB(mean = R_t Λ_t, disp = k)``, ``Λ_t = Σ_s w_s I_{t-s}``.
``sse``
    ``I_t ~ NB(mean = R_t Λ_t, disp = k Λ_t)``, same ``Λ_t``. In NumPy's parameterisation the
    success probability ``k / (k + R_t)`` is free of ``Λ_t`` — the NB-closure property that
    makes ``k`` an individual-level offspring dispersion.
``ssi``
    ``I_t ~ Poisson(R_t Σ_s w_s Y_{t-s})`` with ``Y_t | I_t ~ Gamma(k I_t, k)`` and
    ``Y_t = 0`` whenever ``I_t = 0``.
``cori``
    ``I_t ~ Poisson(R_t Λ_t)``.

Seeding
-------
``initial_counts`` is the observed prefix to condition on; simulation continues from the day
after it ends. For SSI the latent infectivities of the seeded days are drawn from
``Gamma(k I_u, k)``, which is *exactly* their conditional law given the counts: in the joint
density ``p(I, Y) = Π_u Gamma(Y_u; k I_u, k) · Π_t Poisson(I_t; R_t Σ_s w_s Y_{t-s})`` the
infectivity priors are conditionally independent given ``I``. (The same identity is what makes
the naive Monte-Carlo marginalisation used in the tests unbiased.)

``initial_infectivity`` overrides that draw with a supplied ``Y`` path. The RAC reset state of
§5.1 is a *posterior* state, not a prior one, so a simulator that redrew the seeded
infectivities from the prior would be answering a different question; passing the same ``Y``
to the analytic calculator and to the simulator is what makes the §6.4 equality check a
matched-conditioning comparison rather than a filtering-versus-smoothing one.

Onset-anchored models
---------------------
The onset models have two interchangeable simulators, which are the computational version of
the equivalence proof in §4 of the implementation plan:

``simulate_onset_anchored_convenient``
    Draws each onset directly from ``Poisson(Σ_a f_inc,a E_{t-a})``. This is the form used by
    the PyMC builders and is the fastest way to generate synthetic onset histories.
``simulate_onset_anchored_natural``
    Draws explicit infection counts ``J_t ~ Poisson(E_t)`` and independently assigns them an
    incubation delay. This exposes transmission events, and is therefore the form used to pin
    the RAC/RAT distinction. Poisson splitting makes it exactly equivalent to the convenient
    form, not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from end_of_outbreak import renewal
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import (
    ModelSpecification,
    TransmissionParameters,
    specification_of,
)


@dataclass(frozen=True)
class NaiveSimulation:
    """Replicated trajectories from one of the infection-anchored models."""

    counts: NDArray[np.int64]
    """``(n_replicates, n_days)`` simulated daily counts, seeded prefix included."""

    infectivity: NDArray[np.float64] | None
    """``(n_replicates, n_days)`` latent ``Y_t`` for SSI; ``None`` for the other models."""

    @property
    def n_replicates(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.counts.shape[1])

    @property
    def total_cases(self) -> NDArray[np.int64]:
        """Outbreak size of each replicate."""
        return self.counts.sum(axis=1)


@dataclass(frozen=True)
class OnsetSimulation:
    """Replicated trajectories from an onset-anchored model."""

    onsets: NDArray[np.int64]
    """``(n_replicates, n_days)`` symptom-onset counts, seeded prefix included."""

    expected_infections: NDArray[np.float64]
    """``E_t`` for every transmission day in the returned window."""

    latent: NDArray[np.float64] | None
    """``lambda_tilde`` (SSE-SO) or ``Y`` (SSI-SO); ``None`` for Cori-SO."""

    infections: NDArray[np.int64] | None = None
    """Explicit ``J_t`` in the natural form; ``None`` in the convenient form."""

    @property
    def counts(self) -> NDArray[np.int64]:
        """Alias used by code that treats infection- and onset-anchored simulations alike."""
        return self.onsets

    @property
    def n_replicates(self) -> int:
        return int(self.onsets.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.onsets.shape[1])

    @property
    def total_cases(self) -> NDArray[np.int64]:
        """Number of symptom onsets in each replicate."""
        return self.onsets.sum(axis=1)


def simulate_naive(
    model: str | ModelSpecification,
    parameters: TransmissionParameters,
    *,
    serial_interval: NDArray[np.float64],
    n_days: int,
    switch_day: int,
    initial_counts: NDArray[np.int64] | tuple[int, ...] = (1,),
    initial_infectivity: NDArray[np.float64] | None = None,
    n_replicates: int = 1,
    rng: np.random.Generator | None = None,
) -> NaiveSimulation:
    """Simulate replicate trajectories of ``dlo``, ``sse``, ``ssi`` or ``cori``.

    Parameters
    ----------
    model
        Model name or :class:`~end_of_outbreak.model_specifications.ModelSpecification`.
    parameters
        ``R_pre``, ``R_post`` and (except for ``cori``) ``k``.
    serial_interval
        Discrete serial interval ``w`` stored from lag 1.
    n_days
        Length of each trajectory, seeded prefix included.
    switch_day
        Day on which ``R`` changes from ``R_pre`` to ``R_post`` (§5.7).
    initial_counts
        Observed prefix to condition on. The default ``(1,)`` is the sole imported index case
        of §3.1.
    initial_infectivity
        Latent ``Y`` for the seeded days, for SSI only: either one path of length
        ``len(initial_counts)`` shared by every replicate, or one per replicate. Must vanish
        wherever ``initial_counts`` does. Omit it to draw from ``Gamma(k I_u, k)``, which is
        the exact conditional law given the counts alone; supply it to condition on a
        *posterior* state, as the RAC checks of §6.4 do.
    n_replicates
        Number of independent trajectories.
    rng
        NumPy generator; a fresh default one is used if omitted.
    """
    specification = specification_of(model)
    if specification.anchoring != "infections":
        raise ValueError(
            f"model {specification.name!r} is onset-anchored; use "
            "simulate_onset_anchored_convenient or simulate_onset_anchored_natural"
        )
    rng = np.random.default_rng() if rng is None else rng
    w = np.asarray(serial_interval, dtype=np.float64)
    seed = np.asarray(initial_counts, dtype=np.int64)

    if w.ndim != 1 or w.size == 0:
        raise ValueError("serial_interval must be a non-empty one-dimensional array")
    if seed.ndim != 1 or seed.size == 0:
        raise ValueError("initial_counts must be a non-empty one-dimensional array")
    if (seed < 0).any() or seed[0] < 1:
        raise ValueError("initial_counts must be non-negative and start with at least one case")
    if n_days < seed.size:
        raise ValueError(f"n_days ({n_days}) is shorter than initial_counts ({seed.size})")
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least one")

    k = parameters.require_k(specification) if specification.has_dispersion else None
    R_by_day = renewal.reproduction_number_by_day(
        parameters.R_pre, parameters.R_post, n_days=n_days, switch_day=switch_day
    )

    counts = np.zeros((n_replicates, n_days), dtype=np.int64)
    counts[:, : seed.size] = seed

    # SSI is driven by the latent infectivities rather than by the counts themselves. The
    # seeded days' infectivities are drawn from their exact conditional law given the counts.
    uses_latent_infectivity = specification.latent_variable is not None
    if initial_infectivity is not None and not uses_latent_infectivity:
        raise ValueError(
            f"model {specification.name!r} has no latent infectivity to seed; it is driven by "
            "the counts themselves"
        )
    infectivity: NDArray[np.float64] | None = None
    if uses_latent_infectivity:
        assert k is not None  # guaranteed: every latent model here has a dispersion
        infectivity = np.zeros((n_replicates, n_days), dtype=np.float64)
        if initial_infectivity is None:
            for day in np.flatnonzero(seed > 0):
                infectivity[:, day] = rng.gamma(k * seed[day], 1.0 / k, size=n_replicates)
        else:
            infectivity[:, : seed.size] = _validated_seed_infectivity(
                initial_infectivity, seed, n_replicates=n_replicates
            )
        driving = infectivity
    else:
        driving = counts.astype(np.float64)

    for day in range(seed.size, n_days):
        lags = min(day, w.size)
        force_of_infection = driving[:, day - lags : day] @ w[:lags][::-1]
        new_counts = draw_counts(
            specification,
            force_of_infection,
            R=float(R_by_day[day]),
            k=k,
            rng=rng,
        )
        counts[:, day] = new_counts
        if uses_latent_infectivity:
            assert infectivity is not None and k is not None
            positive = new_counts > 0
            if positive.any():
                infectivity[positive, day] = rng.gamma(
                    k * new_counts[positive], 1.0 / k, size=int(positive.sum())
                )
        else:
            driving[:, day] = new_counts

    return NaiveSimulation(counts=counts, infectivity=infectivity)


def simulate_onset_anchored(
    model: str | ModelSpecification,
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    n_days: int,
    switch_day: int,
    initial_onsets: NDArray[np.int64] | tuple[int, ...] = (1,),
    initial_latent: NDArray[np.float64] | None = None,
    n_replicates: int = 1,
    form: str = "convenient",
    rng: np.random.Generator | None = None,
) -> OnsetSimulation:
    """Simulate ``sse_so``, ``ssi_so`` or ``cori_so`` in either equivalent form.

    ``form="convenient"`` draws the onset process directly and accepts an observed prefix.
    ``form="natural"`` exposes infection counts and consequently starts from the sole imported
    index-case onset of §3.1; conditioning an explicit infection-allocation history on a longer
    onset prefix would require that unobserved allocation history as additional state.
    """
    if form == "convenient":
        return simulate_onset_anchored_convenient(
            model,
            parameters,
            delays=delays,
            n_days=n_days,
            switch_day=switch_day,
            initial_onsets=initial_onsets,
            initial_latent=initial_latent,
            n_replicates=n_replicates,
            rng=rng,
        )
    if form == "natural":
        return simulate_onset_anchored_natural(
            model,
            parameters,
            delays=delays,
            n_days=n_days,
            switch_day=switch_day,
            initial_onsets=initial_onsets,
            initial_latent=initial_latent,
            n_replicates=n_replicates,
            rng=rng,
        )
    raise ValueError(f"form must be 'convenient' or 'natural', got {form!r}")


def simulate_onset_anchored_convenient(
    model: str | ModelSpecification,
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    n_days: int,
    switch_day: int,
    initial_onsets: NDArray[np.int64] | tuple[int, ...] = (1,),
    initial_latent: NDArray[np.float64] | None = None,
    n_replicates: int = 1,
    rng: np.random.Generator | None = None,
) -> OnsetSimulation:
    """Simulate the inference-convenient onset recursion of §3.5.

    The incubation period starts at lag one, so the onset on day ``t`` is drawn entirely from
    ``E_{<t}``; its latent and ``E_t`` can then be drawn without an algebraic loop even though
    the TOST distribution may carry mass at lag zero.
    """
    specification, seed, k, R_by_day, rng = _onset_simulation_inputs(
        model,
        parameters,
        delays=delays,
        n_days=n_days,
        switch_day=switch_day,
        initial_onsets=initial_onsets,
        n_replicates=n_replicates,
        rng=rng,
    )
    onsets = np.zeros((n_replicates, n_days), dtype=np.int64)
    onsets[:, : seed.size] = seed
    expected_infections = np.zeros((n_replicates, n_days), dtype=np.float64)
    latent = (
        None
        if specification.latent_variable is None
        else np.zeros((n_replicates, n_days), dtype=np.float64)
    )

    supplied = _validated_onset_seed_latent(
        specification,
        initial_latent,
        seed,
        delays=delays,
        n_replicates=n_replicates,
    )
    for day in range(seed.size):
        _draw_onset_latent_and_expected_infections(
            specification,
            day,
            onsets=onsets,
            latent=latent,
            expected_infections=expected_infections,
            R=float(R_by_day[day]),
            k=k,
            tost=delays.tost,
            supplied=None if supplied is None else supplied[:, day],
            rng=rng,
        )

    for day in range(seed.size, n_days):
        lags = min(day, delays.incubation.size)
        mean_onsets = expected_infections[:, day - lags : day] @ delays.incubation[:lags][::-1]
        onsets[:, day] = rng.poisson(mean_onsets)
        _draw_onset_latent_and_expected_infections(
            specification,
            day,
            onsets=onsets,
            latent=latent,
            expected_infections=expected_infections,
            R=float(R_by_day[day]),
            k=k,
            tost=delays.tost,
            supplied=None,
            rng=rng,
        )

    return OnsetSimulation(
        onsets=onsets,
        expected_infections=expected_infections,
        latent=latent,
    )


def simulate_onset_anchored_natural(
    model: str | ModelSpecification,
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    n_days: int,
    switch_day: int,
    initial_onsets: NDArray[np.int64] | tuple[int, ...] = (1,),
    initial_latent: NDArray[np.float64] | None = None,
    n_replicates: int = 1,
    rng: np.random.Generator | None = None,
) -> OnsetSimulation:
    """Simulate explicit infections and their incubation allocations (§4).

    Independent Poisson splitting is used in place of drawing ``J_t`` followed by one large
    multinomial: the lag-specific counts are independent
    ``Poisson(E_t f_inc,a)`` and their sum is exactly ``Poisson(E_t)``. The resulting onset
    trajectory therefore has the same law as :func:`simulate_onset_anchored_convenient` while
    retaining the actual transmission counts needed for RAT.
    """
    specification, seed, k, R_by_day, rng = _onset_simulation_inputs(
        model,
        parameters,
        delays=delays,
        n_days=n_days,
        switch_day=switch_day,
        initial_onsets=initial_onsets,
        n_replicates=n_replicates,
        rng=rng,
    )
    if seed.size != 1:
        raise ValueError(
            "the natural-form simulator needs the explicit infection-allocation state to "
            "condition on more than the sole day-0 index case; use the convenient form for "
            "an observed onset prefix"
        )

    onsets = np.zeros((n_replicates, n_days), dtype=np.int64)
    onsets[:, 0] = seed[0]
    expected_infections = np.zeros((n_replicates, n_days), dtype=np.float64)
    infections = np.zeros((n_replicates, n_days), dtype=np.int64)
    latent = (
        None
        if specification.latent_variable is None
        else np.zeros((n_replicates, n_days), dtype=np.float64)
    )
    supplied = _validated_onset_seed_latent(
        specification,
        initial_latent,
        seed,
        delays=delays,
        n_replicates=n_replicates,
    )

    for day in range(n_days):
        _draw_onset_latent_and_expected_infections(
            specification,
            day,
            onsets=onsets,
            latent=latent,
            expected_infections=expected_infections,
            R=float(R_by_day[day]),
            k=k,
            tost=delays.tost,
            supplied=(supplied[:, 0] if supplied is not None and day == 0 else None),
            rng=rng,
        )

        # Draw every incubation allocation, including the portion that appears beyond this
        # simulation window. Their sum is the explicit infection count J_t.
        for offset, probability in enumerate(delays.incubation, start=1):
            allocated = rng.poisson(expected_infections[:, day] * probability)
            infections[:, day] += allocated
            onset_day = day + offset
            if onset_day < n_days:
                onsets[:, onset_day] += allocated

    return OnsetSimulation(
        onsets=onsets,
        expected_infections=expected_infections,
        latent=latent,
        infections=infections,
    )


def _onset_simulation_inputs(
    model: str | ModelSpecification,
    parameters: TransmissionParameters,
    *,
    delays: OnsetAnchoredDelays,
    n_days: int,
    switch_day: int,
    initial_onsets: NDArray[np.int64] | tuple[int, ...],
    n_replicates: int,
    rng: np.random.Generator | None,
) -> tuple[
    ModelSpecification,
    NDArray[np.int64],
    float | None,
    NDArray[np.float64],
    np.random.Generator,
]:
    """Validate common onset-simulator inputs and resolve their derived values."""
    specification = specification_of(model)
    if specification.anchoring != "onsets":
        raise ValueError(f"model {specification.name!r} is infection-anchored; use simulate_naive")
    seed = np.asarray(initial_onsets, dtype=np.int64)
    if seed.ndim != 1 or seed.size == 0:
        raise ValueError("initial_onsets must be a non-empty one-dimensional array")
    if (seed < 0).any() or seed[0] < 1:
        raise ValueError("initial_onsets must be non-negative and start with at least one case")
    if n_days < seed.size:
        raise ValueError(f"n_days ({n_days}) is shorter than initial_onsets ({seed.size})")
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least one")
    for weights, name in ((delays.tost, "f_tost"), (delays.incubation, "f_inc")):
        if weights.ndim != 1 or weights.size == 0 or (weights < 0).any():
            raise ValueError(f"{name} must be a non-empty non-negative one-dimensional array")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"{name} must sum to one")
    k = parameters.require_k(specification) if specification.has_dispersion else None
    R_by_day = renewal.reproduction_number_by_day(
        parameters.R_pre, parameters.R_post, n_days=n_days, switch_day=switch_day
    )
    return (
        specification,
        seed,
        k,
        R_by_day,
        np.random.default_rng() if rng is None else rng,
    )


def _draw_onset_latent_and_expected_infections(
    specification: ModelSpecification,
    day: int,
    *,
    onsets: NDArray[np.int64],
    latent: NDArray[np.float64] | None,
    expected_infections: NDArray[np.float64],
    R: float,
    k: float | None,
    tost: NDArray[np.float64],
    supplied: NDArray[np.float64] | None,
    rng: np.random.Generator,
) -> None:
    """Draw one day's latent after its onset is known, then compute ``E_t``."""
    lags = min(day + 1, tost.size)
    if specification.name in ("sse_so", "cori_so"):
        tost_sum = onsets[:, day + 1 - lags : day + 1] @ tost[:lags][::-1]
        if specification.name == "cori_so":
            expected_infections[:, day] = R * tost_sum
            return
        assert latent is not None and k is not None
        if supplied is None:
            positive = tost_sum > 0.0
            latent[positive, day] = rng.gamma(
                k * tost_sum[positive], 1.0 / k, size=int(positive.sum())
            )
        else:
            latent[:, day] = supplied
        expected_infections[:, day] = R * latent[:, day]
        return

    assert specification.name == "ssi_so" and latent is not None and k is not None
    if supplied is None:
        positive = onsets[:, day] > 0
        latent[positive, day] = rng.gamma(
            k * onsets[positive, day], 1.0 / k, size=int(positive.sum())
        )
    else:
        latent[:, day] = supplied
    expected_infections[:, day] = R * (latent[:, day + 1 - lags : day + 1] @ tost[:lags][::-1])


def _validated_onset_seed_latent(
    specification: ModelSpecification,
    initial_latent: NDArray[np.float64] | None,
    seed: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    n_replicates: int,
) -> NDArray[np.float64] | None:
    """Broadcast and validate a retained latent path for the seeded onset prefix."""
    if initial_latent is None:
        return None
    if specification.latent_variable is None:
        raise ValueError(f"model {specification.name!r} has no latent block to seed")
    supplied = np.asarray(initial_latent, dtype=np.float64)
    if supplied.ndim == 1:
        supplied = np.broadcast_to(supplied, (n_replicates, supplied.size))
    if supplied.shape != (n_replicates, seed.size):
        raise ValueError(
            f"initial_latent must have shape ({seed.size},) or "
            f"({n_replicates}, {seed.size}), got {supplied.shape}"
        )
    if (supplied < 0).any():
        raise ValueError("initial_latent must be non-negative")
    if specification.name == "ssi_so" and (supplied[:, seed == 0] != 0).any():
        raise ValueError("SSI-SO initial_latent must vanish on days with no onsets")
    if specification.name == "sse_so":
        scale = renewal.delay_weighted_sum(seed.astype(np.float64), delays.tost, first_lag=0)
        if (supplied[:, scale == 0.0] != 0).any():
            raise ValueError("SSE-SO initial_latent must vanish where the TOST sum is zero")
    return supplied


def _validated_seed_infectivity(
    initial_infectivity: NDArray[np.float64],
    seed: NDArray[np.int64],
    *,
    n_replicates: int,
) -> NDArray[np.float64]:
    """A supplied seed path, broadcast to one row per replicate and checked against the counts."""
    supplied = np.asarray(initial_infectivity, dtype=np.float64)
    if supplied.ndim == 1:
        supplied = np.broadcast_to(supplied, (n_replicates, supplied.size))
    if supplied.shape != (n_replicates, seed.size):
        raise ValueError(
            f"initial_infectivity must have shape ({seed.size},) or "
            f"({n_replicates}, {seed.size}), got {supplied.shape}"
        )
    if (supplied < 0).any():
        raise ValueError("initial_infectivity must be non-negative")
    if (supplied[:, seed == 0] != 0).any():
        raise ValueError(
            "initial_infectivity must vanish on days with no cases: Y_u | I_u = 0 is a point "
            "mass at zero, so a positive value there is not a state the model can be in"
        )
    return supplied


def draw_counts(
    specification: ModelSpecification,
    force_of_infection: NDArray[np.float64],
    *,
    R: float,
    k: float | None,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """One day's counts for every replicate. Zero force of infection gives zero counts.

    Public because the particle filter needs it too: under incomplete reporting the true counts
    are latent, so the filter has to *draw* each day rather than read it, and it must draw from
    the same law the model generates from. Reusing this is what keeps the two from drifting.
    """
    new_counts = np.zeros(force_of_infection.size, dtype=np.int64)
    driven = force_of_infection > 0.0
    if not driven.any():
        return new_counts
    mean_incidence = R * force_of_infection[driven]

    if specification.name == "cori" or specification.name == "ssi":
        new_counts[driven] = rng.poisson(mean_incidence)
    elif specification.name == "dlo":
        assert k is not None
        # Constant dispersion: the NB "number of successes" is k on every day, and only the
        # success probability moves with the force of infection.
        new_counts[driven] = rng.negative_binomial(k, k / (k + mean_incidence))
    else:  # sse
        assert k is not None
        # Dispersion proportional to the force of infection: the success probability is the
        # same on every day, and the counts pool by NB closure.
        new_counts[driven] = rng.negative_binomial(k * force_of_infection[driven], k / (k + R))
    return new_counts

"""Forward simulators for the infection-anchored models.

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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from end_of_outbreak import renewal
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
        raise NotImplementedError(
            f"model {specification.name!r} is onset-anchored; the onset-anchored simulators "
            "land in Stage 8"
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
        new_counts = _draw_counts(
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


def _draw_counts(
    specification: ModelSpecification,
    force_of_infection: NDArray[np.float64],
    *,
    R: float,
    k: float | None,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """One day's counts for every replicate. Zero force of infection gives zero counts."""
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

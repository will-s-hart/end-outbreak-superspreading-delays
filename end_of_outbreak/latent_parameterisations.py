"""Strategies for the latent Gamma blocks, and the machinery to build one.

Every latent block in the project has the same shape. A vector of independent Gammas

    Y_u ~ Gamma(shape = k · scale_u, rate = k),

with ``scale_u`` a **known constant** (``I_u`` for SSI, ``D_u`` for SSI-SO,
``λ_u = Σ_s f_tost,s D_{u-s}`` for SSE-SO), drives a Poisson observation model in which each
``Y_u`` enters linearly. Sampling that block is the main technical risk of the project (§6.3
of the implementation plan), and this module holds the candidate strategies plus the one
routine that builds a block under any of them.

Two headings, because they answer different questions
-----------------------------------------------------
*Initialisation strategies* change where the sampler starts, not what it explores:

``rescale_to_unit_mean``
    Sample ``Ỹ_u ~ Gamma(k scale_u, k scale_u)`` and recover ``Y_u = scale_u · Ỹ_u``. Since
    PyMC already samples positive variables on the log scale, this is a pure **translation**
    there: the Gamma *shape* is unchanged, so the log-density curvature NUTS has to navigate
    is identical. It moves every coordinate's starting point to the same place, nothing more.
``initialise_at_geometric_mean``
    PyMC starts a Gamma at its **arithmetic** mean ``α/β``. For a small shape that is far
    above the typical value — ``E[log Y] = ψ(α) − log β`` is roughly ``−1/α`` while
    ``log(α/β)`` is ``O(1)`` — so the chain begins in the far right tail of exactly the
    coordinates that are hardest. Starting at ``exp(ψ(α))/β`` puts it at the typical value.

*Genuine reparameterisations* change the space the sampler works in:

``inverse_cdf``
    Sample ``u_u ~ Uniform(0, 1)`` and set ``Y_u = F⁻¹(u_u)``. A bounded, flat-prior,
    constant-geometry space. The ``u`` are i.i.d. uniform under the **prior** only — the
    likelihood couples them exactly as it couples the Gammas — but the geometry is far
    tamer. PyTensor differentiates the Gamma quantile with respect to ``u`` but not its shape,
    so fixed-``k`` fits run the whole sampled block under NUTS, while estimated-``k`` fits use
    Slice for ``k`` and NUTS for the ``R`` parameters and latent uniforms. That compound sampler
    is the configuration the estimated-``k`` benchmark validated.
``marginalise_uncoupled``
    Integrate out, in closed form, every latent that the data constrain only through the
    ``exp(−Σ_j μ_j)`` factor. See below: this is exact, not an approximation.

Composability, and one provable no-op
-------------------------------------
The switches combine freely, with one exception worth stating rather than benchmarking:
**inverse-CDF and mean-1 rescaling do not compose into anything new.** The Gamma quantile
function is scale-equivariant, ``scale · F⁻¹_{Gamma(α, α)}(u) = F⁻¹_{Gamma(α, k)}(u)`` for
``α = k · scale``, so rescaling composes with the ``u`` map as a constant factor and leaves the
sampled variable's target identical. Stacking them is exactly a no-op; ``tests`` pins that.

Marginalising the uncoupled latents
-----------------------------------
The log-likelihood of a Poisson observation model splits into a ``−Σ_j μ_j`` part, which is
always **linear** in the latents and therefore factorises across them, and a
``Σ_j D_j log μ_j`` part, which couples them. A latent that appears in no ``log μ_j`` term —
because every observation day it can reach carries a zero count — is therefore conditionally
independent of everything else given the parameters, with likelihood contribution
``exp(−c_u Y_u)`` for a known coefficient ``c_u``. Its Gamma prior is conjugate to that:

    ∫ Gamma(y; k scale_u, k) e^{−c_u y} dy = (1 + c_u/k)^{−k scale_u},
    Y_u | data ~ Gamma(shape = k scale_u, rate = k + c_u).

So the latent can be removed from the sampler and replaced by a ``pm.Potential`` carrying the
exact factor, with **no approximation whatsoever**, and recovered afterwards from the closed
form (:func:`conditional_posterior`). On the Équateur series this removes 52 of SSE-SO's 110
latents — precisely the pathological tail, whose shapes run down to 2.2 × 10⁻⁶ — and leaves 58
whose smallest shape is 0.024 at ``k = 0.18``.

Dropping negligible latents
---------------------------
``negligible_threshold`` is the separate, *approximate* knob of §6.3: latents whose ``scale_u``
falls below it are removed outright with ``Y_u`` set to 0, at a cost bounded by
``max(R) · Σ_dropped scale_u``. It is applied only to latents that survive marginalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.special
from numpy.typing import NDArray

Reparameterisation = Literal["centred", "inverse_cdf"]

LOG_UNDERFLOW = float(np.log(np.finfo(np.float64).tiny))
"""Smallest ``log x`` a double can represent (about −708).

Relevant because ``E[log Y] ≈ −1/α`` for a small Gamma shape: below ``α ≈ 1.4 × 10⁻³`` the
whole typical set of the latent lies outside the representable range, so there is no starting
point to hand the sampler and no coordinate value that is not effectively zero. That is a hard
numerical wall, not a tuning problem — see the SSE-SO block, whose untouched shapes reach
2.7 × 10⁻⁶.
"""


@dataclass(frozen=True)
class LatentParameterisation:
    """One point in the strategy space of §6.3, as four independent switches."""

    name: str
    reparameterisation: Reparameterisation = "centred"
    rescale_to_unit_mean: bool = False
    marginalise_uncoupled: bool = False
    initialise_at_geometric_mean: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.reparameterisation == "inverse_cdf" and self.rescale_to_unit_mean:
            raise ValueError(
                "inverse-CDF and mean-1 rescaling do not compose: the Gamma quantile function "
                "is scale-equivariant, so rescaling leaves the sampled u unchanged. Stacking "
                "them is a no-op — use the plain inverse_cdf parameterisation."
            )

    @property
    def changes_the_starting_point(self) -> bool:
        """Whether this is an *initialisation* strategy (§6.3's first heading)."""
        return self.rescale_to_unit_mean or self.initialise_at_geometric_mean

    @property
    def changes_the_space(self) -> bool:
        """Whether this is a *genuine reparameterisation* (§6.3's second heading)."""
        return self.reparameterisation != "centred" or self.marginalise_uncoupled


CENTRED = LatentParameterisation(
    name="centred",
    description="Baseline: pm.Gamma latents sampled as written, with PyMC's default start.",
)

CENTRED_GEOMETRIC_INIT = LatentParameterisation(
    name="centred_geometric_init",
    initialise_at_geometric_mean=True,
    description="Baseline geometry, started at exp(E[log Y]) instead of at E[Y].",
)

UNIT_MEAN = LatentParameterisation(
    name="unit_mean",
    rescale_to_unit_mean=True,
    description="Mean-1 rescaling: every coordinate starts at the same place on the log scale.",
)

UNIT_MEAN_GEOMETRIC_INIT = LatentParameterisation(
    name="unit_mean_geometric_init",
    rescale_to_unit_mean=True,
    initialise_at_geometric_mean=True,
    description="Both initialisation fixes at once, to check they are not redundant.",
)

INVERSE_CDF = LatentParameterisation(
    name="inverse_cdf",
    reparameterisation="inverse_cdf",
    description="Sample Uniform(0, 1) and push through the Gamma quantile function.",
)

MARGINALISED = LatentParameterisation(
    name="marginalised",
    marginalise_uncoupled=True,
    description="Centred, with every latent the data cannot resolve integrated out exactly.",
)

MARGINALISED_UNIT_MEAN = LatentParameterisation(
    name="marginalised_unit_mean",
    rescale_to_unit_mean=True,
    marginalise_uncoupled=True,
    description="Exact marginalisation stacked on the mean-1 initialisation fix.",
)

MARGINALISED_INVERSE_CDF = LatentParameterisation(
    name="marginalised_inverse_cdf",
    reparameterisation="inverse_cdf",
    marginalise_uncoupled=True,
    description="Exact marginalisation stacked on the inverse-CDF reparameterisation.",
)

PARAMETERISATIONS: dict[str, LatentParameterisation] = {
    parameterisation.name: parameterisation
    for parameterisation in (
        CENTRED,
        CENTRED_GEOMETRIC_INIT,
        UNIT_MEAN,
        UNIT_MEAN_GEOMETRIC_INIT,
        INVERSE_CDF,
        MARGINALISED,
        MARGINALISED_UNIT_MEAN,
        MARGINALISED_INVERSE_CDF,
    )
}
"""The candidates the Stage-3 benchmark compares, by name."""


def parameterisation_of(
    parameterisation: str | LatentParameterisation,
) -> LatentParameterisation:
    """Resolve a name to a :class:`LatentParameterisation`; pass one through unchanged.

    ``None`` is deliberately **not** accepted. ``config/config.yaml`` carries
    ``latent_parameterisation`` explicitly so that no script can silently fall back to a
    default that was never benchmarked.
    """
    if isinstance(parameterisation, LatentParameterisation):
        return parameterisation
    try:
        return PARAMETERISATIONS[parameterisation]
    except KeyError as error:
        known = ", ".join(sorted(PARAMETERISATIONS))
        raise KeyError(
            f"unknown latent parameterisation {parameterisation!r}; known: {known}"
        ) from error


# ---------------------------------------------------------------------------------------
# Which latents are sampled, which are integrated out, which are dropped
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentClassification:
    """How each position of a latent block is treated. Positions index the block, not days."""

    sampled: NDArray[np.int64]
    """Positions given a random variable."""

    marginalised: NDArray[np.int64]
    """Positions integrated out exactly, replaced by a ``pm.Potential``."""

    dropped: NDArray[np.int64]
    """Positions removed approximately, with the latent set to 0 (``negligible_threshold``)."""

    inactive: NDArray[np.int64]
    """Positions whose ``scale_u`` is 0, so the latent is identically 0 anyway."""

    n_latents: int = field(default=0)

    @property
    def has_sampled_latents(self) -> bool:
        return self.sampled.size > 0


def classify_latents(
    *,
    scale: NDArray[np.float64],
    couples_to_observations: NDArray[np.bool_],
    parameterisation: LatentParameterisation,
    negligible_threshold: float = 0.0,
) -> LatentClassification:
    """Split a latent block into sampled, exactly marginalised, dropped and inactive parts.

    Parameters
    ----------
    scale
        The known per-latent multiplier: the Gamma is ``Gamma(k · scale_u, k)``.
    couples_to_observations
        Whether latent ``u`` appears in some ``log μ_j`` term, i.e. can reach an observation
        day with a **positive** count. Latents that do not are exactly marginalisable.
    parameterisation
        Supplies ``marginalise_uncoupled``.
    negligible_threshold
        Latents with ``0 < scale_u <= negligible_threshold`` that are *not* marginalised are
        dropped outright. ``0.0`` drops nothing.
    """
    scale = np.asarray(scale, dtype=np.float64)
    couples_to_observations = np.asarray(couples_to_observations, dtype=bool)
    if scale.shape != couples_to_observations.shape:
        raise ValueError("scale and couples_to_observations must have the same shape")
    if negligible_threshold < 0.0:
        raise ValueError("negligible_threshold must be non-negative")

    positions = np.arange(scale.size, dtype=np.int64)
    active = scale > 0.0
    marginalised = active & ~couples_to_observations & parameterisation.marginalise_uncoupled
    dropped = active & ~marginalised & (scale <= negligible_threshold)
    return LatentClassification(
        sampled=positions[active & ~marginalised & ~dropped],
        marginalised=positions[marginalised],
        dropped=positions[dropped],
        inactive=positions[~active],
        n_latents=int(scale.size),
    )


# ---------------------------------------------------------------------------------------
# Building the block
# ---------------------------------------------------------------------------------------


def build_gamma_latent_block(
    name: str,
    *,
    k: Any,
    scale: NDArray[np.float64],
    coupling: Any,
    classification: LatentClassification,
    parameterisation: LatentParameterisation,
    dims: str,
) -> Any:
    """Build one latent Gamma block inside the enclosing ``pm.Model``.

    Parameters
    ----------
    name
        Name of the recovered latent vector (``Y``, ``lambda_tilde``).
    k
        Dispersion — a float or a PyMC random variable.
    scale
        Per-latent Gamma shape multiplier; ``alpha = k · scale``.
    coupling
        Symbolic vector ``c_u``: the coefficient of ``Y_u`` in ``Σ_j μ_j``. Only its
        marginalised entries are used, but it is passed whole so callers need not subset it.
    classification
        Output of :func:`classify_latents`, whose ``sampled`` positions must match the
        model coordinate named by ``dims``.
    parameterisation
        The strategy to build under.
    dims
        Model dimension naming the sampled positions.

    Returns
    -------
    A symbolic vector of length ``classification.n_latents``, zero at every position that is
    marginalised, dropped or inactive.
    """
    sampled = classification.sampled
    values = pt.zeros(classification.n_latents)

    if sampled.size > 0:
        sampled_scale = np.asarray(scale, dtype=np.float64)[sampled]
        drawn = _sampled_gamma(
            name,
            k=k,
            scale=sampled_scale,
            dims=dims,
            parameterisation=parameterisation,
        )
        values = pt.set_subtensor(values[sampled], drawn)

    marginalised = classification.marginalised
    if marginalised.size > 0:
        # The exact conjugate factor (1 + c_u/k)^(-k scale_u) for every latent the data
        # constrain only through exp(-Σ_j μ_j). No approximation; see the module docstring.
        marginalised_scale = np.asarray(scale, dtype=np.float64)[marginalised]
        pm.Potential(
            f"{name}_marginalised",
            -(k * marginalised_scale * pt.log1p(coupling[marginalised] / k)).sum(),
        )

    return values


def declare_inverse_cdf_uniform(name: str, *, dims: str) -> Any:
    """The free ``Uniform(0, 1)`` behind an inverse-CDF block.

    Split out from :func:`_sampled_gamma` because its law does not mention the Gamma scale, so
    it can be declared **before** anything the scale depends on. That is what lets the
    under-reporting models — whose scale is a function of the latent true counts — order their
    graph at all; see :mod:`end_of_outbreak.reporting`. It is also why under-reporting requires
    an inverse-CDF strategy: under ``centred`` or ``rescale_to_unit_mean`` the free variable's
    own prior involves the scale, and no ordering exists.
    """
    uniform: Any = pm.Uniform(f"{name}_uniform", lower=0.0, upper=1.0, dims=dims)
    return uniform


def gamma_from_uniform(uniform: Any, *, k: Any, scale: Any, guard_zero_scale: bool = False) -> Any:
    """``Gamma(k · scale, k)`` evaluated at its quantile ``uniform``.

    ``scale`` may be symbolic, which is what the under-reporting path needs.

    ``guard_zero_scale`` is for that path alone. A latent whose scale is a *random* count can
    have ``scale_u = 0`` on a day with no true case, where the block degenerates to a point
    mass at zero and the Gamma quantile is undefined. Flooring the shape keeps the gradient
    finite and the switch pins the value at the correct 0. The complete-reporting builders pass
    ``False``, so their graph is exactly what it was before this argument existed — the scales
    there are data, and :func:`classify_latents` has already dropped every zero.
    """
    alpha = k * scale
    if not guard_zero_scale:
        return pm.icdf(pm.Gamma.dist(alpha=alpha, beta=k), uniform)
    floored = pt.maximum(alpha, np.finfo(np.float64).tiny)
    return pt.switch(pt.gt(alpha, 0.0), pm.icdf(pm.Gamma.dist(alpha=floored, beta=k), uniform), 0.0)


def _sampled_gamma(
    name: str,
    *,
    k: Any,
    scale: NDArray[np.float64],
    dims: str,
    parameterisation: LatentParameterisation,
) -> Any:
    """The sampled part of a block: ``Gamma(k · scale, k)`` under one parameterisation."""
    alpha = k * scale
    if parameterisation.reparameterisation == "inverse_cdf":
        uniform = declare_inverse_cdf_uniform(name, dims=dims)
        return pm.Deterministic(name, gamma_from_uniform(uniform, k=k, scale=scale), dims=dims)
    if parameterisation.rescale_to_unit_mean:
        # Gamma(alpha, alpha) has mean 1; multiplying by the known constant `scale` recovers
        # the original law exactly. An affine change of variables by a constant, so no
        # Jacobian is involved -- and on the log scale it is a pure translation.
        unit_mean: Any = pm.Gamma(f"{name}_unit_mean", alpha=alpha, beta=alpha, dims=dims)
        return pm.Deterministic(name, scale * unit_mean, dims=dims)
    gamma: Any = pm.Gamma(name, alpha=alpha, beta=k, dims=dims)
    return gamma


def free_variable_name(name: str, parameterisation: LatentParameterisation) -> str:
    """Name of the *sampled* variable behind a latent block called ``name``.

    The recovered latent is a ``pm.Deterministic`` under the rescaled and inverse-CDF
    parameterisations, so the free variable — the one ``initvals`` and the sampler statistics
    refer to — is not always ``name``.
    """
    if parameterisation.reparameterisation == "inverse_cdf":
        return f"{name}_uniform"
    if parameterisation.rescale_to_unit_mean:
        return f"{name}_unit_mean"
    return name


def suggested_initial_values(
    name: str,
    *,
    k: float,
    sampled_scale: NDArray[np.float64],
    parameterisation: LatentParameterisation,
) -> dict[str, NDArray[np.float64]]:
    """``initvals`` for a latent block, or an empty dict if PyMC's default is being used.

    Only ``initialise_at_geometric_mean`` produces anything. It starts each latent at
    ``exp(E[log Y]) = exp(ψ(α))/β`` rather than at PyMC's default ``α/β``: for a small shape
    the two differ by orders of magnitude, and the arithmetic mean sits far out in the right
    tail of the very coordinates that are hardest to sample.

    Parameters
    ----------
    name
        Name of the latent block, as passed to :func:`build_gamma_latent_block`.
    k
        Must be a number: the initial point is computed before sampling, so when ``k`` is
        estimated the caller supplies a representative value (its prior median).
    sampled_scale
        ``scale_u`` for the **sampled** latents only, in the order the model's coordinate
        lists them (see :func:`end_of_outbreak.pymc_models.model_days`).
    parameterisation
        The strategy the block was built under.
    """
    sampled_scale = np.asarray(sampled_scale, dtype=np.float64)
    if not parameterisation.initialise_at_geometric_mean or sampled_scale.size == 0:
        return {}
    alpha = k * sampled_scale
    log_geometric_mean = scipy.special.digamma(alpha) - np.log(
        alpha if parameterisation.rescale_to_unit_mean else k
    )
    if not np.isfinite(log_geometric_mean).all() or (log_geometric_mean < LOG_UNDERFLOW).any():
        smallest = float(alpha.min())
        raise ValueError(
            f"the typical value of a Gamma({smallest:.3g}) latent is about "
            f"exp({-1 / smallest:.3g}), which underflows to zero in double precision, so "
            "there is no initial point to supply. This is a property of the block itself, not "
            "of the initialisation strategy: any shape below about 1.4e-3 has its whole "
            "typical set outside the representable range. Marginalise the uncoupled latents, "
            "or drop them."
        )
    return {free_variable_name(name, parameterisation): np.exp(log_geometric_mean)}


def conditional_posterior(
    *,
    k: float | NDArray[np.float64],
    scale: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(shape, rate)`` of a marginalised latent's exact conditional posterior.

    A latent removed by :func:`build_gamma_latent_block` is not lost: conditional on the
    parameters it is ``Gamma(k · scale_u, k + c_u)``, so a posterior draw of it can be
    reconstructed after the fit from a parameter draw. That is what the RAC calculators need
    in order to rebuild the state at the conditioning day.

    Everything here is elementwise, so ``k`` and ``coupling`` may carry a leading draw axis
    (against a ``scale`` that does not) and a whole posterior is converted at once.
    """
    scale = np.asarray(scale, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64)
    return k * scale, k + coupling

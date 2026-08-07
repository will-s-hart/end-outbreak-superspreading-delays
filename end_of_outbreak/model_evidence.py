"""Marginal likelihoods and posterior model probabilities (§6.5).

Panels 1C and 3C compare the models by ``p(D_{1:110} | model)`` — the observed series with
**everything** integrated out: the parameters, and for the latent models the latents too. This
module computes that number, and turns a set of them into posterior model probabilities under
the uniform prior over models of §6.2.

The density being normalised
----------------------------
The unnormalised density whose normalising constant is the evidence is the **built model's own
joint log-density**, priors times latent priors times likelihood. So the evidence is computed
from ``pymc_models.build_model`` called with the arguments the fit was run with, rather than
from a likelihood written out again here: that guarantees the integral being estimated is the
one that was actually sampled, and it is the reason ``R_pre``, ``R_post`` and ``k`` have to be
passed exactly as they were passed to :func:`end_of_outbreak.fitting.fit_model` — a ``float``
where the parameter was fixed and contributes no prior factor, a
:class:`~end_of_outbreak.model_specifications.LogNormalPrior` where it was estimated and does.
A fixed parameter is a constant in the graph, so neither the prior nor the value can be
recovered from the draws.

The integral is done in **PyMC's unconstrained coordinates** — ``log R``, and the interval
transform of the inverse-CDF uniforms — with the Jacobian included, because those are the
coordinates the posterior draws live in and the ones the Gaussian proposals below are
defensible in. The change of variables leaves the normalising constant alone; getting its
*direction* wrong would shift every evidence by a constant, which is exactly the quantity being
estimated, so :class:`UnconstrainedTarget` is pinned against
:func:`end_of_outbreak.pymc_models.compile_joint_logp` in the tests rather than trusted.

Marginalised latents do not change the answer
---------------------------------------------
Under the default ``marginalised_inverse_cdf`` parameterisation the builder integrates the
uncoupled latents out in closed form, so the built model has *fewer* value variables than under
plain ``inverse_cdf`` — 30 rather than 31 for SSI on the real series. The normalising constant
is nevertheless identical, because that marginalisation is exact: the ``pm.Potential`` carries
the whole factor ``(1 + c_u/k)^{−k·scale_u}`` those latents contribute. Nothing has to be
reconstructed here; unlike the RAC calculators, the evidence never needs the removed latents
back. The invariance is a test rather than a claim.

The three estimators
--------------------
``bridge_sampling``
    Meng & Wong's optimal bridge, in the log-space iterative form of Gronau et al. (2017), with
    a moment-matched multivariate-normal proposal. The default, and what the pipeline uses.
``prior_monte_carlo``
    Draw from the prior, average the likelihood. Unbiased and needs no posterior at all, but its
    variance is dictated by how far the posterior has moved from the prior, so on a real series
    it is a cross-check with a wide interval rather than a competitor.
``importance_sampling``
    A moment-matched *independent* (diagonal) normal proposal. Deliberately a different family
    from the bridge proposal, so that agreement between the two is evidence about the answer
    rather than about a shared covariance estimate.

Everything is on the log scale throughout; nothing is exponentiated except inside a
``logsumexp``.

What the resulting model probabilities do and do not say
--------------------------------------------------------
They compare the models **as specified, priors included** — nothing more (§6.5). For Fig. 1 in
particular, ``k = 0.18`` is applied to DLO deliberately, as a demonstration of a mistake made in
the literature, so panel 1C is a statement about a set of models one of which is knowingly
miscalibrated. It is not clean evidence about the mechanism of transmission heterogeneity, and
the report has to say so in the caption as well as in the methods.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy.linalg
import scipy.special
from numpy.typing import NDArray
from pymc.blocking import DictToArrayBijection

from end_of_outbreak import pymc_models
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.latent_parameterisations import LatentParameterisation
from end_of_outbreak.model_specifications import (
    LogNormalPrior,
    ModelSpecification,
    specification_of,
)

BRIDGE_SAMPLING = "bridge_sampling"
PRIOR_MONTE_CARLO = "prior_monte_carlo"
IMPORTANCE_SAMPLING = "importance_sampling"

ESTIMATORS: tuple[str, ...] = (BRIDGE_SAMPLING, PRIOR_MONTE_CARLO, IMPORTANCE_SAMPLING)
"""The estimators :func:`log_evidence` offers, the default first."""

_BRIDGE_TOLERANCE = 1e-10
"""Convergence threshold on ``|log r_{t+1} − log r_t|`` for the bridge recursion, in nats."""

_BRIDGE_MAX_ITERATIONS = 1000

_MINIMUM_VARIANCE = 1e-12
"""Floor on a proposal variance, so a coordinate the data pin exactly does not divide by zero."""

_IMPORTANCE_SAMPLING_ESS_WARNING = 0.01
"""Warn when the importance weights' effective sample size falls below this fraction."""


# ---------------------------------------------------------------------------------------
# The result type
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogEvidence:
    """One estimate of ``log p(D_{1:110} | model)``, with the error it was measured to."""

    model: str
    """Specification name, e.g. ``"ssi"``."""

    estimator: str
    """One of :data:`ESTIMATORS`."""

    log_evidence: float
    standard_error: float
    """Standard error **on the log scale**, or ``nan`` if the estimator cannot supply one.

    What it covers differs by estimator and is documented on each; in every case it is the
    Monte-Carlo error of this estimate given these posterior draws, and in no case does it
    cover an error in the posterior itself.
    """

    n_draws: int
    """Monte-Carlo draws the estimate averages over.

    The two proposal-based estimators count their proposal draws; bridge sampling counts both
    sample sets, since its recursion consumes the posterior half and the proposal draws
    together.
    """


# ---------------------------------------------------------------------------------------
# The unnormalised density, in the coordinates PyMC samples
# ---------------------------------------------------------------------------------------


class UnconstrainedTarget:
    """A built model's joint log-density as a function of its unconstrained value variables.

    PyMC samples ``log R`` rather than ``R``, and the interval transform of a ``Uniform(0, 1)``
    rather than the uniform itself. Those are the coordinates the posterior draws live in, and
    the only ones in which a Gaussian proposal is a sensible thing to fit — so the whole of this
    module works there, with the transforms' Jacobians folded into the density so that its
    integral over ``R^d`` is still ``p(D | model)``.

    Three log-densities are exposed, all as functions of a ``(n_points, dimension)`` array of
    unconstrained coordinates:

    ``log_density``
        The whole joint: priors, latent priors, the exact-marginalisation potential and the
        likelihood, with the Jacobian. This is the integrand.
    ``log_prior``
        The free random variables' prior terms alone, with the Jacobian.
    ``log_likelihood``
        The difference — the observation term plus the marginalisation potential, the Jacobians
        having cancelled. It is what prior Monte Carlo averages. The potential belongs on this
        side rather than with the prior: it carries the removed latents' prior *and* their
        contribution to the likelihood, and its integral against the sampled block's prior is
        the evidence itself.

    Exposed rather than kept private because the direct-marginalisation checks of §6.5 —
    :func:`log_evidence_by_quadrature` in the tests and in ``validation/`` — need the same
    integrand that bridge sampling is estimating the normalising constant of.
    """

    def __init__(self, model: pm.Model) -> None:
        layout = DictToArrayBijection.map(model.initial_point()).point_map_info
        random_variable_by_value_name = {model.rvs_to_values[rv].name: rv for rv in model.free_RVs}
        value_variable_by_name = {variable.name: variable for variable in model.value_vars}

        self._model = model
        self._names = [str(name) for name, _, _, _ in layout]
        self._shapes = [tuple(int(extent) for extent in shape) for _, shape, _, _ in layout]
        sizes = [int(size) for _, _, size, _ in layout]
        if any(dtype != np.float64 for _, _, _, dtype in layout):
            raise ValueError(
                "every value variable must be float64: the compiled densities are called with "
                "input checking disabled, which is only safe for an exactly-typed point"
            )

        self.dimension = int(sum(sizes))
        if self.dimension == 0:
            raise ValueError(
                "the model has no free variables, so there is nothing to marginalise and the "
                "evidence is the likelihood itself — use pymc_models.compile_observation_logp"
            )
        offsets = np.cumsum([0, *sizes])
        self._slices = [slice(int(start), int(stop)) for start, stop in itertools.pairwise(offsets)]
        self._random_variables = [random_variable_by_value_name[name] for name in self._names]
        self._forward = [_forward_transform(model, rv) for rv in self._random_variables]

        inputs = [value_variable_by_name[name] for name in self._names]
        self._joint = _compile(inputs, model.logp(jacobian=True))
        self._prior = _compile(inputs, model.logp(vars=model.free_RVs, jacobian=True))

    # -- evaluation ----------------------------------------------------------------------

    def log_density(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """``log p(D, latents, parameters)`` at each row, Jacobian included."""
        return self._evaluate(self._joint, points)

    def log_prior(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """The free variables' prior terms alone, Jacobian included."""
        return self._evaluate(self._prior, points)

    def log_likelihood(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """``log p(D | latents, parameters)``, the marginalisation potential included."""
        return self.log_density(points) - self.log_prior(points)

    def _evaluate(
        self, compiled: Callable[..., Any], points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        rows = self._as_rows(points)
        return np.array([float(compiled(*self._unpack(row))) for row in rows], dtype=np.float64)

    def _as_rows(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        rows = np.ascontiguousarray(points, dtype=np.float64)
        rows = rows[None, :] if rows.ndim == 1 else rows
        if rows.ndim != 2 or rows.shape[1] != self.dimension:
            raise ValueError(
                f"points must have {self.dimension} unconstrained coordinates per row, got "
                f"shape {np.shape(points)}"
            )
        return rows

    def _unpack(self, row: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """One raveled point as the per-variable arrays the compiled densities expect."""
        return [
            row[window].reshape(shape)
            for window, shape in zip(self._slices, self._shapes, strict=True)
        ]

    # -- getting draws into these coordinates --------------------------------------------

    def posterior_points(self, idata: Any) -> NDArray[np.float64]:
        """The posterior draws of ``idata``, pushed forward into these coordinates.

        Chains are stacked **chain-major**, matching
        :func:`end_of_outbreak.risk_of_additional_cases.posterior_state`, so that the
        ``(chain, draw)`` structure can be recovered by reshaping — which the bridge standard
        error needs in order to charge for the draws' autocorrelation.

        Raises if the fit does not carry every free variable of the built model. That mismatch
        almost always means a parameter was fixed in one and estimated in the other, which would
        otherwise produce a plausible-looking evidence for a model nobody fitted.
        """
        posterior = idata.posterior
        columns = []
        for random_variable, shape, forward in zip(
            self._random_variables, self._shapes, self._forward, strict=True
        ):
            name = random_variable.name
            if name not in posterior.data_vars:
                available = ", ".join(sorted(posterior.data_vars))
                raise ValueError(
                    f"the fit has no {name!r} in its posterior, but the model built here has it "
                    f"as a free variable. The posterior holds: {available}. Pass R_pre, R_post "
                    "and k exactly as they were passed to fitting.fit_model — a float where the "
                    "parameter was fixed, a LogNormalPrior where it was estimated"
                )
            natural = np.asarray(
                posterior[name].stack(draw_index=("chain", "draw")).transpose("draw_index", ...),
                dtype=np.float64,
            )
            if natural.shape[1:] != shape:
                raise ValueError(
                    f"the fit's {name!r} has shape {natural.shape[1:]} per draw but the model "
                    f"built here expects {shape}; the two were built from different data or a "
                    "different latent parameterisation"
                )
            mapped = natural if forward is None else np.asarray(forward(natural))
            columns.append(mapped.reshape(natural.shape[0], -1))
        return np.concatenate(columns, axis=1)

    def prior_points(self, n_draws: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """``n_draws`` independent draws from the prior, in these coordinates."""
        drawn = pm.draw(self._random_variables, draws=int(n_draws), random_seed=rng)
        sampled = drawn if isinstance(drawn, list) else [drawn]
        columns = []
        for values, shape, forward in zip(sampled, self._shapes, self._forward, strict=True):
            natural = np.asarray(values, dtype=np.float64).reshape((int(n_draws), *shape))
            mapped = natural if forward is None else np.asarray(forward(natural))
            columns.append(mapped.reshape(int(n_draws), -1))
        return np.concatenate(columns, axis=1)


def _compile(inputs: list[Any], output: Any) -> Callable[..., Any]:
    """Compile a log-density with per-call input validation switched off.

    Measured on the real-series SSI model: 1.4 ms per call through PyTensor's input filtering,
    0.013 ms without it. The estimators here evaluate the density tens of thousands of times, so
    the difference decides whether a prior-Monte-Carlo run takes seconds or minutes.
    ``UnconstrainedTarget._unpack`` is the only thing that ever builds these arguments, and the
    constructor has already refused any model whose value variables are not float64.
    """
    compiled = pytensor.function(inputs, output, on_unused_input="ignore")
    compiled.trust_input = True
    return compiled


def _forward_transform(model: pm.Model, random_variable: Any) -> Callable[..., Any] | None:
    """Map natural-scale draws of one random variable to their unconstrained image.

    ``None`` when PyMC samples the variable untransformed. The compiled function takes an extra
    leading draw axis, which works because every transform this project uses — the log transform
    on ``R`` and ``k``, the interval transform on the inverse-CDF uniforms — is elementwise; a
    structured transform such as a simplex would need the draws fed through one at a time.
    """
    transform = model.rvs_to_transforms.get(random_variable)
    if transform is None:
        return None
    n_dimensions = int(random_variable.type.ndim) + 1
    placeholder = pt.tensor(
        f"{random_variable.name}_draws",
        dtype=random_variable.dtype,
        shape=(None,) * n_dimensions,
    )
    return pytensor.function(
        [placeholder],
        transform.forward(placeholder, *random_variable.owner.inputs),
        on_unused_input="ignore",
    )


# ---------------------------------------------------------------------------------------
# Gaussian proposals in the unconstrained space
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _GaussianProposal:
    """A multivariate normal, held by its mean and a lower-triangular Cholesky factor."""

    mean: NDArray[np.float64]
    cholesky: NDArray[np.float64]

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def log_density(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        deviation = np.atleast_2d(np.asarray(points, dtype=np.float64)) - self.mean
        whitened = scipy.linalg.solve_triangular(self.cholesky, deviation.T, lower=True)
        log_determinant = float(np.log(np.diag(self.cholesky)).sum())
        return (
            -0.5 * (whitened**2).sum(axis=0)
            - log_determinant
            - 0.5 * self.dimension * np.log(2.0 * np.pi)
        )

    def draw(self, n_draws: int, rng: np.random.Generator) -> NDArray[np.float64]:
        standard = rng.standard_normal((int(n_draws), self.dimension))
        return self.mean + standard @ self.cholesky.T


def _fit_gaussian_proposal(points: NDArray[np.float64], *, independent: bool) -> _GaussianProposal:
    """Moment-match a normal to the unconstrained draws, full-covariance or diagonal.

    The diagonal fit is the *independent* proposal of §6.5 — the companion repo's
    independent-Gamma proposal, transplanted to coordinates where the defensible family is
    normal. Keeping the two estimators on different families is deliberate: agreement between
    bridge sampling and importance sampling is then evidence about the evidence, not about a
    shared covariance estimate.

    The Cholesky factor is taken after adding the smallest ridge that makes the covariance
    numerically positive definite. A ridge is needed in practice because posterior draws of a
    latent block can be very nearly collinear; it widens the proposal slightly, which costs
    variance and never bias.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    mean = points.mean(axis=0)
    if independent:
        variance = np.maximum(points.var(axis=0, ddof=1), _MINIMUM_VARIANCE)
        return _GaussianProposal(mean=mean, cholesky=np.diag(np.sqrt(variance)))

    covariance = np.atleast_2d(np.cov(points, rowvar=False, ddof=1))
    scale = float(np.trace(covariance)) / covariance.shape[0]
    for exponent in range(-12, -2):
        ridge = 0.0 if exponent == -12 else scale * 10.0**exponent
        try:
            cholesky = scipy.linalg.cholesky(
                covariance + ridge * np.eye(covariance.shape[0]), lower=True
            )
        except scipy.linalg.LinAlgError:
            continue
        return _GaussianProposal(mean=mean, cholesky=cholesky)
    raise ValueError(
        "the posterior covariance of the unconstrained draws is not positive definite even "
        "after ridging; the fit is probably degenerate"
    )


# ---------------------------------------------------------------------------------------
# The three estimators
# ---------------------------------------------------------------------------------------


def _log_mean_exp(values: NDArray[np.float64]) -> float:
    return float(scipy.special.logsumexp(values) - np.log(values.size))


def _self_normalised_standard_error(log_weights: NDArray[np.float64]) -> tuple[float, float]:
    """``(standard error on the log scale, effective sample size)`` of a weighted average.

    For weights ``w_i`` the estimate is ``log mean(w)``; the delta method gives a log-scale
    standard error of ``sd(w) / (mean(w) √n)``, which rearranges to ``√(1/ESS − 1/n)`` with the
    usual ``ESS = (Σw)² / Σw²``. Writing it that way keeps the whole computation in logs and
    makes the diagnosis and the error the same number, which is the point: an importance
    estimate is exactly as trustworthy as its effective sample size.
    """
    n_draws = int(log_weights.size)
    log_total = float(scipy.special.logsumexp(log_weights))
    log_total_squared = float(scipy.special.logsumexp(2.0 * log_weights))
    effective = float(np.exp(2.0 * log_total - log_total_squared))
    variance = 1.0 / effective - 1.0 / n_draws
    return float(np.sqrt(max(variance, 0.0))), effective


def _prior_monte_carlo(
    target: UnconstrainedTarget, *, n_draws: int, rng: np.random.Generator
) -> tuple[float, float, int]:
    """Average the likelihood over draws from the prior.

    Unbiased and needs no posterior, which is what makes it a genuinely independent check: it
    shares nothing with bridge sampling except the built model. Its variance is set by how far
    the data have moved the posterior from the prior, so on a long series the interval it
    reports is wide and *should* be — a narrow one would mean the prior draws had never found
    the posterior at all, which is the failure mode the effective sample size in the standard
    error is there to expose.
    """
    points = target.prior_points(n_draws, rng)
    log_likelihood = target.log_likelihood(points)
    standard_error, _ = _self_normalised_standard_error(log_likelihood)
    return _log_mean_exp(log_likelihood), standard_error, int(n_draws)


def _importance_sampling(
    target: UnconstrainedTarget,
    posterior_points: NDArray[np.float64],
    *,
    n_proposal_draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Importance sampling from an independent normal matched to the posterior marginals."""
    proposal = _fit_gaussian_proposal(posterior_points, independent=True)
    draws = proposal.draw(n_proposal_draws, rng)
    log_weights = target.log_density(draws) - proposal.log_density(draws)
    standard_error, effective = _self_normalised_standard_error(log_weights)
    if effective < _IMPORTANCE_SAMPLING_ESS_WARNING * n_proposal_draws:
        warnings.warn(
            f"the independent-normal proposal gives an effective sample size of "
            f"{effective:.1f} out of {n_proposal_draws} draws; the posterior's coordinates are "
            "correlated enough that this estimate is dominated by a handful of weights. Treat "
            "it as a rough agreement check on bridge sampling, not as an estimate in its own "
            "right",
            RuntimeWarning,
            stacklevel=3,
        )
    return _log_mean_exp(log_weights), standard_error, int(n_proposal_draws)


def _bridge_sampling(
    target: UnconstrainedTarget,
    posterior_points: NDArray[np.float64],
    *,
    n_chains: int,
    n_proposal_draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Meng & Wong's optimal bridge estimator, iterated to its fixed point.

    The proposal is a full-covariance normal moment-matched to the **first half of each chain**,
    and the recursion runs on the second half. Fitting the proposal on draws that do not enter
    the estimate removes the small bias of tuning a proposal on the very sample it is then
    weighted against (Gronau et al. 2017); splitting within each chain rather than across chains
    keeps both halves representative of the whole posterior, which matters when only two chains
    were run.
    """
    per_chain = posterior_points.reshape(n_chains, -1, posterior_points.shape[1])
    if per_chain.shape[1] < 4:
        raise ValueError(
            f"bridge sampling needs at least 4 draws per chain to split the fitting draws from "
            f"the estimation draws, got {per_chain.shape[1]}"
        )
    midpoint = per_chain.shape[1] // 2
    fitting_points = per_chain[:, :midpoint].reshape(-1, posterior_points.shape[1])
    estimation_points = per_chain[:, midpoint:]

    proposal = _fit_gaussian_proposal(fitting_points, independent=False)
    posterior_half = estimation_points.reshape(-1, posterior_points.shape[1])
    proposal_draws = proposal.draw(n_proposal_draws, rng)

    # ℓ = log(unnormalised target) − log(proposal), on each sample set in turn.
    log_ratio_posterior = target.log_density(posterior_half) - proposal.log_density(posterior_half)
    log_ratio_proposal = target.log_density(proposal_draws) - proposal.log_density(proposal_draws)
    estimate = _bridge_recursion(log_ratio_posterior, log_ratio_proposal)
    standard_error = _bridge_standard_error(
        log_ratio_posterior,
        log_ratio_proposal,
        log_evidence=estimate,
        n_chains=n_chains,
        n_per_chain=estimation_points.shape[1],
    )
    return estimate, standard_error, int(posterior_half.shape[0] + n_proposal_draws)


def _bridge_recursion(
    log_ratio_posterior: NDArray[np.float64], log_ratio_proposal: NDArray[np.float64]
) -> float:
    """The fixed point of Meng & Wong's optimal bridge, iterated in log space.

    With ``ℓ_i = log q(θ_i)/g(θ_i)`` on posterior draws and ``ℓ*_j`` on proposal draws,

        ``r ← [N₂⁻¹ Σ_j e^{ℓ*_j} / (s₁ e^{ℓ*_j} + s₂ r)] / [N₁⁻¹ Σ_i 1 / (s₁ e^{ℓ_i} + s₂ r)]``

    with ``s₁ = N₁/(N₁+N₂)``. Everything is shifted by ``ℓ* = median(ℓ)`` before exponentiating,
    which keeps ``r`` near 1 whatever the absolute scale of the evidence — on the real series the
    unshifted ratio is ``e^{−80}`` or smaller and the recursion would stall at zero.

    Note that this is *not* the recursion in ``sse-ssi-pmo/src/sse_ssi_pmo/evidence.py``, which
    divides through by ``e^{ℓ}`` rather than by ``g``; that form has a different fixed point.
    """
    n_posterior = int(log_ratio_posterior.size)
    n_proposal = int(log_ratio_proposal.size)
    log_s1 = np.log(n_posterior / (n_posterior + n_proposal))
    log_s2 = np.log(n_proposal / (n_posterior + n_proposal))

    shift = float(np.median(log_ratio_posterior))
    shifted_posterior = log_ratio_posterior - shift
    shifted_proposal = log_ratio_proposal - shift

    log_r = 0.0
    for _ in range(_BRIDGE_MAX_ITERATIONS):
        log_numerator = scipy.special.logsumexp(
            shifted_proposal - np.logaddexp(log_s1 + shifted_proposal, log_s2 + log_r)
        ) - np.log(n_proposal)
        log_denominator = scipy.special.logsumexp(
            -np.logaddexp(log_s1 + shifted_posterior, log_s2 + log_r)
        ) - np.log(n_posterior)
        updated = float(log_numerator - log_denominator)
        if abs(updated - log_r) < _BRIDGE_TOLERANCE:
            return updated + shift
        log_r = updated
    warnings.warn(
        f"the bridge recursion did not converge to {_BRIDGE_TOLERANCE:g} in "
        f"{_BRIDGE_MAX_ITERATIONS} iterations; the estimate is the last iterate",
        RuntimeWarning,
        stacklevel=3,
    )
    return log_r + shift


def _bridge_standard_error(
    log_ratio_posterior: NDArray[np.float64],
    log_ratio_proposal: NDArray[np.float64],
    *,
    log_evidence: float,
    n_chains: int,
    n_per_chain: int,
) -> float:
    """Frühwirth-Schnatter's relative-MSE error for a bridge estimate, on the log scale.

    The relative mean-squared error of the *evidence* decomposes into a proposal-sample term and
    a posterior-sample term,

        ``RE² = N₂⁻¹ Var(f₁)/E(f₁)² + (ρ/N₁) Var(f₂)/E(f₂)²``,

    with ``f₁`` and ``f₂`` the two bridge weight series and ``ρ`` the integrated autocorrelation
    time of ``f₂`` — the posterior draws are an MCMC sample, so treating them as independent
    would understate the error. ``ρ`` is taken as ``N₁ / ESS(f₂)`` from ArviZ's effective sample
    size, which is the same quantity the ``bridgesampling`` R package obtains from a spectral
    density at zero. ``√RE²`` is the reported log-scale standard error.

    What it does **not** cover: any error in the posterior draws themselves. A fit that has not
    converged, or a latent parameterisation that is exploring the wrong geometry, gives a
    confidently wrong evidence with a small standard error. That is what the agreement checks
    against the other two estimators, and against direct quadrature for DLO and SSE, are for.
    """
    n_posterior = int(log_ratio_posterior.size)
    n_proposal = int(log_ratio_proposal.size)
    s1 = n_posterior / (n_posterior + n_proposal)
    s2 = n_proposal / (n_posterior + n_proposal)

    # f₁ on the proposal draws and f₂ on the posterior draws, both written so that the only
    # exponential taken is of a centred log-ratio.
    centred_proposal = log_ratio_proposal - log_evidence
    centred_posterior = log_ratio_posterior - log_evidence
    with np.errstate(over="ignore"):
        f1 = np.exp(centred_proposal) / (s1 * np.exp(centred_proposal) + s2)
        f2 = 1.0 / (s1 * np.exp(centred_posterior) + s2)
    if not (np.isfinite(f1).all() and np.isfinite(f2).all()):
        return float("nan")

    autocorrelation_time = 1.0
    if n_per_chain >= 4:
        effective = float(az.ess(f2.reshape(n_chains, n_per_chain)))
        if np.isfinite(effective) and effective > 0.0:
            autocorrelation_time = max(1.0, n_posterior / effective)

    relative_mse = (
        f1.var(ddof=1) / f1.mean() ** 2 / n_proposal
        + autocorrelation_time * f2.var(ddof=1) / f2.mean() ** 2 / n_posterior
    )
    return float(np.sqrt(max(relative_mse, 0.0)))


# ---------------------------------------------------------------------------------------
# Direct numerical marginalisation, for the models whose parameter space is small enough
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuadratureEvidence:
    """A deterministic marginal likelihood, with the diagnostic that says whether to trust it."""

    log_evidence: float

    log_boundary_ratio: float
    """``max log density on the grid boundary − max log density anywhere on the grid``.

    The quadrature integrates over a finite box, so the estimate is a lower bound short by the
    mass outside it. This number is how much is being ignored: at ``−50`` the boundary integrand
    is ``2 × 10⁻²²`` of the peak and the truncation is far below every other error here. A value
    near zero means the box missed the posterior and the answer is meaningless.
    """

    n_points: int
    """Grid points per axis."""


def log_evidence_by_quadrature(
    target: UnconstrainedTarget,
    *,
    centre: NDArray[np.float64],
    scale: NDArray[np.float64],
    n_standard_deviations: float = 12.0,
    n_points: int = 401,
) -> QuadratureEvidence:
    """Marginalise the whole unconstrained space on a tensor-product Simpson grid.

    This is the acceptance criterion of §6.5 for DLO and SSE: with no latents and ``k`` fixed
    the integral is two-dimensional, so it can be done deterministically and bridge sampling
    held against the answer rather than against another sampler. It shares only the integrand
    with the Monte-Carlo estimators — none of the proposal fitting, the weighting or the
    recursion — which is what makes it a check of them.

    Parameters
    ----------
    target
        The integrand, from :class:`UnconstrainedTarget`.
    centre, scale
        Location and spread of the box, per unconstrained coordinate; the posterior draws' mean
        and standard deviation are the natural choice. They localise the grid only — the box is
        ``centre ± n_standard_deviations · scale`` and ``log_boundary_ratio`` reports whether
        that was wide enough, so a poor choice is visible rather than silent.
    n_standard_deviations
        Half-width of the box in units of ``scale``.
    n_points
        Grid points per axis; forced up to the next odd number, as Simpson's rule requires.

    Returns
    -------
    :class:`QuadratureEvidence`
    """
    if target.dimension > 3:
        raise ValueError(
            f"a tensor-product grid is only tractable in a few dimensions; this model has "
            f"{target.dimension}. Direct marginalisation is the check for DLO and SSE, whose "
            "parameter space is two-dimensional at fixed k; use bridge sampling elsewhere"
        )
    centre = np.asarray(centre, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if centre.size != target.dimension or scale.size != target.dimension:
        raise ValueError(
            f"centre and scale must give one value per unconstrained coordinate "
            f"({target.dimension})"
        )
    n_points = int(n_points) + 1 - int(n_points) % 2

    axes = [
        np.linspace(c - n_standard_deviations * s, c + n_standard_deviations * s, n_points)
        for c, s in zip(centre, scale, strict=True)
    ]
    weights = [_simpson_weights(axis) for axis in axes]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    log_density = target.log_density(points)

    weight = weights[0]
    for further in weights[1:]:
        weight = np.multiply.outer(weight, further)
    log_evidence = float(scipy.special.logsumexp(log_density, b=weight.reshape(-1)))

    shaped = log_density.reshape([n_points] * target.dimension)
    boundary = np.zeros_like(shaped, dtype=bool)
    for axis in range(target.dimension):
        index: list[Any] = [slice(None)] * target.dimension
        index[axis] = [0, n_points - 1]
        boundary[tuple(index)] = True
    return QuadratureEvidence(
        log_evidence=log_evidence,
        log_boundary_ratio=float(shaped[boundary].max() - shaped.max()),
        n_points=n_points,
    )


def _simpson_weights(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Composite Simpson weights ``h/3 · [1, 4, 2, ..., 4, 1]`` for an odd, uniform grid."""
    spacing = float(axis[1] - axis[0])
    weights = np.ones(axis.size, dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * spacing / 3.0


# ---------------------------------------------------------------------------------------
# The public entry points
# ---------------------------------------------------------------------------------------


def log_evidence(
    model: str | ModelSpecification,
    idata: Any,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None = None,
    latent_parameterisation: str | LatentParameterisation | None = None,
    negligible_latent_threshold: float = 0.0,
    estimator: str = BRIDGE_SAMPLING,
    n_proposal_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> LogEvidence:
    """``log p(D | model)`` with every parameter and latent marginalised.

    Parameters
    ----------
    model
        Model name or :class:`~end_of_outbreak.model_specifications.ModelSpecification`.
    idata
        The ``InferenceData`` from the fit of *this* model. Bridge sampling and importance
        sampling build their proposals from its draws; prior Monte Carlo needs no posterior at
        all and uses ``idata`` only for the default draw count.
    counts, delays, switch_day, negligible_latent_threshold
        As passed to :func:`end_of_outbreak.fitting.fit_model`. The model is rebuilt from them,
        so that the integral being estimated is the one that was sampled.
    R_pre, R_post, k
        **Exactly** as they were passed to the fit: a ``float`` where the parameter was fixed,
        a :class:`~end_of_outbreak.model_specifications.LogNormalPrior` where it was estimated.
        The evidence is an integral against the prior, so the prior is part of the question and
        cannot be recovered from the draws; a fixed parameter, being a constant in the graph,
        contributes no factor at all.
    latent_parameterisation
        Required for a model with latents, as everywhere in this package. Which one is used does
        not change the answer — the marginalised strategies integrate their latents out exactly
        — but it does change how many coordinates the estimators work in.
    estimator
        One of :data:`ESTIMATORS`. ``bridge_sampling`` is the default and what the pipeline
        uses; the other two are the §6.5 agreement checks.
    n_proposal_draws
        Proposal draws (bridge, importance sampling) or prior draws (prior Monte Carlo).
        Defaults to the number of posterior draws in ``idata``. That is a sensible default for
        the two proposal-based estimators and a low one for prior Monte Carlo, whose weights are
        far more degenerate — pass a larger number there and read the standard error.
    rng
        Generator for the proposal and prior draws.

    Returns
    -------
    :class:`LogEvidence`
    """
    if estimator not in ESTIMATORS:
        known = ", ".join(ESTIMATORS)
        raise ValueError(f"unknown estimator {estimator!r}; known estimators: {known}")
    specification = specification_of(model)
    built = pymc_models.build_model(
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
    target = UnconstrainedTarget(built)
    generator = np.random.default_rng() if rng is None else rng
    posterior = idata.posterior
    n_posterior_draws = int(posterior.sizes["chain"] * posterior.sizes["draw"])
    n_draws = n_posterior_draws if n_proposal_draws is None else int(n_proposal_draws)

    if estimator == PRIOR_MONTE_CARLO:
        estimate, standard_error, used = _prior_monte_carlo(target, n_draws=n_draws, rng=generator)
    else:
        points = target.posterior_points(idata)
        if estimator == BRIDGE_SAMPLING:
            estimate, standard_error, used = _bridge_sampling(
                target,
                points,
                n_chains=int(posterior.sizes["chain"]),
                n_proposal_draws=n_draws,
                rng=generator,
            )
        else:
            estimate, standard_error, used = _importance_sampling(
                target, points, n_proposal_draws=n_draws, rng=generator
            )
    return LogEvidence(
        model=specification.name,
        estimator=estimator,
        log_evidence=float(estimate),
        standard_error=float(standard_error),
        n_draws=int(used),
    )


def posterior_model_probabilities(log_evidences: Mapping[str, float]) -> dict[str, float]:
    """Normalise log evidences into posterior model probabilities, uniform prior over models.

    §6.2 fixes the prior over models at uniform within each analysis — 1/3 for Figs. 1–2, 1/4
    for Figs. 3–4 — so the prior cancels and the probabilities are the normalised evidences. The
    report must state that, because posterior model probabilities are not interpretable without
    it, and §6.5 records the further caveat that they compare the models *as specified*: the
    same ``k`` prior means an individual-level offspring dispersion in SSE/SSI and a day-level
    incidence dispersion in DLO.

    The normalisation goes through ``logsumexp``, so evidences hundreds of nats apart give an
    exact 1 and an exact 0 rather than an overflow.

    Parameters
    ----------
    log_evidences
        Model name to ``log p(D | model)``, all on the same data and the same day window.

    Returns
    -------
    The probabilities, in the input's key order, summing to 1.
    """
    if not log_evidences:
        return {}
    names = list(log_evidences)
    values = np.array([float(log_evidences[name]) for name in names], dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("log evidences must not be nan; a failed estimate is not a comparison")
    total = float(scipy.special.logsumexp(values))
    if not np.isfinite(total):
        raise ValueError(
            "every model has log evidence −inf, so the posterior model probabilities are "
            "undefined; the models cannot all be impossible on data they were fitted to"
        )
    return {name: float(np.exp(value - total)) for name, value in zip(names, values, strict=True)}

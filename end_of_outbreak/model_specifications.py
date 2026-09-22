"""What the five models are, and the priors they share.

This module is deliberately free of PyMC and of NumPy random state: it holds the *descriptions*
that the model builders (:mod:`end_of_outbreak.pymc_models`), the simulators
(:mod:`end_of_outbreak.forward_simulation`) and later the RAC calculators all dispatch on, so
that none of them has to hard-code model names or re-derive a prior's parameters.

The five models
---------------

============  =====================  ===================================  ==================
Name          Anchored on            Overdispersion acts at the level of  Latents
============  =====================  ===================================  ==================
``dlo``       infections (naive)     the day                              none
``sse``       infections (naive)     the transmission event               none
``ssi``       infections (naive)     the individual                       ``Y_t``, 31
``sse_so``    symptom onsets         the transmission event               ``λ̃_t``
``ssi_so``    symptom onsets         the individual                       ``Y_t``
============  =====================  ===================================  ==================

``cori`` is registered alongside them as the Poisson (``k → ∞``) limit. It is not one of the
five models being compared; it exists because "``k → ∞`` collapses SSE and SSI onto Cori" is
one of the project's validation checks (§4.4, §6.4 of the implementation plan).

The naive models set ``I_t := C_t`` — they treat onset dates as infection dates, which is
exactly the practice the project sets out to quantify. The onset-anchored models set
``D_t := C_t`` and model the incubation period explicitly.

Priors
------
Priors must be **proper and identical across the models within an analysis** (§6.2): the
posterior model probabilities are Bayes factors, which diffuse or improper priors would make
arbitrary. :class:`LogNormalPrior` carries the project's parameterisation — a median plus a
quantile — and exposes both a SciPy and a PyMC view of it.

Identical priors make the Bayes factors *well defined*, not *neutral*. ``k`` is an
individual-level offspring dispersion in SSE/SSI but a day-level incidence dispersion in DLO
(§5.4), so placing the same prior on both is a deliberate choice; see §6.5 for what the
resulting model probabilities do and do not say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.stats

Anchoring = Literal["infections", "onsets"]
"""Whether the observed counts are read as infection dates or as symptom-onset dates."""

OverdispersionLevel = Literal["none", "day", "event", "individual"]
"""The level at which the excess variance acts — the axis the project compares along."""


# ---------------------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogNormalPrior:
    """A log-normal prior, parameterised by its median and log-scale standard deviation.

    Construct with :meth:`from_median_and_quantile` (or :meth:`from_config`) rather than
    directly: the project specifies priors by a median and a tail quantile, e.g. ``R_pre``
    with median 1 and 2.5th percentile 0.2, and ``k`` with median 0.18 and 95% interval
    ``(0.018, 1.8)``.
    """

    median: float
    sigma_log: float
    """Standard deviation of ``log X``."""

    def __post_init__(self) -> None:
        if not self.median > 0:
            raise ValueError(f"median must be positive, got {self.median}")
        if not self.sigma_log > 0:
            raise ValueError(f"sigma_log must be positive, got {self.sigma_log}")

    @classmethod
    def from_median_and_quantile(
        cls, *, median: float, quantile: float, probability: float
    ) -> LogNormalPrior:
        """The log-normal with this median whose ``probability`` quantile is ``quantile``."""
        if not median > 0 or not quantile > 0:
            raise ValueError("median and quantile must both be positive")
        if not 0.0 < probability < 1.0 or probability == 0.5:
            raise ValueError(
                f"probability must lie strictly in (0, 1) and not be 0.5, got {probability}"
            )
        sigma_log = (np.log(quantile) - np.log(median)) / scipy.stats.norm.ppf(probability)
        return cls(median=float(median), sigma_log=float(sigma_log))

    @classmethod
    def from_config(cls, block: dict[str, Any]) -> LogNormalPrior:
        """Build from a ``config.yaml`` prior block.

        Expects ``distribution: lognormal``, a ``median``, and at least one of
        ``quantile_025`` / ``quantile_975``. When both are given they must agree to within
        1% in ``sigma_log``: a log-normal is symmetric in log space, so a mismatch means the
        configured interval is not achievable and the intent is ambiguous.
        """
        distribution = block.get("distribution", "lognormal")
        if distribution != "lognormal":
            raise ValueError(f"only lognormal priors are supported, got {distribution!r}")
        median = float(block["median"])

        candidates = [
            cls.from_median_and_quantile(median=median, quantile=float(block[key]), probability=p)
            for key, p in (("quantile_025", 0.025), ("quantile_975", 0.975))
            if block.get(key) is not None
        ]
        if not candidates:
            raise ValueError("prior block needs a 'quantile_025' or a 'quantile_975' entry")
        if len(candidates) == 2:
            first, second = candidates
            if abs(first.sigma_log - second.sigma_log) > 0.01 * first.sigma_log:
                raise ValueError(
                    f"quantile_025 and quantile_975 imply different log-scale spreads "
                    f"({first.sigma_log:.4g} vs {second.sigma_log:.4g}); a log-normal is "
                    "symmetric in log space, so no single prior matches both"
                )
        return candidates[0]

    def frozen(self) -> Any:
        """The corresponding frozen ``scipy.stats.lognorm``."""
        return scipy.stats.lognorm(s=self.sigma_log, scale=self.median)

    def pymc_kwargs(self) -> dict[str, float]:
        """``mu``/``sigma`` keyword arguments for ``pm.LogNormal``."""
        return {"mu": float(np.log(self.median)), "sigma": self.sigma_log}


# ---------------------------------------------------------------------------------------
# Model descriptions
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpecification:
    """A structural description of one model — no parameter values, no data."""

    name: str
    """Lower-case identifier used in file names, config and the Snakemake wildcards."""

    label: str
    """Display name for figures and tables."""

    anchoring: Anchoring
    overdispersion_level: OverdispersionLevel

    latent_variable: str | None
    """Name of the model's latent block, or ``None`` if the likelihood is closed form."""

    description: str

    @property
    def has_latents(self) -> bool:
        """Whether inference must sample a latent block."""
        return self.latent_variable is not None

    @property
    def has_dispersion(self) -> bool:
        """Whether the model has a dispersion parameter ``k`` at all (Cori does not)."""
        return self.overdispersion_level != "none"


DLO = ModelSpecification(
    name="dlo",
    label="DLO",
    anchoring="infections",
    overdispersion_level="day",
    latent_variable=None,
    description=(
        "Day-level overdispersion: I_t ~ NB(mean = R_t Λ_t, disp = k) with k constant across "
        "days and independent of the force of infection. The negative-binomial variant of "
        "Supplementary Analysis 1 of Thompson et al. (2024). Because the dispersion does not "
        "scale with the force of infection, k here indexes day-to-day variation in aggregate "
        "incidence rather than an individual-level offspring distribution."
    ),
)

SSE = ModelSpecification(
    name="sse",
    label="SSE",
    anchoring="infections",
    overdispersion_level="event",
    latent_variable=None,
    description=(
        "Superspreading events: I_t ~ NB(mean = R_t Λ_t, disp = k Λ_t), equivalently "
        "λ_t ~ Gamma(k Λ_t, k) with I_t ~ Poisson(R_t λ_t). The dispersion scales with the "
        "force of infection, which is what makes k an individual-level offspring dispersion. "
        "The only one of the five models whose likelihood is closed form despite the "
        "overdispersion acting below the day level."
    ),
)

SSI = ModelSpecification(
    name="ssi",
    label="SSI",
    anchoring="infections",
    overdispersion_level="individual",
    latent_variable="Y",
    description=(
        "Superspreading individuals: each case carries a latent infectivity with mean 1 and "
        "variance 1/k, aggregated over the day-t cohort as Y_t | I_t ~ Gamma(k I_t, k), and "
        "I_t ~ Poisson(R_t Σ_s w_s Y_{t-s}). Marginally the offspring distribution is the "
        "same NB(R, k) as SSE; the models differ only in mechanism."
    ),
)

CORI = ModelSpecification(
    name="cori",
    label="Cori",
    anchoring="infections",
    overdispersion_level="none",
    latent_variable=None,
    description=(
        "The Poisson renewal model of Cori et al. (2013): I_t ~ Poisson(R_t Λ_t). Not one of "
        "the five models under comparison — it is the k → ∞ limit of both SSE and SSI, and "
        "exists here as the target of that validation check."
    ),
)

SSE_SO = ModelSpecification(
    name="sse_so",
    label="SSE-SO",
    anchoring="onsets",
    overdispersion_level="event",
    latent_variable="lambda_tilde",
    description=(
        "Onset-anchored SSE: λ̃_t ~ Gamma(k Σ_s f_tost,s D_{t-s}, k), E_t = R_t λ̃_t, and "
        "D_t ~ Poisson(Σ_a f_inc,a E_{t-a}). Transmission is referenced to the infector's own "
        "onset and the incubation period is modelled explicitly. Unlike the infection-anchored "
        "SSE this has no closed-form likelihood: each observed onset pools contributions from "
        "many days, each carrying its own independent Gamma."
    ),
)

SSI_SO = ModelSpecification(
    name="ssi_so",
    label="SSI-SO",
    anchoring="onsets",
    overdispersion_level="individual",
    latent_variable="Y",
    description=(
        "Onset-anchored SSI: Y_t | D_t ~ Gamma(k D_t, k), E_t = R_t Σ_s f_tost,s Y_{t-s}, and "
        "D_t ~ Poisson(Σ_a f_inc,a E_{t-a}). The infectivity prior is Gamma(k D_t, k), not the "
        "Gamma(k I_t, k) written in starter_docs/models.jpeg, which is a transcription slip."
    ),
)

CORI_SO = ModelSpecification(
    name="cori_so",
    label="Cori-SO",
    anchoring="onsets",
    overdispersion_level="none",
    latent_variable=None,
    description=(
        "Onset-anchored Poisson renewal: E_t = R_t Σ_s f_tost,s D_{t-s} and "
        "D_t ~ Poisson(Σ_a f_inc,a E_{t-a}). The k → ∞ limit of SSE-SO and SSI-SO, and the "
        "warm-up case for the §4 equivalence arguments. Not a compared model."
    ),
)

NAIVE_MODELS: tuple[ModelSpecification, ...] = (DLO, SSE, SSI)
"""The three infection-anchored models compared in analyses 1 and 2."""

ONSET_ANCHORED_MODELS: tuple[ModelSpecification, ...] = (SSE_SO, SSI_SO)
"""The two onset-anchored models added to the comparison in analyses 3 and 4."""

MODEL_SPECIFICATIONS: dict[str, ModelSpecification] = {
    specification.name: specification
    for specification in (DLO, SSE, SSI, CORI, SSE_SO, SSI_SO, CORI_SO)
}
"""Every model this package can build, by name."""


def specification_of(model: str | ModelSpecification) -> ModelSpecification:
    """Resolve a model name to its :class:`ModelSpecification`; pass one through unchanged."""
    if isinstance(model, ModelSpecification):
        return model
    try:
        return MODEL_SPECIFICATIONS[model]
    except KeyError as error:
        known = ", ".join(sorted(MODEL_SPECIFICATIONS))
        raise KeyError(f"unknown model {model!r}; known models: {known}") from error


# ---------------------------------------------------------------------------------------
# Parameter values
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransmissionParameters:
    """One parameter draw: the two reproduction numbers and the dispersion.

    Used by the simulators and, from Stage 4b, by the particle filter and PMMH sampler, so
    that a posterior draw can be passed around as a single object.
    """

    R_pre: float
    R_post: float
    k: float | None = None
    """``None`` for Cori, which has no dispersion parameter."""

    def __post_init__(self) -> None:
        if not self.R_pre > 0 or not self.R_post > 0:
            raise ValueError("R_pre and R_post must be positive")
        if self.k is not None and not self.k > 0:
            raise ValueError("k must be positive when given")

    def require_k(self, specification: ModelSpecification) -> float:
        """``k``, raising a model-specific error if it is missing but needed."""
        if self.k is None:
            raise ValueError(f"model {specification.name!r} needs a dispersion parameter k")
        return self.k

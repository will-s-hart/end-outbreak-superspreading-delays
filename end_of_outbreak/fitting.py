"""One place that turns a model description plus data into an ``InferenceData``.

Every analysis is a single fit to the complete record — days 0–110 — because RAC is a
retrospective quantity and one fit serves every conditioning day (§5.1, §5.6). So this module
is deliberately thin: resolve the sampler settings, build the model, supply an initial point if
the latent parameterisation asks for one, sample, and record on the result the few facts a
downstream calculator cannot recover from the draws alone.

Those facts matter. The RAC calculators need to know which latent parameterisation the fit used
— it decides which latents were integrated out and therefore have to be rebuilt (§6.3) — and,
in the fixed-``k`` analyses, the value ``k`` was held at, since a fixed parameter is a constant
in the graph rather than a variable in the posterior. Both travel with the fit as attributes so
that a results file is self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
from numpy.typing import NDArray

from end_of_outbreak import latent_parameterisations as lp
from end_of_outbreak import pymc_models
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import (
    LogNormalPrior,
    ModelSpecification,
    specification_of,
)

MODEL_ATTRIBUTE = "model"
PARAMETERISATION_ATTRIBUTE = "latent_parameterisation"
FIXED_DISPERSION_ATTRIBUTE = "fixed_k"
SWITCH_DAY_ATTRIBUTE = "switch_day"
THRESHOLD_ATTRIBUTE = "negligible_latent_threshold"


@dataclass(frozen=True)
class SamplerSettings:
    """NUTS settings for one fit, as carried per analysis in ``config/config.yaml``."""

    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.9
    seed: int | None = None

    @classmethod
    def from_config(cls, block: dict[str, Any]) -> SamplerSettings:
        """Build from an analysis's ``sampler:`` block."""
        return cls(
            draws=int(block["draws"]),
            tune=int(block["tune"]),
            chains=int(block["chains"]),
            target_accept=float(block["target_accept"]),
            seed=None if block.get("seed") is None else int(block["seed"]),
        )


def fit_model(
    model: str | ModelSpecification,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    switch_day: int,
    R_pre: float | LogNormalPrior,
    R_post: float | LogNormalPrior,
    k: float | LogNormalPrior | None = None,
    latent_parameterisation: str | lp.LatentParameterisation | None = None,
    negligible_latent_threshold: float = 0.0,
    sampler: SamplerSettings | None = None,
    progressbar: bool = False,
) -> az.InferenceData:
    """Fit one model to the whole series and return the draws, self-describing.

    Parameters
    ----------
    model, counts, delays, switch_day, R_pre, R_post, k, latent_parameterisation,
    negligible_latent_threshold
        Passed through to :func:`end_of_outbreak.pymc_models.build_model`; see there for the
        conventions. ``latent_parameterisation`` is required for a model with latents.
    sampler
        NUTS settings; the defaults of :class:`SamplerSettings` if omitted.
    progressbar
        Off by default, since fits are normally run from the pipeline.
    """
    specification = specification_of(model)
    settings = SamplerSettings() if sampler is None else sampler
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
    initial_values = _initial_values(
        specification,
        built,
        counts,
        delays=delays,
        k=k,
        latent_parameterisation=latent_parameterisation,
    )

    with built:
        idata = pm.sample(
            draws=settings.draws,
            tune=settings.tune,
            chains=settings.chains,
            cores=settings.chains,
            target_accept=settings.target_accept,
            random_seed=settings.seed,
            initvals=initial_values or None,
            progressbar=progressbar,
        )

    idata.attrs.update(
        {
            MODEL_ATTRIBUTE: specification.name,
            SWITCH_DAY_ATTRIBUTE: int(switch_day),
            THRESHOLD_ATTRIBUTE: float(negligible_latent_threshold),
            PARAMETERISATION_ATTRIBUTE: (
                ""
                if latent_parameterisation is None
                else lp.parameterisation_of(latent_parameterisation).name
            ),
            # A fixed k is a constant in the graph, so it is nowhere in the draws; record it,
            # or the RAC calculators have no way to recover the value the fit was run at.
            FIXED_DISPERSION_ATTRIBUTE: float(k) if isinstance(k, float | int) else np.nan,
        }
    )
    return idata


def _initial_values(
    specification: ModelSpecification,
    built: pm.Model,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    k: float | LogNormalPrior | None,
    latent_parameterisation: str | lp.LatentParameterisation | None,
) -> dict[Any, Any]:
    """``initvals`` for the latent block, empty unless the strategy asks for a starting point.

    Only ``initialise_at_geometric_mean`` produces anything, and it needs a numeric ``k``: the
    initial point is computed before sampling, so when ``k`` is estimated the prior median
    stands in for it.
    """
    if specification.latent_variable is None or latent_parameterisation is None:
        return {}
    parameterisation = lp.parameterisation_of(latent_parameterisation)
    dimension = pymc_models.latent_dimension(specification)
    sampled_days = pymc_models.model_days(built, dimension)
    scale = pymc_models.latent_scale_by_day(specification, counts, delays=delays)[sampled_days]
    representative_k = k.median if isinstance(k, LogNormalPrior) else float(k if k else 1.0)
    return dict(
        lp.suggested_initial_values(
            specification.latent_variable,
            k=representative_k,
            sampled_scale=scale,
            parameterisation=parameterisation,
        )
    )


def save_fit(idata: az.InferenceData, path: str | Path) -> Path:
    """Write a fit to netCDF, creating the directory if need be. Attributes travel with it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(destination))
    return destination


def load_fit(path: str | Path) -> az.InferenceData:
    """Read a fit back, with the attributes :func:`fit_model` recorded on it."""
    return az.from_netcdf(str(path))


def fitted_parameterisation(idata: az.InferenceData) -> str | None:
    """The latent parameterisation a fit was run under, or ``None`` if it had no latents."""
    name = idata.attrs.get(PARAMETERISATION_ATTRIBUTE, "")
    return str(name) or None


def fitted_dispersion(idata: az.InferenceData) -> float | None:
    """The value ``k`` was fixed at, or ``None`` if it was estimated (or absent)."""
    value = idata.attrs.get(FIXED_DISPERSION_ATTRIBUTE, np.nan)
    return None if value is None or np.isnan(float(value)) else float(value)

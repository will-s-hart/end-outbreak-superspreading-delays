"""One place that turns a model description plus data into a posterior ``DataTree``.

This module is deliberately thin: resolve the sampler settings, build the model, supply an
initial point if the latent parameterisation asks for one, sample, and record on the result the
few facts a downstream calculator cannot recover from the draws alone.

It is called on **windows**, not only on the whole record. RAC is a real-time quantity, so
:mod:`end_of_outbreak.refit_risk` calls this once per conditioning day with ``counts[:t + 1]``;
the evidence and dispersion steps call it once on the complete record. Nothing here needs to
know which is happening — the window is just the data it is given.

The recorded facts matter. The RAC calculators need to know which latent parameterisation the
fit used — it decides which latents were integrated out of the likelihood, and therefore which
have to be integrated back out of the risk (§6.3) — and, in the fixed-``k`` analyses, the value
``k`` was held at, since a fixed parameter is a constant in the graph rather than a variable in
the posterior. Both travel with the fit as attributes so that a results file is self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr
from numpy.typing import NDArray

from end_of_outbreak import latent_parameterisations as lp
from end_of_outbreak import pymc_models, reporting
from end_of_outbreak.delay_distributions import GammaDelay, OnsetAnchoredDelays
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
REPORTING_PROBABILITY_ATTRIBUTE = "reporting_probability"
REPORTING_DELAY_MEAN_ATTRIBUTE = "reporting_delay_mean"
REPORTING_DELAY_SD_ATTRIBUTE = "reporting_delay_sd"
AS_OF_DAY_ATTRIBUTE = "as_of_day"

NETCDF_ENGINE = "h5netcdf"
"""A fit has groups, so it is written as NETCDF4, which needs an HDF5 backend.

Named rather than left to xarray's default so that reading and writing cannot end up on
different engines, and so the dependency the pipeline actually rests on is visible from the code
that rests on it. PyMC returns an ``xarray.DataTree`` — ``arviz.InferenceData`` is a deprecated
alias for exactly that in arviz 1.x — so the I/O here is xarray's, not arviz's.
"""


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
    reporting_model: reporting.ReportingModel | None = None,
    as_of_day: int | None = None,
    sampler: SamplerSettings | None = None,
    progressbar: bool = False,
) -> xr.DataTree:
    """Fit one model to the whole series and return the draws, self-describing.

    Parameters
    ----------
    model, counts, delays, switch_day, R_pre, R_post, k, latent_parameterisation,
    negligible_latent_threshold, reporting_model, as_of_day
        Passed through to :func:`end_of_outbreak.pymc_models.build_model`; see there for the
        conventions. ``latent_parameterisation`` is required for a model with latents.
    sampler
        NUTS settings; the defaults of :class:`SamplerSettings` if omitted. Step assignment is
        left to PyMC — that is what the sampler benchmark validated — with the single exception
        of the latent true-count block under incomplete reporting, which is handed
        :class:`end_of_outbreak.reporting.SingleSiteCountMetropolis` because PyMC's own choice
        for it silently stops moving on a series this long.
    progressbar
        Off by default, since fits are normally run from the pipeline.
    """
    specification = specification_of(model)
    settings = SamplerSettings() if sampler is None else sampler
    resolved_reporting = (
        reporting.COMPLETE_REPORTING if reporting_model is None else reporting_model
    )
    resolved_as_of_day = len(counts) - 1 if as_of_day is None else int(as_of_day)
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
        reporting_model=resolved_reporting,
        as_of_day=resolved_as_of_day,
    )
    initial_values = _initial_values(
        specification,
        built,
        counts,
        delays=delays,
        k=k,
        latent_parameterisation=latent_parameterisation,
        reporting_model=resolved_reporting,
    )

    # `target_accept` is a NUTS setting, and PyMC rejects it outright when no variable is
    # sampled by NUTS. That happens only under incomplete reporting with every continuous
    # parameter fixed — the latent counts are then the whole free block, and Metropolis has no
    # acceptance target to hit.
    overrides: dict[str, Any] = (
        {"target_accept": settings.target_accept} if _has_continuous_variables(built) else {}
    )
    # The one step PyMC would get wrong; everything else it assigns around this. See
    # `reporting.SingleSiteCountMetropolis` for what the default does to a long series.
    count_step = reporting.count_block_step(built)
    if count_step is not None:
        overrides["step"] = [count_step]
    with built:
        idata = pm.sample(
            draws=settings.draws,
            tune=settings.tune,
            chains=settings.chains,
            cores=settings.chains,
            random_seed=settings.seed,
            initvals=initial_values or None,
            progressbar=progressbar,
            **overrides,
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
            # Likewise the reporting assumption: it is an input, not a parameter, and the risk
            # path has to know whether the driving series is the data or the latent totals.
            REPORTING_PROBABILITY_ATTRIBUTE: float(resolved_reporting.probability),
            REPORTING_DELAY_MEAN_ATTRIBUTE: (
                np.nan if resolved_reporting.delay is None else resolved_reporting.delay.mean
            ),
            REPORTING_DELAY_SD_ATTRIBUTE: (
                np.nan if resolved_reporting.delay is None else resolved_reporting.delay.sd
            ),
            AS_OF_DAY_ATTRIBUTE: resolved_as_of_day,
        }
    )
    return idata


def _has_continuous_variables(built: pm.Model) -> bool:
    """Whether anything in the model is sampled by a gradient-based step.

    Read off ``free_RVs`` rather than ``value_vars``: it is a plain list, it carries the same
    dtypes, and it does not materialise anything the sampler has not asked for yet.
    """
    return any(
        not np.issubdtype(np.dtype(variable.dtype), np.integer) for variable in built.free_RVs
    )


def _initial_values(
    specification: ModelSpecification,
    built: pm.Model,
    counts: NDArray[np.int64],
    *,
    delays: OnsetAnchoredDelays,
    k: float | LogNormalPrior | None,
    latent_parameterisation: str | lp.LatentParameterisation | None,
    reporting_model: reporting.ReportingModel,
) -> dict[Any, Any]:
    """``initvals`` for the latent block, empty unless the strategy asks for a starting point.

    Only ``initialise_at_geometric_mean`` produces anything, and it needs a numeric ``k``: the
    initial point is computed before sampling, so when ``k`` is estimated the prior median
    stands in for it.
    """
    if specification.latent_variable is None or latent_parameterisation is None:
        return {}
    parameterisation = lp.parameterisation_of(latent_parameterisation)
    if not reporting_model.is_complete:
        # The Gamma scale is a function of the latent true counts, so there is no
        # data-determined starting point to compute. The inverse-CDF uniforms this path
        # requires start at 0.5, which is the median of every latent and needs no help.
        if parameterisation.initialise_at_geometric_mean:
            raise ValueError(
                f"latent parameterisation {parameterisation.name!r} starts the block at a "
                "value computed from the latent scale, which incomplete reporting makes random"
            )
        return {}
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


def save_fit(idata: xr.DataTree, path: str | Path) -> Path:
    """Write a fit to netCDF, creating the directory if need be. Attributes travel with it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(destination), engine=NETCDF_ENGINE)
    return destination


def load_fit(path: str | Path) -> xr.DataTree:
    """Read a fit back, with the attributes :func:`fit_model` recorded on it.

    Loaded eagerly rather than left lazy. A fit is a few megabytes, and a lazily-open handle on
    a results file is a standing invitation to the classic bug where re-running the rule that
    produced it truncates the file out from under a reader.
    """
    return xr.open_datatree(str(path), engine=NETCDF_ENGINE).load()


def fitted_parameterisation(idata: xr.DataTree) -> str | None:
    """The latent parameterisation a fit was run under, or ``None`` if it had no latents."""
    name = idata.attrs.get(PARAMETERISATION_ATTRIBUTE, "")
    return str(name) or None


def fitted_dispersion(idata: xr.DataTree) -> float | None:
    """The value ``k`` was fixed at, or ``None`` if it was estimated (or absent)."""
    value = idata.attrs.get(FIXED_DISPERSION_ATTRIBUTE, np.nan)
    return None if value is None or np.isnan(float(value)) else float(value)


def fitted_reporting(idata: xr.DataTree) -> reporting.ReportingModel:
    """The reporting assumption a fit was run under.

    Fits written before this attribute existed carry no reporting probability and were all
    completely reported, so that is what an absent attribute means.
    """
    probability = float(idata.attrs.get(REPORTING_PROBABILITY_ATTRIBUTE, 1.0))
    mean = float(idata.attrs.get(REPORTING_DELAY_MEAN_ATTRIBUTE, np.nan))
    sd = float(idata.attrs.get(REPORTING_DELAY_SD_ATTRIBUTE, np.nan))
    delay = None if np.isnan(mean) or np.isnan(sd) else GammaDelay(mean=mean, sd=sd)
    return reporting.ReportingModel(probability=probability, delay=delay)


def fitted_as_of_day(idata: xr.DataTree, counts: NDArray[np.int64]) -> int:
    """The day a fit's window was observed on, defaulting to its last day."""
    value = idata.attrs.get(AS_OF_DAY_ATTRIBUTE)
    return len(counts) - 1 if value is None else int(value)

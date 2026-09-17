"""The shared body of every ``scripts/run_<analysis>.py``.

Everything that varies between the four analyses of §8 lives in ``config/config.yaml`` — which
models are compared, whether ``k`` is fixed or estimated, the sampler settings and the seeds — so
the run scripts themselves differ only in the analysis they name. Stage 5 kept the whole driver
inside ``run_naive_models_fixed_k.py`` deliberately, on the grounds that the shape of the
duplication is only visible once there are two of them; Stage 7 is the second, so it is here.

A run script is now a docstring, an analysis name and a one-line ``main``::

    ANALYSIS = "naive_models_estimated_k"

    def main(argv: list[str] | None = None) -> None:
        analysis_driver.main(ANALYSIS, argv)

This module cannot live in ``scripts/utils.py``, which the ``figure`` rule depends on: fit- and
results-driving logic there would make every restyle a reason to re-run MCMC. It is named
instead in ``FIT_CORE``, ``RAC_CORE``, ``EVIDENCE_CORE`` and ``DISPERSION_CORE`` at the top of
the ``Snakefile``, alongside the package modules whose changes must invalidate those tiers.

Four subcommands, one per pipeline rule::

    python scripts/run_<analysis>.py fit        --model ssi --output ..._posterior.nc
    python scripts/run_<analysis>.py rac        --model ssi --posterior ... --output ..._rac.csv
    python scripts/run_<analysis>.py evidence   --posteriors ... --output model_evidence.json
    python scripts/run_<analysis>.py dispersion --posteriors ... --output ..._posteriors.json

``dispersion`` applies only to the analyses that estimate ``k``, and says so rather than writing
an empty file when it is pointed at a fixed-``k`` fit.

**``rac`` is a tier-1 step under the default method.** RAC(t) conditions on the record through
day ``t``, so the estimator refits the model once per conditioning day and the "recomputing a
risk curve never re-runs a fit" separation cannot hold for it. The other three steps keep it: a
fit, an evidence and a dispersion summary are all properties of the model given the whole record.
The ``--method`` flag selects between the per-day refit and the filtering approximation; the
default comes from ``config/config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from end_of_outbreak import (
    configuration,
    filtered_risk,
    fitting,
    model_evidence,
    outbreak_data,
    refit_risk,
    reporting,
)
from end_of_outbreak import posterior_comparison as pc
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import LogNormalPrior, specification_of

DISPERSION_VARIABLE = "k"
"""Name the model builders give the dispersion parameter in the posterior."""

REFIT_DAILY = "refit_daily"
SINGLE_FIT_FILTERED = "single_fit_filtered"
RAC_METHODS = (REFIT_DAILY, SINGLE_FIT_FILTERED)


# ---------------------------------------------------------------------------------------
# Config → the objects every subcommand needs
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSetting:
    """One analysis's configuration, resolved into the objects the subcommands take."""

    name: str
    config: dict[str, Any]
    """The whole config: the latent parameterisation and the negligible-latent threshold sit
    outside the per-analysis blocks, being properties of the *implementation* settled once by
    the Stage-3 benchmark rather than of any one analysis."""

    block: dict[str, Any]
    data: outbreak_data.OutbreakData
    delays: OnsetAnchoredDelays

    @classmethod
    def load(cls, name: str, *, config_path: str | Path, data_path: str | Path) -> AnalysisSetting:
        """Read the config and the onset series, and build the delay triple."""
        config = configuration.load_config(config_path)
        block = configuration.analysis_config(config, name)
        return cls(
            name=name,
            config=config,
            block=block,
            data=outbreak_data.load_onset_data(data_path),
            delays=configuration.onset_anchored_delays_from_config({"shared": block["shared"]}),
        )

    @property
    def models(self) -> list[str]:
        """The models this analysis compares, in the order the figures show them."""
        return list(self.block["models"])

    @property
    def latent_parameterisation(self) -> str:
        return str(self.config["latent_parameterisation"])

    @property
    def negligible_latent_threshold(self) -> float:
        return float(self.config["negligible_latent_threshold"])

    @property
    def rac(self) -> dict[str, Any]:
        """The ``rac:`` block: which estimator, from which day, and the filter's settings.

        An analysis may override individual keys with a ``rac:`` block of its own — the
        under-reporting sweeps use it to start their curves at the ERT's arrival — but the
        estimator and the convergence criteria stay global, because they are properties of the
        implementation rather than of any one analysis.
        """
        return dict(self.config["rac"]) | dict(self.block.get("rac") or {})

    def reporting_model(self) -> reporting.ReportingModel:
        """How true cases become reported ones; complete reporting unless an analysis says so."""
        return reporting.ReportingModel.from_config(self.block.get("reporting"))

    def rac_method(self, override: str | None = None) -> str:
        """The estimator to use, from the command line if given and the config otherwise."""
        method = override or str(self.rac["method"])
        if method not in RAC_METHODS:
            raise ValueError(f"unknown rac method {method!r}; expected one of {RAC_METHODS}")
        return method

    def reproduction_number_priors(self) -> tuple[LogNormalPrior, LogNormalPrior]:
        """``R_pre`` and ``R_post``'s priors, shared by every model so the evidences compare.

        The priors themselves, whether or not this analysis estimates them: the figures draw
        them behind the posteriors. :meth:`reproduction_numbers` is what the builders take.
        """
        priors = self.block["shared"]["priors"]
        return (
            LogNormalPrior.from_config(priors["R_pre"]),
            LogNormalPrior.from_config(priors["R_post"]),
        )

    def reproduction_numbers(self) -> tuple[float | LogNormalPrior, float | LogNormalPrior]:
        """``R_pre`` and ``R_post`` as the builders take them, fixed where the analysis says so.

        The same fixed-or-estimated switch :meth:`dispersion` applies to ``k``, and it carries
        the same consequence: a fixed reproduction number is a constant in the PyMC graph, so
        it is nowhere in the draws and has to be handed back to the RAC calculators. The fit
        records it for exactly that reason (``fitting.fitted_reproduction_numbers``).
        """
        R_pre_prior, R_post_prior = self.reproduction_number_priors()
        fixed_R_pre = self.block.get("fixed_R_pre")
        fixed_R_post = self.block.get("fixed_R_post")
        return (
            R_pre_prior if fixed_R_pre is None else float(fixed_R_pre),
            R_post_prior if fixed_R_post is None else float(fixed_R_post),
        )

    def switch_day(self) -> int:
        """The day ``R`` switches on, in each model's own time index.

        The ERT arrival day unless the analysis overrides it. An override is how a sensitivity
        analysis shifts the switch, and how the no-switchpoint variants disable it altogether:
        a day at or past the end of the window leaves ``R_pre`` in force throughout
        (:func:`end_of_outbreak.renewal.switch_index`).
        """
        override = self.block.get("switch_day")
        return self.data.ert_arrival_day if override is None else int(override)

    def dispersion(self) -> float | LogNormalPrior:
        """``k`` as the builders take it: a float where fixed, a prior where estimated.

        A fixed ``k`` becomes a constant in the PyMC graph rather than a random variable, which
        is why it has to be handed back to the RAC calculators explicitly later — it is nowhere
        in the draws.
        """
        fixed = self.block.get("fixed_k")
        if fixed is not None:
            return float(fixed)
        return LogNormalPrior.from_config(self.block["k_prior"])

    def require_model(self, model: str) -> str:
        """Reject a model that is not part of this analysis, rather than fitting it anyway."""
        if model not in self.models:
            known = ", ".join(self.models)
            raise ValueError(f"{self.name} does not include model {model!r}; it compares {known}")
        return model

    def model_entropy(self, model: str) -> list[int]:
        """Seed entropy for one model of this analysis: the sampler seed and the model's index.

        A different stream per model, since sharing one would correlate results that are meant
        to be independent.
        """
        seed = self.block["sampler"].get("seed")
        return [0 if seed is None else int(seed), self.models.index(model)]

    def reconstruction_rng(self, model: str) -> np.random.Generator:
        """Generator for the Monte-Carlo steps of the evidence estimators (§6.5)."""
        return np.random.default_rng(self.model_entropy(model))

    def daily_fit_seed(self, model: str) -> Callable[[int], int]:
        """Sampler seed for the day-``t`` refit, reproducible and distinct per model and day.

        Re-using one seed across the 110 days would correlate the curve's Monte-Carlo error from
        day to day, which is exactly what the standard error beside it is there to measure.
        """
        entropy = self.model_entropy(model)

        def seed(day: int) -> int:
            return int(np.random.SeedSequence([*entropy, day]).generate_state(1)[0])

        return seed

    def load_posteriors(self, paths: list[Path]) -> dict[str, xr.DataTree]:
        """Fits keyed by the model each one records itself as, in this analysis's order."""
        fits = {}
        for path in paths:
            idata = fitting.load_fit(path)
            fits[self.require_model(str(idata.attrs[fitting.MODEL_ATTRIBUTE]))] = idata
        return {model: fits[model] for model in self.models if model in fits}


def setting_from_arguments(args: argparse.Namespace) -> AnalysisSetting:
    """The :class:`AnalysisSetting` a parsed command line asks for."""
    return AnalysisSetting.load(args.analysis, config_path=args.config, data_path=args.data)


# ---------------------------------------------------------------------------------------
# Tier 1 — the fit
# ---------------------------------------------------------------------------------------


def command_fit(args: argparse.Namespace) -> None:
    """Fit one model to the complete record and save the draws."""
    setting = setting_from_arguments(args)
    model = setting.require_model(args.model)
    R_pre, R_post = setting.reproduction_numbers()

    idata = fitting.fit_model(
        model,
        setting.data.onsets,
        delays=setting.delays,
        switch_day=setting.switch_day(),
        R_pre=R_pre,
        R_post=R_post,
        k=setting.dispersion(),
        latent_parameterisation=setting.latent_parameterisation,
        negligible_latent_threshold=setting.negligible_latent_threshold,
        reporting_model=setting.reporting_model(),
        sampler=fitting.SamplerSettings.from_config(setting.block["sampler"]),
        progressbar=args.progressbar,
    )
    fitting.save_fit(idata, args.output)
    print(f"{model}: {int(idata.posterior.sizes['chain'])} chains saved to {args.output}")


# ---------------------------------------------------------------------------------------
# Tier 1 — the risk of additional cases, one fit per conditioning day
# ---------------------------------------------------------------------------------------


def command_rac(args: argparse.Namespace) -> None:
    """Estimate the RAC (and, for the onset models, RAT) curve.

    Under ``refit_daily`` this is a **tier-1** step: the estimand conditions on the record
    through day ``t``, so it fits the model once per conditioning day. The alternative reuses a
    single full-record fit and filters the latents, which is faster and approximate.

    The curves and their Monte-Carlo standard errors come from a single evaluation of the
    per-draw log-probabilities: RAC(t) is a posterior *average*, so it carries Monte-Carlo
    error, and a curve published without it cannot be compared with another one.
    """
    setting = setting_from_arguments(args)
    model = setting.require_model(args.model)
    data = setting.data
    method = setting.rac_method(args.method)
    first_day = int(setting.rac["first_day"])
    days = refit_risk.conditioning_days(data.onsets.size, first_day=first_day)
    if method == REFIT_DAILY and args.diagnostics is None:
        raise ValueError(
            f"{method} runs one fit per conditioning day, so --diagnostics is required: a "
            "curve built from 110 fits nobody has looked at is not a result"
        )
    setting.rac_method(args.method)  # reject an unknown --method before hours of sampling

    if method == REFIT_DAILY:
        estimate, diagnostics = _rac_by_refitting(
            setting, model, days=days, posterior=args.posterior, jobs=int(args.jobs)
        )
    else:
        estimate, diagnostics = _rac_by_filtering(
            setting, model, days=days, posterior=args.posterior
        )

    cases = estimate.risk_of_additional_cases()
    transmission = estimate.risk_of_additional_transmission()
    case_error, transmission_error = estimate.standard_errors()
    columns: dict[str, Any] = {
        "date": [data.date_of(int(day)) for day in estimate.days],
        "day": estimate.days,
        "model": model,
        "risk_of_additional_cases": cases.risk,
        "monte_carlo_standard_error": case_error,
    }
    if model.endswith("_so"):
        # RAC and RAT coincide under every infection-anchored model, so writing both columns
        # there would suggest a comparison the models cannot make.
        columns["risk_of_additional_transmission"] = transmission.risk
        columns["transmission_monte_carlo_standard_error"] = transmission_error
    pd.DataFrame(columns).to_csv(configuration.ensure_parent(args.output), index=False)
    if args.diagnostics is not None:
        diagnostics.insert(0, "method", method)
        diagnostics.to_csv(configuration.ensure_parent(args.diagnostics), index=False)

    print(
        f"{model}: RAC over days {int(estimate.days[0])}–{int(estimate.days[-1])} by {method}; "
        f"written to {args.output}"
    )
    _report_convergence(setting, model, diagnostics, path=args.diagnostics)


def _report_convergence(
    setting: AnalysisSetting, model: str, diagnostics: pd.DataFrame, *, path: Path | None
) -> None:
    """Act on the acceptance criteria, after the curve is safely on disk.

    The curve is written either way, and is identical either way: the criteria decide whether a
    shaky-looking set of fits stops the build, not what the answer is. Refusing to write an
    already-computed curve would throw away an hour of sampling and tell nobody anything the
    diagnostics table does not already say — and Snakemake deletes a failed job's outputs, so
    the evidence would go with it. Hence: write, then judge, and always name the days.
    """
    suspect = diagnostics.attrs.get("suspect")
    if not suspect:
        return
    criteria = setting.rac.get("convergence", {})
    listed = ", ".join(
        f"day {day.day} (R̂ {day.max_r_hat:.4f}, {day.divergences} divergences)"
        for day in suspect[:10]
    )
    more = "" if len(suspect) <= 10 else f", and {len(suspect) - 10} more"
    summary = (
        f"{model}: {len(suspect)} of {len(diagnostics)} conditioning-day fits fail the "
        f"acceptance criteria (R̂ > {criteria.get('max_r_hat')}, divergences > "
        f"{criteria.get('divergence_fraction'):.0%} of draws): {listed}{more}."
    )
    if str(criteria.get("on_failure", "warn")) == "error":
        raise RuntimeError(f"{summary} Full table: {path}.")
    print(f"WARNING: {summary} Full table: {path}.", file=sys.stderr)


def _rac_by_refitting(
    setting: AnalysisSetting, model: str, *, days: np.ndarray, posterior: Path, jobs: int = 1
) -> tuple[rac.DailyRiskEstimate, pd.DataFrame]:
    """The gold standard: one fit per conditioning day, with every day's diagnostics kept.

    The full-record fit the pipeline already holds is reused for the last conditioning day
    rather than repeated, so the end of the curve and the evidence and dispersion summaries
    describe the same posterior.

    ``jobs`` spreads the days over worker processes. It changes the wall time and nothing else:
    each day carries a seed derived from the analysis and the day, so the curve is the same
    whether one process or eight produced it.
    """
    data = setting.data
    R_pre, R_post = setting.reproduction_numbers()
    reported = {int(day) for day in np.linspace(days[0], days[-1], 12).round()}
    result = refit_risk.risk_by_refitting(
        model,
        data.onsets,
        delays=setting.delays,
        switch_day=setting.switch_day(),
        R_pre=R_pre,
        R_post=R_post,
        k=setting.dispersion(),
        latent_parameterisation=setting.latent_parameterisation,
        negligible_latent_threshold=setting.negligible_latent_threshold,
        reporting_model=setting.reporting_model(),
        days=days,
        sampler=fitting.SamplerSettings.from_config(setting.block["sampler"]),
        seed_for_day=setting.daily_fit_seed(model),
        final_day_fit=fitting.load_fit(posterior),
        on_day=lambda day: _report_day(model, day, reported),
        n_jobs=jobs,
    )
    diagnostics = pd.DataFrame(
        {
            "day": [day.day for day in result.diagnostics],
            "divergences": [day.divergences for day in result.diagnostics],
            "max_r_hat": [day.max_r_hat for day in result.diagnostics],
            "min_ess_bulk": [day.min_ess_bulk for day in result.diagnostics],
            "seconds": [day.seconds for day in result.diagnostics],
        }
    )
    # Carried on the frame rather than raised here, so that `command_rac` writes the curve and
    # the table before anything decides whether to stop the build.
    criteria = setting.rac.get("convergence", {})
    diagnostics.attrs["suspect"] = result.suspect_days(
        max_r_hat=float(criteria.get("max_r_hat", 1.02)),
        divergence_fraction=float(criteria.get("divergence_fraction", 0.01)),
    )
    return result.estimate, diagnostics


def _report_day(model: str, day: refit_risk.DayDiagnostics, reported: set[int]) -> None:
    """Progress for a long tier-1 step, at a dozen days rather than all of them."""
    if day.day in reported:
        print(
            f"  {model} day {day.day}: {day.seconds:.1f}s, R̂ {day.max_r_hat:.3f}, "
            f"{day.divergences} divergences",
            flush=True,
        )


def _rac_by_filtering(
    setting: AnalysisSetting, model: str, *, days: np.ndarray, posterior: Path
) -> tuple[rac.DailyRiskEstimate, pd.DataFrame]:
    """The approximation: one full-record fit, latents filtered to each conditioning day."""
    data = setting.data
    idata = fitting.load_fit(posterior)
    fixed_R_pre, fixed_R_post = fitting.fitted_reproduction_numbers(idata)
    state = rac.posterior_state(
        model,
        idata,
        data.onsets,
        delays=setting.delays,
        switch_day=setting.switch_day(),
        latent_parameterisation=fitting.fitted_parameterisation(idata),
        fixed_R_pre=fixed_R_pre,
        fixed_R_post=fixed_R_post,
        fixed_k=fitting.fitted_dispersion(idata),
        negligible_latent_threshold=setting.negligible_latent_threshold,
        reporting_model=fitting.fitted_reporting(idata),
    )
    filtering = setting.rac["filtering"]
    # Thinning exists only to bound the cost of running a filter per draw, so a model with no
    # latent block keeps every draw and reproduces the closed form exactly.
    if specification_of(model).has_latents or not setting.reporting_model().is_complete:
        state = filtered_risk.thin_draws(state, int(filtering["n_draws"]))
    result = filtered_risk.risk_by_filtering(
        model,
        state,
        counts=data.onsets,
        delays=setting.delays,
        switch_day=setting.switch_day(),
        days=days,
        n_particles=int(filtering["n_particles"]),
        reporting_model=fitting.fitted_reporting(idata),
        as_of_day=fitting.fitted_as_of_day(idata, data.onsets),
        rng=setting.reconstruction_rng(model),
    )
    # DLO and SSE run no filter — they have no latent state — so there is nothing per-day to
    # diagnose, and the file says so rather than inventing columns.
    diagnostics = pd.DataFrame({"day": result.estimate.days})
    if result.diagnostics is not None:
        diagnostics["min_effective_sample_size"] = result.diagnostics.min_effective_sample_size
        diagnostics["mean_effective_sample_size"] = result.diagnostics.mean_effective_sample_size
        diagnostics["resample_fraction"] = result.diagnostics.resample_fraction
    return result.estimate, diagnostics


# ---------------------------------------------------------------------------------------
# Tier 2 — model evidence and posterior model probabilities
# ---------------------------------------------------------------------------------------


def command_evidence(args: argparse.Namespace) -> None:
    """Marginal likelihoods for every model in the analysis, and the probabilities they imply.

    What the model-probability panel then reports is a comparison of the models **as specified,
    priors included** (§6.5): identical priors make the Bayes factors well defined, not neutral,
    because ``k`` means an individual-level offspring dispersion in SSE/SSI and a day-level
    incidence dispersion in DLO. The caption has to say so; this file only computes the numbers.
    """
    setting = setting_from_arguments(args)
    R_pre, R_post = setting.reproduction_numbers()
    k = setting.dispersion()

    estimates = {
        model: model_evidence.log_evidence(
            model,
            idata,
            setting.data.onsets,
            delays=setting.delays,
            switch_day=setting.switch_day(),
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent_parameterisation=fitting.fitted_parameterisation(idata),
            negligible_latent_threshold=setting.negligible_latent_threshold,
            estimator=args.estimator,
            rng=setting.reconstruction_rng(model),
        )
        for model, idata in setting.load_posteriors(args.posteriors).items()
    }
    ordered = list(estimates)
    probabilities = model_evidence.posterior_model_probabilities(
        {model: estimates[model].log_evidence for model in ordered}
    )
    _write_json(
        args.output,
        {
            "analysis": setting.name,
            "models": ordered,
            "estimator": args.estimator,
            "log_evidence": {model: estimates[model].log_evidence for model in ordered},
            "log_evidence_standard_error": {
                model: estimates[model].standard_error for model in ordered
            },
            "n_draws": {model: estimates[model].n_draws for model in ordered},
            "posterior_model_probability": probabilities,
        },
    )

    summary = ", ".join(f"{model} {probabilities[model]:.3g}" for model in ordered)
    print(f"posterior model probabilities ({args.estimator}): {summary} → {args.output}")


# ---------------------------------------------------------------------------------------
# Tier 2 — the dispersion posteriors, for the analyses that estimate `k`
# ---------------------------------------------------------------------------------------


def command_dispersion(args: argparse.Namespace) -> None:
    """Summarise each model's ``k`` posterior and measure how far the three of them differ.

    This is the quantitative half of the Stage-7 result. The models are handed the *same*
    prior on ``k`` (§6.2), so wherever their posteriors end up apart it is the data speaking
    about a symbol that means different things in different mechanisms — a day-level incidence
    dispersion in DLO, an individual-level offspring dispersion in SSE/SSI (§5.4). Every pair
    is compared, rather than a designated reference model, so the same file serves Analysis 4
    where there is no DLO to be the reference.
    """
    setting = setting_from_arguments(args)
    prior = setting.dispersion()
    if not isinstance(prior, LogNormalPrior):
        raise ValueError(
            f"{setting.name} holds k fixed at {prior:g}, so there is no k posterior to "
            "summarise. This subcommand belongs to the analyses that estimate k."
        )

    draws = {
        model: _dispersion_draws(idata, model)
        for model, idata in setting.load_posteriors(args.posteriors).items()
    }
    summaries = {model: pc.summarise_posterior(values) for model, values in draws.items()}
    pairs = {
        f"{first}_vs_{second}": pc.compare_posteriors(draws[first], draws[second]).as_dict()
        for index, first in enumerate(draws)
        for second in list(draws)[index + 1 :]
    }
    _write_json(
        args.output,
        {
            "analysis": setting.name,
            "models": list(draws),
            "prior": {
                "median": prior.median,
                "sigma_log": prior.sigma_log,
                "interval_lower": float(prior.frozen().ppf(0.025)),
                "interval_upper": float(prior.frozen().ppf(0.975)),
            },
            "posterior": {model: summary.as_dict() for model, summary in summaries.items()},
            "divergence": pairs,
        },
    )

    located = ", ".join(f"{model} {summary.median:.3g}" for model, summary in summaries.items())
    print(f"posterior median k: {located} → {args.output}")


def _dispersion_draws(idata: xr.DataTree, model: str) -> np.ndarray:
    """One fit's ``k`` draws, chains flattened, with a useful error when ``k`` was fixed."""
    posterior = idata.posterior
    if DISPERSION_VARIABLE not in posterior.data_vars:
        raise ValueError(
            f"the {model} fit has no {DISPERSION_VARIABLE!r} in its posterior; it was run with "
            "k fixed, which makes k a constant in the graph rather than a sampled variable"
        )
    return np.asarray(posterior.data_vars[DISPERSION_VARIABLE]).reshape(-1)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a tier-2 summary, creating its directory."""
    with open(configuration.ensure_parent(path), "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------


def add_jobs_argument(parser: argparse.ArgumentParser) -> None:
    """The worker-process count for ``refit_daily``, shared with ``scripts/run_risk_curves.py``.

    A command-line flag rather than a config value, and deliberately absent from the
    ``Snakefile``'s ``rac`` params. How many cores a machine has changes how long the curve
    takes and not one number in it, so recording it as a rerun trigger would re-run hours of
    MCMC to reproduce a byte-identical file. The pipeline passes Snakemake's ``{threads}``.
    """
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker processes to spread the conditioning days over (refit_daily only)",
    )


def parse_arguments(analysis: str, argv: list[str] | None = None) -> argparse.Namespace:
    """The four subcommands, with the analysis baked in by the calling run script."""
    parser = argparse.ArgumentParser(description=f"pipeline steps for the {analysis} analysis")
    parser.set_defaults(analysis=analysis)
    subcommands = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        subparser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
        subparser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
        subparser.add_argument("--output", type=Path, required=True)
        return subparser

    fit = common(subcommands.add_parser("fit", help="MCMC fit for one model (tier 1)"))
    fit.add_argument("--model", required=True)
    fit.add_argument("--progressbar", action="store_true")
    fit.set_defaults(handler=command_fit)

    risk = common(
        subcommands.add_parser("rac", help="risk of additional cases (tier 1 under refit_daily)")
    )
    risk.add_argument("--model", required=True)
    risk.add_argument(
        "--posterior",
        type=Path,
        required=True,
        help="the full-record fit: the whole state under single_fit_filtered, and the last "
        "conditioning day's fit under refit_daily",
    )
    risk.add_argument(
        "--diagnostics",
        type=Path,
        help="where to write the per-day sampler diagnostics (refit_daily only)",
    )
    risk.add_argument(
        "--method",
        choices=RAC_METHODS,
        help="override config's rac.method; the default refits per conditioning day",
    )
    add_jobs_argument(risk)
    risk.set_defaults(handler=command_rac)

    evidence = common(subcommands.add_parser("evidence", help="model evidence (tier 2)"))
    evidence.add_argument("--posteriors", type=Path, nargs="+", required=True)
    evidence.add_argument(
        "--estimator",
        choices=model_evidence.ESTIMATORS,
        default=model_evidence.BRIDGE_SAMPLING,
        help="the other two are agreement checks (§6.5); see validation/run_evidence_validation.py",
    )
    evidence.set_defaults(handler=command_evidence)

    dispersion = common(
        subcommands.add_parser("dispersion", help="k posteriors and their divergence (tier 2)")
    )
    dispersion.add_argument("--posteriors", type=Path, nargs="+", required=True)
    dispersion.set_defaults(handler=command_dispersion)

    return parser.parse_args(argv)


def main(analysis: str, argv: list[str] | None = None) -> None:
    """Run one subcommand of one analysis."""
    arguments = parse_arguments(analysis, argv)
    arguments.handler(arguments)

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

Four subcommands, one per pipeline rule, so that recomputing a risk curve never re-runs a fit::

    python scripts/run_<analysis>.py fit        --model ssi --output ..._posterior.nc
    python scripts/run_<analysis>.py rac        --model ssi --posterior ... --output ..._rac.csv
    python scripts/run_<analysis>.py evidence   --posteriors ... --output model_evidence.json
    python scripts/run_<analysis>.py dispersion --posteriors ... --output ..._posteriors.json

``dispersion`` applies only to the analyses that estimate ``k``, and says so rather than writing
an empty file when it is pointed at a fixed-``k`` fit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from end_of_outbreak import configuration, fitting, model_evidence, outbreak_data
from end_of_outbreak import posterior_comparison as pc
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import LogNormalPrior

DISPERSION_VARIABLE = "k"
"""Name the model builders give the dispersion parameter in the posterior."""


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

    def reproduction_number_priors(self) -> tuple[LogNormalPrior, LogNormalPrior]:
        """``R_pre`` and ``R_post``, shared by every model so the evidences compare."""
        priors = self.block["shared"]["priors"]
        return (
            LogNormalPrior.from_config(priors["R_pre"]),
            LogNormalPrior.from_config(priors["R_post"]),
        )

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

    def reconstruction_rng(self, model: str) -> np.random.Generator:
        """Generator for the latent reconstruction, seeded reproducibly and distinctly per model.

        The default parameterisation integrates some latents out exactly, so they have to be
        drawn back from their conditional before a RAC curve can be computed (§6.3). That draw
        is Monte Carlo, so it needs a seed that lives in the config like every other one — and a
        different stream per model, since sharing one would correlate curves that are meant to
        be independent.
        """
        seed = self.block["sampler"].get("seed")
        entropy = [0 if seed is None else int(seed), self.models.index(model)]
        return np.random.default_rng(entropy)

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
    R_pre, R_post = setting.reproduction_number_priors()

    idata = fitting.fit_model(
        model,
        setting.data.onsets,
        delays=setting.delays,
        switch_day=setting.data.ert_arrival_day,
        R_pre=R_pre,
        R_post=R_post,
        k=setting.dispersion(),
        latent_parameterisation=setting.latent_parameterisation,
        negligible_latent_threshold=setting.negligible_latent_threshold,
        sampler=fitting.SamplerSettings.from_config(setting.block["sampler"]),
        progressbar=args.progressbar,
    )
    fitting.save_fit(idata, args.output)
    print(f"{model}: {int(idata.posterior.sizes['chain'])} chains saved to {args.output}")


# ---------------------------------------------------------------------------------------
# Tier 2 — the risk of additional cases
# ---------------------------------------------------------------------------------------


def command_rac(args: argparse.Namespace) -> None:
    """Turn one fit into its RAC curve.

    The curve and its Monte-Carlo standard error come from a single evaluation of the per-draw
    log-probabilities: RAC(t) is a posterior *average*, so it carries Monte-Carlo error, and a
    curve published without it cannot be compared with another one.
    """
    setting = setting_from_arguments(args)
    model = setting.require_model(args.model)
    data = setting.data
    idata = fitting.load_fit(args.posterior)

    state = rac.posterior_state(
        model,
        idata,
        data.onsets,
        delays=setting.delays,
        switch_day=data.ert_arrival_day,
        latent_parameterisation=fitting.fitted_parameterisation(idata),
        fixed_k=fitting.fitted_dispersion(idata),
        negligible_latent_threshold=setting.negligible_latent_threshold,
        rng=setting.reconstruction_rng(model),
    )
    log_probability = rac.log_probability_of_no_further_cases(
        model,
        counts=data.onsets,
        serial_interval=setting.delays.serial_interval,
        R_pre=state.R_pre,
        k=state.k,
        infectivity=state.infectivity,
    )
    curve = rac.RiskCurve(
        days=data.day_index,
        risk=1.0 - np.exp(log_probability).mean(axis=0),
        n_draws=state.n_draws,
    )
    frame = pd.DataFrame(
        {
            "date": [data.date_of(int(day)) for day in curve.days],
            "day": curve.days,
            "model": model,
            "risk_of_additional_cases": curve.risk,
            "monte_carlo_standard_error": rac.monte_carlo_standard_error(
                log_probability, n_chains=state.n_chains
            ),
        }
    )
    frame.to_csv(configuration.ensure_parent(args.output), index=False)

    crossings = " ".join(
        f"{threshold:g}→{_crossing_text(curve, threshold, data)}" for threshold in (0.05, 0.01)
    )
    print(f"{model}: RAC first settles below {crossings}; written to {args.output}")


def _crossing_text(curve: rac.RiskCurve, threshold: float, data: outbreak_data.OutbreakData) -> str:
    """The date a curve settles below a threshold, or a marker that it never does."""
    day = curve.first_day_below(threshold)
    return "never" if day is None else f"day {day} ({data.date_of(day).isoformat()})"


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
    R_pre, R_post = setting.reproduction_number_priors()
    k = setting.dispersion()

    estimates = {
        model: model_evidence.log_evidence(
            model,
            idata,
            setting.data.onsets,
            delays=setting.delays,
            switch_day=setting.data.ert_arrival_day,
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

    risk = common(subcommands.add_parser("rac", help="risk of additional cases (tier 2)"))
    risk.add_argument("--model", required=True)
    risk.add_argument("--posterior", type=Path, required=True)
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

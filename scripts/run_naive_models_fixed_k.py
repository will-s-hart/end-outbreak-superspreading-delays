"""Analysis 1 — the three naive models with ``k`` held at 0.18 (→ Fig. 1).

DLO, SSE and SSI all anchored at symptom onsets-as-infections, all driven by the same serial
interval, and all handed the *same* individual-level dispersion estimate. That last part is the
point of the analysis rather than an approximation: `k = 0.18` is an offspring-distribution
dispersion, so applying it to DLO — where the dispersion acts on a whole day's aggregate
incidence — is a deliberate reproduction of a mistake made in the literature (§1 aim 2, §5.4).
The comparison of the resulting RAC curves is the headline of Fig. 1.

Three subcommands, one per pipeline tier, so that restyling a figure never re-runs MCMC and
recomputing a risk curve never re-runs a fit::

    python scripts/run_naive_models_fixed_k.py fit      --model ssi --output ..._posterior.nc
    python scripts/run_naive_models_fixed_k.py rac      --model ssi --output ..._rac.csv
    python scripts/run_naive_models_fixed_k.py evidence --posteriors ... --output ...json

Everything that varies between the four analyses is read from ``config/config.yaml`` under the
analysis named by :data:`ANALYSIS`, so the three estimated-``k`` and onset-anchored siblings of
Stages 7 and 9 differ from this file in that constant and in their docstrings. Factor a shared
driver when the second one lands and the real shape of the duplication is visible — not before.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from end_of_outbreak import configuration, fitting, model_evidence, outbreak_data
from end_of_outbreak import risk_of_additional_cases as rac
from end_of_outbreak.delay_distributions import OnsetAnchoredDelays
from end_of_outbreak.model_specifications import LogNormalPrior

ANALYSIS = "naive_models_fixed_k"


# ---------------------------------------------------------------------------------------
# Config → the objects every subcommand needs
# ---------------------------------------------------------------------------------------


def analysis_setting(
    config_path: str | Path, data_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], outbreak_data.OutbreakData, OnsetAnchoredDelays]:
    """The whole config, the analysis's block, the onset series, and the delay triple.

    The full config comes back too because the latent parameterisation and the negligible-latent
    threshold sit outside the per-analysis blocks: they are properties of the *implementation*
    chosen once by the Stage-3 benchmark, so all four analyses share them.
    """
    config = configuration.load_config(config_path)
    analysis = configuration.analysis_config(config, ANALYSIS)
    data = outbreak_data.load_onset_data(data_path)
    delays = configuration.onset_anchored_delays_from_config({"shared": analysis["shared"]})
    return config, analysis, data, delays


def reproduction_number_priors(analysis: dict[str, Any]) -> tuple[LogNormalPrior, LogNormalPrior]:
    """The ``R_pre`` and ``R_post`` priors, shared by every model so the evidences compare."""
    priors = analysis["shared"]["priors"]
    return (
        LogNormalPrior.from_config(priors["R_pre"]),
        LogNormalPrior.from_config(priors["R_post"]),
    )


def dispersion(analysis: dict[str, Any]) -> float | LogNormalPrior:
    """``k`` as the builders take it: a float where it is fixed, a prior where it is estimated.

    A fixed ``k`` becomes a constant in the PyMC graph rather than a random variable, which is
    why it has to be handed back to the RAC calculators explicitly later — it is nowhere in the
    draws.
    """
    fixed = analysis.get("fixed_k")
    if fixed is not None:
        return float(fixed)
    return LogNormalPrior.from_config(analysis["k_prior"])


def require_model(analysis: dict[str, Any], model: str) -> str:
    """Reject a model that is not part of this analysis, rather than fitting it anyway."""
    if model not in analysis["models"]:
        known = ", ".join(analysis["models"])
        raise ValueError(f"{ANALYSIS} does not include model {model!r}; it compares {known}")
    return model


def reconstruction_rng(analysis: dict[str, Any], model: str) -> np.random.Generator:
    """Generator for the latent reconstruction, seeded reproducibly and distinctly per model.

    The default parameterisation integrates some latents out exactly, so they have to be drawn
    back from their conditional before a RAC curve can be computed (§6.3). That draw is Monte
    Carlo, so it needs a seed that lives in the config like every other one — and a different
    stream per model, since sharing one would correlate curves that are meant to be independent.
    """
    seed = analysis["sampler"].get("seed")
    entropy = [0 if seed is None else int(seed), analysis["models"].index(model)]
    return np.random.default_rng(entropy)


# ---------------------------------------------------------------------------------------
# Tier 1 — the fit
# ---------------------------------------------------------------------------------------


def command_fit(args: argparse.Namespace) -> None:
    """Fit one model to the complete record and save the draws."""
    config, analysis, data, delays = analysis_setting(args.config, args.data)
    model = require_model(analysis, args.model)
    R_pre, R_post = reproduction_number_priors(analysis)

    idata = fitting.fit_model(
        model,
        data.onsets,
        delays=delays,
        switch_day=data.ert_arrival_day,
        R_pre=R_pre,
        R_post=R_post,
        k=dispersion(analysis),
        latent_parameterisation=config["latent_parameterisation"],
        negligible_latent_threshold=float(config["negligible_latent_threshold"]),
        sampler=fitting.SamplerSettings.from_config(analysis["sampler"]),
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
    config, analysis, data, delays = analysis_setting(args.config, args.data)
    model = require_model(analysis, args.model)
    idata = fitting.load_fit(args.posterior)

    state = rac.posterior_state(
        model,
        idata,
        data.onsets,
        delays=delays,
        switch_day=data.ert_arrival_day,
        latent_parameterisation=fitting.fitted_parameterisation(idata),
        fixed_k=fitting.fitted_dispersion(idata),
        negligible_latent_threshold=float(config["negligible_latent_threshold"]),
        rng=reconstruction_rng(analysis, model),
    )
    log_probability = rac.log_probability_of_no_further_cases(
        model,
        counts=data.onsets,
        serial_interval=delays.serial_interval,
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

    What panel C then reports is a comparison of the models **as specified, priors included**
    (§6.5). For this analysis one of those specifications applies an individual-level ``k`` to a
    day-level mechanism on purpose, so the evidence is a statement about that set of models and
    not clean evidence about the mechanism of transmission heterogeneity. The caption has to say
    so; this file only computes the numbers.
    """
    config, analysis, data, delays = analysis_setting(args.config, args.data)
    R_pre, R_post = reproduction_number_priors(analysis)
    k = dispersion(analysis)
    threshold = float(config["negligible_latent_threshold"])

    estimates = {}
    for path in args.posteriors:
        idata = fitting.load_fit(path)
        model = require_model(analysis, str(idata.attrs[fitting.MODEL_ATTRIBUTE]))
        estimates[model] = model_evidence.log_evidence(
            model,
            idata,
            data.onsets,
            delays=delays,
            switch_day=data.ert_arrival_day,
            R_pre=R_pre,
            R_post=R_post,
            k=k,
            latent_parameterisation=fitting.fitted_parameterisation(idata),
            negligible_latent_threshold=threshold,
            estimator=args.estimator,
            rng=reconstruction_rng(analysis, model),
        )

    ordered = [model for model in analysis["models"] if model in estimates]
    probabilities = model_evidence.posterior_model_probabilities(
        {model: estimates[model].log_evidence for model in ordered}
    )
    payload = {
        "analysis": ANALYSIS,
        "models": ordered,
        "estimator": args.estimator,
        "log_evidence": {model: estimates[model].log_evidence for model in ordered},
        "log_evidence_standard_error": {
            model: estimates[model].standard_error for model in ordered
        },
        "n_draws": {model: estimates[model].n_draws for model in ordered},
        "posterior_model_probability": probabilities,
    }
    with open(configuration.ensure_parent(args.output), "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    summary = ", ".join(f"{model} {probabilities[model]:.3g}" for model in ordered)
    print(f"posterior model probabilities ({args.estimator}): {summary} → {args.output}")


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()

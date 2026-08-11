"""Compute RAC, RAT and RST together from one sequence of conditioning-day fits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import analysis_driver
import pandas as pd

from end_of_outbreak import configuration, outbreak_data, refit_risk
from end_of_outbreak import risk_of_additional_cases as rac


def command_risk(args: argparse.Namespace) -> None:
    """Run the selected estimator once and write every applicable risk metric."""
    setting = analysis_driver.setting_from_arguments(args)
    model = setting.require_model(args.model)
    data = setting.data
    method = setting.rac_method(args.method)
    days = refit_risk.conditioning_days(data.onsets.size, first_day=int(setting.rac["first_day"]))
    if method == analysis_driver.REFIT_DAILY and args.diagnostics is None:
        raise ValueError(
            f"{method} runs one fit per conditioning day, so --diagnostics is required: a "
            "curve built from 110 fits nobody has looked at is not a result"
        )

    if method == analysis_driver.REFIT_DAILY:
        estimate, diagnostics = analysis_driver._rac_by_refitting(
            setting, model, days=days, posterior=args.posterior, jobs=int(args.jobs)
        )
    else:
        estimate, diagnostics = analysis_driver._rac_by_filtering(
            setting, model, days=days, posterior=args.posterior
        )

    columns = risk_columns(estimate, model=model, data=data)
    pd.DataFrame(columns).to_csv(configuration.ensure_parent(args.output), index=False)

    if args.diagnostics is not None:
        diagnostics.insert(0, "method", method)
        diagnostics.to_csv(configuration.ensure_parent(args.diagnostics), index=False)
    print(
        f"{model}: risks over days {int(estimate.days[0])}–{int(estimate.days[-1])} by "
        f"{method}; written to {args.output}"
    )
    analysis_driver._report_convergence(setting, model, diagnostics, path=args.diagnostics)


def risk_columns(
    estimate: rac.DailyRiskEstimate, *, model: str, data: outbreak_data.OutbreakData
) -> dict[str, Any]:
    """CSV columns for every metric the model defines."""
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
        columns["risk_of_additional_transmission"] = transmission.risk
        columns["transmission_monte_carlo_standard_error"] = transmission_error
    if model != "dlo":
        columns["risk_of_sustained_transmission"] = estimate.risk_of_sustained_transmission().risk
        columns["sustained_transmission_monte_carlo_standard_error"] = (
            estimate.sustained_transmission_standard_error()
        )
    return columns


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(command="rac")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--model", required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--method", choices=analysis_driver.RAC_METHODS)
    parser.add_argument("--output", type=Path, required=True)
    analysis_driver.add_jobs_argument(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    command_risk(parse_arguments(argv))


if __name__ == "__main__":
    main()

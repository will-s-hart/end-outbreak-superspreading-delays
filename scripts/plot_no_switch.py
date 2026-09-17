"""RAC for the four onset-comparison models with the R switchpoint removed.

Two exploratory analyses share this script, because they differ only in whether there is a
parameter posterior to draw. Both remove the switch, so neither carries the ``R_pre``/``R_post``
pair the report's fixed-``k`` figure devotes two panels to, and neither draws posterior model
probabilities: under ``no_switch_fixed_R`` there is nothing left to integrate over.

- ``no_switch_fixed_R`` fixes ``R`` as well as ``k``, so the figure is the RAC panel alone.
- ``no_switch_single_R`` estimates one ``R`` per model, which gets a panel above the curves.
  It is read off ``R_pre``: with the switch past the end of the window that *is* the single
  reproduction number, and ``R_post`` reaches no day and merely returns its prior.

Both titles state that the reset to ``R_pre`` is a no-op here, so that no reader takes these
curves for the reset predictive the report's analyses report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import figure_panels
import matplotlib.pyplot as plt
import pandas as pd
import utils
import xarray as xr
from matplotlib.figure import Figure

from end_of_outbreak import configuration, outbreak_data
from end_of_outbreak.model_specifications import LogNormalPrior

FIXED_R = "no_switch_fixed_R"
SINGLE_R = "no_switch_single_R"
ANALYSES = (FIXED_R, SINGLE_R)

# Removing the switchpoint is what these figures are for, and it brings the onset-anchored
# curves onto the naive ones — SSI and SSI-SO to within the line width. That coincidence is the
# result, but four solid lines would show it as a curve that failed to draw, so the two
# onset-anchored models are dashed. The report's figures keep every model solid.
ONSET_ANCHORED_LINESTYLE = {"sse_so": "--", "ssi_so": "--"}


def build_fixed_R_figure(
    curves: dict[str, pd.DataFrame],
    data: outbreak_data.OutbreakData,
    *,
    fixed_k: float,
    fixed_R: float,
) -> Figure:
    """The RAC panel alone: with every parameter fixed there is no posterior to show."""
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 4.2))
    # One panel, so no letter: there is nothing for a caption to cross-refer to.
    figure_panels.risk_curve_panel(
        figure.add_subplot(1, 1, 1), curves, data, linestyles=ONSET_ANCHORED_LINESTYLE
    )
    figure.suptitle(
        f"Équateur 2018: no $R$ switchpoint, $R$ fixed at {fixed_R:g} and $k$ at {fixed_k:g} "
        "— the reset is a no-op, so this is a forward predictive",
        fontsize=9.5,
    )
    return figure


def build_single_R_figure(
    curves: dict[str, pd.DataFrame],
    posteriors: dict[str, xr.DataTree],
    data: outbreak_data.OutbreakData,
    *,
    R_prior: LogNormalPrior,
    fixed_k: float,
) -> Figure:
    """One estimated ``R`` above the RAC curves it produced."""
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 6.4))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.45)
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(grid[0, 0]),
        # `R_pre` is the single R: the switch sits past the end of the window.
        {model: utils.posterior_draws(posteriors[model], "R_pre") for model in curves},
        prior=R_prior,
        xlabel="$R$",
        title="One reproduction number for the whole outbreak",
        letter="A",
        legend=True,
    )
    figure_panels.risk_curve_panel(
        figure.add_subplot(grid[1, 0]),
        curves,
        data,
        letter="B",
        linestyles=ONSET_ANCHORED_LINESTYLE,
    )
    figure.suptitle(
        f"Équateur 2018: no $R$ switchpoint, a single estimated $R$ and $k$ fixed at "
        f"{fixed_k:g} — the reset is a no-op, so this is a forward predictive",
        fontsize=9.5,
    )
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    analysis = configuration.analysis_config(config, args.analysis)
    models = list(analysis["models"])
    data = outbreak_data.load_onset_data(args.data)
    curves = utils.read_risk_curves(args.results_dir, models)
    fixed_k = float(analysis["fixed_k"])

    if args.analysis == FIXED_R:
        figure = build_fixed_R_figure(
            curves, data, fixed_k=fixed_k, fixed_R=float(analysis["fixed_R_pre"])
        )
    else:
        figure = build_single_R_figure(
            curves,
            utils.read_posteriors(args.results_dir, models),
            data,
            R_prior=LogNormalPrior.from_config(analysis["shared"]["priors"]["R_pre"]),
            fixed_k=fixed_k,
        )
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--analysis", choices=ANALYSES, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

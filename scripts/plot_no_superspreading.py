"""Superspreading and onset anchoring, each with the other held fixed.

Four models, two by two: the Poisson limits ``cori`` and ``cori_so``, fitted in this analysis,
beside the SSI and SSI-SO of Analysis 3 (``compared_with`` in the config). Read along a row of
that square and anchoring varies with superspreading held fixed; down a column, superspreading
varies with anchoring held fixed. So the figure answers two questions at once: whether the
superspreading models are better supported than the Poisson ones (the pie, normalised over all
four), and whether superspreading or onset anchoring moves the risk further (the curves).

The borrowed models are drawn from Analysis 3's own results rather than refitted here, so they
are the curves of Fig. 4 exactly; the pie reads the combined evidence file, which checks that
the two analyses fitted their models under the same assumptions before normalising over both.

One analysis, so the analysis is hard-coded rather than taken as a flag. This is deliberately
*not* an extra branch in ``plot_onset_models_fixed_k.py``: that script reads
``analysis["fixed_k"]`` for its suptitle, which this analysis has no value for.

The RAC panel is :func:`figure_panels.risk_curve_panel` rather than ``risk_metrics_panel``,
because the settling-day markers are the result here: the gaps between them are the two effects
the figure exists to compare.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import figure_panels
import matplotlib.pyplot as plt
import pandas as pd
import utils
import xarray as xr
from matplotlib.figure import Figure

from end_of_outbreak import configuration, outbreak_data
from end_of_outbreak.model_specifications import LogNormalPrior

ANALYSIS = "onset_models_no_superspreading"

COMBINED_EVIDENCE_FILE = "combined_model_evidence.json"
"""Written by ``scripts/run_combined_evidence.py``: the four models' probabilities, together."""


def build_figure(
    *,
    curves: dict[str, pd.DataFrame],
    posteriors: dict[str, xr.DataTree],
    evidence: dict[str, Any],
    data: outbreak_data.OutbreakData,
    R_pre_prior: LogNormalPrior,
    R_post_prior: LogNormalPrior,
) -> Figure:
    """The fixed-``k`` four-panel layout, over all four models.

    The two ``R`` posteriors and the model probabilities across the top, the RAC curves along
    the bottom. There is no ``k`` panel: the Poisson limits have no ``k``, and SSI and SSI-SO
    hold it at Analysis 3's value, which the caption states.

    Colours are the report's palette throughout, where the Poisson limits are greys: dark for the
    naive one, mid for the onset-anchored one. Grey reads as "no superspreading", and SSI and
    SSI-SO keep the colours they have in every other figure.
    """
    utils.apply_house_style()
    figure = plt.figure(figsize=(9.4, 6.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.6, wspace=0.35)
    models = list(curves)
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(grid[0, 0]),
        {model: utils.posterior_draws(posteriors[model], "R_pre") for model in models},
        prior=R_pre_prior,
        xlabel="$R_{\\mathrm{pre}}$",
        title="Before ERT arrival",
        letter="A",
        legend=True,
    )
    figure_panels.parameter_posterior_panel(
        figure.add_subplot(grid[0, 1]),
        {model: utils.posterior_draws(posteriors[model], "R_post") for model in models},
        prior=R_post_prior,
        xlabel="$R_{\\mathrm{post}}$",
        title="After ERT arrival",
        letter="B",
    )
    figure_panels.model_probability_panel(
        figure.add_subplot(grid[0, 2]), evidence, models, letter="C"
    )
    # `first_day` left unset: the whole window, so that the climb over the opening weeks and
    # the descents are all visible, and the settling markers can be read off directly.
    figure_panels.risk_curve_panel(figure.add_subplot(grid[1, :]), curves, data, letter="D")
    figure.suptitle(
        "Équateur 2018: superspreading and onset anchoring, each with the other held fixed",
        fontsize=9.5,
    )
    return figure


def read_combined_evidence(results_dir: Path) -> dict[str, Any]:
    """The pie's probabilities, normalised over both analyses' models; no fallback."""
    path = results_dir / COMBINED_EVIDENCE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"no combined model evidence at {path}. The pie normalises over this analysis's "
            "models and the ones it borrows, which `rule combined_evidence` computes — run "
            f"`snakemake {path}` first. A single analysis's model_evidence.json is not a "
            "substitute: it normalises over half the models."
        )
    with open(path) as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    analysis = configuration.analysis_config(config, ANALYSIS)
    own_models = list(analysis["models"])
    other = str(analysis["compared_with"]["analysis"])
    borrowed = list(analysis["compared_with"]["models"])
    own_dir, other_dir = args.results_root / ANALYSIS, args.results_root / other
    priors = analysis["shared"]["priors"]
    figure = build_figure(
        curves=utils.read_risk_curves(own_dir, own_models)
        | utils.read_risk_curves(other_dir, borrowed),
        posteriors=utils.read_posteriors(own_dir, own_models)
        | utils.read_posteriors(other_dir, borrowed),
        evidence=read_combined_evidence(own_dir),
        data=outbreak_data.load_onset_data(args.data),
        R_pre_prior=LogNormalPrior.from_config(priors["R_pre"]),
        R_post_prior=LogNormalPrior.from_config(priors["R_post"]),
    )
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=configuration.REPO_ROOT / "results",
        help="holds this analysis's results and those of the analysis it borrows models from",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

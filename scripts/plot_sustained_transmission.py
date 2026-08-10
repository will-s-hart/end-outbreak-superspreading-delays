"""RAC, RAT and RST for the infection- and onset-anchored branching models."""

from __future__ import annotations

import argparse
from pathlib import Path

import figure_panels
import matplotlib.pyplot as plt
import utils

from end_of_outbreak import configuration, outbreak_data

ANALYSES = ("onset_models_fixed_k", "onset_models_estimated_k")
NAIVE_MODELS = ["sse", "ssi"]
ONSET_MODELS = ["sse_so", "ssi_so"]


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    data = outbreak_data.load_onset_data(args.data)
    directory = args.results_root / args.analysis
    utils.apply_house_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True, sharey=True)

    figure_panels.risk_metrics_panel(
        axes[0],
        utils.read_risk_curves(directory, NAIVE_MODELS),
        data,
        title="Infection-anchored models",
        letter="A",
        first_day=data.ert_arrival_day,
        infection_anchored=True,
    )
    figure_panels.risk_metrics_panel(
        axes[1],
        utils.read_risk_curves(directory, ONSET_MODELS),
        data,
        title="Onset-anchored models",
        letter="B",
        first_day=data.ert_arrival_day,
        infection_anchored=False,
    )
    figure.subplots_adjust(wspace=0.24)
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--results-root", type=Path, default=configuration.REPO_ROOT / "results")
    parser.add_argument("--analysis", choices=ANALYSES, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

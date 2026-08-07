"""Supplementary figure — RAC, RAT and the latent incubation-pipeline gap (§5.5)."""

from __future__ import annotations

import argparse
from pathlib import Path

import figure_panels
import matplotlib.pyplot as plt
import utils

from end_of_outbreak import configuration, outbreak_data

ANALYSES = ("onset_models_fixed_k", "onset_models_estimated_k")
ONSET_MODELS = ["sse_so", "ssi_so"]


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    data = outbreak_data.load_onset_data(args.data)
    utils.apply_house_style()
    figure, axes = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True)
    for ax, analysis, title, letter in zip(
        axes,
        ANALYSES,
        ("Dispersion fixed at $k=0.18$", "Dispersion estimated"),
        ("A", "B"),
        strict=True,
    ):
        curves = utils.read_risk_curves(args.results_root / analysis, ONSET_MODELS)
        figure_panels.rac_rat_panel(
            ax,
            curves,
            data,
            title=title,
            letter=letter,
            first_day=data.ert_arrival_day,
        )
    figure.suptitle(
        "Risk of additional cases and transmission: contribution of the incubation pipeline",
        fontsize=9.5,
    )
    figure.subplots_adjust(hspace=0.35)
    utils.save_figure(figure, pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--data", type=Path, default=outbreak_data.DEFAULT_DATA_FILE)
    parser.add_argument("--results-root", type=Path, default=configuration.REPO_ROOT / "results")
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

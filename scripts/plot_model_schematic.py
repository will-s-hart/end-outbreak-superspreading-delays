"""The methods schematic — what separates the infection- and onset-anchored models.

A schematic, and the one figure in the report that draws no data. Every position on every
timeline is illustrative: the two lanes of panel A are laid out roughly in proportion to the
fitted delays, but nothing here is read from ``results/`` and nothing here is measured. That is
also why it carries **no number**. The displacement in panel D is one incubation period, and
the report's caption quotes its length through ``\\resultnum`` like every other number in the
document; putting ``11`` on the canvas would hard-code a config value into a PDF.

The Poisson (Cori / Cori-SO) forms are drawn, so that the anchoring argument is not entangled
with the overdispersion mechanism: the retained-transmission and pipeline structure of panels B
and C is common to all five compared models, and only the exponents change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import utils
from matplotlib.axes import Axes
from matplotlib.figure import Figure

INFECTION_ANCHORED = "#0072B2"
ONSET_ANCHORED = "#CC79A7"
LATENT = "0.45"
"""Two Okabe–Ito colours for the two anchoring conventions, and grey for what is unobserved.

Deliberately *not* ``utils.MODEL_COLOURS``: those name the five compared models, and this figure
is about the convention a model is built on rather than about any one of them.
"""

# Illustrative event positions for panel A, in days on an arbitrary axis.
INFECTOR_ONSET = 0.0
TRANSMISSION = 3.9
INFECTEE_ONSET = 15.3


def _bare(ax: Axes) -> None:
    """Strip an axis back to a blank canvas: this figure has no measured coordinates."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _timeline(ax: Axes, y: float, x0: float, x1: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops={"arrowstyle": "-|>", "color": "0.75", "lw": 0.9, "shrinkA": 0, "shrinkB": 0},
    )


def _event(
    ax: Axes, x: float, y: float, *, colour: str, filled: bool = True, size: float = 7.0
) -> None:
    """One event on a timeline; hollow markers are the states no one observes."""
    ax.plot(
        [x],
        [y],
        marker="o",
        markersize=size,
        markerfacecolor=colour if filled else "white",
        markeredgecolor=colour,
        markeredgewidth=1.3,
        linestyle="none",
        clip_on=False,
        zorder=4,
    )


def _span_arrow(ax: Axes, x0: float, x1: float, y: float, *, colour: str, label: str) -> None:
    """A straight arrow at a fixed height with its label centred above it.

    Straight rather than curved: an ``arc3`` connection is computed in display coordinates, so
    its height depends on the panel's aspect and the labels drift over the arrows as soon as the
    figure is resized.
    """
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.2, "shrinkA": 0, "shrinkB": 0},
        zorder=3,
    )
    ax.text(0.5 * (x0 + x1), y + 0.18, label, ha="center", va="bottom", fontsize=7.4, color=colour)


def _heading(ax: Axes, x: float, y: float, text: str, colour: str) -> None:
    ax.text(x, y, text, fontsize=8, color=colour, fontweight="bold", ha="left", va="top")


def _note(ax: Axes, x: float, y: float, text: str, *, ha: str = "center", va: str = "top") -> None:
    ax.text(x, y, text, fontsize=7.2, color="0.35", ha=ha, va=va)


def _transmission_pair_panel(ax: Axes) -> None:
    """Panel A: one infector–infectee pair, dated by each convention."""
    ax.set_xlim(-3.0, 21.0)
    ax.set_ylim(0.0, 10.0)
    _bare(ax)

    _heading(
        ax,
        -3.0,
        9.9,
        "Infection-anchored (Cori):  onset dates used as infection dates",
        INFECTION_ANCHORED,
    )
    y = 7.6
    _timeline(ax, y, -2.5, 20.5)
    _event(ax, INFECTOR_ONSET, y, colour=INFECTION_ANCHORED)
    _event(ax, INFECTEE_ONSET, y, colour=INFECTION_ANCHORED)
    _span_arrow(
        ax,
        INFECTOR_ONSET,
        INFECTEE_ONSET,
        y + 0.8,
        colour=INFECTION_ANCHORED,
        label="serial interval $w_s$",
    )
    _note(ax, INFECTOR_ONSET, y - 0.45, "infector")
    _note(ax, INFECTEE_ONSET, y - 0.45, "infectee")

    _heading(
        ax,
        -3.0,
        5.8,
        "Onset-anchored (Cori-SO):  transmission dated from the infector's onset",
        ONSET_ANCHORED,
    )
    y = 3.0
    _timeline(ax, y, -2.5, 20.5)
    _event(ax, INFECTOR_ONSET, y, colour=ONSET_ANCHORED)
    _event(ax, TRANSMISSION, y, colour=LATENT, filled=False, size=6.5)
    _event(ax, INFECTEE_ONSET, y, colour=ONSET_ANCHORED)
    _span_arrow(
        ax,
        INFECTOR_ONSET,
        TRANSMISSION,
        y + 0.8,
        colour=ONSET_ANCHORED,
        label="onset to transmission $f^{\\mathrm{tost}}$",
    )
    _span_arrow(
        ax,
        TRANSMISSION,
        INFECTEE_ONSET,
        y + 0.8,
        colour=LATENT,
        label="incubation $f^{\\mathrm{inc}}$",
    )
    _note(ax, INFECTOR_ONSET, y - 0.45, "infector onset")
    _note(ax, TRANSMISSION, y - 0.45, "infection (latent)")
    _note(ax, INFECTEE_ONSET, y - 0.45, "infectee onset")

    ax.text(
        20.5,
        0.3,
        "same serial interval:  $w = f^{\\mathrm{tost}} * f^{\\mathrm{inc}}$",
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="0.2",
    )
    utils.panel_label(ax, "A")


def _reset_state_panel(ax: Axes, *, onset_anchored: bool, letter: str) -> None:
    """Panels B and C: the state each convention carries past the last day of the record."""
    colour = ONSET_ANCHORED if onset_anchored else INFECTION_ANCHORED
    ax.set_xlim(-13.0, 15.0)
    ax.set_ylim(0.0, 10.0)
    _bare(ax)
    ax.set_title(
        "Onset-anchored" if onset_anchored else "Infection-anchored",
        color=colour,
        fontweight="bold",
        fontsize=8.5,
        pad=4,
    )

    ax.plot([0, 0], [2.4, 8.4], color="0.35", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    _note(ax, -0.6, 8.5, "record ends, day $t$", ha="right", va="bottom")

    y = 5.7
    _timeline(ax, y, -12.5, 14.5)
    for day in (-10, -7, -4):
        _event(ax, day, y, colour=colour, size=6.5)
    _note(ax, -7, y - 0.45, "observed cases")
    _span_arrow(
        ax,
        -4,
        11.0,
        y + 1.4,
        colour=colour,
        label=("$W(t)$" if onset_anchored else "$\\Lambda(t)$") + ": transmission still to come",
    )

    y = 4.0
    if onset_anchored:
        for day in (2.0, 5.5, 9.0):
            _event(ax, day, y, colour=LATENT, filled=False, size=6.5)
        ax.text(10.6, y, "$M(t)$", fontsize=7.4, color="0.35", ha="left", va="center")
        _note(ax, 6.3, y - 0.5, "infected, not yet\nsymptomatic")
    else:
        _note(ax, 7.8, y, "$M(t) = 0$:  no hidden cases", va="center")

    ax.text(
        -12.5,
        0.9,
        (
            "P(no further case) $= e^{-R^{\\star}W(t)}\\,e^{-M(t)}$"
            if onset_anchored
            else "P(no further case) $= e^{-R^{\\star}\\Lambda(t)}$"
        ),
        fontsize=7.8,
        color="0.12",
        ha="left",
        va="bottom",
    )
    ax.text(
        -12.5,
        0.0,
        "RAC $>$ RAT" if onset_anchored else "RAC $=$ RAT",
        fontsize=7.8,
        color=colour,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    utils.panel_label(ax, letter)


def _intervention_panel(ax: Axes) -> None:
    """Panel D: the day the reproduction number switches, in each convention's own time index."""
    ax.set_xlim(-22.0, 30.0)
    ax.set_ylim(0.0, 10.0)
    _bare(ax)

    transmission_y, onset_y = 7.0, 2.6
    incubation = 11.0  # illustrative: one incubation period on this panel's arbitrary axis
    for y, name in ((transmission_y, "transmission"), (onset_y, "recorded onsets")):
        _timeline(ax, y, -21.5, 29.5)
        ax.text(-21.5, y + 0.35, name, fontsize=7.2, color="0.55", ha="left", va="bottom")

    ax.plot([0, 0], [1.4, 9.2], color="0.35", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    _note(ax, 0.0, 9.4, "control measures begin", va="bottom")

    # Onset-anchored: R switches for the transmission happening that day, and the recorded
    # onsets follow one incubation period later.
    _event(ax, 0.0, transmission_y, colour=ONSET_ANCHORED)
    _event(ax, incubation, onset_y, colour=ONSET_ANCHORED, filled=False, size=6.5)
    ax.annotate(
        "",
        xy=(incubation, onset_y),
        xytext=(0.0, transmission_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": ONSET_ANCHORED,
            "lw": 1.2,
            "shrinkA": 5,
            "shrinkB": 5,
        },
        zorder=3,
    )
    ax.text(
        incubation + 0.8,
        0.5 * (transmission_y + onset_y),
        "onsets react one\nincubation period later",
        fontsize=7.2,
        color=ONSET_ANCHORED,
        ha="left",
        va="center",
    )
    ax.text(
        1.0,
        transmission_y + 0.9,
        "onset-anchored: $R$ switches here",
        fontsize=7.4,
        color=ONSET_ANCHORED,
        ha="left",
        va="bottom",
    )

    # Infection-anchored: R switches for the onsets recorded that day, which dates the same
    # transmission an incubation period earlier than it happened.
    _event(ax, 0.0, onset_y, colour=INFECTION_ANCHORED)
    _event(ax, -incubation, transmission_y, colour=INFECTION_ANCHORED, filled=False, size=6.5)
    ax.annotate(
        "",
        xy=(-incubation, transmission_y),
        xytext=(0.0, onset_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": INFECTION_ANCHORED,
            "lw": 1.2,
            "linestyle": (0, (3, 2)),
            "shrinkA": 5,
            "shrinkB": 5,
        },
        zorder=3,
    )
    ax.text(
        -incubation - 0.8,
        0.5 * (transmission_y + onset_y),
        "implies transmission changed\none incubation period earlier",
        fontsize=7.2,
        color=INFECTION_ANCHORED,
        ha="right",
        va="center",
    )
    ax.text(
        1.0,
        onset_y - 0.9,
        "infection-anchored: $R$ switches here",
        fontsize=7.4,
        color=INFECTION_ANCHORED,
        ha="left",
        va="top",
    )
    utils.panel_label(ax, "D")


def build_figure() -> Figure:
    """Assemble the four panels: the transmission pair, the two reset states, the switch."""
    utils.apply_house_style()
    figure = plt.figure(figsize=(7.2, 6.9))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.3, wspace=0.14)
    _transmission_pair_panel(figure.add_subplot(grid[0, :]))
    _reset_state_panel(figure.add_subplot(grid[1, 0]), onset_anchored=False, letter="B")
    _reset_state_panel(figure.add_subplot(grid[1, 1]), onset_anchored=True, letter="C")
    _intervention_panel(figure.add_subplot(grid[2, :]))
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    utils.save_figure(build_figure(), pdf=args.output_pdf, png=args.output_png)
    print(f"wrote {args.output_pdf} and {args.output_png}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

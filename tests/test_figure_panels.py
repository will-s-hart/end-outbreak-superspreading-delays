"""What the risk-metric panels draw, and what they refuse to.

The plotting layer is mostly drawing, which a test cannot usefully judge. The exception is
:func:`figure_panels.risk_metrics_panel`, which decides per model whether RAC and RAT are one
curve or two. Getting that wrong produces a figure that looks entirely reasonable — a missing
RAT line reads as a result — so the decision is pinned here, on synthetic curves over the real
calendar, by what ends up on the axes rather than by how it was reached.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from end_of_outbreak import outbreak_data, risk_curves

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    scripts = REPO_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", scripts / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


figure_panels = load_script("figure_panels")
utils = figure_panels.utils
plot_sustained_transmission = load_script("plot_sustained_transmission")
plot_underreporting = load_script("plot_underreporting")


@pytest.fixture(scope="module")
def data() -> outbreak_data.OutbreakData:
    return outbreak_data.load_onset_data()


def curve(
    data: outbreak_data.OutbreakData, model: str, *, columns: tuple[str, ...]
) -> pd.DataFrame:
    """A declining curve over the whole window, carrying only the named estimand columns."""
    days = np.arange(len(data.dates))
    risk = np.clip(1.0 - (days - 40) / 40, 0.0, 1.0)
    frame = pd.DataFrame({"date": data.dates, "day": days, "model": model})
    for scale, column in enumerate(columns):
        frame[column] = risk * 0.9**scale
    return frame


INFECTION_ANCHORED_COLUMNS = (utils.RISK_COLUMN, utils.SUSTAINED_RISK_COLUMN)
ONSET_ANCHORED_COLUMNS = (
    utils.RISK_COLUMN,
    utils.TRANSMISSION_RISK_COLUMN,
    utils.SUSTAINED_RISK_COLUMN,
)


def all_four(data: outbreak_data.OutbreakData) -> dict[str, pd.DataFrame]:
    """The four compared models, each with exactly the columns ``analysis_driver`` writes."""
    return {
        "sse": curve(data, "sse", columns=INFECTION_ANCHORED_COLUMNS),
        "ssi": curve(data, "ssi", columns=INFECTION_ANCHORED_COLUMNS),
        "sse_so": curve(data, "sse_so", columns=ONSET_ANCHORED_COLUMNS),
        "ssi_so": curve(data, "ssi_so", columns=ONSET_ANCHORED_COLUMNS),
    }


def drawn(ax) -> dict[str, str]:
    """Each labelled curve on the axes, mapped to its linestyle. Backdrop rules are unlabelled."""
    return {
        line.get_label(): line.get_linestyle()
        for line in ax.lines
        if not line.get_label().startswith("_")
    }


def panel(data, curves, **kwargs):
    figure, ax = plt.subplots()
    figure_panels.risk_metrics_panel(ax, curves, data, title="t", letter="A", **kwargs)
    return figure, ax


def test_rac_and_rat_are_one_curve_under_infection_anchoring_and_two_under_onset(data):
    figure, ax = panel(data, all_four(data), metrics=("rac", "rat"))
    assert drawn(ax) == {
        "SSE RAC/RAT": "-",
        "SSI RAC/RAT": "-",
        "SSE-SO RAC": "-",
        "SSE-SO RAT": "--",
        "SSI-SO RAC": "-",
        "SSI-SO RAT": "--",
    }
    plt.close(figure)


def test_all_three_estimands_keep_the_default_linestyles(data):
    figure, ax = panel(data, all_four(data))
    lines = drawn(ax)
    assert lines["SSE RST"] == ":" and lines["SSE-SO RST"] == ":"
    assert "SSE RAT" not in lines and lines["SSE RAC/RAT"] == "-"
    assert len(lines) == 2 * 2 + 3 * 2
    plt.close(figure)


def test_a_single_estimand_panel_takes_the_linestyle_override(data):
    figure, ax = panel(data, all_four(data), metrics=("rst",), metric_linestyles={"rst": "-"})
    assert drawn(ax) == {
        "SSE RST": "-",
        "SSI RST": "-",
        "SSE-SO RST": "-",
        "SSI-SO RST": "-",
    }
    plt.close(figure)


def test_infection_anchored_rat_alone_is_the_rac_column_in_rat_style(data):
    frame = curve(data, "sse", columns=INFECTION_ANCHORED_COLUMNS)
    figure, ax = panel(data, {"sse": frame}, metrics=("rat",))
    (line,) = [line for line in ax.lines if line.get_label() == "SSE RAT"]
    assert line.get_linestyle() == "--"
    np.testing.assert_array_equal(line.get_ydata(), frame[utils.RISK_COLUMN])
    plt.close(figure)


def test_a_model_without_the_requested_column_is_refused_by_name(data):
    """DLO defines no transmission lineages, so its curve file has no RST to draw."""
    dlo = curve(data, "dlo", columns=(utils.RISK_COLUMN,))
    figure, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"dlo .*risk_of_sustained_transmission"):
        figure_panels.risk_metrics_panel(ax, {"dlo": dlo}, data, title="t", letter="A")
    plt.close(figure)


def test_an_onset_anchored_file_missing_rat_is_refused_not_drawn_as_one_line(data):
    """The convention comes from the specification, so a lost column cannot pass as a result."""
    frame = curve(data, "sse_so", columns=INFECTION_ANCHORED_COLUMNS)
    figure, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"sse_so .*risk_of_additional_transmission"):
        figure_panels.risk_metrics_panel(
            ax, {"sse_so": frame}, data, title="t", letter="A", metrics=("rac", "rat")
        )
    plt.close(figure)


def test_the_settling_marker_is_the_crossing_the_report_quotes(data):
    frame = curve(data, "sse", columns=(utils.RISK_COLUMN,))
    expected = risk_curves.RiskCurve(
        days=np.asarray(frame["day"]), risk=np.asarray(frame[utils.RISK_COLUMN]), n_draws=0
    ).first_day_below(figure_panels.SETTLING_THRESHOLD)
    assert expected is not None
    assert figure_panels.settling_day(frame) == expected


def test_the_sustained_transmission_figure_splits_by_estimand(data):
    figure = plot_sustained_transmission.build_figure(curves=all_four(data), data=data)
    rac_rat, rst = (ax for ax in figure.axes if ax.get_title())
    assert len(drawn(rac_rat)) == 6
    assert set(drawn(rst).values()) == {"-"} and len(drawn(rst)) == 4
    plt.close(figure)


def test_the_reporting_figure_has_a_sweep_row_and_an_estimand_row(data):
    curves = {probability: all_four(data) for probability in (1.0, 0.8, 0.6)}
    figure = plot_underreporting.build_figure(curves=curves, data=data)
    naive, onset, at_80, at_60 = (ax for ax in figure.axes if ax.get_title())
    assert {label.split(",")[0] for label in drawn(naive)} == {"SSE", "SSI"}
    assert {label.split(",")[0] for label in drawn(onset)} == {"SSE-SO", "SSI-SO"}
    for ax, percent in ((at_80, "80%"), (at_60, "60%")):
        assert percent in ax.get_title()
        assert len(drawn(ax)) == 6
    plt.close(figure)


def test_a_log_scale_parameter_panel_draws_the_density_of_log_x():
    """On a log axis the curve must be ``p(x)·x``, or equal areas stop being equal probability."""
    from end_of_outbreak.model_specifications import LogNormalPrior

    prior = LogNormalPrior.from_config(
        {"distribution": "lognormal", "median": 0.18, "quantile_025": 0.018, "quantile_975": 1.8}
    )
    draws = {"sse_so": np.random.default_rng(1).lognormal(np.log(0.18), 1.0, 4000)}
    figure, ax = plt.subplots()
    figure_panels.parameter_posterior_panel(
        ax, draws, prior=prior, xlabel="$k$", title="t", log_scale=True
    )
    assert ax.get_xscale() == "log"
    (prior_line,) = [line for line in ax.lines if line.get_label() == "prior"]
    x = np.asarray(prior_line.get_xdata(), dtype=np.float64)
    y = np.asarray(prior_line.get_ydata(), dtype=np.float64)
    np.testing.assert_allclose(y, prior.frozen().pdf(x) * x)
    # A density of log x integrates to (nearly) one over log x across the drawn range.
    assert float(np.trapezoid(y, np.log(x))) == pytest.approx(1.0, abs=0.05)
    plt.close(figure)

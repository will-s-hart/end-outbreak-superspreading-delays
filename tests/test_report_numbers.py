"""The report's numbers: the artefacts agree, and the formatting decisions hold.

Two kinds of check, and the first is the one that matters.

**The committed report and the committed macro file must agree.** Stage 10's acceptance
criterion is that every figure is included and every number in the prose traces to a file in
``results/``. That is a property of two text files, so it is checkable without running anything:
every ``\\resultnum`` key the report uses must be defined, every figure it includes must exist,
and every file the macros were read from must still be there. Without this test the failure mode
is a renamed analysis leaving a stale number in the prose, which is invisible on the page.

**The formatting helpers.** Rounding and rendering are presentation decisions the script owns, so
that the same quantity cannot be quoted to two decimal places in one section and three in
another. They are pure functions of their arguments and are pinned here.

The script is loaded by path rather than imported: ``scripts/`` is not a package, and the run
scripts are deliberately standalone programs invoked from Snakemake ``shell:`` directives.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_TEX = REPO_ROOT / "report" / "report.tex"
NUMBERS_TEX = REPO_ROOT / "results" / "report_numbers.tex"


def load_script(name: str) -> ModuleType:
    """Load a ``scripts/`` program as a module, without putting ``scripts/`` on the path."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report_numbers = load_script("run_report_numbers")


# ---------------------------------------------------------------------------------------
# The committed artefacts agree
# ---------------------------------------------------------------------------------------


def report_body() -> str:
    """``report.tex`` with its comments stripped.

    The file's own header explains the ``\\resultnum`` convention and writes the macro out to do
    so, which is not a use of it. Comments are the only place that happens, so dropping them is
    both sufficient and exact.
    """
    return "\n".join(
        re.sub(r"(?<!\\)%.*$", "", line) for line in REPORT_TEX.read_text().split("\n")
    )


def defined_keys() -> set[str]:
    return set(re.findall(r"\\defresultnum\{([^}]*)\}", NUMBERS_TEX.read_text()))


def used_keys() -> set[str]:
    return set(re.findall(r"\\resultnum\{([^}]*)\}", report_body()))


def test_every_number_the_report_quotes_is_defined():
    missing = sorted(used_keys() - defined_keys())
    assert not missing, (
        f"report.tex quotes {len(missing)} numbers that results/report_numbers.tex does not "
        f"define: {missing[:10]}. Re-run the report_numbers rule, or fix the keys — a number "
        "must never be typed into the prose."
    )


def test_the_report_quotes_at_least_one_number_per_analysis():
    """Guards against the keys and the prose drifting apart silently in the other direction.

    A report that stopped using a whole analysis's numbers would still pass the check above.
    """
    used = used_keys()
    for analysis in (
        "naivemodelsfixedk",
        "naivemodelsestimatedk",
        "onsetmodelsfixedk",
        "onsetmodelsestimatedk",
    ):
        assert any(key.startswith(f"{analysis}.") for key in used), (
            f"the report quotes nothing from the {analysis} analysis"
        )


def test_every_included_figure_exists():
    included = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", report_body())
    assert included, "the report includes no figures"
    for relative in included:
        # Paths in the report are relative to report/, which is where latexmk runs it.
        assert (REPORT_TEX.parent / relative).exists(), f"missing figure {relative}"


def test_every_source_the_numbers_were_read_from_still_exists():
    sources = re.findall(r"^%   (\S+)$", NUMBERS_TEX.read_text(), flags=re.MULTILINE)
    assert sources, "the macro file records no provenance"
    for source in sources:
        assert (REPO_ROOT / source).exists(), f"{source} is recorded as a source but is gone"


def test_no_key_is_defined_twice():
    definitions = re.findall(r"\\defresultnum\{([^}]*)\}", NUMBERS_TEX.read_text())
    assert len(definitions) == len(set(definitions))


@pytest.mark.parametrize("setting", ["draws", "tune", "chains"])
def test_the_sampler_settings_the_methods_states_once_are_shared(setting):
    """The methods quotes one analysis's draws, tuning and chains for all four.

    That sentence is only true while the four agree, and nothing in LaTeX could notice if a
    config change made them differ — so the claim is pinned here instead.
    """
    values = {
        analysis: value
        for analysis in (
            "naivemodelsfixedk",
            "naivemodelsestimatedk",
            "onsetmodelsfixedk",
            "onsetmodelsestimatedk",
        )
        for value in [
            re.search(
                rf"\\defresultnum\{{{analysis}\.{setting}\}}\{{([^}}]*)\}}",
                NUMBERS_TEX.read_text(),
            )
        ]
        if value is not None
    }
    assert len(values) == 4
    assert len({match.group(1) for match in values.values()}) == 1, (
        f"the analyses no longer agree on {setting}: "
        f"{ {name: match.group(1) for name, match in values.items()} }. The methods section "
        "states it once, for all four, so it now needs one figure per analysis."
    )


# ---------------------------------------------------------------------------------------
# The formatting decisions
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("naive_models_fixed_k", "naivemodelsfixedk"),
        ("onset_models_estimated_k", "onsetmodelsestimatedk"),
        ("sse_so", "sseso"),
        ("R_pre", "rpre"),
    ],
)
def test_slug_drops_everything_but_letters_and_digits(name, expected):
    """Keys are expanded through ``\\csname``, so they carry no underscores."""
    assert report_numbers.slug(name) == expected


def test_dates_are_written_as_prose():
    assert report_numbers.date_text(datetime.date(2018, 4, 5)) == "5 April 2018"


def test_small_probabilities_stay_readable():
    assert report_numbers.small_probability(0.49) == "0.49"
    assert "10^{-10}" in report_numbers.small_probability(3.2e-10)
    assert "<10^{-300}" in report_numbers.small_probability(0.0)


def test_differences_carry_their_sign_and_are_set_in_math_mode():
    assert report_numbers.signed(7.4123, 2) == "\\ensuremath{+7.41}"
    assert report_numbers.signed(-0.5, 1) == "\\ensuremath{-0.5}"
    # A gain that rounds away is written without a sign rather than as "+0.00", which would
    # read as a measured increase.
    assert report_numbers.signed(0.001, 2) == "\\ensuremath{0.00}"


def test_a_key_cannot_be_defined_twice():
    numbers = report_numbers.NumberFile()
    numbers.set("a.b", "1")
    with pytest.raises(ValueError, match="defined twice"):
        numbers.set("a.b", "2")


def test_rendered_file_is_sorted_and_carries_its_provenance():
    numbers = report_numbers.NumberFile()
    numbers.set("z.key", "1")
    numbers.set("a.key", "2")
    rendered = numbers.render(sources=["results/whatever.json"])
    assert rendered.index("\\defresultnum{a.key}") < rendered.index("\\defresultnum{z.key}")
    assert "results/whatever.json" in rendered


# ---------------------------------------------------------------------------------------
# The one piece of method the collation borrows
# ---------------------------------------------------------------------------------------


def synthetic_outbreak(n_days: int = 11):
    """A short series with the shape the real one has: a few onsets, then a case-free tail."""
    from end_of_outbreak import outbreak_data

    onsets = np.zeros(n_days, dtype=np.int64)
    onsets[[0, 2, 3]] = 1
    return outbreak_data.OutbreakData(
        dates=pd.date_range("2018-04-05", periods=n_days, freq="D"),
        onsets=onsets,
        ert_arrival_day=3,
        ert_withdrawal_day=n_days - 1,
    )


def risk_frame(risk, transmission_risk=None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "day": np.arange(len(risk), dtype=np.int64),
            report_numbers.RISK_COLUMN: np.asarray(risk, dtype=np.float64),
            report_numbers.STANDARD_ERROR_COLUMN: np.full(len(risk), 1e-3),
        }
    )
    if transmission_risk is not None:
        frame[report_numbers.TRANSMISSION_RISK_COLUMN] = np.asarray(
            transmission_risk, dtype=np.float64
        )
        frame["transmission_monte_carlo_standard_error"] = np.full(len(risk), 2e-3)
    return frame


def test_crossings_are_the_settling_definition_the_figures_use():
    """The curve dips below 0.05 and comes back up; the crossing is the *last* time it does.

    "Settles below" is a definition the report states, and it must not be able to drift between
    the marker on a panel and the number in the text, so both go through ``RiskCurve``.
    """
    data = synthetic_outbreak()
    risk = [1.0, 1.0, 0.9, 0.04, 0.30, 0.20, 0.10, 0.04, 0.03, 0.02, 0.005]
    numbers = report_numbers.NumberFile()
    report_numbers.add_risk_curve(numbers, risk_frame(risk), data, "x", reference_day=6)
    rendered = numbers.render(sources=[])
    assert "\\defresultnum{x.rac05.day}{7}" in rendered
    assert "\\defresultnum{x.rac01.day}{10}" in rendered
    assert "\\defresultnum{x.rac.final}{0.005}" in rendered
    assert "\\defresultnum{x.rac.reference}{0.100}" in rendered


def test_a_curve_that_never_settles_says_so_rather_than_inventing_a_day():
    data = synthetic_outbreak()
    numbers = report_numbers.NumberFile()
    report_numbers.add_risk_curve(numbers, risk_frame([0.9] * 11), data, "x", reference_day=6)
    rendered = numbers.render(sources=[])
    assert "\\defresultnum{x.rac05.day}{never}" in rendered
    assert "\\defresultnum{x.rac05.date}{never}" in rendered


def test_the_rat_keys_appear_only_where_the_rat_column_does():
    data = synthetic_outbreak()
    risk = [1.0, 0.90, 0.80, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.040, 0.0050]
    transmission = [1.0, 0.85, 0.65, 0.35, 0.30, 0.20, 0.10, 0.04, 0.03, 0.020, 0.0010]

    naive = report_numbers.NumberFile()
    report_numbers.add_risk_curve(naive, risk_frame(risk), data, "x", reference_day=6)
    assert "x.rat" not in naive.render(sources=[])
    assert "x.gap" not in naive.render(sources=[])

    onset = report_numbers.NumberFile()
    report_numbers.add_risk_curve(onset, risk_frame(risk, transmission), data, "x", reference_day=6)
    rendered = onset.render(sources=[])
    # RAC >= RAT always, and the gap is the pipeline: widest where the two curves diverge most.
    assert "\\defresultnum{x.gap.max}{0.250}" in rendered
    assert "\\defresultnum{x.gap.day}{3}" in rendered
    # A case-based declaration waits for the pipeline to drain: RAC reaches 0.05 on day 9, RAT
    # two days earlier on day 7.
    assert "\\defresultnum{x.rat05.day}{7}" in rendered
    assert "\\defresultnum{x.rac05.day}{9}" in rendered
    assert "\\defresultnum{x.gap.shift05}{2}" in rendered


def test_a_reference_day_outside_the_window_is_an_error_rather_than_a_crash():
    data = synthetic_outbreak()
    with pytest.raises(ValueError, match="reference day"):
        report_numbers.add_risk_curve(
            report_numbers.NumberFile(), risk_frame([0.5] * 11), data, "x", reference_day=90
        )


def test_missing_results_fail_loudly_rather_than_falling_back(tmp_path):
    """A blank in the prose is invisible; a missing file has to stop the build."""
    with pytest.raises(FileNotFoundError, match="no results file"):
        report_numbers.read_json(tmp_path / "model_evidence.json")
    with pytest.raises(FileNotFoundError, match="no RAC curve"):
        report_numbers.read_risk_curve(tmp_path / "ssi_rac.csv")

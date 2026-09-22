"""Reading ``config/config.yaml`` into the objects the rest of the package takes.

Config parsing lives in the package rather than beside the scripts for two reasons. It is
shared by *both* script trees — the pipeline analyses in ``scripts/`` and the validation
studies in ``validation/`` — and neither should have to reach into the other's directory for
it. And it is a real dependency of the results: a change to how the delay triple is built
must re-run the fits, which only happens if the Snakemake rules can name the module in their
``input:`` lists.

The other half of the config lives on the objects themselves, where the parsing is specific to
one type: :meth:`~end_of_outbreak.model_specifications.LogNormalPrior.from_config` and
:meth:`~end_of_outbreak.fitting.SamplerSettings.from_config`. This module holds what is left —
loading the file, slicing an analysis out of it, and assembling the delay triple.

Presentation-only helpers (figure styling and the like) do **not** belong here; they go in
``scripts/utils.py``, which arrives with the plotting scripts in Stage 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from end_of_outbreak import delay_distributions

REPO_ROOT = Path(__file__).resolve().parents[1]
"""The repository root: this is a flat package, so it is the package directory's parent."""

DEFAULT_CONFIG_FILE = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_FILE) -> dict[str, Any]:
    """Load the pipeline configuration."""
    with open(path) as handle:
        return yaml.safe_load(handle)


def analysis_config(config: dict[str, Any], analysis: str) -> dict[str, Any]:
    """The block for one analysis, with ``shared`` merged in under a ``shared`` key."""
    try:
        block = dict(config["analyses"][analysis])
    except KeyError as error:
        known = ", ".join(config["analyses"])
        raise KeyError(f"unknown analysis {analysis!r}; known analyses: {known}") from error
    block["shared"] = config["shared"]
    return block


NO_SWITCH = "never"
"""``switch_day: never``: ``R_pre`` stays in force on every day of the window."""


def switch_day_from_config(block: dict[str, Any], *, ert_arrival_day: int, n_days: int) -> int:
    """The day ``R`` switches on, in each model's own time index, for one analysis block.

    The ERT's arrival unless the block says otherwise. An integer moves the switch, and must
    fall inside the window. ``never`` removes it, and resolves to ``n_days`` — past the last
    day, where :func:`end_of_outbreak.renewal.switch_index` leaves ``R_pre`` in force
    throughout.

    The spelling is deliberate. The no-switch variants used to write ``switch_day: 111``, one
    past the end of a 111-day window, and that number would silently have become a late switch
    the moment the window grew. So an integer at or past the end of the window is refused
    rather than read as "no switch": that meaning has one name, and it does not depend on how
    long the window is.
    """
    configured = block.get("switch_day")
    if configured is None:
        return ert_arrival_day
    if configured == NO_SWITCH:
        return n_days
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise ValueError(f"switch_day must be a day index or {NO_SWITCH!r}, got {configured!r}")
    if not 1 <= configured < n_days:
        raise ValueError(
            f"switch_day {configured} lies outside the window's days 1..{n_days - 1}. To "
            f"remove the switch altogether, write `switch_day: {NO_SWITCH}`."
        )
    return configured


def gamma_delay_from_config(block: dict[str, Any]) -> delay_distributions.GammaDelay:
    """Build a :class:`GammaDelay` from a ``{mean, sd}`` config block."""
    return delay_distributions.GammaDelay(mean=float(block["mean"]), sd=float(block["sd"]))


def onset_anchored_delays_from_config(
    config: dict[str, Any],
) -> delay_distributions.OnsetAnchoredDelays:
    """Build the discretised ``(w, f_tost, f_inc)`` triple from the shared config block."""
    shared = config["shared"]
    return delay_distributions.build_onset_anchored_delays(
        serial_interval=gamma_delay_from_config(shared["serial_interval"]),
        incubation=gamma_delay_from_config(shared["incubation_period"]),
        max_lag=int(shared["max_lag"]),
        tolerance=float(shared["serial_interval_tolerance"]),
    )


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of ``path`` if needed, and return ``path``."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

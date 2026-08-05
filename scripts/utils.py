"""Shared helpers for the analysis and plotting scripts.

Presentation- and I/O-level only. Anything reusable as *method* belongs in the
``end_of_outbreak`` package, so that the Snakemake rules can depend on it precisely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from end_of_outbreak import delay_distributions

REPO_ROOT = Path(__file__).resolve().parents[1]
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

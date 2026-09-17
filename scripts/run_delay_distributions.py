"""Write the discrete delay distributions the models use, for their supplementary figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from end_of_outbreak import configuration, delay_distributions


def _records(
    name: str, weights: NDArray[np.float64], *, first_lag: int
) -> list[dict[str, float | int | str]]:
    return [
        {"distribution": name, "lag": first_lag + index, "probability": probability}
        for index, probability in enumerate(weights)
    ]


def delay_distribution_frame(
    delays: delay_distributions.OnsetAnchoredDelays,
) -> pd.DataFrame:
    """Long-form table of each discrete model input and the implied serial interval."""
    rows = []
    rows.extend(
        _records(
            "incubation",
            delays.incubation,
            first_lag=delay_distributions.INCUBATION_FIRST_LAG,
        )
    )
    rows.extend(_records("tost", delays.tost, first_lag=delay_distributions.TOST_FIRST_LAG))
    rows.extend(
        _records(
            "serial_interval",
            delays.serial_interval,
            first_lag=delay_distributions.SERIAL_INTERVAL_FIRST_LAG,
        )
    )
    rows.extend(
        _records(
            "implied_serial_interval",
            delays.implied_serial_interval(),
            first_lag=delay_distributions.SERIAL_INTERVAL_FIRST_LAG,
        )
    )
    return pd.DataFrame.from_records(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    delays = configuration.onset_anchored_delays_from_config(config)
    output = configuration.ensure_parent(args.output)
    delay_distribution_frame(delays).to_csv(output, index=False)
    print(f"wrote {output}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

"""Under-reporting at 80% — SSE-SO and SSI-SO with ``k = 0.18`` (→ the reporting figure).

The middle rung of the sweep; see ``run_underreporting_60.py`` for what the reporting layer
does to the model.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "underreporting_80"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

"""Under-reporting at 60% — SSE-SO and SSI-SO with ``k = 0.18`` (→ the reporting figure).

The onset series is read as *reported* onsets: the true ones are latent, thinned into what was
observed by an independent per-case binomial. RAC still asks about a further **true** case, so
lowering the reporting probability can only raise it — the tail of zeros that ends an outbreak
may be hiding cases. The 100% panel of the figure is Analysis 3, which is the same model.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "underreporting_60"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

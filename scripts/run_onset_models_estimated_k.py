"""Analysis 4 — naive and onset-anchored SSE/SSI with ``k`` estimated (→ Fig. 4).

This is the direct test of whether treating onsets as infections attenuates day-level
dispersion: SSE and SSE-SO fit the same record under a common prior, while SSI and SSI-SO test
whether individual-level dispersion is comparatively invariant to the incubation regrouping.
The shared analysis driver computes the fits, RAC/RAT, evidence and every pairwise ``k``
posterior comparison.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "onset_models_estimated_k"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

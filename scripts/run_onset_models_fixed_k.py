"""Analysis 3 — naive and onset-anchored SSE/SSI with ``k = 0.18`` (→ Fig. 3).

The four models share the observed onset series and its implied onset-to-onset serial interval.
SSE/SSI read onsets as infection dates; SSE-SO/SSI-SO reconstruct transmission and incubation
explicitly. The headline output is RAC. The same tier-2 files carry RAT for the onset models,
which the supplementary figure reports separately.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "onset_models_fixed_k"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

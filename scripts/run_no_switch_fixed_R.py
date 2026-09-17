"""Exploratory: the onset comparison with no switchpoint and R held at 0.95 throughout.

One of the two variants that ask how much of the naive/onset gap in the fixed-``k`` onset
analysis is the switchpoint. Here nothing is estimated but the latents: ``k`` is fixed, ``R``
is fixed at the same value before and after, and the switch is moved past the end of the
window. Any gap that survives is therefore the retained-state arithmetic alone — ``Λ(t)`` and
``Λ_Y(t)`` against ``W(t) + M(t)`` — with the two anchorings handed identical parameters.

With ``R`` constant the reset to ``R_pre`` is a no-op, so RAC here is a forward posterior
predictive rather than the reset predictive of the report's analyses. SSE has no free variable
at all under these settings, and its curve is deterministic.

Exploratory: no report figure and no report number depend on this.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "no_switch_fixed_R"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

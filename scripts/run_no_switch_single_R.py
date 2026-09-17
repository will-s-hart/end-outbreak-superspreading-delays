"""Exploratory: the onset comparison with no switchpoint and one estimated R per model.

The companion to ``run_no_switch_fixed_R.py``. The switch is again moved past the end of the
window, so a single ``R`` — ``R_pre`` — governs the whole outbreak, but each model now infers
it for itself. Against the fixed-``R`` variant this adds back exactly one thing: the difference
between what the two anchoring conventions conclude the reproduction number was.

``R_post`` is still declared, and reaches no day, so it returns its prior. That is what makes
this variant need no new builder: an unreached parameter is exact, merely wasted.

With ``R`` constant the reset to ``R_pre`` is a no-op, so RAC here is a forward posterior
predictive rather than the reset predictive of the report's analyses.

Exploratory: no report figure and no report number depend on this.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "no_switch_single_R"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

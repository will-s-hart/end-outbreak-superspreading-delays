"""A risk curve and the rule for when it has settled.

Deliberately separate from :mod:`end_of_outbreak.risk_of_additional_cases`, and deliberately
free of package imports. The estimand module computes RAC and RAT, and under the real-time
estimand doing so drives MCMC (:mod:`end_of_outbreak.refit_risk`); this module is what the
*figure* and *report* tiers consume. Keeping them in one file would put a presentation-facing
definition on the dependency list of ~1500 fits, so that editing the marker on a panel would
invalidate every one of them.

The one piece of method the figure tier borrows is :meth:`RiskCurve.first_day_below`: "the day a
curve settles below a threshold" is a definition the report quotes, so the marker on the panel
and the number in the text must come from the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RiskCurve:
    """A risk over the conditioning days: one posterior-averaged probability per day.

    RAC(t) is a single number, not a distribution: it is the posterior *probability* of a
    further case, so the average over draws happens inside it (§5.1, and eq. (5) of Thompson
    et al. (2024), which does the same integral in closed form).
    """

    days: NDArray[np.int64]
    risk: NDArray[np.float64]
    """The probability of at least one further event after day ``t``."""

    n_draws: int

    @property
    def probability_of_no_further_cases(self) -> NDArray[np.float64]:
        """``1 − risk(t)``, the quantity the closed forms of §5.3 actually evaluate."""
        return 1.0 - self.risk

    def first_day_below(self, threshold: float) -> int | None:
        """First conditioning day on which the risk falls below ``threshold`` and stays there.

        The decision-relevant summary of the curve: the report quantifies onset-anchoring by
        how far this date moves at the 0.05 and 0.01 thresholds (§5.7, Stage 9).
        """
        above = np.flatnonzero(self.risk >= threshold)
        settled = 0 if above.size == 0 else int(above[-1]) + 1
        return None if settled >= self.risk.size else int(self.days[settled])

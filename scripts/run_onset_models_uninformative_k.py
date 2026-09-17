"""Supplementary: Analysis 4 again, under a ``k`` prior a decade wider on each side.

Under Analysis 4's prior the onset-anchored ``k`` posteriors land on its median, and releasing
``k`` buys them almost no evidence. That is what data agreeing with the literature value would
look like, and also what data saying nothing about ``k`` would look like. This tells the two
apart: the models, record and median are unchanged, and only the prior's width moves. A posterior
that widens with it was the prior; one that stays put is the data.

Expect the model probabilities to move as well. A wider prior spreads its mass over values the
data exclude, and the evidence charges for that, so the pie is not comparable with Analysis 4's.

Supplementary: no report figure and no report number depend on this yet.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "onset_models_uninformative_k"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

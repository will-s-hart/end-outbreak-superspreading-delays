"""Analysis 2 — the three naive models with ``k`` estimated (→ Fig. 2 and its supplement).

The same three models, the same onsets and the same serial interval as Analysis 1, with one
change: ``k`` is given a prior — log-normal, median 0.18, 95% interval ``(0.09, 0.36)`` — and
estimated rather than imposed. That single change is what turns Analysis 1's demonstration into
a measurement. Analysis 1 shows what happens *if* a literature estimate of individual-level
offspring dispersion is transplanted into a day-level model; this analysis asks the data where
each model's ``k`` actually wants to sit, and the three answers diverge because the symbol means
different things in the three mechanisms (§5.4, §1 aim 2).

Two consequences for the pipeline, both handled by :mod:`analysis_driver`:

- ``k`` is a sampled variable here rather than a constant in the graph, so the RAC calculators
  take it from the draws instead of from the fit's ``fixed_k`` attribute, and the risk curves
  integrate over the dispersion posterior as well as over ``R_pre``.
- The ``dispersion`` subcommand applies, and writes ``dispersion_posteriors.json``: the three
  posterior summaries and every pairwise divergence between them. Those are the numbers the
  report quotes, so they are computed once here rather than in the plotting script.

Everything that varies between the four analyses is read from ``config/config.yaml`` under the
analysis named by :data:`ANALYSIS`::

    python scripts/run_naive_models_estimated_k.py fit --model ssi --output ..._posterior.nc
    python scripts/run_naive_models_estimated_k.py rac --model ssi --output ..._rac.csv
    python scripts/run_naive_models_estimated_k.py evidence   --posteriors ... --output ...json
    python scripts/run_naive_models_estimated_k.py dispersion --posteriors ... --output ...json
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "naive_models_estimated_k"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

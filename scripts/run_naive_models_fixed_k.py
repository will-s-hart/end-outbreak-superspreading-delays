"""Analysis 1 — the three naive models with ``k`` held at 0.18 (→ Fig. 1).

DLO, SSE and SSI all anchored at symptom onsets-as-infections, all driven by the same serial
interval, and all handed the *same* individual-level dispersion estimate. That last part is the
point of the analysis rather than an approximation: `k = 0.18` is an offspring-distribution
dispersion, so applying it to DLO — where the dispersion acts on a whole day's aggregate
incidence — is a deliberate reproduction of a mistake made in the literature (§1 aim 2, §5.4).
The comparison of the resulting RAC curves is the headline of Fig. 1.

Analysis 2 (:mod:`run_naive_models_estimated_k`) is the same three models with ``k`` estimated
under a common prior instead of imposed, which is the closer thing to a comparison of
mechanisms; the pair is the argument that ``k`` is not portable between them.

Everything that varies between the four analyses is read from ``config/config.yaml`` under the
analysis named by :data:`ANALYSIS`, so the whole body of this script is
:mod:`analysis_driver` — see there for the subcommands and what each pipeline tier does::

    python scripts/run_naive_models_fixed_k.py fit  --model ssi --output ..._posterior.nc
    python scripts/run_naive_models_fixed_k.py rac  --model ssi --output ..._rac.csv
    python scripts/run_naive_models_fixed_k.py evidence --posteriors ... --output ...json
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "naive_models_fixed_k"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

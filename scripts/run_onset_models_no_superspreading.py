"""Supplementary: the anchoring comparison with superspreading removed.

``cori`` against ``cori_so`` — the ``k → ∞`` Poisson limits of the two mechanisms. Every
existing measurement of the naive-versus-onset-anchored difference is made in models that also
carry superspreading, so the difference and the overdispersion are always entangled. Here they
are not: the two models differ in the anchoring convention and in nothing else at all.

These are the only models in the project with no dispersion parameter, which is why the
analysis block gives neither ``fixed_k`` nor ``k_prior`` and ``AnalysisSetting.dispersion()``
returns ``None`` for it. ``evidence`` still applies — the two share the ``R`` priors, so the
Bayes factor is well defined — but ``dispersion`` does not, and no rule asks for it.

Its figure, in the report's supplement, sets the two beside Analysis 3's SSI and SSI-SO, which
are reused from that analysis rather than refitted here (``compared_with`` in the config), with
the model probabilities normalised over all four by ``run_combined_evidence.py``.
"""

from __future__ import annotations

import analysis_driver

ANALYSIS = "onset_models_no_superspreading"


def main(argv: list[str] | None = None) -> None:
    analysis_driver.main(ANALYSIS, argv)


if __name__ == "__main__":
    main()

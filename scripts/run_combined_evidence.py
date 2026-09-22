"""Posterior model probabilities over models drawn from two analyses.

The superspreading/anchoring figure sets the Poisson limits ``cori`` and ``cori_so`` beside
Analysis 3's SSI and SSI-SO, and asks two things of the four: whether the superspreading models
are better supported than the Poisson ones, and how the superspreading and anchoring effects
compare. A posterior model probability is relative to the set it is normalised over, and each
analysis's ``model_evidence.json`` normalises only its own models — so the pie needs the four
normalised together, which is what this writes.

Nothing requires the four evidences to come from one analysis. Each is an estimate of
``p(D | model)`` for a fully specified model, and together they form the comparison one analysis
of four models would have made **provided** the data are the same, the parameters the models
share have the same priors, and nothing else about the fits differs. The ``R`` priors are
shared by every analysis by construction; this checks the rest from the config — the switch
day, any fixed reproduction number, the reporting assumption — and refuses a pair that differs,
rather than renormalising evidences that answer different questions. ``k`` is the exception by
design: it is what the models disagree about, and the Poisson limits have none.

Reusing Analysis 3's evidences, rather than refitting SSI and SSI-SO inside this analysis, is
also what makes the pie's SSI-SO the same SSI-SO as in Fig. 4, to the last digit.

Tier 2 and seconds: it reads two evidence files and opens no posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from end_of_outbreak import configuration, model_evidence

SETTINGS_THAT_MUST_AGREE: tuple[str, ...] = (
    "fixed_R_pre",
    "fixed_R_post",
    "switch_day",
    "reporting",
)
"""Block keys that change what a model's evidence is an evidence *of*, apart from ``k``."""


def combine(
    config: dict[str, Any], analysis: str, evidence: dict[str, Any], compared: dict[str, Any]
) -> dict[str, Any]:
    """The analysis's own models followed by the ones it borrows, normalised together."""
    block = config["analyses"][analysis]
    borrowed = block.get("compared_with")
    if borrowed is None:
        raise ValueError(f"{analysis} has no `compared_with` block, so there is nothing to combine")
    other = str(borrowed["analysis"])
    other_block = config["analyses"][other]

    for name, payload in ((analysis, evidence), (other, compared)):
        if payload["analysis"] != name:
            raise ValueError(
                f"the evidence file given for {name} was written for {payload['analysis']}"
            )
    differing = [key for key in SETTINGS_THAT_MUST_AGREE if block.get(key) != other_block.get(key)]
    if differing:
        raise ValueError(
            f"{analysis} and {other} differ in {', '.join(differing)}, so their evidences are "
            "of models fitted under different assumptions and one normalisation over them "
            "would not be a comparison of anything"
        )
    if evidence["estimator"] != compared["estimator"]:
        raise ValueError(
            f"{analysis} used {evidence['estimator']} and {other} used {compared['estimator']}"
        )

    sources = {model: analysis for model in evidence["models"]}
    for model in borrowed["models"]:
        if model not in compared["models"]:
            known = ", ".join(compared["models"])
            raise ValueError(f"{other} has no evidence for {model}; it compares {known}")
        if model in sources:
            raise ValueError(f"{model} is in both {analysis} and {other}")
        sources[model] = other

    def field(name: str) -> dict[str, Any]:
        return {
            model: (evidence if source == analysis else compared)[name][model]
            for model, source in sources.items()
        }

    log_evidence = field("log_evidence")
    return {
        "analysis": analysis,
        "models": list(sources),
        "sources": sources,
        "estimator": evidence["estimator"],
        "log_evidence": log_evidence,
        "log_evidence_standard_error": field("log_evidence_standard_error"),
        "n_draws": field("n_draws"),
        "posterior_model_probability": model_evidence.posterior_model_probabilities(log_evidence),
    }


def read_evidence(results_root: Path, analysis: str) -> dict[str, Any]:
    path = results_root / analysis / "model_evidence.json"
    if not path.exists():
        raise FileNotFoundError(f"no model evidence at {path}; run the `evidence` rule first")
    with open(path) as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = configuration.load_config(args.config)
    other = str(config["analyses"][args.analysis]["compared_with"]["analysis"])
    combined = combine(
        config,
        args.analysis,
        read_evidence(args.results_root, args.analysis),
        read_evidence(args.results_root, other),
    )
    with open(configuration.ensure_parent(args.output), "w") as handle:
        json.dump(combined, handle, indent=2)
        handle.write("\n")
    summary = ", ".join(
        f"{model} {probability:.3g}"
        for model, probability in combined["posterior_model_probability"].items()
    )
    print(f"posterior model probabilities over both analyses: {summary} → {args.output}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG_FILE)
    parser.add_argument("--results-root", type=Path, default=configuration.REPO_ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

# Pipeline entry point.
#
#   pixi run pipeline        # reproduce every result and figure
#   pixi run pipeline-dry    # explain what would re-run, and why
#
# Three tiers per analysis, so a change at one tier never re-runs the tiers above it:
#
#   1. fit      results/<analysis>/<model>_posterior.nc     minutes-hours (MCMC)
#   2. rac      results/<analysis>/<model>_rac.csv          seconds-minutes
#      evidence results/<analysis>/model_evidence.json
#   3. figure   figures/<analysis>/*.pdf, *.png             seconds
#
# Tier 1 carries the model in a wildcard rather than fitting a whole analysis at once, so
# re-fitting SSE-SO does not re-fit SSI.
#
# Fine-grained code dependencies: Snakemake's `code` rerun-trigger hashes only a rule's own
# run/shell body and does not follow Python imports, so each rule lists in its `input:`
# exactly the package modules it uses -- never the package directory, never a glob. Keep the
# package modules genuinely separable so these lists stay narrow.


configfile: "config/config.yaml"


CONFIG_FILE = "config/config.yaml"
PKG = "end_of_outbreak"


def code(*modules):
    """Package modules a rule depends on, as paths."""
    return [f"{PKG}/{m}.py" for m in modules]


FIT_CORE = code(
    "outbreak_data",
    "delay_distributions",
    "renewal",
    "model_specifications",
    "pymc_models",
    "latent_parameterisations",
    "fitting",
)
RAC_CORE = code("risk_of_additional_cases", "renewal", "delay_distributions")
EVIDENCE_CORE = code("model_evidence", "renewal", "delay_distributions")
PLOT_CORE = ["scripts/utils.py"]

ANALYSES = config["analyses"]
ONSETS_CSV = config["shared"]["data_file"]


# Everything a rule's result depends on must appear in its `params:`. The default profile
# drops the `mtime` rerun trigger (committed outputs lose their mtimes on clone), and the
# `input` trigger tracks the *set* of input files rather than their contents -- so editing
# config.yaml does NOT invalidate a rule unless the changed value is recorded here.
def analysis_params(analysis):
    """Config values that change what a fit for `analysis` produces."""
    block = ANALYSES[analysis]
    return {
        "fixed_k": block.get("fixed_k"),
        "k_prior": block.get("k_prior"),
        "sampler": block["sampler"],
        "latent_parameterisation": config.get("latent_parameterisation"),
    }


# Delay distributions, the analysis window and the R-switch day feed every tier.
SHARED_PARAMS = config["shared"]


wildcard_constraints:
    analysis="|".join(ANALYSES),
    model="[a-z0-9_]+",


def models_of(analysis):
    return ANALYSES[analysis]["models"]


def posteriors_of(wildcards):
    return [
        f"results/{wildcards.analysis}/{model}_posterior.nc"
        for model in models_of(wildcards.analysis)
    ]


def racs_of(wildcards):
    return [
        f"results/{wildcards.analysis}/{model}_rac.csv"
        for model in models_of(wildcards.analysis)
    ]


# ---------------------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------------------

# Empty until Stage 5 lands the first analysis scripts; populated then with the four main
# figures, the supplementary figures and the compiled report.
MAIN_TARGETS = []


rule all:
    input:
        MAIN_TARGETS,


# ---------------------------------------------------------------------------------------
# Tier 1 -- MCMC fits (one job per model)
# ---------------------------------------------------------------------------------------


rule fit:
    input:
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script=lambda wildcards: ANALYSES[wildcards.analysis]["run_script"],
        code=FIT_CORE,
    output:
        "results/{analysis}/{model}_posterior.nc",
    params:
        analysis=lambda wildcards: analysis_params(wildcards.analysis),
        shared=SHARED_PARAMS,
    shell:
        "python {input.script} fit"
        " --model {wildcards.model}"
        " --data {input.data}"
        " --config {input.config}"
        " --output {output}"


# ---------------------------------------------------------------------------------------
# Tier 2 -- quantities derived from the posteriors
# ---------------------------------------------------------------------------------------


# The risk of additional cases (RAC) is the project's headline quantity. For the
# onset-anchored models the output also carries the supplementary risk of additional
# transmission (RAT) as a second column -- one derived-results file per model, named for the
# headline quantity. Under the naive models the two coincide by assumption.
rule rac:
    input:
        posterior="results/{analysis}/{model}_posterior.nc",
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script=lambda wildcards: ANALYSES[wildcards.analysis]["run_script"],
        code=RAC_CORE,
    output:
        "results/{analysis}/{model}_rac.csv",
    params:
        analysis=lambda wildcards: analysis_params(wildcards.analysis),
        shared=SHARED_PARAMS,
    shell:
        "python {input.script} rac"
        " --model {wildcards.model}"
        " --posterior {input.posterior}"
        " --data {input.data}"
        " --config {input.config}"
        " --output {output}"


rule evidence:
    input:
        posteriors=posteriors_of,
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script=lambda wildcards: ANALYSES[wildcards.analysis]["run_script"],
        code=EVIDENCE_CORE,
    output:
        "results/{analysis}/model_evidence.json",
    params:
        analysis=lambda wildcards: analysis_params(wildcards.analysis),
        shared=SHARED_PARAMS,
    shell:
        "python {input.script} evidence"
        " --posteriors {input.posteriors}"
        " --data {input.data}"
        " --config {input.config}"
        " --output {output}"


# ---------------------------------------------------------------------------------------
# Tier 3 -- figures
# ---------------------------------------------------------------------------------------


rule figure:
    input:
        racs=racs_of,
        posteriors=posteriors_of,
        evidence="results/{analysis}/model_evidence.json",
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script=lambda wildcards: ANALYSES[wildcards.analysis]["plot_script"],
        code=PLOT_CORE,
    output:
        pdf="figures/{analysis}/{analysis}.pdf",
        png="figures/{analysis}/{analysis}.png",
    params:
        shared=SHARED_PARAMS,
    shell:
        "python {input.script}"
        " --results-dir results/{wildcards.analysis}"
        " --data {input.data}"
        " --config {input.config}"
        " --output-pdf {output.pdf}"
        " --output-png {output.png}"


# ---------------------------------------------------------------------------------------
# Convenience aggregates
# ---------------------------------------------------------------------------------------


rule fits:
    input:
        [
            f"results/{analysis}/{model}_posterior.nc"
            for analysis in ANALYSES
            for model in models_of(analysis)
        ],


rule results:
    input:
        [
            f"results/{analysis}/{model}_rac.csv"
            for analysis in ANALYSES
            for model in models_of(analysis)
        ],
        [f"results/{analysis}/model_evidence.json" for analysis in ANALYSES],


rule figures:
    input:
        [f"figures/{analysis}/{analysis}.pdf" for analysis in ANALYSES],

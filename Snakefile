# Pipeline entry point.
#
#   pixi run pipeline        # reproduce every result and figure
#   pixi run pipeline-dry    # explain what would re-run, and why
#
# Three tiers per analysis, so a change at one tier never re-runs the tiers above it:
#
#   1. fit        results/<analysis>/<model>_posterior.nc         minutes-hours (MCMC)
#   2. rac        results/<analysis>/<model>_rac.csv              seconds-minutes
#      evidence   results/<analysis>/model_evidence.json
#      dispersion results/<analysis>/dispersion_posteriors.json   (analyses that estimate k)
#   3. figure     figures/<analysis>/*.pdf, *.png                 seconds
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


# The run scripts are a docstring and an analysis name apiece: the whole body of every
# `fit`/`rac`/`evidence`/`dispersion` step lives in `scripts/analysis_driver.py`, so every
# results tier depends on it exactly as it does on the package modules below. It cannot live in
# `scripts/utils.py`, which `PLOT_CORE` names -- results-driving logic there would make a
# restyle a reason to re-run MCMC.
RUN_DRIVER = ["scripts/analysis_driver.py"]

# `configuration` parses config/config.yaml into the delay triple, the priors and the sampler
# settings, so every tier depends on it: a change to how a config value is read changes results
# exactly as a change to the value itself would.
FIT_CORE = RUN_DRIVER + code(
    "configuration",
    "outbreak_data",
    "delay_distributions",
    "renewal",
    "model_specifications",
    "pymc_models",
    "latent_parameterisations",
    "fitting",
)
# The RAC step rebuilds the latents the fit integrated out, which needs the model's block
# structure (`pymc_models`, `latent_parameterisations`), and -- from Stage 8, for the
# onset-anchored models, whose RAC has no closed form -- the simulators too.
RAC_CORE = RUN_DRIVER + code(
    "configuration",
    "risk_of_additional_cases",
    "renewal",
    "delay_distributions",
    "outbreak_data",
    "model_specifications",
    "latent_parameterisations",
    "pymc_models",
    "forward_simulation",
    "fitting",
)
EVIDENCE_CORE = RUN_DRIVER + code(
    "configuration",
    "model_evidence",
    "renewal",
    "delay_distributions",
    "outbreak_data",
    "model_specifications",
    "latent_parameterisations",
    "pymc_models",
    "fitting",
)
# The `k` posteriors and the divergences between them, for the analyses that estimate `k`.
# Nothing here rebuilds a model: the summaries are of draws a fit already produced.
DISPERSION_CORE = RUN_DRIVER + code(
    "configuration",
    "posterior_comparison",
    "outbreak_data",
    "delay_distributions",
    "model_specifications",
    "fitting",
)
# The figure tier is deliberately the narrowest list. A plotting script reads what tier 2 wrote
# and decides what it looks like; the one piece of *method* it borrows is
# `RiskCurve.first_day_below`, the rule for when a curve has settled below a threshold, which
# the report quotes and so must not be reimplemented beside the panel. `risk_of_additional_cases`
# is named for that and not for its own imports: nothing else in the modelling stack can change
# a figure without first changing a results file.
PLOT_CORE = code(
    "configuration", "outbreak_data", "model_specifications", "risk_of_additional_cases"
) + ["scripts/utils.py", "scripts/figure_panels.py"]

ANALYSES = config["analyses"]
ONSETS_CSV = config["shared"]["data_file"]

# Analyses whose run and plot scripts exist. Keep this explicit rather than deriving it from
# the config: a configured future analysis must not become a target before its scripts land.
IMPLEMENTED_ANALYSES = [
    "naive_models_fixed_k",
    "naive_models_estimated_k",
    "onset_models_fixed_k",
    "onset_models_estimated_k",
]


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
        "negligible_latent_threshold": config.get("negligible_latent_threshold"),
    }


# Delay distributions, the analysis window and the R-switch day feed every tier.
SHARED_PARAMS = config["shared"]


wildcard_constraints:
    analysis="|".join(ANALYSES),
    model="[a-z0-9_]+",


def models_of(analysis):
    return ANALYSES[analysis]["models"]


def estimates_dispersion(analysis):
    """Whether this analysis gives `k` a prior rather than holding it at a literature value."""
    return ANALYSES[analysis].get("fixed_k") is None


def dispersion_summary_of(wildcards):
    """The `k` summary file, for the analyses that have one; nothing for the fixed-`k` ones."""
    if not estimates_dispersion(wildcards.analysis):
        return []
    return [f"results/{wildcards.analysis}/dispersion_posteriors.json"]


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

# One main figure per implemented analysis. Grows with `IMPLEMENTED_ANALYSES`; the §5.5 RAT
# supplement of the onset-anchored analyses and the compiled report join it at Stages 9 and 10.
MAIN_TARGETS = [
    f"figures/{analysis}/{analysis}.{extension}"
    for analysis in IMPLEMENTED_ANALYSES
    for extension in ("pdf", "png")
]
RAT_FIGURE_TARGETS = [
    f"figures/onset_models_rat/onset_models_rat.{extension}" for extension in ("pdf", "png")
]


rule all:
    input:
        MAIN_TARGETS,
        RAT_FIGURE_TARGETS,


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


# The `k` posteriors of the analyses that estimate it, and every pairwise divergence between
# them: the models share a prior on `k` by design (§6.2), so where their posteriors end up
# apart is aim 2 measured rather than demonstrated. The numbers the report quotes -- and the
# medians the `k` panel puts in its legend -- come from here and nowhere else.
rule dispersion:
    input:
        posteriors=posteriors_of,
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script=lambda wildcards: ANALYSES[wildcards.analysis]["run_script"],
        code=DISPERSION_CORE,
    output:
        "results/{analysis}/dispersion_posteriors.json",
    params:
        analysis=lambda wildcards: analysis_params(wildcards.analysis),
        shared=SHARED_PARAMS,
    shell:
        "python {input.script} dispersion"
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
        dispersion=dispersion_summary_of,
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


rule rat_figure:
    input:
        racs=[
            f"results/{analysis}/{model}_rac.csv"
            for analysis in ("onset_models_fixed_k", "onset_models_estimated_k")
            for model in ("sse_so", "ssi_so")
        ],
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script="scripts/plot_onset_models_rat.py",
        code=PLOT_CORE,
    output:
        pdf="figures/onset_models_rat/onset_models_rat.pdf",
        png="figures/onset_models_rat/onset_models_rat.png",
    params:
        shared=SHARED_PARAMS,
    shell:
        "python {input.script}"
        " --results-root results"
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
            for analysis in IMPLEMENTED_ANALYSES
            for model in models_of(analysis)
        ],


rule results:
    input:
        [
            f"results/{analysis}/{model}_rac.csv"
            for analysis in IMPLEMENTED_ANALYSES
            for model in models_of(analysis)
        ],
        [f"results/{analysis}/model_evidence.json" for analysis in IMPLEMENTED_ANALYSES],
        [
            f"results/{analysis}/dispersion_posteriors.json"
            for analysis in IMPLEMENTED_ANALYSES
            if estimates_dispersion(analysis)
        ],


rule figures:
    input:
        MAIN_TARGETS,
        RAT_FIGURE_TARGETS,

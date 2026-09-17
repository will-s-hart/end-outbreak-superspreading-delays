# Pipeline entry point.
#
#   pixi run pipeline        # reproduce every result and figure
#   pixi run pipeline-dry    # explain what would re-run, and why
#   pixi run pipeline-present # after a cluster pull, render figures/report only
#
# Three tiers per analysis, so a change at one tier never re-runs the tiers above it:
#
#   1. fit        results/<analysis>/<model>_posterior.nc         seconds-minutes (MCMC)
#      rac        results/<analysis>/<model>_rac.csv              minutes-hours (MCMC)
#   2. evidence   results/<analysis>/model_evidence.json          seconds-minutes
#      dispersion results/<analysis>/dispersion_posteriors.json   (analyses that estimate k)
#      report_numbers  results/report_numbers.tex                 seconds (spans the analyses)
#   3. figure     figures/<analysis>/*.pdf, *.png                 seconds
#      report     report/report.pdf                               seconds (needs latexmk)
#
# `rac` sits in tier 1, not tier 2, and that is not an oversight. RAC(t) conditions on the
# record through day t, so the estimator fits the model once per conditioning day -- about 110
# fits per model. The tier split still holds where it can: `fit`, `evidence` and `dispersion`
# describe the model given the whole record, and restyling a figure re-runs no MCMC. What it
# cannot hold for is a quantity whose definition *is* a fit per day. Selecting
# `rac.method: single_fit_filtered` in the config moves `rac` back to seconds, at the cost of
# an approximation -- see config/config.yaml and end_of_outbreak/filtered_risk.py.
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
    "reporting",
    "fitting",
)
# The RAC step drives a fit per conditioning day (`refit_risk` -> `fitting`, `pymc_models`) or
# a particle filter per posterior draw (`filtered_risk` -> `particle_filter`), and integrates
# the latents the fit left out of the posterior back out of the risk, which needs the model's
# block structure (`pymc_models`, `latent_parameterisations`). `risk_curves` is deliberately
# absent: the curve container and the "settles below" rule belong to the figure and report
# tiers, and naming them here would put a presentation-facing definition on the dependency list
# of every fit.
#
# `parallel` is absent for the same reason `rac.convergence` is absent from the params: it
# decides which process runs a fit, not what the fit returns. Every conditioning day carries a
# seed fixed before any of them start, so the curve is identical at any worker count, and
# `tests/test_parallel.py` pins that. Listing it would make a scheduling tweak a reason to
# re-run hours of MCMC for a byte-identical file.
RAC_CORE = RUN_DRIVER + code(
    "configuration",
    "branching_process",
    "risk_of_additional_cases",
    "refit_risk",
    "filtered_risk",
    "particle_filter",
    "forward_simulation",
    "reporting",
    "renewal",
    "delay_distributions",
    "outbreak_data",
    "model_specifications",
    "latent_parameterisations",
    "pymc_models",
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
# The report quotes no number of its own: `scripts/run_report_numbers.py` collects them all from
# what the other tier-2 rules wrote and emits LaTeX macros, and `report.tex` expands those. That
# makes it a tier-2 rule like the rest -- it opens posteriors and summarises them -- so it lists
# the modules it summarises with. `posterior_comparison` and `risk_curves` are here for the same
# reason they are elsewhere: the medians and the "settles below" rule the report quotes have to
# be the ones the pipeline computed.
REPORT_NUMBERS_CORE = ["scripts/run_report_numbers.py"] + code(
    "configuration",
    "outbreak_data",
    "delay_distributions",
    "model_specifications",
    "posterior_comparison",
    "risk_curves",
    "pymc_models",
    # The latent-block counts have to be laid out the way the fit laid them out, which under
    # incomplete reporting means reading the reporting assumption back off the fit.
    "fitting",
    "reporting",
)
# The figure tier is deliberately the narrowest list. A plotting script reads what tier 2 wrote
# and decides what it looks like; the one piece of *method* it borrows is
# `RiskCurve.first_day_below`, the rule for when a curve has settled below a threshold, which
# the report quotes and so must not be reimplemented beside the panel. That rule lives in
# `risk_curves`, which has no package imports at all -- the estimand module cannot appear here,
# because under `refit_daily` it drives every fit, and a restyle must never reach the MCMC.
PLOT_CORE = code("configuration", "outbreak_data", "model_specifications", "risk_curves") + [
    "scripts/utils.py",
    "scripts/figure_panels.py",
]
DELAY_RESULTS_CORE = ["scripts/run_delay_distributions.py"] + code(
    "configuration", "delay_distributions"
)
DELAY_FIGURE_CORE = code("configuration") + ["scripts/utils.py"]
# The schematic draws no data at all -- no results file, no config value, no package module. Its
# only dependency is the house style, so this is the shortest list in the file.
SCHEMATIC_FIGURE_CORE = ["scripts/utils.py"]

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

# The reporting sweeps are analyses for `rule fit` and `rule rac`, but not for the tiers above:
# they have no comparison figure of their own (they share one), no model evidence and no
# dispersion summary. So they stay out of `IMPLEMENTED_ANALYSES`, which is what drives those,
# and are listed separately for the rules that do reach them.
RAC_ONLY_ANALYSES = [
    "underreporting_60",
    "underreporting_80",
]
SAMPLED_ANALYSES = IMPLEMENTED_ANALYSES + RAC_ONLY_ANALYSES

# The no-switchpoint variants: analyses for `rule fit`, `rule rac` and their own figure rule,
# and for nothing else. They are exploratory -- they ask how much of the naive/onset gap is the
# switchpoint, by removing it -- so they feed no report figure and no report number, and they
# stay out of both lists above. Nothing builds them unless you name the target:
#
#     pixi run pipeline-no-switch
#     snakemake --profile config/snakemake_profile results/no_switch_fixed_R/sse_rac.csv
#
# `wildcard_constraints` is built from `ANALYSES`, so the generic rules already reach them.
# Promoting one to a report analysis means adding it to `IMPLEMENTED_ANALYSES` and giving it a
# `plot_script`; until then `rule all` cannot see it.
EXPLORATORY_ANALYSES = [
    "no_switch_fixed_R",
    "no_switch_single_R",
]


# Everything a rule's result depends on must appear in its `params:`. Note what that is and is
# not protecting against: the default profile drops the `mtime` rerun trigger (committed outputs
# lose their mtimes on clone), but the surviving `input` trigger is per-file and content-based,
# and `config/config.yaml` is an `input:` of every rule. So *any* edit to it -- including one to
# a value named in no `params:` at all -- marks every fit stale, reported as
# `Updated input files: config/config.yaml`. The same goes for the package modules each rule
# lists, which is exactly what makes those lists worth keeping narrow.
#
# These `params:` are therefore a second line of defence rather than the mechanism. Keep them
# complete anyway: dropping the config file from the `input:` lists would make them
# load-bearing, and that trade is not worth making until they can be trusted alone.
def analysis_params(analysis):
    """Config values that change what a fit for `analysis` produces.

    Shared by every rule that samples, so keep it to what a *fit* depends on. Settings that
    only affect a later step belong in that step's own `params:` -- otherwise changing the
    particle count of an approximation nobody is running would re-fit every model.
    """
    block = ANALYSES[analysis]
    return {
        "fixed_k": block.get("fixed_k"),
        "k_prior": block.get("k_prior"),
        # A fixed reproduction number and a moved switch day each change every draw, so both
        # belong here for the same reason `fixed_k` does. Absent from all but the exploratory
        # analyses, where they read as None.
        "fixed_R_pre": block.get("fixed_R_pre"),
        "fixed_R_post": block.get("fixed_R_post"),
        "switch_day": block.get("switch_day"),
        "sampler": block["sampler"],
        "latent_parameterisation": config.get("latent_parameterisation"),
        "negligible_latent_threshold": config.get("negligible_latent_threshold"),
        "reporting": block.get("reporting"),
    }


# What the `rac` step produces depends on which estimator is selected, where the curve starts,
# and -- under the filtering estimator -- how many draws and particles it uses.
#
# `rac.convergence` is deliberately absent. It decides whether a curve whose fits look shaky
# stops the build; it does not change a single number in the curve, so putting it here would
# make tightening a threshold re-run hours of MCMC to reproduce the identical output.
RAC_PARAMS = {
    key: config.get("rac", {}).get(key) for key in ("method", "first_day", "filtering")
}


# An analysis may override any of those, and the under-reporting sweeps override
# `first_day`. It belongs here rather than in `analysis_params`, which `rule fit` shares:
# where a curve starts changes no fit.
def rac_params_of(analysis):
    return RAC_PARAMS | dict(ANALYSES[analysis].get("rac") or {})


# Delay distributions, the analysis window and the R-switch day feed every tier.
SHARED_PARAMS = config["shared"]


wildcard_constraints:
    analysis="|".join(ANALYSES),
    model="[a-z0-9_]+",
    rst_setting="fixed_k|estimated_k",


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

# One comparison figure per implemented analysis. Grows with `IMPLEMENTED_ANALYSES`.
ANALYSIS_FIGURE_TARGETS = [
    f"figures/{analysis}/{analysis}.{extension}"
    for analysis in IMPLEMENTED_ANALYSES
    for extension in ("pdf", "png")
]
SUSTAINED_TRANSMISSION_ANALYSES = {
    "fixed_k": "onset_models_fixed_k",
    "estimated_k": "onset_models_estimated_k",
}
SUSTAINED_TRANSMISSION_TARGETS = {
    setting: [
        f"figures/sustained_transmission_{setting}/sustained_transmission_{setting}.{extension}"
        for extension in ("pdf", "png")
    ]
    for setting in SUSTAINED_TRANSMISSION_ANALYSES
}
# The reporting sweep. 100% is Analysis 3 rather than a rerun: at pi = 1 the model *is* that
# model, so reusing the committed curves both saves ~156 fits and makes the figure's baseline
# the paper's headline result rather than a Monte-Carlo-different twin of it.
UNDERREPORTING_ANALYSES = {
    1.0: "onset_models_fixed_k",
    0.8: "underreporting_80",
    0.6: "underreporting_60",
}
UNDERREPORTING_MODELS = ("sse_so", "ssi_so")
UNDERREPORTING_TARGETS = [
    f"figures/underreporting/underreporting.{extension}" for extension in ("pdf", "png")
]
# Main Fig. 1, the methods schematic. It is a main target like the rest, so that `figures` and
# the report build it, but it depends on nothing any other tier produces.
SCHEMATIC_FIGURE_TARGETS = [
    f"figures/model_schematic/model_schematic.{extension}" for extension in ("pdf", "png")
]
MAIN_TARGETS = (
    SCHEMATIC_FIGURE_TARGETS
    + ANALYSIS_FIGURE_TARGETS
    + SUSTAINED_TRANSMISSION_TARGETS["fixed_k"]
    + UNDERREPORTING_TARGETS
)
RST_SUPPLEMENTARY_TARGETS = SUSTAINED_TRANSMISSION_TARGETS["estimated_k"]
DELAY_FIGURE_TARGETS = [
    f"figures/delay_distributions/delay_distributions.{extension}"
    for extension in ("pdf", "png")
]
SUPPLEMENTARY_FIGURE_TARGETS = DELAY_FIGURE_TARGETS + RST_SUPPLEMENTARY_TARGETS
# The compiled methods-and-results document, and the macro file every number in it expands from.
REPORT_NUMBERS = "results/report_numbers.tex"
REPORT_TARGET = "report/report.pdf"


rule all:
    input:
        MAIN_TARGETS,
        SUPPLEMENTARY_FIGURE_TARGETS,
        REPORT_TARGET,


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
        script="scripts/run_risk_curves.py",
        code=RAC_CORE,
    output:
        curve="results/{analysis}/{model}_rac.csv",
        # One row per conditioning day: divergences, the worst R-hat and the smallest bulk ESS
        # over every variable that day's fit sampled. A hundred and ten fits per model is a
        # hundred and ten chances for one to go quietly wrong, and the step fails rather than
        # writing a curve none of whose fits anyone has looked at.
        diagnostics="results/{analysis}/{model}_rac_diagnostics.csv",
    params:
        analysis=lambda wildcards: analysis_params(wildcards.analysis),
        rac=lambda wildcards: rac_params_of(wildcards.analysis),
        shared=SHARED_PARAMS,
    # One worker per core, over the conditioning days -- the fits themselves are sequential
    # (`fitting.fit_model` samples with `cores=1`), so this is the whole of the step's
    # parallelism. Snakemake caps it at the job's own `-j`, so a local `-j4` gives four.
    #
    # Deliberately NOT in `params:`: the curve is identical at any thread count, and recording
    # it as a rerun trigger would re-run hours of MCMC to reproduce a byte-identical file. Use
    # the profile's `set-threads` to change it without touching this rule.
    threads: 8
    shell:
        "python {input.script}"
        " --analysis {wildcards.analysis}"
        " --model {wildcards.model}"
        " --posterior {input.posterior}"
        " --data {input.data}"
        " --config {input.config}"
        " --diagnostics {output.diagnostics}"
        " --jobs {threads}"
        " --output {output.curve}"


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


# Every number the report quotes, in one macro file. It spans the analyses rather than sitting
# inside one, so it takes the implemented list on the command line: the Snakefile owns that list
# and the script must not grow a second copy of it.
rule report_numbers:
    input:
        posteriors=[
            f"results/{analysis}/{model}_posterior.nc"
            for analysis in SAMPLED_ANALYSES
            for model in models_of(analysis)
        ],
        racs=[
            f"results/{analysis}/{model}_rac.csv"
            for analysis in SAMPLED_ANALYSES
            for model in models_of(analysis)
        ],
        evidence=[f"results/{analysis}/model_evidence.json" for analysis in IMPLEMENTED_ANALYSES],
        dispersion=[
            f"results/{analysis}/dispersion_posteriors.json"
            for analysis in IMPLEMENTED_ANALYSES
            if estimates_dispersion(analysis)
        ],
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        code=REPORT_NUMBERS_CORE,
    output:
        REPORT_NUMBERS,
    params:
        analyses=IMPLEMENTED_ANALYSES,
        rac_only=RAC_ONLY_ANALYSES,
        per_analysis={analysis: analysis_params(analysis) for analysis in SAMPLED_ANALYSES},
        shared=SHARED_PARAMS,
    shell:
        "python scripts/run_report_numbers.py"
        " --analyses {params.analyses}"
        " --rac-only-analyses {params.rac_only}"
        " --data {input.data}"
        " --config {input.config}"
        " --results-root results"
        " --output {output}"


rule delay_distributions:
    input:
        config=CONFIG_FILE,
        script="scripts/run_delay_distributions.py",
        code=DELAY_RESULTS_CORE,
    output:
        "results/delay_distributions.csv",
    params:
        shared=SHARED_PARAMS,
    shell:
        "python {input.script}"
        " --config {input.config}"
        " --output {output}"


# ---------------------------------------------------------------------------------------
# Tier 3 -- figures and the report
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


rule sustained_transmission_figure:
    input:
        racs=lambda wildcards: [
            f"results/{SUSTAINED_TRANSMISSION_ANALYSES[wildcards.rst_setting]}/{model}_rac.csv"
            for model in ("sse", "ssi", "sse_so", "ssi_so")
        ],
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script="scripts/plot_sustained_transmission.py",
        code=PLOT_CORE,
    output:
        pdf=(
            "figures/sustained_transmission_{rst_setting}/"
            "sustained_transmission_{rst_setting}.pdf"
        ),
        png=(
            "figures/sustained_transmission_{rst_setting}/"
            "sustained_transmission_{rst_setting}.png"
        ),
    params:
        analysis=lambda wildcards: SUSTAINED_TRANSMISSION_ANALYSES[wildcards.rst_setting],
        shared=SHARED_PARAMS,
    shell:
        "python {input.script}"
        " --analysis {params.analysis}"
        " --results-root results"
        " --data {input.data}"
        " --config {input.config}"
        " --output-pdf {output.pdf}"
        " --output-png {output.png}"


# `figures/{analysis}/{analysis}.{pdf,png}` is also `rule figure`'s output pattern, so the two
# would be ambiguous for these two analyses. The constraint below keeps this rule to them and
# the `ruleorder` settles which one Snakemake reaches for. This rule exists at all because the
# variants take no `model_evidence.json`: `no_switch_fixed_R` fixes every parameter, leaving
# nothing to integrate over, so neither figure draws a posterior-model-probability panel.
ruleorder: no_switch_figure > figure


rule no_switch_figure:
    wildcard_constraints:
        analysis="|".join(EXPLORATORY_ANALYSES),
    input:
        racs=racs_of,
        # Only the single-R variant has a parameter posterior to draw.
        posteriors=lambda wildcards: (
            posteriors_of(wildcards) if wildcards.analysis == "no_switch_single_R" else []
        ),
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script="scripts/plot_no_switch.py",
        code=PLOT_CORE,
    output:
        pdf="figures/{analysis}/{analysis}.pdf",
        png="figures/{analysis}/{analysis}.png",
    params:
        shared=SHARED_PARAMS,
    shell:
        "python {input.script}"
        " --analysis {wildcards.analysis}"
        " --results-dir results/{wildcards.analysis}"
        " --data {input.data}"
        " --config {input.config}"
        " --output-pdf {output.pdf}"
        " --output-png {output.png}"


rule underreporting_figure:
    input:
        racs=[
            f"results/{analysis}/{model}_rac.csv"
            for analysis in UNDERREPORTING_ANALYSES.values()
            for model in UNDERREPORTING_MODELS
        ],
        data=ONSETS_CSV,
        config=CONFIG_FILE,
        script="scripts/plot_underreporting.py",
        code=PLOT_CORE,
    output:
        pdf="figures/underreporting/underreporting.pdf",
        png="figures/underreporting/underreporting.png",
    params:
        analyses=UNDERREPORTING_ANALYSES,
        shared=SHARED_PARAMS,
    shell:
        "python {input.script}"
        " --results-root results"
        " --data {input.data}"
        " --config {input.config}"
        " --output-pdf {output.pdf}"
        " --output-png {output.png}"


rule schematic_figure:
    input:
        script="scripts/plot_model_schematic.py",
        code=SCHEMATIC_FIGURE_CORE,
    output:
        pdf="figures/model_schematic/model_schematic.pdf",
        png="figures/model_schematic/model_schematic.png",
    shell:
        "python {input.script}"
        " --output-pdf {output.pdf}"
        " --output-png {output.png}"


rule delay_figure:
    input:
        delays="results/delay_distributions.csv",
        script="scripts/plot_delay_distributions.py",
        code=DELAY_FIGURE_CORE,
    output:
        pdf="figures/delay_distributions/delay_distributions.pdf",
        png="figures/delay_distributions/delay_distributions.png",
    shell:
        "python {input.script}"
        " --delays {input.delays}"
        " --output-pdf {output.pdf}"
        " --output-png {output.png}"


# The methods-and-results document. It is tier 3 like the figures: it consumes only the PDFs and
# the macro file, and recompiling it can never re-run a fit.
#
# `latexmk` and `pdflatex` come from a TeX distribution, which pixi does not manage -- this is
# the one rule in the pipeline with an external toolchain dependency, so it says so rather than
# failing with a bare "command not found".
rule report:
    input:
        tex="report/report.tex",
        numbers=REPORT_NUMBERS,
        figures=[
            target
            for target in MAIN_TARGETS + SUPPLEMENTARY_FIGURE_TARGETS
            if target.endswith(".pdf")
        ],
    output:
        REPORT_TARGET,
    shell:
        "command -v latexmk >/dev/null || {{ "
        "echo 'latexmk not found: the report rule needs a TeX distribution (e.g. MacTeX or "
        "TeX Live), which pixi does not provide. Everything else in the pipeline runs without "
        "it.' >&2; exit 1; }}; "
        "latexmk -pdf -interaction=nonstopmode -halt-on-error -cd {input.tex}"


# ---------------------------------------------------------------------------------------
# Convenience aggregates
# ---------------------------------------------------------------------------------------


rule fits:
    input:
        [
            f"results/{analysis}/{model}_posterior.nc"
            for analysis in SAMPLED_ANALYSES
            for model in models_of(analysis)
        ],


rule results:
    input:
        [
            f"results/{analysis}/{model}_rac.csv"
            for analysis in SAMPLED_ANALYSES
            for model in models_of(analysis)
        ],
        [f"results/{analysis}/model_evidence.json" for analysis in IMPLEMENTED_ANALYSES],
        [
            f"results/{analysis}/dispersion_posteriors.json"
            for analysis in IMPLEMENTED_ANALYSES
            if estimates_dispersion(analysis)
        ],
        "results/delay_distributions.csv",
        REPORT_NUMBERS,


# The whole of the exploratory tier, in one target. Deliberately not reachable from `rule all`
# or `rule figures`: these analyses answer a question about the models, not one the report asks.
rule no_switch:
    input:
        [
            f"figures/{analysis}/{analysis}.{extension}"
            for analysis in EXPLORATORY_ANALYSES
            for extension in ("pdf", "png")
        ],


rule figures:
    input:
        MAIN_TARGETS,
        SUPPLEMENTARY_FIGURE_TARGETS,

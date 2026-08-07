# Pipeline scripts

The four report analyses, and nothing else. Every script here is invoked by a Snakemake rule
and writes to `results/` or `figures/`; if an output is not a report input, it does not belong
in this tree — see `validation/` for benchmarks and cross-checks.

| Script | Tier | Writes |
| --- | --- | --- |
| `analysis_driver.py` | 1 and 2 | — the whole body of every `run_*.py`, shared |
| `run_naive_models_fixed_k.py` | 1 and 2 | `results/naive_models_fixed_k/{model}_posterior.nc`, `..._rac.csv`, `model_evidence.json` |
| `run_naive_models_estimated_k.py` | 1 and 2 | the same, plus `results/naive_models_estimated_k/dispersion_posteriors.json` |
| `run_onset_models_fixed_k.py` | 1 and 2 | the four-model fixed-`k` Analysis 3 outputs, including RAT in the SO-model RAC files |
| `run_onset_models_estimated_k.py` | 1 and 2 | the four-model estimated-`k` Analysis 4 outputs and dispersion comparisons |
| `plot_naive_models_fixed_k.py` | 3 | `figures/naive_models_fixed_k/*.pdf`, `*.png` |
| `plot_naive_models_estimated_k.py` | 3 | the same for `naive_models_estimated_k` |
| `plot_onset_models_fixed_k.py` | 3 | `figures/onset_models_fixed_k/*.pdf`, `*.png` |
| `plot_onset_models_estimated_k.py` | 3 | the same for `onset_models_estimated_k` |
| `plot_onset_models_rat.py` | 3 | the two-panel supplementary RAC/RAT figure |
| `figure_panels.py` | 3 | the panels the analysis figures are assembled from |
| `utils.py` | — | presentation-only helpers shared by the plotting scripts |

Stage 9 added the two onset-anchored analyses to `IMPLEMENTED_ANALYSES` and the `rat_figure`
rule for §5.5's supplementary comparison. That explicit list, not the config, remains what
`rule all` and the convenience aggregates are built from.

Conventions, all of them load-bearing:

- **Compute-and-save and load-and-plot are separate scripts**, so restyling a figure never
  re-runs MCMC. The run scripts carry one subcommand per tier (`fit`, `rac`, `evidence`, and
  `dispersion` where `k` is estimated) and each Snakemake rule invokes exactly one of them.
- **Named for what they do**, never for figure numbers.
- **`argparse`, invoked from `shell:`**, never through Snakemake's `script:` directive, so each
  stays runnable and debuggable on its own.
- **Config parsing is in the package** (`end_of_outbreak.configuration`), not here, because the
  validation scripts need it too and because the Snakemake rules have to be able to name it in
  their `input:` lists. `utils.py` and `figure_panels.py` are for figure styling and panel
  drawing — things whose only effect is on tier 3.
- **A plotting script computes nothing.** Every number it draws was written to `results/` by a
  tier-2 rule, down to the medians and credible intervals in the `k` panel's legend. The one
  exception is deliberate and documented: `figure_panels` borrows `RiskCurve.first_day_below`
  from the package, because "the day a curve settles below a threshold" is a definition the
  report quotes and must not be able to drift between the marker on the panel and the number in
  the text.
- **A missing input fails loudly.** `utils.read_model_evidence` and `read_dispersion_summary`
  raise rather than falling back when their file is absent: a pie chart of placeholder numbers
  is indistinguishable from a real one on the page.

**The run scripts are a docstring and an analysis name apiece** — Stage 7 factored the shared
body into `analysis_driver.py`, once the second analysis made its real shape visible. It could
not go in `utils.py`: that file is in `PLOT_CORE`, and putting fit-driving logic in it would
make every restyle a reason to re-run MCMC. `analysis_driver.py` is named instead in `FIT_CORE`,
`RAC_CORE`, `EVIDENCE_CORE` and `DISPERSION_CORE`. The plot scripts stay one per analysis,
because the panel *arrangement* is what differs between figures; what they share is
`figure_panels.py`.

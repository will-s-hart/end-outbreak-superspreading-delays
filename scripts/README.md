# Pipeline scripts

The four report analyses, the two exploratory no-switchpoint variants, and nothing else. Every
script here is invoked by a Snakemake rule and writes to `results/` or `figures/`; if an output
is neither a report input nor an analysis of the outbreak, it does not belong in this tree —
see `validation/` for benchmarks and cross-checks, whose subject is the implementation.

| Script | Tier | Writes |
| --- | --- | --- |
| `analysis_driver.py` | 1 and 2 | shared fit/evidence/dispersion machinery and risk acquisition helpers |
| `run_risk_curves.py` | 1 | RAC/RAT/RST together, without repeating any conditioning-day fit |
| `run_naive_models_fixed_k.py` | 1 and 2 | `results/naive_models_fixed_k/{model}_posterior.nc`, `..._rac.csv`, `..._rac_diagnostics.csv`, `model_evidence.json` |
| `run_naive_models_estimated_k.py` | 1 and 2 | the same, plus `results/naive_models_estimated_k/dispersion_posteriors.json` |
| `run_onset_models_fixed_k.py` | 1 and 2 | the four-model fixed-`k` Analysis 3 outputs, including RST and RAT where applicable |
| `run_onset_models_estimated_k.py` | 1 and 2 | the four-model estimated-`k` Analysis 4 outputs and dispersion comparisons |
| `run_delay_distributions.py` | 2 | `results/delay_distributions.csv` — the four distributions drawn in Supplementary Fig. S1 |
| `run_report_numbers.py` | 2 | `results/report_numbers.tex` — every number the report quotes, as LaTeX macros |
| `plot_naive_models_fixed_k.py` | 3 | `figures/naive_models_fixed_k/*.pdf`, `*.png` |
| `plot_naive_models_estimated_k.py` | 3 | the same for `naive_models_estimated_k` |
| `plot_onset_models_fixed_k.py` | 3 | `figures/onset_models_fixed_k/*.pdf`, `*.png` |
| `plot_onset_models_estimated_k.py` | 3 | the same for `onset_models_estimated_k` |
| `run_underreporting_60.py`, `run_underreporting_80.py` | 1 | the same tier-1 outputs, with the onsets read as *reported* counts |
| `plot_underreporting.py` | 3 | `figures/underreporting/*` — the reporting sweep |
| `run_no_switch_fixed_R.py`, `run_no_switch_single_R.py` | 1 | the exploratory variants' tier-1 outputs, with the `R` switchpoint removed |
| `plot_no_switch.py` | 3 | `figures/no_switch_*/*` — both variants, from one script taking `--analysis` |
| `plot_model_schematic.py` | 3 | Main Fig. 1: the infection- vs onset-anchored schematic. Draws no data, so it takes no input but the house style |
| `plot_delay_distributions.py` | 3 | Supplementary Fig. S1: incubation, TOST and serial-interval distributions |
| `plot_sustained_transmission.py` | 3 | Main Fig. 6 (fixed `k`) and Supplementary Fig. S2 (estimated `k`): RAC/RAT/RST by anchoring |
| `figure_panels.py` | 3 | the panels the analysis figures are assembled from |
| `utils.py` | — | presentation-only helpers shared by the plotting scripts |

The two onset-anchored analyses supply both the infection- and onset-anchored panels of the RST
figures. That explicit list, not the config, remains what
`rule all` and the convenience aggregates are built from — and Stage 10's `run_report_numbers.py`
takes it on the command line rather than growing a second copy of it, because it is the one
script that spans the analyses.

**The two no-switchpoint variants are exploratory and stay out of the report.** They are in
`config/config.yaml` but in neither `IMPLEMENTED_ANALYSES` nor `SAMPLED_ANALYSES`, so `rule all`
and `run_report_numbers.py` never see them and nothing builds them unless the target is named
(`pixi run pipeline-no-switch`). They have no `plot_script:` key, because `rule figure` would
demand a `model_evidence.json` that `no_switch_fixed_R` — every parameter fixed — has nothing to
compute; `rule no_switch_figure` draws them instead. Promoting one to a report analysis means
adding it to `IMPLEMENTED_ANALYSES` and giving it a `plot_script`.

Conventions, all of them load-bearing:

- **Compute-and-save and load-and-plot are separate scripts**, so restyling a figure never
  re-runs MCMC. The run scripts carry one subcommand per pipeline rule (`fit`, `rac`, `evidence`,
  and `dispersion` where `k` is estimated) and each Snakemake rule invokes exactly one of them.
- **`rac` is a tier-1 step**, not tier 2. RAC, RAT and RST all condition on the record through the
  conditioning day, so the curve is one MCMC fit per day — about 110 per model — and it writes
  `..._rac_diagnostics.csv` beside the curve, one row per day, failing outright if any of those
  fits did not converge. `--method single_fit_filtered` swaps in the fast approximation for
  prototyping; it is not a results path. **`--jobs N` spreads the days over `N` worker
  processes** and changes nothing else — each day's seed is fixed before any of them start — so
  the pipeline passes Snakemake's `{threads}` and it appears in no rule's `params:`.
- **Named for what they do**, never for figure numbers.
- **The figure layout follows the number of parameter posteriors, and the RAC panel's *view*
  follows the layout.** A fixed-`k` figure has two (`R_pre`, `R_post`), so it is four panels with
  the RAC spanning the bottom row; an estimated-`k` figure has three, so it is five, with the pie
  dropping beside the RAC curves. The no-switchpoint variants take the rule to its other end:
  `no_switch_single_R` has one, so it is two stacked panels, and `no_switch_fixed_R` has none at
  all, so it is the RAC panel by itself. Where the RAC panel shares its row it starts at the ERT's
  arrival (`risk_curve_panel(first_day=...)`). **`first_day` trims the view only** — every curve
  is still drawn over the whole window and the settling markers are untouched, so a trimmed panel
  can never show a different crossing date from an untrimmed one.
- **Both formats come from one render**, so the PDF and the PNG cannot disagree.
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
  the text. `plot_model_schematic.py` takes the rule to its limit: it draws no data, so it reads
  no file and prints no number — the incubation period its panel D turns on is quoted in the
  report's caption through `\resultnum`, not lettered onto the canvas where a config change could
  not reach it.
- **A missing input fails loudly.** `utils.read_model_evidence` and `read_dispersion_summary`
  raise rather than falling back when their file is absent: a pie chart of placeholder numbers
  is indistinguishable from a real one on the page. `run_report_numbers.read_json` does the same,
  for the same reason — and the report's `\resultnum` raises a *LaTeX* error on an undefined key,
  so the equivalent failure in the prose stops the build rather than printing nothing.

**The run scripts are a docstring and an analysis name apiece** — Stage 7 factored the shared
body into `analysis_driver.py`, once the second analysis made its real shape visible. It could
not go in `utils.py`: that file is in `PLOT_CORE`, and putting fit-driving logic in it would
make every restyle a reason to re-run MCMC. `analysis_driver.py` is named instead in `FIT_CORE`,
`RAC_CORE`, `EVIDENCE_CORE` and `DISPERSION_CORE`. The plot scripts stay one per analysis,
because the panel *arrangement* is what differs between figures; what they share is
`figure_panels.py`.

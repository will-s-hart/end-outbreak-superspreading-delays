# Pipeline scripts

The four report analyses, and nothing else. Every script here is invoked by a Snakemake rule
and writes to `results/` or `figures/`; if an output is not a report input, it does not belong
in this tree — see `validation/` for benchmarks and cross-checks.

| Script | Tier | Writes |
| --- | --- | --- |
| `run_naive_models_fixed_k.py` | 1 and 2 | `results/naive_models_fixed_k/{model}_posterior.nc`, `..._rac.csv`, `model_evidence.json` |
| `plot_naive_models_fixed_k.py` | 3 | `figures/naive_models_fixed_k/*.pdf`, `*.png` |
| `utils.py` | — | presentation-only helpers shared by the plotting scripts |

Stage 7 adds the estimated-`k` naive analysis and Stage 9 the two onset-anchored ones. Add each
to `IMPLEMENTED_ANALYSES` at the top of the `Snakefile` as it lands — that list, not the config,
is what `rule all` and the convenience aggregates are built from, so an analysis whose scripts do
not yet exist cannot turn a target into a missing-input error.

Conventions, all of them load-bearing:

- **Compute-and-save and load-and-plot are separate scripts**, so restyling a figure never
  re-runs MCMC. The run scripts carry one subcommand per tier (`fit`, `rac`, `evidence`) and
  each Snakemake rule invokes exactly one of them.
- **Named for what they do**, never for figure numbers.
- **`argparse`, invoked from `shell:`**, never through Snakemake's `script:` directive, so each
  stays runnable and debuggable on its own.
- **Config parsing is in the package** (`end_of_outbreak.configuration`), not here, because the
  validation scripts need it too and because the Snakemake rules have to be able to name it in
  their `input:` lists. `utils.py` is for figure styling and similar — things whose only effect
  is on tier 3.
- **A plotting script computes nothing.** Every number it draws was written to `results/` by a
  tier-2 rule. The one exception is deliberate and documented in `utils.py`: the plot scripts
  borrow `RiskCurve.first_day_below` from the package, because "the day a curve settles below a
  threshold" is a definition the report quotes and must not be able to drift between the marker
  on the panel and the number in the text.
- **A missing input fails loudly.** `utils.read_model_evidence` raises rather than falling back
  when `model_evidence.json` is absent: a pie chart of placeholder numbers is indistinguishable
  from a real one on the page.

The four run scripts differ only in the analysis they name and in what their docstrings say
about it. That duplication is deliberate for now — factor a shared driver when the second one
lands in Stage 7 and the real shape of it is visible, not before. Whatever comes out of that,
it cannot live in `utils.py`: that file is in `PLOT_CORE`, and putting fit-driving logic in it
would make every restyle a reason to re-run MCMC.

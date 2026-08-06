# Pipeline scripts

The four report analyses, and nothing else. Every script here is invoked by a Snakemake rule
and writes to `results/` or `figures/`; if an output is not a report input, it does not belong
in this tree — see `validation/` for benchmarks and cross-checks.

Arriving with Stage 5:

| Script | Tier | Writes |
| --- | --- | --- |
| `run_<analysis>.py` | 1 and 2 | `results/<analysis>/<model>_posterior.nc`, `..._rac.csv`, `model_evidence.json` |
| `plot_<analysis>.py` | 3 | `figures/<analysis>/*.pdf`, `*.png` |
| `utils.py` | — | presentation-only helpers shared by the plotting scripts |

Conventions, all of them load-bearing:

- **Compute-and-save and load-and-plot are separate scripts**, so restyling a figure never
  re-runs MCMC.
- **Named for what they do**, never for figure numbers.
- **`argparse`, invoked from `shell:`**, never through Snakemake's `script:` directive, so each
  stays runnable and debuggable on its own.
- **Config parsing is in the package** (`end_of_outbreak.configuration`), not here, because the
  validation scripts need it too and because the Snakemake rules have to be able to name it in
  their `input:` lists. `utils.py` is for figure styling and similar — things whose only effect
  is on tier 3.

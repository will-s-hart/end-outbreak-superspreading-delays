# Validation studies

Cross-checks, benchmarks and one-off measurements. **Nothing here is a results path**: no figure
in the report and no number in `results/` is produced by this directory, and no Snakemake rule
depends on it. That separation is the point — the scripts here are run on demand, and their
outputs are evidence about the *implementation*, not about the outbreak.

| Script | What it answers | Written to |
| --- | --- | --- |
| `run_sampler_benchmark.py` | Which latent parameterisation should the project use? (Stage 3, §6.3) | `results/sampler_benchmark.md` + two CSVs |
| `run_rac_validation.py` | Are the RAC calculators and the reset state right, is the latent marginalisation exact, and how far apart are the two estimators? (§5.1/§6.4) | `results/rac_validation.md` + six CSVs and a figure |
| `run_evidence_validation.py` | Are the marginal likelihoods right, on the real series? (Stage 6, §6.5) | `results/evidence_validation.md` + three CSVs |
| `run_onset_particle_mcmc_check.py` | Do the onset filters, matched RAC state and PMMH route agree with PyMC? (Stage 8, §6.6) | `results/onset_particle_mcmc_validation.md` + three CSVs |
| `run_rst_validation.py` | Do the analytic RST formulas agree with independent Galton–Watson continuation simulation? | `results/rst_validation.md` + `results/rst_validation.csv` |

Each study has a written summary in `results/` beside its data. Read that first: it carries the
conclusion, what was falsified along the way, and what the study does *not* cover. The CSVs are
the raw runs behind it.

## Conventions

- **Where things go.** A new check adds `validation/run_<name>.py` and writes to
  `validation/results/`. `scripts/` and `results/` are for the four report analyses only, so a
  reader of `results/` never has to work out which files feed a figure.
- **The outputs are committed**, like `results/` and `figures/`, so a claim in `AGENTS.md` or the
  report can be traced to the run behind it without re-running hours of MCMC.
- **Write the summary.** A CSV of runs nobody has interpreted is not a validation; the
  `.md` beside it is the deliverable, and it should say what would have counted as a failure.
- **Config plumbing is shared** with the pipeline scripts, through the package
  (`end_of_outbreak.configuration`) rather than through either tree reaching into the other.
  Only *outputs* are kept apart, not method.
- **Slow is fine here.** Checks that need MCMC on the real series live here rather than in
  `tests/`, which is what keeps `pixi run test` to about twenty seconds. The same comparisons on
  short synthetic histories *are* in `tests/`, as fast regression tests.

`ruff` and `ty` cover this directory (`pixi run check`), so it is held to the same standard as
the package.

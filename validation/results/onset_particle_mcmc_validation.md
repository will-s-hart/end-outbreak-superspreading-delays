# Onset-anchored particle-filter and PMMH validation

## Outcome

- The onset-filter PMMH tuning criterion is comfortably met: the largest measured
  `Var(log L_hat)` is **0.081**, below the practical 1–3 range.
- With conditioning matched — both sides conditioned on the whole record — particle-smoothed and
  MCMC RAC differ by at most **0.0214**. That is the check on the arithmetic, and it
  now also checks the exact latent marginalisation: the MCMC side integrates the latents the fit
  removed out of the risk in closed form rather than drawing them.
- The filtering column is produced by `end_of_outbreak.filtered_risk`, the module the pipeline
  would use under `rac.method: single_fit_filtered`, rather than re-derived here. It differs from
  the matched smoothing curve by up to **0.6517** — a difference of *estimand*,
  not of implementation, and the same one `rac_method_comparison.csv` measures against the
  refitting estimand.
- PMMH and PyMC 95% intervals overlap for **6 of
  6** synthetic-model parameter comparisons. PMMH acceptance rates are
  0.26–0.29.

## Files

- `onset_smc_variance.csv` — repeated real-record SMC likelihood estimates.
- `onset_rac_filtering.csv` — matched smoothing equality check, and the filtering curve beside it.
- `onset_particle_mcmc_synthetic.csv` — independent PMMH/PyMC posterior comparison on simulator output.

The main analyses remain PyMC-based. These runs validate the independent particle route and do
not feed `results/`, figures, or the Snakefile.

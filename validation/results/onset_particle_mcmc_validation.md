# Stage-8 onset particle-filter and PMMH validation

## Outcome

- The onset-filter PMMH tuning criterion is comfortably met: the largest measured
  `Var(log L_hat)` is **0.081**, below the practical 1–3 range.
- With conditioning matched, particle-smoothed and MCMC-smoothed RAC differ by at most
  **0.0214**. Filtering remains a distinct sensitivity estimand; its signed curve
  is recorded in `onset_rac_filtering.csv` rather than treated as an equality check.
- PMMH and PyMC 95% intervals overlap for **6 of
  6** synthetic-model parameter comparisons. PMMH acceptance rates are
  0.26–0.29.

## Files

- `onset_smc_variance.csv` — repeated real-record SMC likelihood estimates.
- `onset_rac_filtering.csv` — matched smoothing equality check and filtering sensitivity.
- `onset_particle_mcmc_synthetic.csv` — independent PMMH/PyMC posterior comparison on simulator output.

The main analyses remain PyMC-based. These runs validate the independent particle route and do
not feed `results/`, figures, or the Snakefile.

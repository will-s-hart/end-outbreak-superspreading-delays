# Validation of risk of sustained transmission

**Conclusion:** the semi-analytic RST formulas agree with independent Galton--Watson
continuation simulation. The largest discrepancy is **1.49
simulation standard errors** (ssi); the largest absolute discrepancy is
**0.0023**.

Each replicate begins from a fixed retained weight of 0.35. Onset-anchored
replicates additionally carry a Poisson incubation pipeline of mean 0.2. Future
transmission runs at `R = 2` and `k = 0.18` for the overdispersed models. A replicate
is resolved as surviving when one generation reaches 200 infections; the
largest upper bound on subsequently becoming extinct is
7.90e-18.

The CSV beside this file records the analytic and simulated values, Monte-Carlo standard errors,
and truncation bounds for SSE, SSI, their onset-anchored forms, and both Poisson limits. DLO is
absent because it does not define independent offspring families and hence has no corresponding
Galton--Watson extinction event.

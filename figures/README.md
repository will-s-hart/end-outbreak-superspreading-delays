Committed figures (PDF + PNG), one subdirectory per analysis, drawn by `scripts/plot_*.py` from
what the earlier tiers wrote to `results/`.

| Path | What it is |
| --- | --- |
| `<analysis>/<analysis>.pdf`, `.png` | The analysis's main figure. |
| `delay_distributions/delay_distributions.pdf`, `.png` | Supplementary Fig. S1: incubation, TOST and target/implied serial intervals. |
| `onset_models_rat/onset_models_rat.pdf`, `.png` | Supplementary Fig. S2: RAC/RAT comparison for Analyses 3–4. |

Both formats are written from a single render, so they cannot disagree. Restyling is tier 3 and
costs seconds: nothing here re-runs a fit.

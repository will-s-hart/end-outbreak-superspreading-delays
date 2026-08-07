Committed figures (PDF + PNG), one subdirectory per analysis, drawn by `scripts/plot_*.py` from
what the tier-2 rules wrote to `results/`.

| Path | What it is |
| --- | --- |
| `<analysis>/<analysis>.pdf`, `.png` | The analysis's main figure. |
| `<analysis>/<analysis>_supplementary.pdf`, `.png` | A standalone panel displaced from the main figure, where there is one. |

Both formats are written from a single render, so they cannot disagree. Restyling is tier 3 and
costs seconds: nothing here re-runs a fit.

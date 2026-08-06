Committed analysis outputs, one subdirectory per analysis plus `checks/` for validation studies
that are not part of the pipeline.

| Path | What it is |
| --- | --- |
| `checks/sampler_benchmark.md` | The Stage-3 written comparison: which latent parameterisation the project uses, and why. |
| `checks/sampler_benchmark.csv` | Its 64 fixed-`k` runs, from `scripts/run_sampler_benchmark.py`. |
| `checks/sampler_benchmark_estimated_k.csv` | The 10 estimated-`k` runs (analyses 2 and 4). |

The per-analysis subdirectories (`naive_models_fixed_k/` and the rest) arrive with Stage 5.

Git does not preserve mtimes, so a fresh clone looks stale to Snakemake. The default profile in
`config/snakemake_profile/` drops the `mtime` rerun trigger for that reason; `snakemake --touch`
restores mtime consistency after cloning if you want it.

"""Rewrite committed fits through the current on-disk format, in place.

A maintenance tool, not a Snakemake rule. It exists because the format a fit is *stored* in
changed — :data:`fitting.NETCDF_COMPRESSION` — while the numbers in it did not, and re-running
the pipeline to pick that up would mean thousands of MCMC fits for a byte-level change.

Run it wherever fits live, including the cluster, and then re-stamp provenance::

    python scripts/recompress_fits.py --apply
    snakemake --touch                     # locally
    snakemake --touch results             # on the cluster, over `hpc ssh`

Without the `--touch`, Snakemake sees changed input bytes and marks every curve stale.

It rewrites the container, never the contents: it does not thin. `sampler.thin` decides how
many draws a fit is written with, and that is a property of the run that produced it, so
changing it means refitting rather than rewriting.

Idempotent: a fit already in the current format is rewritten to the same content. Nothing is
replaced until the rewrite has been read back and checked array by array against the original,
because these are committed results and the failure mode of a silent dtype or fill-value change
is a wrong number rather than a crash.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

from end_of_outbreak import configuration, fitting


def assert_same_attributes(written: dict, original: dict, where: object) -> None:
    """Compared as arrays, so that NaN matches NaN.

    ``fixed_k`` and the two ``fixed_R_*`` attributes are NaN on every analysis that estimates
    them, which a plain ``==`` reports as a difference in every such fit.
    """
    assert set(written) == set(original), where
    for key, value in original.items():
        np.testing.assert_array_equal(
            np.asarray(written[key]), np.asarray(value), err_msg=f"{where} attribute {key}"
        )


def rewrite(path: Path, *, apply: bool) -> tuple[int, int]:
    """Rewrite one fit beside itself, verify it, and replace only if ``apply``."""
    original = xr.open_datatree(str(path), engine=fitting.NETCDF_ENGINE).load()
    before = path.stat().st_size
    temporary = path.with_suffix(".nc.rewriting")
    fitting.save_fit(original, temporary)

    written = xr.open_datatree(str(temporary), engine=fitting.NETCDF_ENGINE).load()
    assert sorted(written.groups) == sorted(original.groups), path
    assert_same_attributes(written.attrs, original.attrs, path)
    for group in original.groups:
        if group == "/":
            continue
        was, now = original[group].to_dataset(), written[group].to_dataset()
        assert_same_attributes(now.attrs, was.attrs, (path, group))
        assert set(now.data_vars) == set(was.data_vars), (path, group)
        for name, variable in was.data_vars.items():
            np.testing.assert_array_equal(np.asarray(now[name]), np.asarray(variable))
            assert now[name].dims == variable.dims, (path, name)
            assert now[name].dtype == variable.dtype, (path, name)
        for name, coordinate in was.coords.items():
            np.testing.assert_array_equal(np.asarray(now.coords[name]), np.asarray(coordinate))

    after = temporary.stat().st_size
    if apply:
        os.replace(temporary, path)
    else:
        temporary.unlink()
    return before, after


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    total_before = total_after = largest = 0
    for path in sorted(args.results_root.rglob("*_posterior.nc")):
        before, after = rewrite(path, apply=args.apply)
        total_before += before
        total_after += after
        largest = max(largest, after)
        print(f"{before / 1048576:8.1f} -> {after / 1048576:8.1f} MB  {path}")
    print(f"\ntotal {total_before / 1048576:.1f} -> {total_after / 1048576:.1f} MB")
    print(f"largest fit {largest / 1048576:.1f} MB")
    print("verified: every array, dtype, dimension, coordinate and attribute unchanged")
    if not args.apply:
        print("dry run; pass --apply to replace the files")
    else:
        print("now re-stamp provenance with `snakemake --touch`, or every curve looks stale")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=configuration.REPO_ROOT / "results")
    parser.add_argument(
        "--apply", action="store_true", help="replace the files; without it, only verify"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()

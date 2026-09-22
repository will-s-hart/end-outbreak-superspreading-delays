"""Running independent PyMC fits in parallel worker processes.

Where the parallelism lives
---------------------------
A fit has two axes that could be run in parallel: its chains, and — in the one step that runs
many fits, :mod:`end_of_outbreak.refit_risk` — its conditioning days. PyMC will do the first for
you, and this project deliberately does not let it: :func:`end_of_outbreak.fitting.fit_model`
always samples with ``cores=1``. Parallelism is taken one level up, over the days.

Three reasons, in increasing order of how much they cost:

- **Process startup.** ``pm.sample(cores=4)`` spawns four processes per fit and each imports
  PyMC and PyTensor and unpickles the compiled log-density. A refit curve is 130 fits, so that
  is 520 process starts to do 130 fits' worth of work. Here the pool is built once and the
  workers are reused for the whole curve.
- **The chain count caps the speedup.** Four chains can use four cores and no more, however
  many the machine has. Days are the abundant axis — 110 of them — so the pool can be sized to
  the hardware rather than to the model.
- **The idle tail.** Chains of the same fit finish at different times, and ``pm.sample`` cannot
  return until the slowest one does. A day-level pool starts the next day's fit the moment a
  worker is free.

None of this changes a number. Chain ``c`` of a fit is seeded from ``random_seed`` by
``pm.sample`` itself, identically whether the chains are run in one process or four, and the
day-``t`` fits of a refit curve were always independent — ``seed_for_day`` gave each its own
stream long before any of them ran in parallel. ``tests/test_parallel.py`` pins both.

What a worker has to be given
-----------------------------
Two things, and getting either wrong turns the parallelism into a slowdown:

- **Its own local compiler caches.** PyTensor caches compiled C modules and its Numba linker
  uses Numba's on-disk cache. Eight workers compiling the same never-seen-before graph at once
  must share neither cache — especially when their defaults live on a cluster network
  filesystem. :func:`prepare_worker` gives each worker private PyTensor, Numba and Matplotlib
  caches under node-local temporary storage and removes them at exit.
- **Single-threaded numeric libraries.** BLAS and OpenMP size their own thread pools from the
  core count, so eight workers each helpfully spawning eight threads oversubscribes the machine
  by a factor of eight. joblib's ``inner_max_num_threads`` sets the environment variables in the
  child *before* it imports NumPy, which is the only point at which they still take effect;
  :func:`prepare_worker` sets them again for anything loky does not cover.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""The thread-pool controls of every numeric backend the stack might link against.

Set to one in a worker, and with ``setdefault`` rather than assignment: an operator who has
deliberately exported one of these knows better than we do what their machine wants.
"""

_WORKER_PREPARED = False
"""Whether this process has already been given its own caches. Workers are reused across tasks,
so the setup must be idempotent and must not build a fresh compile directory per fit."""


def prepare_worker() -> None:
    """Give this process its own compile directory and single-threaded numeric pools.

    Called **only in worker processes**, and idempotent because a worker serves many fits. The
    serial path deliberately does not call it: there is no cache to contend for with one
    process, and repointing PyTensor's compile directory would throw away a warm cache and
    leave the caller's session compiling from scratch for the rest of its life.
    """
    global _WORKER_PREPARED
    if _WORKER_PREPARED:
        return
    _WORKER_PREPARED = True

    for variable in THREAD_LIMIT_VARIABLES:
        os.environ.setdefault(variable, "1")

    # This must be local rather than a child of PyTensor's configured base directory: on the
    # cluster that base is on the network filesystem, where an otherwise valid Numba cache read
    # can fail with ESTALE while several workers compile graphs concurrently.
    root = Path(tempfile.mkdtemp(prefix=f"end_of_outbreak_worker_{os.getpid()}_"))
    compiledir = root / "pytensor"
    numba_cache_dir = root / "numba"
    matplotlib_configdir = root / "matplotlib"
    compiledir.mkdir()
    numba_cache_dir.mkdir()
    matplotlib_configdir.mkdir()

    # Set the environment before importing either compiler. The explicit Numba assignment also
    # covers an embedding process that happened to import it before calling this initializer.
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_configdir)
    import numba
    import pytensor

    # Numba's config attributes are generated dynamically from environment variables.
    numba.config.CACHE_DIR = str(numba_cache_dir)  # ty: ignore[unresolved-attribute]
    pytensor.config.compiledir = compiledir
    atexit.register(shutil.rmtree, root, ignore_errors=True)


def _prepared_call[TaskT, ResultT](function: Callable[[TaskT], ResultT], task: TaskT) -> ResultT:
    prepare_worker()
    return function(task)


def map_fits[TaskT, ResultT](
    function: Callable[[TaskT], ResultT],
    tasks: Iterable[TaskT],
    *,
    n_jobs: int = 1,
    description: str = "fits",
    on_result: Callable[[ResultT], None] | None = None,
) -> list[ResultT]:
    """Apply ``function`` to every task, in worker processes when asked for more than one.

    Results come back **in completion order, not task order**. Fits of different conditioning
    days take very different times, and holding a finished result back to preserve the input
    order would idle a worker for no reason. Every task therefore has to carry whatever the
    caller needs to put the results back in order — :mod:`end_of_outbreak.refit_risk` carries
    the day.

    ``on_result`` is called in this process as each result arrives, for progress reporting that
    outlives the run: the bar is for a human watching, the callback is for the log.

    ``n_jobs=1`` runs in this process with no pool at all, which is what the tests and any
    interactive use want: a traceback from a failed fit points at the fit, not at a worker.
    """
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least one, got {n_jobs}")
    task_list = list(tasks)
    if n_jobs == 1 or len(task_list) < 2:
        return _collect(
            (function(task) for task in task_list),
            total=len(task_list),
            description=description,
            on_result=on_result,
        )

    from joblib import Parallel, delayed, parallel_config

    # `inner_max_num_threads` is a loky-only setting, and it is the reason for naming loky
    # explicitly rather than taking joblib's default backend.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        # `batch_size=1` because the tasks are minutes long and wildly uneven: batching would
        # hand one worker several slow days while another sat idle.
        # The initializer runs before the worker unpickles a task and therefore before importing
        # the model closure can create a Numba cache locator. `_prepared_call` remains as an
        # idempotent safeguard for backends that do not honour the initializer.
        results = Parallel(
            n_jobs=n_jobs,
            return_as="generator_unordered",
            batch_size=1,
            initializer=prepare_worker,
        )(delayed(_prepared_call)(function, task) for task in task_list)
        return _collect(results, total=len(task_list), description=description, on_result=on_result)


def _collect[ResultT](
    results: Iterable[ResultT],
    *,
    total: int,
    description: str,
    on_result: Callable[[ResultT], None] | None,
) -> list[ResultT]:
    """Drain a stream of results, reporting each as it lands."""
    collected = []
    for result in _progress(results, total=total, description=description):
        if on_result is not None:
            on_result(result)
        collected.append(result)
    return collected


def _progress[ResultT](
    items: Iterable[ResultT], *, total: int, description: str
) -> Iterable[ResultT]:
    """``items`` wrapped in a progress bar.

    ``total`` is passed in rather than measured, because the parallel path hands this a
    generator that yields results as they finish and consuming it to count would defeat the
    purpose. The bar is worth having: a refit curve is the longest thing this project runs, and
    "how far through is it" is otherwise unanswerable without reading a file being written.
    """
    from tqdm.auto import tqdm

    return tqdm(items, total=total, desc=description)

"""Tests for running the conditioning-day fits across worker processes.

Parallelism is a throughput change, so the whole burden of proof is that it changes nothing
else. Three claims, in the order in which a violation would be missed for longest:

1. **The curve does not depend on the worker count.** A refit curve built in one process and
   the same curve built in four must agree to the bit. This is the claim the ``Snakefile``
   rests on when it leaves ``end_of_outbreak.parallel`` out of ``RAC_CORE`` and ``threads`` out
   of the ``rac`` params — if it were false, changing a core count would silently change a
   published number.
2. **Sequential chains are the same chains.** ``fitting.fit_model`` samples with ``cores=1``
   now. PyMC seeds chain ``c`` from ``random_seed`` before it decides how many processes to run
   them in, so the draws are unchanged; this pins that, because if it were false every
   committed posterior in ``results/`` would have quietly gone stale.
3. **Results come back in day order.** The pool returns them as they finish, which for fits of
   very uneven length is emphatically not the order they went in. A curve silently permuted in
   time would still plot, still settle somewhere, and be wrong.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from unittest import mock

import cloudpickle
import numpy as np
import pymc as pm
import pytest

from end_of_outbreak import delay_distributions as dd
from end_of_outbreak import fitting, parallel, refit_risk
from end_of_outbreak.model_specifications import LogNormalPrior

COUNTS = np.array([1, 0, 2, 1, 0, 3, 0, 1, 0, 0, 1, 0, 0, 0])
SWITCH_DAY = 6
K = 0.6
PRIOR = LogNormalPrior.from_median_and_quantile(median=1.0, quantile=0.2, probability=0.025)

SHORT_DELAYS = dd.build_onset_anchored_delays(
    serial_interval=dd.GammaDelay(mean=3.0, sd=2.0),
    incubation=dd.GammaDelay(mean=1.5, sd=1.0),
    max_lag=COUNTS.size,
    tolerance=None,
)

FAST = fitting.SamplerSettings(draws=100, tune=100, chains=2, seed=11)


PARAMETERISATION = "marginalised_inverse_cdf"


def _curve(model: str, *, n_jobs: int, days: np.ndarray) -> refit_risk.RefitRiskResult:
    return refit_risk.risk_by_refitting(
        model,
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        latent_parameterisation=PARAMETERISATION,
        days=days,
        sampler=FAST,
        # The seeds that make the days independent, and therefore reorderable.
        seed_for_day=lambda day: 1000 + day,
        n_jobs=n_jobs,
    )


# --- 1. the curve does not depend on the worker count ---------------------------------------


@pytest.mark.parametrize("model", ["sse", "ssi_so"])
def test_a_curve_built_in_parallel_matches_the_one_built_in_this_process(model):
    """The claim the pipeline rests on: `--jobs` buys wall time and costs no accuracy.

    Bit equality, not a tolerance. Every day carries its own seed, so there is no Monte-Carlo
    slack to allow for: if these differ at all, something about the fit depends on where it ran.
    """
    days = np.array([5, 8, 11, 13])
    serial = _curve(model, n_jobs=1, days=days)
    concurrent = _curve(model, n_jobs=3, days=days)

    np.testing.assert_array_equal(serial.estimate.days, concurrent.estimate.days)
    np.testing.assert_array_equal(
        serial.estimate.log_no_further_cases, concurrent.estimate.log_no_further_cases
    )


def test_the_reused_final_day_survives_the_parallel_route():
    """The one day that is not fitted in a worker still lands in the right place.

    It is evaluated in the parent, from the whole-record fit the pipeline already holds, and
    then appended to results that arrived in completion order — so this is exactly where a
    day could end up attached to the wrong estimate.
    """
    idata = fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        sampler=FAST,
    )
    days = np.array([9, 11, COUNTS.size - 1])

    def curve(n_jobs: int) -> refit_risk.RefitRiskResult:
        return refit_risk.risk_by_refitting(
            "sse",
            COUNTS,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            R_pre=PRIOR,
            R_post=PRIOR,
            k=K,
            days=days,
            sampler=FAST,
            seed_for_day=lambda day: 1000 + day,
            final_day_fit=idata,
            n_jobs=n_jobs,
        )

    serial = curve(1)
    concurrent = curve(3)

    np.testing.assert_array_equal(serial.estimate.days, days)
    np.testing.assert_array_equal(
        serial.estimate.log_no_further_cases, concurrent.estimate.log_no_further_cases
    )


# --- 2. sequential chains are the same chains -----------------------------------------------


def test_sampling_the_chains_in_one_process_gives_the_draws_four_processes_would():
    """``cores`` is a scheduling argument to ``pm.sample``, not a statistical one.

    Bit equality again. If this ever fails, ``results/`` is stale and the ``cores=1`` in
    :func:`end_of_outbreak.fitting.fit_model` is the reason.
    """
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=np.array([0.4, -0.2, 1.1]))
        sequential = pm.sample(
            draws=150, tune=150, chains=4, cores=1, random_seed=7, progressbar=False
        )
        concurrent = pm.sample(
            draws=150, tune=150, chains=4, cores=4, random_seed=7, progressbar=False
        )

    np.testing.assert_array_equal(
        np.asarray(sequential.posterior["mu"]), np.asarray(concurrent.posterior["mu"])
    )


def test_a_fit_asks_pymc_for_one_process_however_many_chains_it_wants(monkeypatch):
    """Pin the call itself, so a future edit cannot reintroduce chain-level processes quietly.

    The bit-equality test above would not catch it — it would still pass — but the pipeline's
    core budget and the pool's thread accounting both assume a fit occupies one core.
    """
    seen = {}
    real_sample = pm.sample

    def record(*args, **kwargs):
        seen.update(kwargs)
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(fitting.pm, "sample", record)
    fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        sampler=fitting.SamplerSettings(draws=50, tune=50, chains=3, seed=1),
    )
    assert seen["cores"] == 1
    assert seen["chains"] == 3


# --- 3. results are put back in order, and workers are set up ---------------------------------


def test_results_come_back_in_day_order_however_they_finish():
    """The curve is a time series, so the sort is load-bearing, not tidiness."""
    days = np.array([3, 6, 9, 12])
    result = _curve("sse", n_jobs=3, days=days)
    np.testing.assert_array_equal(result.estimate.days, days)
    assert [day.day for day in result.diagnostics] == list(days)


def test_the_work_sent_to_a_worker_does_not_carry_the_whole_record_fit():
    """A closure is pickled once per task, so what it closes over is a per-day cost.

    The whole-record fit is tens of megabytes and no worker ever reads it — the day it stands in
    for is evaluated in the parent. Capturing it anyway would be invisible in every result and
    would quietly ship gigabytes over a 110-day curve, so the size is pinned rather than trusted.
    """
    idata = fitting.fit_model(
        "sse",
        COUNTS,
        delays=SHORT_DELAYS,
        switch_day=SWITCH_DAY,
        R_pre=PRIOR,
        R_post=PRIOR,
        k=K,
        sampler=FAST,
    )
    fit_bytes = len(cloudpickle.dumps(idata))
    sent: dict[str, int] = {}
    real_map = parallel.map_fits

    def spy(function, tasks, **kwargs):
        sent["bytes"] = len(cloudpickle.dumps(function))
        return real_map(function, tasks, **kwargs)

    with mock.patch.object(refit_risk.parallel, "map_fits", spy):
        refit_risk.risk_by_refitting(
            "sse",
            COUNTS,
            delays=SHORT_DELAYS,
            switch_day=SWITCH_DAY,
            R_pre=PRIOR,
            R_post=PRIOR,
            k=K,
            days=np.array([11, 12, COUNTS.size - 1]),
            sampler=FAST,
            seed_for_day=lambda day: 1000 + day,
            final_day_fit=idata,
        )
    assert sent["bytes"] < fit_bytes / 4, (
        f"the per-task closure is {sent['bytes']} bytes against a {fit_bytes}-byte fit, so it "
        "is probably carrying the fit"
    )


def test_map_fits_reports_every_result_exactly_once():
    """``on_result`` is what keeps the pipeline log informative; a dropped call loses a day."""
    seen: list[int] = []
    returned = parallel.map_fits(
        lambda task: task * 2, [1, 2, 3, 4], n_jobs=2, on_result=seen.append
    )
    assert sorted(seen) == [2, 4, 6, 8]
    assert sorted(returned) == [2, 4, 6, 8]


def test_a_worker_count_below_one_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        parallel.map_fits(lambda task: task, [1, 2], n_jobs=0)


def test_preparing_a_worker_pins_the_numeric_thread_pools_and_gives_it_its_own_cache():
    """Two settings that turn the pool from a speedup into a slowdown when they are missing.

    Run in a subprocess because that is the only honest way to test it: ``prepare_worker``
    repoints PyTensor's compile directory for the life of the process, so calling it here would
    make the rest of the suite compile from scratch. The second call checks it is idempotent —
    a worker serves many fits, and a fresh cache per fit would be worse than sharing one.
    """
    program = textwrap.dedent(
        """
        import os
        import pytensor
        from end_of_outbreak import parallel

        parallel.prepare_worker()
        first = pytensor.config.compiledir
        parallel.prepare_worker()

        assert pytensor.config.compiledir == first, "not idempotent"
        assert os.path.isdir(first), first
        assert os.environ["MPLCONFIGDIR"]
        for variable in parallel.THREAD_LIMIT_VARIABLES:
            assert os.environ[variable] == "1", variable
        print("ok")
        """
    )
    finished = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert finished.returncode == 0, finished.stderr
    assert "ok" in finished.stdout


def test_a_worker_leaves_an_operators_own_thread_setting_alone():
    """``setdefault``, not assignment: someone who exported this knows their machine."""
    program = textwrap.dedent(
        """
        import os
        from end_of_outbreak import parallel

        parallel.prepare_worker()
        assert os.environ["OMP_NUM_THREADS"] == "3", os.environ["OMP_NUM_THREADS"]
        print("ok")
        """
    )
    finished = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "3"},
    )
    assert finished.returncode == 0, finished.stderr
    assert "ok" in finished.stdout

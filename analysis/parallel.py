"""Worker pools shared by the MMRM, mediation and pathway engines, and the BLAS pinning they use.

The engines parallelize across independent fits or gene-set libraries, so each worker runs
its BLAS single-threaded: threaded BLAS in every worker would oversubscribe the cores.
``forkserver_pool`` pins the threads and starts the pool (MMRM, mediation); the pathway
batch calls ``pin_blas_threads`` before starting its own executor; the Fig 3b rotation null
(``analysis/pipelines/run_disease_reversal.py``), which runs in one process, calls it before it
imports numpy, so its BLAS is single-threaded too. The module imports nothing heavy, so a
script can pin the BLAS threads before it imports numpy.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.pool import Pool

# Thread-pool sizes that OpenBLAS, OpenMP, MKL and numexpr read when they load.
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_blas_threads() -> None:
    """Set every BLAS thread-pool size to one for this process and the workers it starts.

    A library reads the variable when it loads, so this takes effect in processes
    started afterwards and in this one only if called before numpy is imported.
    """
    for var in BLAS_THREAD_VARS:
        os.environ[var] = "1"


def forkserver_pool(n_jobs: int, *, maxtasksperchild: int | None) -> Pool:
    """A pool of single-BLAS-thread workers started from a fork server.

    A forked child inherits the parent's Polars thread-pool state and hangs on its first
    Polars operation, so workers start from a clean fork server instead.

    Args:
        n_jobs: Number of worker processes.
        maxtasksperchild: Tasks a worker runs before it is replaced; None keeps
            workers for the life of the pool.

    Returns:
        The pool; use it as a context manager.
    """
    pin_blas_threads()
    return mp.get_context("forkserver").Pool(n_jobs, maxtasksperchild=maxtasksperchild)


def resolve_n_jobs(n_jobs: int, n_tasks: int | None = None) -> int:
    """Worker count for a pool: ``-1`` means every core but one, never more than the tasks.

    Args:
        n_jobs: Requested workers; ``-1`` for all cores but one.
        n_tasks: Number of tasks to run; caps the count when given.

    Returns:
        The worker count, at least one when ``n_tasks`` is given.
    """
    n = mp.cpu_count() - 1 if n_jobs == -1 else n_jobs
    return n if n_tasks is None else max(1, min(n, n_tasks))


def default_chunksize(n_tasks: int, n_jobs: int) -> int:
    """Tasks per pool chunk: about twenty chunks per worker, one to five tasks each.

    Args:
        n_tasks: Total number of tasks.
        n_jobs: Number of workers.

    Returns:
        Chunk size between 1 and 5.
    """
    return min(5, max(1, n_tasks // (n_jobs * 20)))


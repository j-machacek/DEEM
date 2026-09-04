#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Jan Machaček
#
# This file is part of DEEM, released under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.

"""
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                               Copyright (C) 2023-2025 Jan Machacek
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 05.02.2023
#
# DEEM - Differential Evolution with Elitism and Multi-populations
#
# History
# 05.02.2023, J. Machacek - Initial version
# 23.06.2026, J. Machacek - Added Windows-compatible worker initialization: the
#                           process pool is started with an initializer that
#                           re-establishes ACT.globals in every spawned worker.
#                           On Linux the pool inherits the parent state through
#                           fork; on Windows workers are spawned and need this.
# 23.06.2026, J. Machacek - A) evaluate_cost_function additionally returns the
#                              number of *real* objective calls performed, so the
#                              driver can keep an exact function-evaluation budget
#                              (relevant when each call is an expensive FE run).
#                           B) Optional EvalCache (consulted in the MAIN process
#                              before any worker is dispatched): identical (within a
#                              tolerance) positions are answered from the cache and
#                              never re-evaluated. None by default (off).
# 04.09.2026, J. Machacek - Added reusable process executors across generations
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import numpy as np
from concurrent import futures


def get_worker_data():
    # 23.06.2026, J. Machacek - Initial version
    #
    # On Linux the process pool inherits ACT.globals through fork. On Windows
    # new worker processes are spawned and therefore need explicit initialization.
    try:
        import ACT.globals as _globals
        return {
            'model': _globals.model,
            'free_parameter': _globals.free_parameter,
            'exp_oedometer': _globals.exp_oedometer,
            'exp_triaxCD': _globals.exp_triaxCD,
            'exp_triax_CU': _globals.exp_triaxCU,
            'exp_triax_CUcyc': _globals.exp_triax_CUcyc,
            'exp_triax_CDcyc_HCA': _globals.exp_triax_CDcyc_HCA,
            'exp_triax_CUcyc_HCA': _globals.exp_triax_CUcyc_HCA,
            'exp_USScyc': _globals.exp_USScyc,
            'exp_isotropic_compression': _globals.exp_isotropic_compression,
            'similarity': _globals.similarity,
            'verbose': _globals.verbose,
            'weights': _globals.weights,
            'parent_path': _globals.parent_path,
            'penalty': _globals.penalty,
            'timeout': _globals.timeout,
            'experimental_database': getattr(_globals, 'experimental_database', '-')
            }
    except Exception:
        return None


def initialize_worker(worker_data):
    # 23.06.2026, J. Machacek - Initial version

    if worker_data is None:
        return

    import ACT.globals as _globals
    _globals.model = worker_data['model']
    _globals.free_parameter = worker_data['free_parameter']
    _globals.exp_oedometer = worker_data['exp_oedometer']
    _globals.exp_triaxCD = worker_data['exp_triaxCD']
    _globals.exp_triaxCU = worker_data['exp_triax_CU']
    _globals.exp_triax_CUcyc = worker_data['exp_triax_CUcyc']
    _globals.exp_triax_CDcyc_HCA = worker_data['exp_triax_CDcyc_HCA']
    _globals.exp_triax_CUcyc_HCA = worker_data['exp_triax_CUcyc_HCA']
    _globals.exp_USScyc = worker_data['exp_USScyc']
    _globals.exp_isotropic_compression = worker_data['exp_isotropic_compression']
    _globals.similarity = worker_data['similarity']
    _globals.verbose = worker_data['verbose']
    _globals.weights = worker_data['weights']
    _globals.parent_path = worker_data['parent_path']
    _globals.penalty = worker_data['penalty']
    _globals.timeout = worker_data['timeout']
    _globals.experimental_database = worker_data['experimental_database']
    _globals.initialized = True


def update_particles_group(particle_group, updated_particles):
    for p in particle_group:
        p.update_cost()
    updated_particles.extend(particle_group)


def update_particle(particle):
    particle.update_cost()
    return particle


class EvalCache:
    """
    Lightweight evaluation cache keyed on the position rounded to a relative
    tolerance of the search-space span. Answers identical/near-identical positions
    without a new (expensive) objective call. Lives in the MAIN process, so it is
    compatible with serial and parallel (thread/process) evaluation alike.
    """

    def __init__(self, LB, UB, rel_tol: float = 1e-9, max_size: int = 200000):
        self.LB = np.asarray(LB, dtype=float)
        self.span = np.asarray(UB, dtype=float) - self.LB
        self.span[self.span == 0.0] = 1.0
        self.rel_tol = rel_tol
        self.max_size = max_size
        self._store = {}

    def _key(self, x: np.ndarray):
        scaled = (np.asarray(x, dtype=float) - self.LB) / self.span
        return tuple(np.round(scaled / self.rel_tol).astype(np.int64).tolist())

    def get(self, x: np.ndarray):
        return self._store.get(self._key(x), None)

    def put(self, x: np.ndarray, f: float) -> None:
        if len(self._store) < self.max_size:
            self._store[self._key(x)] = f

    def __len__(self):
        return len(self._store)


def create_evaluation_executor(nworkers: int, mode: str = 'process'):
    """Create one reusable executor for all generations of an optimisation."""
    if nworkers <= 1:
        return None
    if mode == 'thread':
        return futures.ThreadPoolExecutor(max_workers=nworkers)
    worker_data = get_worker_data()
    return futures.ProcessPoolExecutor(
        max_workers=nworkers,
        initializer=initialize_worker,
        initargs=(worker_data,)
    )


def evaluate_cost_function(particles, nworkers: int = 1, mode: str = 'process',
                           cache: 'EvalCache' = None, executor=None):
    """
    Evaluate the cost function for a list of candidate solutions.

    Returns
    -------
    (particles, n_real_evals) : tuple(list, int)
        The evaluated candidate list and the number of *real* objective calls.

    Notes
    -----
    A caller-supplied executor can be reused across generations. When no executor
    is supplied, a temporary pool is created for backward compatibility. Process
    pools use initialize_worker so spawned workers (Windows) re-establish
    ACT.globals before evaluating; on Linux the state is inherited through fork.
    """
    # ---- 1) cache lookup in the main process ----------------------------------
    to_eval = particles
    if cache is not None:
        to_eval = []
        for p in particles:
            cached = cache.get(p.x)
            if cached is None:
                to_eval.append(p)
            else:
                p.f = cached
                p.improved = False
                if not p.initialized:
                    p.fbest = p.fbest0 = p.f
                    p.xbest = p.xbest0 = p.x.copy()
                    p.improved = True
                    p.initialized = True
                elif p.f < p.fbest:
                    p.xbest0 = p.xbest.copy(); p.fbest0 = p.fbest
                    p.xbest = p.x.copy(); p.fbest = p.f
                    p.niter_xbest = 0; p.improved = True
                p.niter_xbest += 1

    n_real = len(to_eval)

    # ---- 2) evaluate the remaining (uncached) candidates ----------------------
    if nworkers == 1 or n_real == 0:
        for p in to_eval:
            p.update_cost()
    elif executor is not None:
        done = list(executor.map(update_particle, to_eval))
        _copy_back(done, to_eval)
    elif mode == "thread":
        with futures.ThreadPoolExecutor(nworkers) as executor:
            done = list(executor.map(update_particle, to_eval))
        _copy_back(done, to_eval)
    else:  # process mode (default) with Windows-compatible worker initialization
        worker_data = get_worker_data()
        with futures.ProcessPoolExecutor(max_workers=nworkers,
                                         initializer=initialize_worker,
                                         initargs=(worker_data,)) as temporary_executor:
            done = list(temporary_executor.map(update_particle, to_eval))
        _copy_back(done, to_eval)

    # ---- 3) populate the cache ------------------------------------------------
    if cache is not None:
        for p in to_eval:
            cache.put(p.x, p.f)

    return particles, n_real


# numeric result fields written by CandidateSolution.update_cost(); copied from the
# worker-evaluated copies back onto the main-process candidates so that object
# identity (and the objective reference) is preserved in the driver. executor.map
# preserves input order, so the copy-back is a simple positional zip.
_RESULT_FIELDS = ('x', 'f', 'xbest', 'xbest0', 'fbest', 'fbest0',
                  'improved', 'initialized', 'niter_xbest')


def _copy_back(done, to_eval):
    for src, dst in zip(done, to_eval):
        for fld in _RESULT_FIELDS:
            setattr(dst, fld, getattr(src, fld))

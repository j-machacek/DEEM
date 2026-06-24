#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Jan Machaček
#
# This file is part of DEEM, released under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.

"""
Self-contained DEEMI optimisation examples.

Runs the improved optimiser DEEMI on the Rastrigin benchmark (global minimum
f(0,...,0) = 0) in four configurations, for problem dimensions nD = 2 and nD = 10:

    1. default          - DEEMI with its default settings
    2. cached           - DEEMI with the evaluation cache enabled (cache_tol)
    3. RBF surrogate    - DEEMI with the radial-basis-function surrogate
    4. GP surrogate     - DEEMI with the Gaussian-process surrogate (LCB selection)

A fixed RNG seed is passed to every run, so all configurations start from the
*same* initial population (DEEM seeds NumPy's global RNG at construction, and the
initial sampling draws from it). The optimisation trajectories then differ because
the cache / surrogate change which candidates are evaluated.

History:
24.06.2026, J. Machacek - Initial version

"""

import io
import sys
import time
import contextlib
import numpy as np

try:
    from DEEM.DEEMI import DEEM
    from DEEM.surrogate import SurrogateManager, GPSurrogateManager
except ModuleNotFoundError:  # allow running the file directly from examples/
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from DEEM.DEEMI import DEEM
    from DEEM.surrogate import SurrogateManager, GPSurrogateManager


# --------------------------------------------------------------------------- #
#  Benchmark objective: Rastrigin (minimise; global minimum 0 at the origin)  #
# --------------------------------------------------------------------------- #
def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


SEED = 42                      # same initialisation for every configuration
BOUND = 5.12                   # Rastrigin is usually evaluated on [-5.12, 5.12]


def build_optimizer(nD, config, maxiter):
    """Create a DEEMI optimiser for the given dimension and configuration."""
    LB = np.full(nD, -BOUND)
    UB = np.full(nD, BOUND)

    common = dict(
        function=rastrigin,
        lower_bound=LB, upper_bound=UB,
        nparticles_max=10 * nD, nparticles_min=10 * nD,
        npop_max=4, npop_min=4,
        maxiter=maxiter,
        seed=SEED,                       # identical starting points
    )

    if config == "default":
        return DEEM(**common)
    if config == "cached":
        return DEEM(**common, cache_tol=1e-8)
    if config == "RBF surrogate":
        sm = SurrogateManager(LB, UB, eval_frac=0.5, explore_frac=0.3)
        return DEEM(**common, surrogate=sm)
    if config == "GP surrogate":
        sm = GPSurrogateManager(LB, UB, eval_frac=0.5, explore_frac=0.3, kappa=1.5)
        return DEEM(**common, surrogate=sm)
    raise ValueError(config)


def run(nD, config, maxiter):
    """Run one configuration quietly and return a summary row."""
    with contextlib.redirect_stdout(io.StringIO()):   # silence DEEM's own log
        opt = build_optimizer(nD, config, maxiter)
        t0 = time.time()
        result = opt.update()
        dt = time.time() - t0
    return {
        "config": config,
        "nD": nD,
        "f": result["f"],
        "nfev": result["nfev"],
        "restarts": result["n_restarts"],
        "cache": result["cache_size"],
        "time": dt,
    }


def main():
    configs = ["default", "cached", "RBF surrogate", "GP surrogate"]
    maxiter_by_dim = {2: 80, 10: 250}

    rows = []
    for nD in (2, 10):
        for cfg in configs:
            rows.append(run(nD, cfg, maxiter_by_dim[nD]))

    header = f"{'config':<15} {'nD':>3} {'best f':>12} {'real nfev':>10} " \
             f"{'restarts':>9} {'cache':>6} {'time/s':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['config']:<15} {r['nD']:>3} {r['f']:>12.5e} {r['nfev']:>10} "
              f"{r['restarts']:>9} {r['cache']:>6} {r['time']:>7.2f}")

    print("\nNotes:")
    print("  * All configurations start from the same population (seed = %d)." % SEED)
    print("  * The cache helps when identical positions recur (rugged/near-discrete")
    print("    cost functions); on a smooth analytic function exact revisits are rare.")
    print("  * Surrogate pre-screening trades solution quality for far fewer real")
    print("    evaluations; it pays off when one evaluation is expensive (minutes+),")
    print("    not on a cheap analytic benchmark like this one.")


if __name__ == "__main__":
    main()

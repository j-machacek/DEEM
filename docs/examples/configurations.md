# Configurations: default, cache and surrogates

This example runs **DEEMI** on the [Rastrigin](https://en.wikipedia.org/wiki/Rastrigin_function)
benchmark (global minimum \(f(\mathbf{0}) = 0\)) in four configurations, for problem
dimensions \(n_D = 2\) and \(n_D = 10\):

1. **default** — DEEMI with its default settings.
2. **cached** — DEEMI with the evaluation cache enabled (`cache_tol`).
3. **RBF surrogate** — DEEMI with the radial-basis-function surrogate.
4. **GP surrogate** — DEEMI with the Gaussian-process surrogate (LCB selection).

A fixed RNG `seed` is passed to every run, so all four configurations start from the
*same* initial population; the trajectories then differ because the cache and the
surrogates change which candidates are evaluated.

!!! warning "Surrogate models are experimental"
    The surrogate models are currently **experimental**. In our experience they do
    not necessarily improve the optimisation process in all cases, and further
    investigation is required. We recommend running the optimisation **without a
    surrogate model first**.

## The script

```python
#!/usr/bin/env python3
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
```

## Running it

```bash
python examples/deemi_benchmarks.py
```

The four-configuration benchmark needs only numpy and scipy.

## Example output

A representative run (numbers vary slightly with the NumPy build):

| config | nD | best f | real nfev | restarts |
| ------ | --: | ------: | --------: | -------: |
| default | 2 | 2.9e-04 | 1640 | 6 |
| cached | 2 | 1.5e-04 | 1633 | 6 |
| RBF surrogate | 2 | 9.5e-04 | 890 | 4 |
| GP surrogate | 2 | 2.8e-04 | 900 | 5 |
| default | 10 | 2.6e-07 | 25200 | 9 |
| cached | 10 | 1.2e-05 | 24983 | 9 |
| RBF surrogate | 10 | 1.0e+00 | 12800 | 2 |
| GP surrogate | 10 | 1.0e+00 | 12850 | 3 |

On this *cheap analytic* benchmark the **default** optimiser is the most accurate.
The **surrogates** reach a worse objective here even though they use far fewer real
evaluations — a concrete illustration of why they are considered experimental and
why we recommend running without a surrogate first. Surrogate pre-screening only
pays off when a single evaluation is genuinely expensive (minutes or more), such as
the finite-element calibration of a constitutive model.

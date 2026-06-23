# Quickstart

This page walks through a minimal optimisation with `DEEM` and then shows the
equivalent call for the improved variant `DEEMI`.

## A first optimisation

The objective is any callable that maps a 1-D NumPy array (a candidate position
$\mathbf{x}$) to a scalar to be **minimised**. Bounds are given as two arrays
`lower_bound` and `upper_bound` of length $n_D$.

```python
from DEEM.DEEM import DEEM
import numpy as np

def rosenbrock(x):
    x = np.asarray(x, float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

nD = 10
lb = np.array([-5.0] * nD)
ub = np.array([ 5.0] * nD)

optimizer = DEEM(
    function=rosenbrock,
    lower_bound=lb,
    upper_bound=ub,
    nparticles_max=10 * nD,     # population size n_c (paper recommends 5–15·nD)
    nparticles_min=10 * nD,
    nswarm_max=10,              # number of sub-populations n_s ...
    nswarm_min=4,               # ... linearly reduced 10 -> 4 (paper default)
    maxiter=500,
)

optimizer.update()             # run the main loop

print("best objective :", optimizer.FBEST)
print("best position  :", optimizer.XBEST)
print("evaluations    :", optimizer.fev)
```

The optimiser exposes its result through attributes after `update()` returns:

| Attribute | Meaning |
| --------- | ------- |
| `optimizer.XBEST` | best position found, $\mathbf{x}_{GB}$ |
| `optimizer.FBEST` | best objective value found |
| `optimizer.fev`   | number of function evaluations performed |
| `optimizer.XBEST_history`, `optimizer.FBEST_history` | improvement history |

## Early termination

By default the run stops after `maxiter` iterations. To stop early once the
objective stops improving, switch the termination criterion to `tolerance`:

```python
optimizer = DEEM(
    function=rosenbrock, lower_bound=lb, upper_bound=ub,
    nparticles_max=10 * nD, nparticles_min=10 * nD,
    nswarm_max=10, nswarm_min=4,
    maxiter=1000,
    termination="tolerance",       # stop when improvement stalls
    tolerance=1e-6,                # TOL
    maxiter_below_tolerance=200,   # n_i^TOL: iterations below TOL before stopping
)
optimizer.update()
```

See [Parameters](../theory/parameters.md) for the full list and the recommended
defaults from the paper.

## Using the improved variant DEEMI

`DEEMI` has the same core interface and adds opt-in improvements (a hybrid
restart, success-history adaptation, an evaluation cache and an optional
surrogate). With its defaults it behaves as the improved algorithm; set
`method_reset="density"` and `adapt=False` to reproduce the original behaviour.

!!! note "Parameter name"

    In `DEEMI` the number of sub-populations is passed as `npop_max` / `npop_min`
    (in the base `DEEM` class it is `nswarm_max` / `nswarm_min`). The two refer to
    the same quantity $n_s$.

```python
from DEEM.DEEMI import DEEM       # note: improved variant, same class name
import numpy as np

nD = 10
opt = DEEM(
    function=rosenbrock,
    lower_bound=np.array([-5.0] * nD),
    upper_bound=np.array([ 5.0] * nD),
    nparticles_max=10 * nD, nparticles_min=10 * nD,
    npop_max=10, npop_min=4,       # n_s  (DEEMI uses npop_*)
    maxiter=500,
    method_reset="hybrid",         # 'hybrid' (default) or 'density' (original)
    adapt=True,                    # success-history adaptation of CR / phi
    seed=1,                        # reproducible run
)
result = opt.update()              # DEEMI's update() also returns a dict
print(result["f"], result["nfev"], result["n_restarts"])
```

Read more on the [DEEMI](../deemi.md) page and in the
[API reference](../reference/deemi.md).

# Constitutive-model calibration

The motivating application of DEEM is the **automatic calibration of advanced
constitutive soil models** with [numgeo-ACT](https://www.numgeo.de). Here a single
objective evaluation runs finite-element simulations of laboratory element tests
(oedometer, drained/undrained monotonic and cyclic triaxial), compares them with
experimental data using a discrete Fréchet distance, and returns a weighted error
$\epsilon$. The cost function is **rugged and discontinuous** — for some parameter
sets the model is not numerically stable — which makes it a hard optimisation
problem and a good match for DEEM's restart and multi-population strategy.

This page is a usage pattern, not a runnable script: the objective wraps an
external FE solver.

## A calibration-style objective

```python
import numpy as np
from DEEM.DEEMI import DEEM

def calibration_error(x):
    """
    Run the element-test simulations for the parameter vector x, compare with
    the experiments, and return the weighted error epsilon (to be minimised).
    Return a large penalty if the model fails to converge for these parameters.
    """
    try:
        eps = run_act_simulations(x)        # your numgeo-ACT evaluation
    except SimulationError:
        return 1e22                          # penalty (matches the `penalty` arg)
    return float(eps)

# parameter bounds for the model (e.g. Hypo-ISA: nD = 15)
LB = np.array([...])
UB = np.array([...])

optimizer = DEEM(
    function=calibration_error,
    lower_bound=LB, upper_bound=UB,
    nparticles_max=10 * len(LB), nparticles_min=10 * len(LB),
    npop_max=10, npop_min=4,
    maxiter=200,                 # calibrations use far fewer iterations than CEC
    nworkers=8,                  # evaluate the population in parallel
    method_reset="hybrid",       # correlation-aware restart (parameters interact)
    adapt=True,
    seed=0,
)
result = optimizer.update()
print("calibrated parameters:", result["x"])
print("final error epsilon  :", result["f"])
```

!!! tip "Why `method_reset="hybrid"` here"

    The model parameters are **interdependent** — similar material behaviour can be
    achieved by different parameter combinations. The hybrid restart re-seeds
    around good basins with a covariance that respects those correlations, instead
    of resetting each dimension independently.

## Parallel evaluation

With `nworkers > 1`, the population is evaluated through a process pool. On a
workstation, the paper reports roughly **40 min per iteration** for the Hypo-ISA
calibration with parallelisation, so spreading the population over the available
cores is essential.

### Windows worker initialization

On Windows, worker processes are *spawned* and do not inherit the parent process
state. The bundled `evaluation.py` starts the pool with an **initializer** that
re-establishes the runtime state the objective needs (for numgeo-ACT, the
`ACT.globals` module) in every worker:

```python
# DEEM/evaluation.py (excerpt)
with futures.ProcessPoolExecutor(
        max_workers=nworkers,
        initializer=initialize_worker,
        initargs=(worker_data,)) as executor:
    ...
```

On Linux this state is inherited through `fork`, so no special handling is
required. If your objective relies on state set at runtime in the main process,
make sure that state is available to the workers — either through the initializer
mechanism, or by running serially with `nworkers=1`.

## Reducing expensive evaluations

For very costly objectives, `DEEMI` offers two opt-in mechanisms to spend the
evaluation budget more carefully:

- **Surrogate pre-screening** (`surrogate=SurrogateManager(LB, UB)`) — evaluate
  only the most promising trial vectors on the real objective each iteration.
- **Evaluation cache** (`cache_tol=1e-6`) — never re-simulate a position that has
  already been evaluated (useful when penalty-triggering points recur).

See the [DEEMI overview](../deemi.md) for details.

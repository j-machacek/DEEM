# Benchmark functions (CEC)

DEEM was validated on the CEC 2015, 2017, 2020 and 2022 benchmark suites. This
page shows how to reproduce a benchmark run using the
[`opfunu`](https://pypi.org/project/opfunu/) library, which provides the CEC test
functions. The scripts mirror `example/DEEM_example.py` in the repository.

!!! note "Benchmark protocol from the paper"

    Population size $n_c = 10\,n_D$, termination at $n_f^{max} = 1000\,n_D$
    evaluations, $n_D = 10$, and 51 repeated runs per function to assess
    reproducibility. Performance is reported in the scaled quantity
    $\left(f_{target}/f_{opt}\right)\cdot 100\%$.

## Running a single CEC function

```python
from DEEM.DEEM import DEEM
import numpy as np
import opfunu


def run_DEEM(func, NDim, NRUNS):
    """Run DEEM `NRUNS` times on an opfunu benchmark `func`; return best values."""
    lb = np.array([-100.0] * NDim)
    ub = np.array([ 100.0] * NDim)

    f_values = []
    for _ in range(NRUNS):
        optimizer = DEEM(
            function=func.evaluate,         # opfunu objective
            lower_bound=lb,
            upper_bound=ub,
            nparticles_max=10 * NDim,
            nparticles_min=10 * NDim,
            nswarm_max=10,
            nswarm_min=4,
            maxiter=1000,
            maxfev=NDim * 10000,
            sampling_method="Random-Uniform",
            nworkers=1,
            tolerance=1e-4,
            maxiter_below_tolerance=1000,
            method_subswarm_reduction="linear",
            method_subswarm_creation="fitness-focused",
            method_boundary="damping-periodic",
        )
        optimizer.update()
        f_values.append(optimizer.FBEST)
    return f_values


if __name__ == "__main__":
    NDim, NRUNS = 10, 1
    all_funcs = opfunu.get_functions_based_classname("2022")
    func = all_funcs[4](ndim=NDim)              # pick one function
    name = func.name.split(":")[0]

    values = run_DEEM(func, NDim, NRUNS)
    with open(f"DEEM_CEC2022_{name}_Ndim={NDim}_Nruns={NRUNS}.txt", "w") as fh:
        fh.write(", ".join(str(v) for v in values) + "\n")
```

## Running a whole suite

```python
if __name__ == "__main__":
    NDim, NRUNS = 10, 1
    for func in opfunu.get_functions_based_classname("2022"):
        name = func.name.split(":")[0]
        values = run_DEEM(func(ndim=NDim), NDim, NRUNS)
        with open(f"DEEM_CEC2022_{name}_Ndim={NDim}_Nruns={NRUNS}.txt", "w") as fh:
            fh.write(", ".join(str(v) for v in values) + "\n")
```

## Benchmarking the improved variant

The same harness works for `DEEMI`; only the import and the sub-population argument
names change (`npop_max` / `npop_min` instead of `nswarm_max` / `nswarm_min`):

```python
from DEEM.DEEMI import DEEM

optimizer = DEEM(
    function=func.evaluate,
    lower_bound=lb, upper_bound=ub,
    nparticles_max=10 * NDim, nparticles_min=10 * NDim,
    npop_max=10, npop_min=4,                 # DEEMI uses npop_*
    maxiter=1000, maxfev=NDim * 10000,
    sampling_method="Random-Uniform",
    method_reset="hybrid", adapt=True, seed=0,
)
optimizer.update()
```

For a fair comparison against the original behaviour, run `DEEMI` with
`method_reset="density"` and `adapt=False`.

# `DEEM` — base algorithm

```python
from DEEM.DEEM import DEEM
```

The base optimiser. Construct it with the objective and bounds, then call
[`update()`](#update) to run the optimisation loop.

## Constructor

```python
DEEM(
    function, lower_bound, upper_bound, X0=None,
    nparticles_max=50, nparticles_min=50,
    nswarm_max=10, nswarm_min=2,
    maxiter=1000, maxfev=100000000,
    sampling_method='LHS',
    nworkers=1,
    tolerance=1e-6, termination='iterations',
    maxiter_below_tolerance=30, log_interval=1,
    method_subswarm_reduction='sigmoid-3',
    method_boundary='damping',
    method_subswarm_creation='equally-distributed',
    method_reset='density',
    niter_reset_global=None,
    penalty=1e22,
)
```

### Required arguments

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `function` | callable | Objective $f(\mathbf{x})$ to **minimise**; takes a 1-D array, returns a float. |
| `lower_bound` | array-like | Lower bounds $\mathbf{LB}$, length $n_D$. |
| `upper_bound` | array-like | Upper bounds $\mathbf{UB}$, length $n_D$. |

### Population and sub-populations

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `X0` | `None` | Optional initial guess; used as the first candidate's start position. |
| `nparticles_max` | `50` | Maximum population size $n_c$. |
| `nparticles_min` | `50` | Minimum population size (set `< max` for L-SHADE-style shrinking). |
| `nswarm_max` | `10` | Maximum number of sub-populations $n_s$. |
| `nswarm_min` | `2` | Minimum number of sub-populations. |
| `method_subswarm_reduction` | `'sigmoid-3'` | Schedule for reducing $n_s$: `constant`, `linear`, `exponential`, `sigmoid-2`, `sigmoid-3`. |
| `method_subswarm_creation` | `'equally-distributed'` | Sub-population creation: `equally-distributed`, `fitness-focused`. |

### Budget and termination

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `maxiter` | `1000` | Maximum number of iterations $n_i^{max}$. |
| `maxfev` | `1e8` | Maximum number of function evaluations $n_f^{max}$. |
| `termination` | `'iterations'` | `'iterations'` runs the full `maxiter`; `'tolerance'` stops early. |
| `tolerance` | `1e-6` | Early-termination tolerance $TOL$. |
| `maxiter_below_tolerance` | `30` | Iterations below $TOL$ before stopping ($n_i^{TOL}$). |
| `niter_reset_global` | `None` | Restart threshold $n_i^{res}$; defaults to `maxiter // 10`. |

### Sampling, boundaries, restart, misc

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `sampling_method` | `'LHS'` | Initial sampling: `LHS`, `Random-Uniform`, `Sobol`, `Halton`, `Grid`. |
| `method_boundary` | `'damping'` | Boundary handling: `clip`, `random`, `damping`, `periodic`, `damping-periodic`, `damping-periodic-random`, `damping-periodic-clip`. |
| `method_reset` | `'density'` | Restart strategy (base algorithm: `density`). |
| `nworkers` | `1` | Parallel workers for evaluation (`1` = serial). |
| `log_interval` | `1` | Iterations between log lines. |
| `penalty` | `1e22` | Value treated as the penalty/infeasible marker. |

## Methods

### `update()` { #update }

Runs the full optimisation loop until a termination criterion is met. Updates the
result attributes in place.

```python
optimizer.update()
```

## Result attributes

After `update()` returns, read the result from:

| Attribute | Description |
| --------- | ----------- |
| `XBEST` | Best position found, $\mathbf{x}_{GB}$. |
| `FBEST` | Best objective value found. |
| `fev` | Number of function evaluations performed. |
| `XBEST_history`, `FBEST_history` | History of improvements to the global best. |
| `iters` | Number of iterations executed. |

!!! tip "Objective convention"

    DEEM **minimises**. For a maximisation problem, pass the negated objective and
    negate `FBEST` afterwards.

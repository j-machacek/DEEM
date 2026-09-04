# `DEEMI` — improved variant

```python
from DEEM.DEEMI import DEEM      # same class name, improved implementation
```

`DEEMI` shares the [base interface](deem.md) and adds the arguments below. Its
defaults enable the improvements; set `method_reset="density"` and `adapt=False`
to reproduce the original behaviour. See the [DEEMI overview](../deemi.md) for the
ideas behind each option.

!!! warning "Sub-population argument name"

    In `DEEMI` the number of sub-populations is passed as **`npop_max` / `npop_min`**
    (the base `DEEM` class uses `nswarm_max` / `nswarm_min`). Both name the same
    quantity $n_s$.

## Constructor

```python
DEEM(
    function, lower_bound, upper_bound, X0=None,
    nparticles_max=50, nparticles_min=50,
    npop_max=10, npop_min=2,
    maxiter=1000, maxfev=100000000,
    sampling_method='LHS',
    nworkers=1,
    tolerance=1e-6, termination='iterations',
    maxiter_below_tolerance=30, log_interval=1,
    method_subswarm_reduction='sigmoid-3',
    method_boundary='damping',
    method_subswarm_creation='equally-distributed',
    method_reset='hybrid',
    niter_reset_global=None,
    penalty=1e22,
    adapt=True,
    cache_tol=None,
    surrogate=None,
    seed=None,
    restart_budget=None,
)
```

### Arguments shared with the base class

All arguments documented in the [`DEEM` reference](deem.md) apply, except that
`nswarm_max` / `nswarm_min` are named `npop_max` / `npop_min` here, and
`method_reset` defaults to `"hybrid"`.

### Arguments added by DEEMI

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `method_reset` | `'hybrid'` | `'hybrid'` (improved restart) or `'density'` (original). |
| `adapt` | `True` | SHADE-style improvement-weighted adaptation of $CR$ and $\phi$. |
| `cache_tol` | `None` | Relative tolerance of the evaluation cache; `None` disables it. |
| `surrogate` | `None` | Surrogate manager for pre-screening (e.g. `SurrogateManager`); `None` disables it. |
| `seed` | `None` | Seed for NumPy's global RNG (reproducibility). |
| `restart_budget` | `None` | Maximum number of global restarts (`None` = unlimited). |

## Methods

### `update()`

Runs the optimisation loop and **returns a result dictionary**:

```python
result = optimizer.update()
```

| Key | Description |
| --- | ----------- |
| `x` | Best position found, $\mathbf{x}_{GB}$. |
| `f` | Best objective value found. |
| `nit` | Number of iterations executed. |
| `nfev` | Number of *real* objective evaluations performed. |
| `n_restarts` | Number of global restarts triggered. |
| `cache_size` | Number of cached evaluations (0 if the cache is off). |
| `time` | Wall-clock run time in seconds. |
| `initial_evaluation_time` | Time spent evaluating the initial population in the constructor. |
| `evaluation_time` | Time spent evaluating candidates inside `update()`. |
| `optimizer_overhead_time` | Time inside `update()` excluding candidate evaluation. |

The result attributes `XBEST`, `FBEST` and `fev` are also available, as in the
base class.

When `nworkers > 1`, DEEMI creates one process pool and reuses it for the initial
population and every generation. `update()` closes this pool automatically,
including when an exception is raised.

### `close()`

Releases the reusable worker pool. Calling `close()` more than once is safe. This
is normally unnecessary because `update()` closes the pool automatically, but it
is useful if an optimiser is constructed and then not run.

## Surrogate managers

!!! warning "Surrogate models are experimental"
    The surrogate models are currently **experimental**. In our experience they do
    not necessarily improve the optimisation process in all cases, and further
    investigation is required. We recommend running the optimisation **without a
    surrogate model first**.

Two interchangeable surrogate controllers are bundled in `DEEM.surrogate`. Both
expose the same `select(optimizer, candidates)` / `observe(evaluated)` interface
and a kNN feasibility bias; they differ in the underlying model.

### `SurrogateManager` (RBF)

```python
from DEEM.surrogate import SurrogateManager

SurrogateManager(LB, UB, eval_frac=0.5, explore_frac=0.3,
                 min_train=40, refit_every=10)
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `LB`, `UB` | — | Search-space bounds (used to scale the surrogate inputs). |
| `eval_frac` | `0.5` | Fraction of candidates evaluated on the real objective each iteration. |
| `explore_frac` | `0.3` | Additional fraction reserved for exploration / novelty. |
| `min_train` | `40` | Minimum number of observations before the surrogate is used. |
| `refit_every` | `10` | Refit the surrogate every *n* iterations. |

A regularised radial-basis-function interpolant; ranks candidates by predicted
value plus a nearest-neighbour novelty score.

### `GPSurrogateManager` (Gaussian process)

```python
from DEEM.surrogate import GPSurrogateManager

GPSurrogateManager(LB, UB, eval_frac=0.5, explore_frac=0.3,
                   kappa=1.5, min_train=40, refit_every=10)
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `LB`, `UB` | — | Search-space bounds. |
| `eval_frac` | `0.5` | Fraction of candidates evaluated on the real objective each iteration. |
| `explore_frac` | `0.3` | Fraction of the budget reserved for the most *uncertain* candidates. |
| `kappa` | `1.5` | Exploration weight of the Lower-Confidence-Bound acquisition $\mu - \kappa\sigma$. |
| `min_train` | `40` | Minimum number of observations before pre-screening starts. |
| `refit_every` | `10` | Refit the GP every *n* new observations. |

A Gaussian process with a squared-exponential kernel; selects by the
Lower-Confidence-Bound rule, so it evaluates candidates that are either promising
(low posterior mean) or uncertain (high posterior std).

Pass either manager to `DEEM(..., surrogate=...)`. Both depend only on numpy and
scipy.

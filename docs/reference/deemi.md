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
    F0=None,
    restart_tail=0.1,
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
| `seed` | `None` | Seed of the optimiser's **own** random number generator: an `int`, a `numpy.random.RandomState` or a `numpy.random.Generator`. `None` draws from NumPy's global RNG (legacy behaviour). See [Random numbers and reproducibility](#random-numbers-and-reproducibility). |
| `restart_budget` | `None` | Maximum number of global restarts (`None` = unlimited). |
| `F0` | `None` | Known objective value of `X0` (e.g. from a pre-flight evaluation); `X0` is then not evaluated again, but still counts as one evaluation. Ignored without `X0`. |
| `restart_tail` | `0.1` | No global restart once less than this fraction of the budget (iterations and evaluations) remains. `0` allows restarts until the end (behaviour before 16.11.2026). |

## Behaviour of the core

These rules hold for `DEEMI` (and, where noted, for the published `DEEM` class).

**Budget.** `maxiter` is the number of generations *after* the initial population
(both classes). A run therefore performs at most `nparticles_max · (maxiter + 1)`
evaluations; `maxfev` stops the run after the generation in which it is reached.
`fev` / `result['nfev']` count the *real* objective calls, including the initial
population; `result['nit']` is the number of generations performed. Before
16.11.2026 the loop ran `maxiter + 1` generations.

**Failed evaluations.** A NaN or infinite objective value is replaced by
`penalty` (both classes). An evaluation counts as *failed* if its value is
`>= penalty`; failed points never enter the elite archive or the mutation pool.
If every evaluation of a generation fails, the generation still completes (the
penalty points are used as references), so a long run survives a region in which
the objective cannot be evaluated.

**Bounds.** `lower_bound`, `upper_bound` and `X0` are validated (both classes):
finite, of equal length, `lower_bound <= upper_bound`, `X0` inside the box.
`lower_bound == upper_bound` is allowed and keeps that parameter fixed.

**Scale invariance.** Diversity measures and the distances used to form
sub-populations are computed on positions rescaled by the bounds, so a parameter
with a range of `1e6` and one with a range of `1e-3` are treated alike. Rescaling a
parameter (e.g. changing its units together with its bounds) does not change the
search. For a box whose dimensions all have the same span the results are
bit-identical to the former raw-coordinate computation. The density restart
(`method_reset='density'`) chooses its exploit fraction from the *normalised*
diversity.

## Random numbers and reproducibility

`seed=<int>` creates a `numpy.random.RandomState` that every random draw of the
run uses: initial sampling (including the scrambling of the Sobol/Halton engines),
the Cauchy draws of the control parameters, boundary handling, Lévy steps and
restarts. Consequences:

- the stream of a seeded run is **identical** to that of earlier versions, which
  called `np.random.seed(seed)`; the generator change alone does not alter
  fixed-seed results (see [changes of fixed-seed results](#changes-of-fixed-seed-results));
- the run is reproducible even if the objective (or any other library) draws from
  `np.random`, and it no longer changes the global RNG state;
- results do not depend on the number of worker processes (`nworkers`).

`seed=None` keeps the legacy behaviour and draws from NumPy's global RNG, so a
preceding `np.random.seed(...)` still makes the run reproducible.

## Changes of fixed-seed results

Compared with the version before 16.11.2026, a run with the same seed and settings
gives

- the **same** result if its box has equal spans in all dimensions, no global
  restart would have fallen into the last 10 % of the budget, and the objective
  never returns NaN/inf (with `restart_tail=0` the second condition disappears);
- a **different** result if the parameter ranges differ (scale invariance), if a
  restart would have happened in the last 10 % of the budget (`restart_tail=0.1`),
  if the objective returns NaN/inf, or with `maxiter` as the binding limit, because
  the loop now performs `maxiter` instead of `maxiter + 1` generations;
- `method_reset='density'` now chooses its exploit fraction from the normalised
  diversity.

The [validation](../validation.md) page shows that these changes are neutral or
better on the benchmark.

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
| `nit` | Number of generations performed after the initial population. |
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

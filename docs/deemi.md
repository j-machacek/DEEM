# DEEMI — the improved variant

`DEEMI` (imported as `from DEEM.DEEMI import DEEM`) is an extended implementation
of the same algorithm. It keeps the published behaviour by default where it
matters, and adds opt-in improvements aimed at **expensive, rugged objectives**
such as constitutive-model calibration, where a single evaluation can take many
minutes and the cost function is discontinuous.

All additions are backward compatible: setting `method_reset="density"` and
`adapt=False` reproduces the original algorithm.

## What DEEMI adds

### Hybrid restart — `method_reset`

In addition to the original density restart (`"density"`), DEEMI offers a
**hybrid** restart (`"hybrid"`, the default):

- the **global best is kept exactly** (true elitism across the restart);
- a diversity-controlled fraction is re-seeded **around the most distinct archive
  elites** using a *shrinkage covariance* estimated from the basin members, so the
  re-seeding respects parameter correlations rather than treating each dimension
  independently;
- the remaining candidates are placed by a **scrambled low-discrepancy (Sobol)**
  sequence for unbiased exploration of the whole box.

This targets the paper's own observation that model parameters are
interdependent, so a per-dimension restart can break correlated parameter sets.

### Success-history adaptation — `adapt`

With `adapt=True` (default), the crossover rate $CR$ and the blend factor $\phi$
are adapted from a **success history weighted by the achieved improvement**
$\Delta f$ (the SHADE idea), instead of the previous absolute-fitness weighting.
The exploration/exploitation balance then responds to what is actually working.

### Optional surrogate pre-screening — `surrogate`

For expensive objectives, an optional surrogate model can rank the trial vectors
each iteration so that **only the most promising fraction (plus an exploration
quota) is evaluated on the real objective**. Two surrogate managers are bundled in
`DEEM.surrogate`, both with a feasibility classifier that biases evaluations away
from regions that previously returned the penalty value (useful when the model is
numerically unstable for some parameter sets):

- **`SurrogateManager`** — a regularised **radial-basis-function (RBF)**
  interpolant. Fast and a good general default; ranks candidates by predicted value
  plus a nearest-neighbour novelty term.
- **`GPSurrogateManager`** — a **Gaussian-process (GP)** surrogate with a
  squared-exponential kernel. Unlike the RBF, it returns a predictive *variance*,
  so it selects with a Lower-Confidence-Bound rule, $\mu(x) - \kappa\,\sigma(x)$,
  evaluating candidates that are either promising (low predicted cost) or uncertain
  (poorly covered by past evaluations). This is the classic surrogate-assisted /
  Bayesian-optimisation acquisition and tends to explore more deliberately on
  expensive, multi-modal problems.

Both share the same interface, so they are interchangeable:

```python
from DEEM.DEEMI import DEEM
from DEEM.surrogate import SurrogateManager, GPSurrogateManager
import numpy as np

LB = np.array([0.0] * 8); UB = np.array([1.0] * 8)

# RBF (default choice):
sm = SurrogateManager(LB, UB, eval_frac=0.5, explore_frac=0.3)

# or a Gaussian process with uncertainty-aware selection:
sm = GPSurrogateManager(LB, UB, eval_frac=0.5, explore_frac=0.3, kappa=1.5)

opt = DEEM(my_expensive_objective, LB, UB,
           nparticles_max=50, nparticles_min=50,
           npop_max=10, npop_min=4, maxiter=60,
           surrogate=sm)
opt.update()
```

Both are dependency-light (numpy + scipy only). On cheap analytic functions
surrogate pre-screening is not worthwhile — its purpose is the expensive-objective
regime (≈40 min per evaluation), where skipping evaluations dominates the cost.

### Evaluation cache — `cache_tol`

When `cache_tol` is set, identical (within a relative tolerance) positions are
answered from a cache instead of being re-simulated. It is **off by default**
(`None`), which reproduces the original number of evaluations. On smooth
landscapes exact revisits are rare, but on rugged / near-discrete calibration cost
functions the same penalty-triggering points recur and the cache avoids paying for
them twice.

### Reproducibility and reporting — `seed`, return value

Passing `seed` seeds NumPy's global RNG for reproducible runs. `DEEMI.update()`
additionally returns a structured result dictionary:

```python
result = opt.update()
# {'x', 'f', 'nit', 'nfev', 'n_restarts', 'cache_size', 'time'}
```

### Restart budget — `restart_budget`

`restart_budget` caps the number of global restarts (`None` = unlimited).

## Parallel evaluation on Windows

`DEEMI` evaluates candidate solutions through a process pool when `nworkers > 1`.
On Windows, worker processes are *spawned* and do not inherit the parent's runtime
state. The bundled `evaluation.py` therefore starts the pool with an
**initializer** that re-establishes the required state (e.g. the `ACT.globals`
used by numgeo-ACT) in every worker. On Linux the state is inherited through
`fork`. See the [calibration example](examples/calibration.md).

## Summary of new constructor arguments

| Argument | Default | Purpose |
| -------- | ------- | ------- |
| `method_reset` | `"hybrid"` | `"hybrid"` (improved) or `"density"` (original) |
| `adapt` | `True` | SHADE-style improvement-weighted adaptation of $CR$ / $\phi$ |
| `cache_tol` | `None` | relative tolerance of the evaluation cache (`None` = off) |
| `surrogate` | `None` | surrogate manager for pre-screening (`None` = off) |
| `seed` | `None` | seed for NumPy's global RNG |
| `restart_budget` | `None` | maximum number of global restarts |

See the [DEEMI API reference](reference/deemi.md) for the full signature.

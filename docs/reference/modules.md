# Supporting modules

Besides the two optimiser classes, the `DEEM` package contains the building blocks
they rely on. Most users will not call these directly, but they are documented
here for completeness and for advanced customisation.

## `DEEM.sampling`

Initial-sampling strategies used to place the first population. The dispatcher
`sampling(nparticles, dim, lower_bound, upper_bound, method)` selects one of:

| `method` | Function | Notes |
| -------- | -------- | ----- |
| `'LHS'` | `latin_hypercube_sampling` | Latin hypercube (default). |
| `'Random-Uniform'` | `random_uniform` | Independent uniform draws. |
| `'Sobol'` | `initial_sampling_sobol` | Low-discrepancy Sobol sequence. |
| `'Halton'` | `halton_sampling` | Low-discrepancy Halton sequence. |
| `'Grid'` | `grid_sampling` | Regular grid. |

## `DEEM.boundary_conditions`

Boundary-repair operators (see [Boundary conditions](../theory/boundary-conditions.md)).
The entry point is

```python
enforce_BC(x, lb, ub, ref, method='random')
```

which dispatches to `enforce_BC_damping`, `enforce_BC_periodic`,
`enforce_BC_random` and their combinations. Accepted `method` strings: `clip`,
`random`, `damping`, `periodic`, `damping-periodic`, `damping-periodic-random`,
`damping-periodic-clip`.

## `DEEM.population`

Defines the data structures and sub-population manager:

- **`CandidateSolution`** — a candidate $c^{\,j}$ holding `x`, `xbest`/`xbest0`,
  `f`/`fbest`/`fbest0`, the objective `function`, and DE parameters `DE_CR`, `phi`.
  Its `update_cost()` evaluates the objective and updates the personal best.
- **`Population`** — builds and manages sub-populations, including the reduction
  schedule (`method_subpop_reduction`) and creation strategy
  (`method_subpop_creation`).

## `DEEM.evaluation`

Evaluates a list of candidate solutions, serially or in parallel.

```python
evaluate_cost_function(particles, nworkers=1, mode='process', cache=None)
```

- `nworkers=1` evaluates serially in the main process.
- `nworkers>1` uses a thread or process pool (`mode='thread'`/`'process'`).
- The process pool is started with a **worker initializer** so that spawned
  workers (Windows) can re-establish the runtime state required by the objective;
  on Linux the state is inherited through `fork`.
- An optional `EvalCache` answers repeated positions without re-evaluating.

In `DEEMI` the function returns `(particles, n_real_evals)` so the driver can keep
an exact evaluation budget.

## `DEEM.toolbox`

Numerical helpers used across the algorithm, including:

- `Density` — the visit-count grid used by the restart;
- `Levy(ndim, beta)` — Mantegna Lévy-flight steps (Eq. 3–4);
- `compute_diversity`, `shannon_entropy` — diversity measures;
- `lehmer_mean`, `weighted_lehmer_mean`, `success_history_lehmer` — parameter
  adaptation;
- `space_filling_sample`, `covariance_seed` — used by the DEEMI hybrid restart;
- `contraction_expansion`, `hashable_array` — utilities.

## `DEEM.surrogate`

Optional, used only by `DEEMI` when a surrogate is supplied:

- **`RBFSurrogate`** — a regularised radial-basis-function surrogate (predict + novelty).
- **`SurrogateManager`** — RBF-based controller: selects which candidates to
  evaluate on the real objective (`select`) and records observations (`observe`),
  with a feasibility classifier that biases evaluations away from penalty regions.
- **`GPSurrogate`** — a Gaussian-process surrogate (squared-exponential kernel)
  that additionally returns a predictive standard deviation (`predict` + `predict_std`).
- **`GPSurrogateManager`** — GP-based controller with the same interface, selecting
  by a Lower-Confidence-Bound rule $\mu - \kappa\sigma$ to balance promising and
  uncertain candidates.

See the [DEEMI reference](deemi.md#surrogate-managers) for the manager arguments.


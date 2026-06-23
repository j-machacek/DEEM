# Parameters

The behaviour of DEEM is controlled by a small set of parameters, grouped into
*primary* parameters and *early-termination* parameters. The population size
$n_c$, the maximum number of iterations $n_i^{max}$ and the maximum number of
function evaluations $n_f^{max}$ are fundamental settings of any optimiser and are
listed separately below.

## Primary parameters

| Symbol | Description | Default | Constructor argument |
| ------ | ----------- | ------- | -------------------- |
| $n_s$ | Number of sub-populations | $10 \rightarrow 4$ | `nswarm_max` / `nswarm_min` (`npop_*` in DEEMI) |
| $\varepsilon^{DIV}$ | Diversity threshold | $10^{-2}$ | (internal) |
| $n_i^{res}$ | Threshold number of iterations for restart | $n_i^{max}/5$ | `niter_reset_global` |
| $\beta$ | Lévy-flight step size | $1.99$ | (internal) |

The notation $10 \rightarrow 4$ means a linear decrease of the number of
sub-populations from 10 to 4 over the run.

## Early-termination parameters

These only apply when the optimisation should stop before reaching $n_i^{max}$.

| Symbol | Description | Default | Constructor argument |
| ------ | ----------- | ------- | -------------------- |
| $TOL$ | Tolerance for early termination | $10^{-6}$ | `tolerance` |
| $n_i^{TOL}$ | Iterations below $TOL$ before stopping | $n_i^{max}/2$ | `maxiter_below_tolerance` |

To enable early termination, set `termination="tolerance"` (the default
`termination="iterations"` runs the full `maxiter`).

## Fundamental settings

| Symbol | Description | Constructor argument |
| ------ | ----------- | -------------------- |
| $n_c$ | Population size | `nparticles_max` / `nparticles_min` |
| $n_i^{max}$ | Maximum number of iterations | `maxiter` |
| $n_f^{max}$ | Maximum number of function evaluations | `maxfev` |

For the population size, the paper recommends a range of
$5\,n_D \le n_c \le 15\,n_D$. In the CEC benchmark study, $n_c = 10\,n_D$ was used,
with termination at $n_f^{max} = 1000\,n_D$ evaluations and $n_D = 10$ dimensions.

## Recommended defaults (from the paper)

These values were used consistently throughout the published study and are a good
starting point:

- **Number of sub-populations:** $n_s = 10 \rightarrow 4$
- **Diversity threshold:** $\varepsilon^{DIV} = 10^{-2}$
- **Restart threshold:** $n_i^{res} = n_i^{max}/5$
- **Lévy-flight step size:** $\beta = 1.99$
- **$\phi_2$ approach:** Cauchy variant (recommended for $n_D > 10$)
- **Population size:** $5\,n_D \le n_c \le 15\,n_D$ (study used $10\,n_D$)

!!! tip "Where these are set"

    The number of sub-populations, the population size, the restart threshold and
    the termination parameters are exposed directly as constructor arguments (see
    the [API reference](../reference/deem.md)). $\varepsilon^{DIV}$ and $\beta$ are
    fixed to their recommended defaults inside the implementation.

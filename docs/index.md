<div class="deem-hero" markdown>
![DEEM — Differential Evolution with Elitism and Multi-populations](logo/deem-logo-light.svg){ .deem-logo-light }
![DEEM — Differential Evolution with Elitism and Multi-populations](logo/deem-logo-dark.svg){ .deem-logo-dark }
</div>

**DEEM** — *Differential Evolution with Elitism and Multi-populations* — is a
heuristic optimiser of the Differential Evolution (DE) family. It combines three
ideas to improve convergence speed, accuracy and reproducibility on rugged,
multi-modal objectives:

- **Sub-populations.** The population is split into sub-populations, each led by
  an *elite* candidate solution, which balances exploration and exploitation.
- **Elitism.** A hierarchy of *global best → elite → ordinary* candidate
  solutions guides the search, with the global best refined by a Lévy-flight step.
- **Diversity-based restart.** When the search stagnates, candidate solutions are
  re-initialised into the least-visited regions of the search space, which
  reduces the chance of being trapped in local minima.

The algorithm was designed for a demanding real-world problem — the automatic
calibration of advanced constitutive soil models, where a single objective
evaluation is a finite-element simulation and the cost function is discontinuous
and highly rugged — but it applies to general bound-constrained, single-objective
optimisation.

This site documents both the **base algorithm (`DEEM`)** and its **improved
variant (`DEEMI`)**, covering the theory and equations, the public interface, and
small usage examples.

!!! quote "Please cite the paper"

    If you use DEEM or DEEMI in your work, please cite:

    Machaček, J., Siegel, S., & Zachert, H. (2025).
    *DEEM — Differential Evolution with Elitism and Multi-populations.*
    **Swarm and Evolutionary Computation, 92, 101818.**
    <https://doi.org/10.1016/j.swevo.2024.101818>

    A BibTeX entry is provided on the [Citation](citation.md) page.

## Quick links

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started/installation.md)** —
  install the package and run your first optimisation.
- :material-function-variant: **[Theory](theory/index.md)** —
  sub-populations, elitism, the position-update equations, boundary handling and
  the restart strategy.
- :material-star-four-points: **[DEEMI](deemi.md)** —
  the improved variant: hybrid restart, success-history adaptation, optional
  surrogate pre-screening (RBF or Gaussian-process) and an evaluation cache.
- :material-code-braces: **[API reference](reference/deem.md)** —
  the constructor arguments and attributes of the `DEEM` and `DEEMI` classes.

</div>

## At a glance

```python
from DEEM.DEEM import DEEM
import numpy as np

# minimise a simple quadratic in 5 dimensions
def sphere(x):
    x = np.asarray(x, float)
    return float(np.sum((x - 0.3) ** 2))

nD = 5
optimizer = DEEM(
    function=sphere,
    lower_bound=np.array([-5.0] * nD),
    upper_bound=np.array([ 5.0] * nD),
    nparticles_max=10 * nD,
    nparticles_min=10 * nD,
    nswarm_max=10,
    nswarm_min=4,
    maxiter=200,
)
optimizer.update()                 # run the optimisation loop

print(optimizer.FBEST)             # best objective value found
print(optimizer.XBEST)             # best position found
```

## Validation

DEEM was validated against benchmark functions from CEC 2015, 2017, 2020 and 2022
and compared with widely used DE variants (DE, DE-JADE, DE-LSHADE). It matches or
outperforms these references on many functions, with the largest gains on
multi-modal and composite problems. See the [paper][paper] for the full study and
[Examples](examples/benchmarks.md) for how to reproduce a benchmark run.

[paper]: https://doi.org/10.1016/j.swevo.2024.101818

## License and authorship

DEEM is developed by Jan Machaček, Institute of Geotechnics, Technical University of Darmstadt.

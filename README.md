# DEEM documentation

![DEEM — Differential Evolution with Elitism and Multi-populations](./docs/logo/deem-logo-light.png)

**DEEM** is a heuristic optimiser of the Differential Evolution (DE) family. It combines three ideas to improve convergence speed, accuracy and reproducibility on rugged,
multi-modal objectives:

- **Sub-populations.** The population is split into sub-populations, each led by an *elite* candidate solution.
- **Elitism.** A hierarchy of *global best → elite → ordinary* candidate solutions guides the search, with the global best refined by a Lévy-flight step.
- **Diversity-based restart.** When the search stagnates, candidate solutions are re-initialised into the least-visited regions of the search space.

**DEEM** is the algorithm that drives the automatic calibration tool [numge-ACT](https://www.numgeo.de/automatic-calibration/) for efficient calibration of advanced soil constitutive models.

## DEEMI, the improved variant

`from DEEM.DEEMI import DEEM` provides the improved implementation (hybrid restart,
success-history adaptation, optional evaluation cache and surrogate pre-screening)
with a robust core: NaN/inf objective values become the penalty, a seeded run uses
its own random number generator (reproducible also with a stochastic objective and
any number of worker processes), bounds are validated, and the search does not
depend on the units of the parameters.

## Validation

`example/deemi_validation.py` benchmarks DEEMI against the published DEEM and
SciPy's `differential_evolution` at equal evaluation budgets (shifted/rotated test
functions, d = 5, 10, 20, 25 seeds, development and hold-out sets, a NaN region and
badly scaled parameters). Method, results and the reproducing commands are on the
[validation page](https://j-machacek.github.io/DEEM/validation/).

```
python example/deemi_validation.py run --suite all --regime all --out validation/results.csv
python example/deemi_validation.py report validation/results.csv --reference deemi --out validation
```

## Documentation

[Access documentation here](https://j-machacek.github.io/DEEM/)

## Citation request

If you use DEEM or DEEMI in your work, please cite:

```
Machaček, J., Siegel, S., & Zachert, H. (2025).
DEEM — Differential Evolution with Elitism and Multi-populations.
Swarm and Evolutionary Computation, 92, 101818.
https://doi.org/10.1016/j.swevo.2024.101818
```
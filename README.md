# DEEM documentation

![DEEM — Differential Evolution with Elitism and Multi-populations](./docs/logo/deem-logo-light.png){ .deem-logo-light }

**DEEM** — *Differential Evolution with Elitism and Multi-populations* — is a heuristic optimiser of the Differential Evolution (DE) family. It combines three ideas to improve convergence speed, accuracy and reproducibility on rugged,
multi-modal objectives:

- **Sub-populations.** The population is split into sub-populations, each led by an *elite* candidate solution.
- **Elitism.** A hierarchy of *global best → elite → ordinary* candidate solutions guides the search, with the global best refined by a Lévy-flight step.
- **Diversity-based restart.** When the search stagnates, candidate solutions are re-initialised into the least-visited regions of the search space.

**DEEM** is the algorithm that drives the automatic calibration tool [numge-ACT](https://www.numgeo.de/automatic-calibration/) for efficient calibration of advanced soil constitutive models.

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
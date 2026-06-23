# Installation

DEEM is a small, dependency-light Python package. It requires:

- **Python** 3.9 or newer
- **NumPy**
- **SciPy** (used for the Cauchy distribution, low-discrepancy sequences and the
  surrogate in DEEMI)

The benchmark examples additionally use [`opfunu`](https://pypi.org/project/opfunu/)
to provide the CEC test functions.

## Get the code

DEEM is distributed as a source package. Clone the repository and place its root
directory on your `PYTHONPATH`:

```bash
git clone https://github.com/j-machacek/DEEM.git
cd DEEM
```

The package layout is:

```text
DEEM/                  # repository root (put this on PYTHONPATH)
└─ DEEM/               # the importable package
   ├─ __init__.py
   ├─ DEEM.py          # base algorithm  -> class DEEM
   ├─ DEEMI.py         # improved variant -> class DEEM
   ├─ population.py
   ├─ evaluation.py
   ├─ boundary_conditions.py
   ├─ sampling.py
   ├─ toolbox.py
   └─ surrogate.py     # optional surrogate model used by DEEMI
```

Because both algorithms live in the `DEEM` package, the base optimiser is imported
as `from DEEM.DEEM import DEEM` and the improved variant as
`from DEEM.DEEMI import DEEM`.

## Install the dependencies

=== "pip"

    ```bash
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install numpy scipy
    pip install opfunu                 # only needed for the benchmark examples
    ```

=== "conda"

    ```bash
    conda create -n deem python=3.12 numpy scipy
    conda activate deem
    pip install opfunu                 # only needed for the benchmark examples
    ```

## Verify the installation

From the repository root:

```python
from DEEM.DEEM import DEEM
import numpy as np

opt = DEEM(lambda x: float(np.sum(np.square(x))),
           lower_bound=np.array([-1.0, -1.0]),
           upper_bound=np.array([ 1.0,  1.0]),
           nparticles_max=20, nparticles_min=20,
           nswarm_max=4, nswarm_min=4, maxiter=50)
opt.update()
print("best:", opt.FBEST)
```

If this prints a small number close to zero, you are ready to go. Continue with
the [Quickstart](quickstart.md).

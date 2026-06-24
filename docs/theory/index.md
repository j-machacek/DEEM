# Theory overview

DEEM is a population-based metaheuristic. It optimises a problem by iteratively
improving a set of candidate solutions (CS) based on an evolutionary process. A
candidate solution $c^{\,j}$ is described by its current position
$\mathbf{x}^{\,j}=[x_1,\dots,x_{n_D}]$ in the $n_D$-dimensional search space, its
**personal best** position to date $\mathbf{x}^{\,j}_{PB}$, and the associated
objective values $f^{\,j}=f(\mathbf{x}^{\,j})$ and
$f^{\,j}_{PB}=f(\mathbf{x}^{\,j}_{PB})$:

$$
c^{\,j} = \{\, \mathbf{x}^{\,j},\ \mathbf{x}^{\,j}_{PB},\ f^{\,j},\ f^{\,j}_{PB} \,\}.
$$

The set of all $n_c$ candidate solutions forms the population
$\mathbf{P}=\{c^1, c^2, \dots, c^{n_c}\}$.

---

## The three core ideas

**Sub-populations.**
The population $\mathbf{P}$ is divided into $n_s$ sub-populations
$\mathbf{S}_i \subset \mathbf{P}$ of size $n_{cs}=n_c/n_s$. Within each
sub-population an **elite** candidate solution $c_{E,S_i}$ is the one with the
best fitness so far. See [Sub-populations](sub-populations.md).

**Elitism / dual-attractor hierarchy.**
Among the sub-population elites, the **globally best** candidate $c_{GB}$ is
identified. The remaining candidates are *ordinary* candidate solutions. The
three roles update their positions differently:

- the **global best** $c_{GB}$ explores its neighbourhood with a Lévy-flight step;
- the **elite** candidates orient towards the global best $\mathbf{x}_{GB}$
  (a *global* attractor);
- the **ordinary** candidates orient towards their sub-population elite
  $\mathbf{x}_{E,PB,S_i}$ (a *local* attractor).

This dual-attractor mechanism lets sub-populations close to $\mathbf{x}_{GB}$
focus on exploitation, while those farther away keep exploring. See
[Position updating](position-updates.md).

**Diversity-based restart.**
When the search stagnates, candidate solutions (except the global best) are
re-initialised into the least-visited regions of the search space. See
[Re-initialisation](reinitialisation.md).

---

## The optimisation loop

Each iteration $i$ performs the following steps:

```text
1.  Sort candidate solutions by personal-best fitness.
2.  Detect stagnation  ->  if triggered, re-initialise (restart).
3.  Form sub-populations and identify elite / global-best candidates.
4.  Update positions:
       - global best : Lévy-flight step          (Eq. 2–4)
       - elite CS    : oriented to global best    (Eq. 5–7)
       - ordinary CS : oriented to subpop elite   (Eq. 10–11)
5.  Enforce boundary conditions                   (Algorithm 3)
6.  Evaluate the objective for all candidate solutions.
7.  Update personal bests, the global best and the elite archive.
8.  Update the density grid and the diversity measure DIV.
9.  Check the termination criteria; otherwise go to 1.
```

The pages in this section describe each component and give the governing
equations. The two parameter tables with the recommended default values are
collected on the [Parameters](parameters.md) page.

---

## Notation

- $\mathbf{x}$ — position
- $\overline{x}_{iD}$ — position scaled to $[0,1]$ in dimension $iD$ 
- $LB_{iD}, UB_{iD}$ — lower / upper bound 
- $n_D$ — number of dimensions 
- $n_c$ — population size 
- $n_s$ — number of sub-populations 
- $CR$ — crossover rate 
- $\phi_1,\phi_2$ — blend coefficients 
- $DIV$ — normalised population diversity 

# Sub-populations

The population $\mathbf{P}$ is divided into $n_s$ sub-populations
$\mathbf{S}_{i=1,\dots,n_s}$ of size $n_{cs}=n_c/n_s$. The division is obtained by
**sorting** the candidate solutions by fitness and then growing each
sub-population around a *seed* (elite) candidate by adding its nearest neighbours
in scaled space.

## Scaled distance

Neighbours are measured by the Euclidean distance in the **scaled** search space,
so that every dimension contributes equally regardless of its bound width:

$$
d(\mathbf{x}^{1}, \mathbf{x}^{2}) =
\sqrt{\sum_{iD=1}^{n_D}\left(\overline{x}^{\,1}_{iD} - \overline{x}^{\,2}_{iD}\right)^2},
\qquad
\overline{x}_{iD} = \frac{x_{iD}-LB_{iD}}{UB_{iD}-LB_{iD}} .
$$

## Sub-population formation

The leading candidate solutions of the sorted population $\mathbf{P}^{sort}$
become the elite "seeds" $c_{E,S_i}$. Each sub-population is then filled with the
$n_{cs}-1$ closest remaining candidates (by the distance above), which are removed
from the pool so they are assigned exactly once.

!!! abstract "Algorithm 1 — Dynamic sub-population formation"

    **Input:** number of sub-populations $n_s$
    **Output:** list of sub-populations

    ```text
    for i = 1 .. n_s:
        create empty sub-population S_i
        identify the best CS in P_sort as elite c_{E,S_i}
        add c_{E,S_i} to S_i and remove it from P_sort
        compute distances d of the personal-best positions of the
            remaining CS in P_sort to x_{E,PB,S_i}            (scaled, Eq. 1)
        add the n_cs - 1 closest CS to S_i
        remove those n_cs - 1 CS from P_sort
    ```

## Why sub-populations help

Splitting the population creates several semi-independent search fronts. The
sub-population that happens to sit near the global best becomes highly
exploitative (its diversity collapses quickly), while sub-populations farther
away keep a higher diversity and continue to explore. This division of labour is
what the dual-attractor [position updates](position-updates.md) exploit.

## Number of sub-populations $n_s$

`DEEM` (and `DEEMI`) allow the number of sub-populations to change over the run,
controlled by `nswarm_max` / `nswarm_min` (`npop_max` / `npop_min` in `DEEMI`) and
the reduction schedule `method_subswarm_reduction`. The empirical study in the
paper found:

- $n_s = 1$ performs worst — sub-populations matter.
- For $n_D = 10$ the algorithm is fairly insensitive to $n_s$.
- For $n_D = 30$ a **linear decrease $n_s = 10 \rightarrow 4$** is the best overall
  choice and is the recommended default.

Available reduction schedules (`method_subswarm_reduction`): `constant`,
`linear`, `exponential`, `sigmoid-2`, `sigmoid-3`. Available creation strategies
(`method_subswarm_creation`): `equally-distributed`, `fitness-focused`.

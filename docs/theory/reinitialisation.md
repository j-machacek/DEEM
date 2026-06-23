# Re-initialisation (restart)

To prevent the algorithm from becoming trapped in a local minimum, DEEM employs a
**diversity-based restart**. When stagnation is detected, the positions of all
candidate solutions — **except the global best** $\mathbf{x}_{GB}$ — are reset to
new locations in the least-visited regions of the search space, restoring
exploration.

## Stagnation criterion

A restart is triggered when either the search has spent too many iterations below
the early-termination tolerance, **or** the diversity has collapsed:

$$
n_i^{TOL} > n_i^{res} \quad\vee\quad DIV < \varepsilon^{DIV},
$$

where $n_i^{TOL}$ is the number of consecutive iterations with an improvement
below $TOL$, $n_i^{res}$ is the restart threshold (default $n_i^{max}/5$), and
$\varepsilon^{DIV}$ is the diversity threshold (default $10^{-2}$). The normalised
diversity $DIV$ is defined on the [position-updates](position-updates.md#population-diversity-div)
page.

## Density-based re-initialisation

New positions are placed in **least-visited** regions, tracked by a density grid:

1. Each dimension $iD$ of the search space is divided into a number of **bins**
   between $LB_{iD}$ and $UB_{iD}$. A density matrix records visit counts per bin.
2. As candidate solutions traverse the search space, the bins corresponding to
   their positions are incremented.
3. On restart, the algorithm finds the bins with the **minimum** visit counts and
   samples new positions within them, kept inside the bounds.

Because the global best is preserved, the elitism guarantees are not lost across a
restart: the best solution found so far always survives.

## Effect on the search

The restart produces characteristic **diversity spikes**: at stagnation, diversity
jumps up as the population is scattered into unexplored regions, then decays again
over the next 20–50 iterations as the sub-populations re-cluster. Combined with the
[multi-population](sub-populations.md) and [elitism](position-updates.md)
mechanisms, this both escapes local minima and keeps refining the best region.

## DEEMI: an improved restart

The base restart is *per-dimension* and density-driven. The improved variant
[DEEMI](../deemi.md) offers an alternative **hybrid** restart
(`method_reset="hybrid"`) that keeps the global best exactly, re-seeds part of the
population around the most distinct archive elites using a shrinkage covariance
(so it respects parameter correlations), and fills the rest with a scrambled
low-discrepancy (Sobol) sequence. The original behaviour remains available with
`method_reset="density"`.

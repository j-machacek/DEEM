# Position updating

In each iteration the position of every candidate solution is updated. The update
rule depends on the candidate's role in the hierarchy: **global best**, **elite**,
or **ordinary**.

## Global best — Lévy-flight step

The globally best candidate refines its position with a Lévy-flight step, which
encourages local exploration around the best solution found so far:

$$
x^{\,i+1}_{iD,GB} = x^{\,i}_{iD,GB} +
\begin{cases}
\min\!\left(l_{iD}\,\Delta^{Levy}_{iD},\; UB_{iD}\right) & \text{if } l_{iD} > 0,\\[4pt]
\max\!\left(l_{iD}\,\Delta^{Levy}_{iD},\; LB_{iD}\right) & \text{if } l_{iD} \le 0,
\end{cases}
$$

where $\Delta^{Levy}_{iD} = 2\%\,\lvert LB_{iD}-UB_{iD}\rvert$ is 2 % of the bound
span and the Lévy step $l_{iD}$ follows Mantegna's algorithm:

$$
l_{iD} = \frac{u_{iD}}{\lvert v_{iD}\rvert^{1/\beta}},
\qquad
\sigma_u = \left(\frac{\Gamma(1+\beta)\sin(\pi\beta/2)}
{\Gamma\!\big((1+\beta)/2\big)\,\beta\,2^{(\beta-1)/2}}\right)^{1/\beta},
\quad \sigma_v = 1 ,
$$

with $u_{iD}\sim\mathcal{N}(0,\sigma_u^2)$, $v_{iD}\sim\mathcal{N}(0,\sigma_v^2)$
and $\Gamma$ the gamma function. The exponent $1<\beta<2$ controls the step size;
**$\beta = 1.99$** is the default, giving small, concentrated steps near the
global best.

## The stochastic mask

Both the elite and the ordinary update blend an attractor $\mathbf{A}$ with a
reference position using a crossover mask controlled by the crossover rate $CR$:

$$
\text{mask} = \big(\text{rand}(\dim) \le CR\big)\ \vee\ \big(\text{range}(\dim)=j_{\text{rand}}\big),
$$

where $j_{\text{rand}}$ is a randomly chosen dimension that is always taken from
the attractor (ensuring at least one component changes).

## Elite candidate solutions

Elite candidates guide their sub-populations towards the global best. Their
position update is

$$
\mathbf{x}^{\,i+1}_{E,PB,S_i} = \text{mask}\cdot \mathbf{A}^{\,i+1}_{E,S_i}
+ (1-\text{mask})\cdot \mathbf{x}_{GB},
$$

with the attractor formed from the candidate's own best, a first reference
position and a differential term:

$$
\mathbf{A}^{\,i+1}_{E,S_i} = \phi_1\,\mathbf{x}_{E,PB,S_i}
+ (1-\phi_1)\,\mathbf{x}_{r_0}
+ \phi_2\,(\mathbf{x}_{r_1}-\mathbf{x}_{r_2}).
$$

The reference vector $\mathbf{R}=[\mathbf{x}_{r_0},\mathbf{x}_{r_1},\mathbf{x}_{r_2}]$
is built from three distinct positions sorted by fitness (so $\mathbf{x}_{r_0}$ is
the best of the three):

- $\mathbf{x}_{r_0}$ is the global best, or a random pick from the top 5 % of
  candidates, distinct from $\mathbf{x}_{E,PB,S_i}$;
- $\mathbf{x}_{r_1},\mathbf{x}_{r_2}$ are drawn from the *unity*
  $\boldsymbol{\Omega}=\mathbf{M}\cup\mathbf{P}_B$, where $\mathbf{M}$ is the
  archive of historical elite positions and $\mathbf{P}_B$ are the personal-best
  positions of the population, all distinct from each other and from
  $\mathbf{x}_{E,PB,S_i}$.

By default the archive size equals the population size.[^eq7]

[^eq7]: In Eq. (7) of the paper the differential term is written with the two
    *trailing* reference positions; the implementation uses the three sorted
    reference positions $\mathbf{x}_{r_0},\mathbf{x}_{r_1},\mathbf{x}_{r_2}$ with
    the differential $\mathbf{x}_{r_1}-\mathbf{x}_{r_2}$, as shown above.

## Ordinary candidate solutions

Ordinary candidates orient towards their sub-population elite (a *local*
attractor) rather than the global best:

$$
\mathbf{x}^{\,i+1}_{j,S_i} = \text{mask}\cdot \mathbf{A}_{j,S_i}
+ (1-\text{mask})\cdot \mathbf{x}_{E,PB,S_i},
$$

$$
\mathbf{A}_{j,S_i} = \phi_1\,\mathbf{x}_{j,PB,S_i}
+ (1-\phi_1)\,\mathbf{x}_{E,PB,S_i}
+ \phi_2\,(\mathbf{x}_{r_1,S_i}-\mathbf{x}_{r_2,S_i}).
$$

Here $\mathbf{x}_{r_1,S_i},\mathbf{x}_{r_2,S_i}$ are two randomly selected,
distinct personal-best positions **from the same sub-population**, ordered so that
$f(\mathbf{x}_{r_1,S_i}) < f(\mathbf{x}_{r_2,S_i})$.

## The blend coefficients $\phi_1$ and $\phi_2$

$\phi_1$ is drawn from a Cauchy distribution with location $1$ and scale $0.2$,
constrained to $0<\phi_1\le 1$:

!!! abstract "Algorithm 2 — Sampling $\phi$ from a Cauchy distribution"

    ```text
    while True:
        phi <- cauchy(mean, scale)
        if   phi < 0:  continue          # resample
        elif phi > 1:  phi <- 1; break   # clip from above
        else:          break
    ```

For $\phi_2$ the paper evaluates four variants; the recommended default for
$n_D>10$ is the **Cauchy** variant with location
$\bar\phi = 0.5 + 0.5\,(1-DIV)$ and scale $0.2$:

| Variant | Definition of $\phi_2$ |
| ------- | ---------------------- |
| $\phi_2^{\phi_1}$ | $\phi_2 = \phi_1$ |
| $\phi_2^{cauchy}$ *(default)* | Cauchy, location $\bar\phi=0.5+0.5(1-DIV)$, scale $0.2$ |
| $\phi_2^{normal}$ | $\mathcal{N}(\bar\phi,\,0.2^2)$ |
| $\phi_2^{gamma}$  | Gamma, shape $2\bar\phi$, scale $1$ |

All variants are passed through Algorithm 2 to enforce $0<\phi_2\le 1$.

## Population diversity $DIV$

The normalised diversity drives both $\phi_2$ and the restart criterion:

$$
DIV = \min\!\left(\frac{DIV}{DIV_0},\,1\right),
\qquad
DIV = \frac{1}{n_c}\sum_{j=1}^{n_c}
\left(\frac{1}{n_D}\sum_{iD=1}^{n_D}
\big\lvert \operatorname{median}(\mathbf{P}_{iD}) - P_{j,iD}\big\rvert\right),
$$

where $\mathbf{P}$ is the matrix of personal-best positions and $DIV_0$ is the
diversity measured at the start of the optimisation. As the population clusters,
$DIV$ falls towards $0$; once it drops below a threshold the algorithm
[restarts](reinitialisation.md).

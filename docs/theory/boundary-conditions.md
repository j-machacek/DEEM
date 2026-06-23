# Boundary conditions

Updated positions may leave the feasible box $[\mathbf{LB},\mathbf{UB}]$. Because
many objectives are undefined outside their bounds — and optimal solutions often
lie *near* the bounds — DEEM repairs infeasible components with a **hybrid**
strategy that mixes *damped reflection* and *periodic* corrections. Mixing the two
increases the diversity of repaired solutions and avoids the bias that a single
rule would introduce.

For each dimension a uniform random number decides which rule is applied, and a
second uniform random number $r=U(0,1)$ scales the correction. The maximum
allowed adjustment is the bound span $\Delta^{max}=UB_{iD}-LB_{iD}$.

## Damped reflection

If the component is out of bounds, it is moved back inside by a random fraction of
the overshoot, capped at $\Delta^{max}$:

$$
x^{\,i+1}_{j,iD} \leftarrow
\begin{cases}
LB_{iD} + r\,\Delta, & \Delta=\min\!\big(LB_{iD}-x^{\,i+1}_{j,iD},\ \Delta^{max}\big), & x^{\,i+1}_{j,iD} < LB_{iD},\\[6pt]
UB_{iD} - r\,\Delta, & \Delta=\min\!\big(x^{\,i+1}_{j,iD}-UB_{iD},\ \Delta^{max}\big), & x^{\,i+1}_{j,iD} > UB_{iD}.
\end{cases}
$$

## Periodic correction

The periodic rule wraps a component that exceeds one boundary to near the opposite
boundary, preserving a continuous search-space topology. With
$\Delta x = \left(\frac{\Delta}{\Delta^{max}} - \big\lfloor \frac{\Delta}{\Delta^{max}}\big\rfloor\right)\Delta^{max}$:

$$
x^{\,i+1}_{j,iD} \leftarrow
\begin{cases}
UB_{iD} - r\,\Delta x, & \Delta = LB_{iD}-x^{\,i+1}_{j,iD}, & x^{\,i+1}_{j,iD} < LB_{iD},\\[6pt]
LB_{iD} + r\,\Delta x, & \Delta = x^{\,i+1}_{j,iD}-UB_{iD}, & x^{\,i+1}_{j,iD} > UB_{iD},
\end{cases}
$$

where $\lfloor\cdot\rfloor$ is the floor function.

## The hybrid rule

!!! abstract "Algorithm 3 — Hybrid boundary-condition enforcement"

    ```text
    function enforce_BC(x, LB, UB, method):
        for iD = 0 .. nD:
            u      <- U(0,1)
            r      <- U(0,1)
            Δmax   <- UB[iD] - LB[iD]
            if u >= 0.5:                      # damped reflection
                if   x[iD] < LB[iD]:  Δ = min(LB[iD]-x[iD], Δmax); x[iD] = LB[iD] + r·Δ
                elif x[iD] > UB[iD]:  Δ = min(x[iD]-UB[iD], Δmax); x[iD] = UB[iD] - r·Δ
            else:                             # periodic
                if   x[iD] < LB[iD]:  Δ = LB[iD]-x[iD]; Δx = frac(Δ/Δmax)·Δmax; x[iD] = UB[iD] - r·Δx
                elif x[iD] > UB[iD]:  Δ = x[iD]-UB[iD]; Δx = frac(Δ/Δmax)·Δmax; x[iD] = LB[iD] + r·Δx
        return x
    ```

## Choosing the method in code

The boundary strategy is selected with the `method_boundary` argument. The
implementation accepts the single rules and several combinations:

`clip`, `random`, `damping`, `periodic`,
`damping-periodic`, `damping-periodic-random`, `damping-periodic-clip`.

The hybrid rule described above corresponds to `damping-periodic` (the per-dimension
coin flip between damped reflection and periodic wrapping).

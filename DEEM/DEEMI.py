#!/usr/bin/env python3
"""
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                            Copyright (C) 2023-2025 Jan Machacek
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 05.02.2023
#
# DEEM - Differential Evolution with Elitism and Multi-populations (improved variant, DEEMI)
#
# For the theory behind it:
# Machaček, J., Siegel, S., & Zachert, H. (2025).
# DEEM — Differential Evolution with Elitism and Multi-populations.
# Swarm and Evolutionary Computation, 92, 101818. https://doi.org/10.1016/j.swevo.2024.101818
#
# History:
# 05.02.2023, J. Machacek - Initial version
# 07.02.2025, J. Machacek - Refactored code
# 23.06.2026, J. Machacek - Improved variant. Changes (all backward compatible,
#                           defaults reproduce the published behaviour unless the
#                           new options are switched on):
#   (1) The per-iteration deepcopy of the candidate list no longer duplicates the
#       objective (see CandidateSolution.__deepcopy__ in population.py). Important
#       for stateful objectives such as the numgeo-ACT FE wrapper.
#   (2) Exact function-evaluation accounting: 'fev' now counts the *real* objective
#       calls returned by evaluate_cost_function (init population + per iteration),
#       so the maxfev budget is honoured exactly. Relevant for expensive (FE) runs.
#   (3) Optional evaluation cache (cache_tol): identical/near-identical positions are
#       answered from a cache instead of being re-simulated.
#   (4) SHADE-style adaptation (adapt=True): the crossover rate CR and the blend
#       factor phi are adapted from a success history weighted by the achieved
#       improvement delta f, instead of the previous absolute-fitness weighting.
#   (5) Improved restart (method_reset='hybrid', default): the global best is always
#       retained exactly (true elitism); a fraction is re-seeded basin-aware with a
#       shrinkage covariance (respects parameter interdependencies) and the rest by
#       scrambled low-discrepancy (Sobol) space filling. The original per-dimension
#       density restart remains available via method_reset='density'.
#   (6) Optional surrogate pre-screening (surrogate=...): trial vectors are ranked by
#       a cheap surrogate and only the most promising fraction (+ an exploration
#       quota) is evaluated on the expensive objective. Default off.
#   (7) Reproducibility (seed=...) and a structured self.result report.
#   (8) Minor clean-ups: dead variables removed, object-list random choice replaced
#       by an index draw, plotting made robust to a changing number of subpopulations.
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import numpy as np
import time
from scipy.stats import cauchy
from copy import deepcopy
from typing import List, Optional

from .population import Population, CandidateSolution
from .boundary_conditions import enforce_BC
from .logger import Logger
from .toolbox import (Density, Levy, hashable_array, weighted_lehmer_mean,
                      success_history_lehmer, space_filling_sample, covariance_seed)
from .sampling import sampling
from .evaluation import evaluate_cost_function, EvalCache


class Position:
    """
    A lightweight container for storing (x, f) pairs.
    """

    def __init__(self, x: np.ndarray, f: float):
        self.x = x
        self.f = f


def bounded_cauchy_draw(location: float, scale: float,
                        lower: float = 0.0, upper: float = 1.0,
                        max_attempts: int = 1000) -> float:
    """
    Draw a sample from a Cauchy distribution ensuring it is within [lower, upper].

    Mirrors Algorithm 2 of the paper: values below 'lower' are resampled, values
    above 'upper' are clipped to 'upper'.
    """
    for _ in range(max_attempts):
        val = cauchy.rvs(loc=location, scale=scale)
        if val < lower:
            continue
        elif val > upper:
            return upper
        return val
    return float(np.clip(location, lower, upper))


class DEEM:
    """
    Differential Evolution with Elitism and Multi-populations (improved variant).

    Attributes:
        function: Objective function to minimise.
        LB, UB: Lower and upper bounds for each dimension.
        candidates: List of CandidateSolution objects.
        population: Population manager for subpopulations.
        result: Dictionary with the final report (filled by update()).
    """

    def __init__(self, function, lower_bound, upper_bound, X0=None,
                 nparticles_max: int = 50, nparticles_min: int = 50, npop_max: int = 10, npop_min: int = 2,
                 maxiter: int = 1000, maxfev: int = 100000000, sampling_method: str = 'LHS',
                 nworkers: int = 1, tolerance: float = 1e-6, termination: str = 'iterations',
                 maxiter_below_tolerance: int = 30, log_interval: int = 1,
                 method_subswarm_reduction: str = 'sigmoid-3', method_boundary: str = 'damping',
                 method_subswarm_creation: str = 'equally-distributed',
                 method_reset: str = 'hybrid', niter_reset_global: int = None, penalty: float = 1e22,
                 adapt: bool = True, cache_tol: Optional[float] = None,
                 surrogate=None, seed: Optional[int] = None,
                 restart_budget: Optional[int] = None):
        """
        Initialize the DEEM optimizer and create the initial candidate solutions.

        New (23.06.2026) keyword arguments
        ----------------------------------
        method_reset : str
            'hybrid' (default, improved) or 'density' (original behaviour).
        adapt : bool
            Enable SHADE-style improvement-weighted adaptation of CR and phi.
        cache_tol : float or None
            Relative tolerance of the evaluation cache. None disables the cache
            (reproduces the original number of evaluations).
        surrogate : object or None
            Optional surrogate manager exposing
            select(optimizer, candidates) -> list_of_indices_to_evaluate and
            observe(evaluated_candidates). None disables surrogate pre-screening.
            See DEEM.surrogate (RBFSurrogate/SurrogateManager and the
            Gaussian-process GPSurrogate/GPSurrogateManager).
        seed : int or None
            Seed for numpy's global RNG (reproducibility).
        restart_budget : int or None
            Maximum number of global restarts (None = unlimited).
        """
        if seed is not None:
            np.random.seed(seed)

        # Handle min/max particles
        self.nparticles_min = min(nparticles_min, nparticles_max)
        self.nparticles_max = nparticles_max
        self.nparticles = self.nparticles_max
        self.nparticles_reset = 0
        self.nworkers = nworkers

        # Penalty for infeasible or penalized solutions
        self.PENALTY = penalty

        # Store settings
        self.function = function
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.fev = 0  # function evaluations (real objective calls)
        self.iters = 0
        self.sampling_method = sampling_method

        # Tolerance tracking
        self.niter_below_tolerance = 0
        self.niter_above_tolerance = 0
        self.niter_reset_global = niter_reset_global if niter_reset_global is not None else self.maxiter // 10
        self.global_reset_iter = 0
        self.tolerance = tolerance
        self.maxiter_below_tolerance = maxiter_below_tolerance
        self.log_interval = log_interval
        self.termination = termination

        # Improved-variant options
        self.adapt = adapt
        self.surrogate = surrogate
        self.method_reset = method_reset
        self.restart_budget = restart_budget
        self.n_restarts = 0

        # Archive
        self.archive_elite: List[Position] = []
        self.archive: List[Position] = []
        self.archive_size = self.nparticles

        # Bounds
        self.method_boundary = method_boundary
        self.LB = np.array(lower_bound, dtype=float)
        self.UB = np.array(upper_bound, dtype=float)
        self.dist_ub_lb = self.UB - self.LB

        self.method_subswarm_reduction = method_subswarm_reduction
        self.method_subswarm_creation = method_subswarm_creation
        self.method_reset = method_reset

        # Global best
        self.XBEST_history: List[np.ndarray] = []
        self.FBEST_history: List[float] = []

        # Population (merged swarm and candidate_solution)
        self.population = Population(npop_max, npop_min, method_subswarm_creation, method_subswarm_reduction)
        self.nswarms = npop_max
        self.npop_max = npop_max

        self.ndim = len(lower_bound)
        self.global_reset_condition = False

        # Density object for re-initialization (only used by method_reset='density')
        self.Density = Density(LB=self.LB, UB=self.UB, num_bins=1000 * self.ndim)

        # Optional evaluation cache
        self.cache = EvalCache(self.LB, self.UB, rel_tol=cache_tol) if cache_tol is not None else None

        # Initialize logger
        self.log = Logger(path='./', lower_bound=self.LB, upper_bound=self.UB)

        # -- HEADER INFO --
        print("")
        print("=======================================================================")
        print("DEEM - Differential Evolution with Elitism and Multi-populations")
        print("(c) Jan Machacek, jan-machacek@outlook.com")
        print("=======================================================================")
        print("")

        # -- Initialize candidate solutions --
        print(f"... generate initial positions: {self.sampling_method}")
        samples = sampling(self.nparticles, self.ndim, self.LB, self.UB, self.sampling_method)
        self.candidates: List[CandidateSolution] = [
            CandidateSolution(x0=samples[i, :], function=self.function)
            for i in range(self.nparticles)
        ]

        # Optional initial guess
        if X0 is not None:
            self.candidates[0].x = np.asarray(X0, dtype=float)

        print("... evaluate fitness of initial positions")
        self.candidates, n_real = evaluate_cost_function(self.candidates, nworkers=self.nworkers, cache=self.cache)
        self.fev += n_real

        # Sort and track best
        self.candidates.sort(key=lambda cs: cs.fbest)
        self.XBEST = self.candidates[0].xbest.copy()
        self.FBEST = self.candidates[0].fbest
        self.XBEST_history.append(self.XBEST)
        self.FBEST_history.append(self.FBEST)
        self.FBEST0 = self.FBEST

        print(f"... best cost: {self.FBEST}\n")

        penalties_count = sum(1 for cs in self.candidates if cs.f == self.PENALTY)
        print(f"penalties = {penalties_count}")

        self.update_archive()

        # Compute initial diversity stats
        pos_matrix_gbest = np.array([cs.xbest for cs in self.candidates])
        self.DIV_GB0 = np.mean(np.mean(np.abs(np.median(pos_matrix_gbest, axis=0) - pos_matrix_gbest), axis=0), axis=0)
        pos_matrix_curr = np.array([cs.x for cs in self.candidates])
        self.DIV_CB0 = np.mean(np.mean(np.abs(np.median(pos_matrix_curr, axis=0) - pos_matrix_curr), axis=0), axis=0)

        self.DIV_NORM_GB = 1.0
        self.DIV_NORM_CB = 1.0
        self.DIV_GB = self.DIV_GB0
        self.DIV_CB = self.DIV_CB0

        # Diversity metrics for each subpopulation
        self.DIV_GB_SUBPOP = [[] for _ in range(self.nswarms)]

        # Optionally plot initial distribution
        self.log.plot_initial_distribution(self.candidates)

    # ------------------------------------------------------------------ #
    #  Restart position generators                                       #
    # ------------------------------------------------------------------ #
    def _restart_positions_density(self, n: int) -> List[np.ndarray]:
        """
        Original restart: exploit near archive elites (per-dimension Gaussian) and
        explore via the density-based least-visited sampler.
        """
        if self.DIV_GB > 0.666:
            n_exploit = int(np.ceil(0.5 * n))
        elif self.DIV_GB > 0.333:
            n_exploit = int(np.ceil(0.25 * n))
        else:
            n_exploit = max(2, int(np.ceil(0.1 * n)))
        n_explore = n - n_exploit
        out = []
        for _ in range(n_exploit):
            idx_choice = np.random.randint(0, max(1, len(self.archive_elite) // 5))
            xref = self.archive_elite[idx_choice].x
            lb_dist = np.abs(xref - self.LB)
            ub_dist = np.abs(xref - self.UB)
            dist = np.minimum(lb_dist, ub_dist)
            R = xref + dist * np.random.normal(loc=0, scale=0.1, size=self.ndim)
            out.append(np.clip(R, self.LB, self.UB))
        for _ in range(n_explore):
            out.append(self.Density.improved_least_visited_position())
        return out

    def _restart_positions_hybrid(self, n: int) -> List[np.ndarray]:
        """
        Improved restart (23.06.2026, J. Machacek).

        - The global best is always retained exactly (true elitism) -> first slot.
        - A diversity-controlled fraction is re-seeded basin-aware around the most
          distinct archive elites using a shrinkage covariance estimated from the
          basin members (respects parameter interdependencies).
        - The remaining candidates are placed by scrambled low-discrepancy (Sobol)
          space filling over the whole search box for unbiased exploration.
        """
        out: List[np.ndarray] = []
        if n <= 0:
            return out
        # 1) keep the global best exactly
        out.append(self.XBEST.copy())
        remaining = n - 1
        if remaining <= 0:
            return out

        # 2) exploit/explore split from normalised diversity
        if self.DIV_NORM_GB > 0.5:
            frac = 0.5
        elif self.DIV_NORM_GB > 0.25:
            frac = 0.33
        else:
            frac = 0.2
        n_exploit = int(np.ceil(frac * remaining))
        n_explore = remaining - n_exploit

        # basin centres = a few distinct archive elites
        elites = [p.x for p in self.archive_elite[:max(1, self.npop_max)]]
        if not elites:
            elites = [self.XBEST]
        members = np.array([p.x for p in self.archive_elite], dtype=float) if self.archive_elite else None

        for k in range(n_exploit):
            centre = elites[k % len(elites)]
            seeded = covariance_seed(centre, members, self.LB, self.UB, n=1, shrink=0.25)
            out.append(seeded[0])

        if n_explore > 0:
            fill = space_filling_sample(n_explore, self.LB, self.UB)
            out.extend([fill[i] for i in range(n_explore)])
        return out

    def _make_restart_candidate(self, x: np.ndarray, is_global_best: bool = False) -> CandidateSolution:
        """
        Build a fresh candidate at position x for a restart. If it is the retained
        global best, its (xbest, fbest) are pre-filled so that elitism is preserved
        even before re-evaluation.
        """
        cs = CandidateSolution(x0=np.asarray(x, dtype=float), function=self.function,
                               subpop_index=self.candidates[0].subpop_index)
        if is_global_best:
            cs.x = self.XBEST.copy()
            cs.xbest = self.XBEST.copy()
            cs.xbest0 = self.XBEST.copy()
            cs.fbest = cs.fbest0 = self.FBEST
            cs.initialized = True
        return cs

    def _restart_positions(self, n: int):
        """Dispatch to the configured restart strategy."""
        if self.method_reset == 'density':
            return [(p, False) for p in self._restart_positions_density(n)]
        # hybrid: first position is the retained global best
        pos = self._restart_positions_hybrid(n)
        flags = [True] + [False] * (len(pos) - 1)
        return list(zip(pos, flags))

    # ------------------------------------------------------------------ #
    #  Position update                                                   #
    # ------------------------------------------------------------------ #
    def positioning(self) -> None:
        """
        Update the position of each candidate solution.
        Uses a combination of the candidate's best position, the global best,
        subpopulation-based best, and random (Cauchy/Levy) perturbations.
        """
        length = self.ndim
        updated_candidates: List[CandidateSolution] = []

        # Sort by best fitness
        self.candidates.sort(key=lambda cs: cs.fbest)

        # Check if reinitialization is required
        self.NPRESET = 0
        self.reset_subswarms = False
        budget_ok = (self.restart_budget is None) or (self.n_restarts < self.restart_budget)
        self.global_reset_condition = budget_ok and (
            self.niter_below_tolerance > self.niter_reset_global or self.DIV_NORM_GB < 1e-2)
        if self.global_reset_condition:
            if (self.iters - self.global_reset_iter) >= self.niter_reset_global:
                self.NPRESET = self.nparticles
                self.global_reset_iter = self.iters
                self.reset_subswarms_iter = self.iters
                self.reset_subswarms = True
                self.n_restarts += 1
            else:
                self.global_reset_condition = False

        # Mark which candidate solutions get re-randomized
        for idx, cs in enumerate(self.candidates):
            cs.randomize = False
            cs.elite = False
            if idx >= (self.nparticles - self.NPRESET) and self.global_reset_condition:
                cs.randomize = True
                cs.iiter_reset = self.iters

        # Create or update subpopulations
        self.reset_subswarms = True
        subpopulations = self.population.create(
            self.candidates, self.LB, self.UB,
            (self.iters - self.global_reset_iter),
            (self.maxiter - self.global_reset_iter),
            self.reset_subswarms
        )
        self.nswarms = len(subpopulations)

        # Merge archive elites and current best solutions
        unity_positions = (self.archive_elite +
                           [Position(cs.xbest, cs.fbest) for cs in self.candidates[:self.nparticles]
                            if cs.fbest != self.PENALTY])

        # Track subpopulation-level diversity
        for subpop_div_list in self.DIV_GB_SUBPOP:
            subpop_div_list.append(0)

        if self.global_reset_condition:
            # -------- global restart --------------------------------------
            for x, is_gb in self._restart_positions(self.nparticles):
                updated_candidates.append(self._make_restart_candidate(x, is_global_best=is_gb))
            nparticles_reset_this_iter = len(updated_candidates)
            self.nparticles = len(self.candidates)

        else:
            # -------- ordinary generation ---------------------------------
            nparticles_reset_this_iter = 0
            for isubpop, subpop in enumerate(subpopulations):
                # Compute subpopulation diversity
                pos_matrix = np.array([cs.xbest for cs in subpop])
                current_div = np.mean(np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0), axis=0)
                self.DIV_GB_SUBPOP[isubpop][-1] = current_div

                base_div = max(self.DIV_GB_SUBPOP[isubpop][0], 1e-12)
                div_subpop_norm = max(0., min(1., current_div / base_div))

                # Subpopulation best (SB)
                SB = min(subpop, key=lambda cs: cs.fbest).xbest

                # --- Adaptation of CR (and phi location) from a success history ---
                crs, phis, dfs = [], [], []
                for cs in subpop:
                    if cs.improved:
                        crs.append(np.mean(cs.DE_CR))
                        phis.append(np.mean(cs.phi))
                        dfs.append(max(0.0, cs.fbest0 - cs.fbest))
                if self.adapt:
                    CR = success_history_lehmer(np.array(crs), np.array(dfs), fallback=0.5)
                    PHI_success = success_history_lehmer(np.array(phis), np.array(dfs), fallback=None) if phis else None
                else:
                    # original behaviour: absolute-fitness weighting
                    list_f = [cs.fbest for cs in subpop if cs.improved]
                    list_f0 = [cs.fbest0 for cs in subpop if cs.improved]
                    CR = (weighted_lehmer_mean(np.array(crs),
                          np.array(list_f) / (np.sum(np.abs(np.array(list_f) - np.array(list_f0))) + 1e-12))
                          if crs else 0.5)
                    PHI_success = None

                # Update each candidate solution in this subpopulation
                for cs in subpop:
                    cs.x0 = cs.x.copy()

                    # DE_CR ~ Cauchy around CR
                    cs.DE_CR = bounded_cauchy_draw(location=CR, scale=0.2)

                    # phi location: diversity-driven, optionally blended with success
                    if isubpop < 2:
                        PHI_div = 0.5 + (0.5 - div_subpop_norm)
                    else:
                        PHI_div = 0.50 + (0.5 - self.DIV_NORM_GB)
                    if PHI_success is not None:
                        PHI = float(np.clip(0.5 * PHI_div + 0.5 * PHI_success, 0.0, 1.0))
                    else:
                        PHI = float(np.clip(PHI_div, 0.0, 1.0))
                    cs.phi = bounded_cauchy_draw(location=PHI, scale=0.1)

                    # second blend factor phi2 (diversity-driven)
                    PHI2 = float(np.clip(0.50 + (0.5 - self.DIV_NORM_GB), 0.0, 1.0))
                    phi2 = bounded_cauchy_draw(location=PHI2, scale=0.2)

                    if cs.randomize:
                        nparticles_reset_this_iter += 1
                        R = self.Density.improved_least_visited_position()
                        cs = CandidateSolution(x0=R, function=cs.function, subpop_index=cs.subpop_index)
                    elif cs.elite:
                        if isubpop != 0:
                            r: List[Position] = [Position(self.XBEST, self.FBEST)]
                            max_tries = 10 * self.nparticles
                            distinct_positions: List[Position] = []
                            while len(distinct_positions) < 3 and max_tries > 0:
                                candidate = unity_positions[np.random.randint(0, len(unity_positions))]
                                if (not np.array_equal(candidate.x, cs.xbest)
                                        and all(not np.array_equal(candidate.x, d.x) for d in distinct_positions)
                                        and not np.array_equal(candidate.x, r[0].x)):
                                    distinct_positions.append(candidate)
                                max_tries -= 1
                            r.extend(distinct_positions)
                            r.sort(key=lambda rp: rp.f)
                            A = (cs.phi * cs.xbest + (1. - cs.phi) * r[0].x + (r[1].x - r[2].x) * phi2) if len(r) >= 3 else cs.xbest
                            j_rand = np.random.randint(0, self.ndim)
                            mask = (np.random.rand(length) <= cs.DE_CR) | (np.arange(length) == j_rand)
                            cs.x = np.where(mask, A, self.XBEST)
                        else:
                            lb_dist = np.abs(cs.xbest - self.LB)
                            ub_dist = np.abs(cs.xbest - self.UB)
                            dx = np.ones(self.ndim)
                            lev = Levy(self.ndim, beta=1.99)
                            for i, ilev in enumerate(lev):
                                if ilev > 0:
                                    dx[i] = min(ilev * self.dist_ub_lb[i] / 50, 0.995 * ub_dist[i])
                                else:
                                    dx[i] = max(ilev * self.dist_ub_lb[i] / 50, -0.995 * lb_dist[i])
                            cs.x = cs.xbest.copy() + dx
                    else:
                        r = []
                        max_tries = 10 * self.nparticles
                        while len(r) < 2 and max_tries > 0:
                            idx_choice = np.random.randint(1, len(subpop))
                            candidate = subpop[idx_choice]
                            if not np.array_equal(candidate.xbest, cs.xbest) and all(not np.array_equal(candidate.xbest, rr.x) for rr in r):
                                r.append(Position(candidate.xbest, candidate.fbest))
                            max_tries -= 1
                        r.sort(key=lambda rp: rp.f)
                        A = (cs.phi * cs.xbest + (1. - cs.phi) * SB + (r[0].x - r[1].x) * phi2) if len(r) >= 2 else cs.xbest
                        j_rand = np.random.randint(0, self.ndim)
                        mask = (np.random.rand(length) <= cs.DE_CR) | (np.arange(length) == j_rand)
                        cs.x = np.where(mask, A, SB)

                    cs.enforce_BC(lb=self.LB, ub=self.UB, ref=SB, method=self.method_boundary)
                    updated_candidates.append(cs)

        # Update the candidates list (CandidateSolution.__deepcopy__ keeps the
        # objective by reference, so this is cheap even for stateful objectives).
        self.candidates = deepcopy(updated_candidates)
        self.nparticles_reset = nparticles_reset_this_iter

        # Expand population if a global reset was triggered and the population is
        # configured to grow back to nparticles_max.
        if self.global_reset_condition and self.nparticles < self.nparticles_max:
            dn = self.nparticles_max - self.nparticles
            for x, is_gb in self._restart_positions(dn):
                self.candidates.append(self._make_restart_candidate(x, is_global_best=is_gb))
            self.nparticles = len(self.candidates)

    # ------------------------------------------------------------------ #
    #  Archive maintenance                                               #
    # ------------------------------------------------------------------ #
    def update_archive(self) -> None:
        """
        Update the elite and general archives with unique candidate solutions.
        """
        unique_xbest = set()
        unique_candidates = []
        if self.iters > 0:
            tmp = [Position(x=cs.xbest, f=cs.fbest)
                   for cs in self.candidates[:self.nparticles]
                   if cs.fbest != self.PENALTY]
        else:
            tmp = [Position(x=cs.xbest0, f=cs.fbest0)
                   for cs in self.candidates[:self.nparticles]
                   if cs.fbest0 != self.PENALTY]

        for pos_obj in (self.archive_elite + tmp):
            xbest_tuple = hashable_array(pos_obj.x)
            if xbest_tuple not in unique_xbest:
                unique_candidates.append(deepcopy(pos_obj))
                unique_xbest.add(xbest_tuple)
        unique_candidates.sort(key=lambda a: a.f)
        self.archive_elite = unique_candidates[:self.nparticles // 2]

        unique_x = set()
        unique_positions = []
        if self.iters > 0:
            tmp2 = [cs for cs in self.candidates if (not cs.improved and cs.f != self.PENALTY)]
        else:
            tmp2 = [cs for cs in self.candidates if cs.f != self.PENALTY]

        for cs in (self.archive + tmp2):
            x_tuple = hashable_array(cs.x)
            if x_tuple not in unique_x:
                unique_positions.append(Position(cs.x, cs.f))
                unique_x.add(x_tuple)
        if len(unique_positions) > self.archive_size:
            np.random.shuffle(unique_positions)
            unique_positions = unique_positions[:2 * self.archive_size]
        self.archive = unique_positions

    # ------------------------------------------------------------------ #
    #  Main loop                                                         #
    # ------------------------------------------------------------------ #
    def update(self) -> dict:
        """
        Main DEEM optimization loop. Returns a structured result dictionary.
        """
        print("--------------------------------------------------------------------------")
        print("{0: >5}  {1: >12}  {2: >14}  {3: >13}  {4: >5}  {5: >5}  {6: >10}  {7: >5}  {8: >5}"
              .format("Iters.", "Best f(x_t)", "f(x_t)-f(x_t0)", "Early stoppage",
                      "Time / s", "FEV", "Candidates/Pop", "COV", "DIV-GB"))
        print("--------------------------------------------------------------------------")
        self.niter_below_tolerance = 0
        start_time = time.time()

        while self.iters <= self.maxiter:
            iter_start_time = time.time()
            self.positioning()

            # Check for duplicate positions and apply Levy-based shift if needed
            visited_positions = set()
            for cs in self.candidates:
                pos_tuple = tuple(cs.x)
                if pos_tuple in visited_positions:
                    lb_dist = np.abs(cs.x - self.LB)
                    ub_dist = np.abs(cs.x - self.UB)
                    dx = np.ones(self.ndim)
                    lev = Levy(self.ndim, beta=1.99)
                    for i, ilev in enumerate(lev):
                        if ilev > 0:
                            dx[i] = min(ilev * self.dist_ub_lb[i] / 50, ub_dist[i])
                        else:
                            dx[i] = max(ilev * self.dist_ub_lb[i] / 50, -lb_dist[i])
                    cs.x = cs.x + dx
                else:
                    visited_positions.add(pos_tuple)

            # Optional surrogate pre-screening: choose which candidates to really
            # evaluate; the rest coast on their previous personal best.
            if self.surrogate is not None:
                eval_idx = self.surrogate.select(self, self.candidates)
                to_eval = [self.candidates[i] for i in eval_idx]
                _, n_real = evaluate_cost_function(to_eval, nworkers=self.nworkers, cache=self.cache)
                self.surrogate.observe(to_eval)
            else:
                self.candidates, n_real = evaluate_cost_function(
                    self.candidates, nworkers=self.nworkers, cache=self.cache)

            self.fev += n_real
            self.candidates.sort(key=lambda cs: cs.fbest)

            prev_best = self.FBEST
            if self.candidates[0].fbest < self.FBEST:
                self.XBEST_history.append(self.XBEST)
                self.FBEST_history.append(self.FBEST)
                self.XBEST = self.candidates[0].xbest.copy()
                self.FBEST = self.candidates[0].fbest

            self.update_archive()
            self.Density.update_density(positions=[cs.x for cs in self.candidates])

            pos_matrix_gbest = np.array([cs.xbest for cs in self.candidates])
            self.DIV_GB = np.mean(np.mean(np.abs(np.median(pos_matrix_gbest, axis=0) - pos_matrix_gbest), axis=0), axis=0)
            pos_matrix_curr = np.array([cs.x for cs in self.candidates])
            self.DIV_CB = np.mean(np.mean(np.abs(np.median(pos_matrix_curr, axis=0) - pos_matrix_curr), axis=0), axis=0)

            self.DIV_NORM_GB = min(self.DIV_GB / max(self.DIV_GB0, 1e-12), 1.0)
            self.DIV_NORM_CB = min(self.DIV_CB / max(self.DIV_CB0, 1e-12), 1.0)

            iter_end_time = time.time()

            # Population size shrinkage if configured (L-SHADE-style)
            if self.nparticles_min != self.nparticles_max:
                iters_local = (self.iters - getattr(self, 'reset_subswarms_iter', 0))
                total_local = max(1, (self.maxiter - getattr(self, 'reset_subswarms_iter', 0)))
                x_frac = iters_local / total_local
                target_nparticles = int(self.nparticles_max * (self.nparticles_min / self.nparticles_max) ** x_frac)
                if self.nparticles > target_nparticles:
                    remove_count = self.nparticles - target_nparticles
                    self.candidates = self.candidates[:-remove_count]
                    self.nparticles = len(self.candidates)

            df = abs(prev_best - self.FBEST)
            if df <= self.tolerance:
                self.niter_below_tolerance += 1
                self.niter_above_tolerance = 0
            else:
                self.niter_below_tolerance = 0
                self.niter_above_tolerance += 1

            self.log.save(self.iters, self.FBEST, self.XBEST,
                          self.candidates, self.nswarms,
                          self.nparticles_reset, self.DIV_CB,
                          self.DIV_GB, self.DIV_GB_SUBPOP)

            if self.iters % self.log_interval == 0:
                extra_reset_flag = "<- RESET" if self.global_reset_condition else ""
                print("{0: >5}  {1: >12.5E}  {2: >14.5E}  {3: >6} / {4: <6}  {5: >9.3E}  {6: >5}  {7: >5} / {8: <5}  {9: <5}  {10: >9.3E} {11: >5}"
                      .format(self.iters, self.FBEST, df,
                              self.niter_below_tolerance, self.maxiter_below_tolerance,
                              (iter_end_time - iter_start_time),
                              self.fev, self.nparticles, self.nswarms,
                              self.nparticles_reset, np.round(self.DIV_NORM_GB, 4), extra_reset_flag))

            if ((self.termination == 'tolerance' and self.niter_below_tolerance >= self.maxiter_below_tolerance)
                    or (self.fev >= self.maxfev)):
                break

            self.FBEST0 = self.FBEST
            self.iters += 1

        end_time = time.time()
        try:
            self.log.plot_results()
        except Exception as exc:   # plotting must never abort an optimisation run
            print(f"[logger] plotting skipped: {exc}")

        # structured report
        self.result = {
            'x': self.XBEST,
            'f': self.FBEST,
            'nit': self.iters,
            'nfev': self.fev,
            'n_restarts': self.n_restarts,
            'cache_size': (len(self.cache) if self.cache is not None else 0),
            'time': end_time - start_time,
        }

        print("---------------------------------------------------------------")
        print(f"Found best position: {self.XBEST}")
        print(f"Best objective     : {self.FBEST:.6E}")
        print(f"Real evaluations   : {self.fev}")
        print(f"Restarts           : {self.n_restarts}")
        print(f"Execution time     : {end_time - start_time:.3f} s")
        print("=======================================================================")
        print("")
        return self.result

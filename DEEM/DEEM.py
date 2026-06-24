#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Jan Machaček
#
# This file is part of DEEM, released under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.

"""
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                            Copyright (C) 2023-2025 Jan Machacek  
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 05.02.2023
#
# DEEM - Differential Evolution with Elitism and Multi-populations
#
# For the theory behind it:
# Machaček, J., Siegel, S., & Zachert, H. (2025). 
# DEEM — Differential Evolution with Elitism and Multi-populations. 
# Swarm and Evolutionary Computation, 92, 101818. https://doi.org/10.1016/j.swevo.2024.101818
# 
# History:
# 05.02.2023, J. Machacek - Initial version
# 07.02.2025, J. Machacek - Refactored code
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import numpy as np
import time
from scipy.stats import cauchy
from copy import deepcopy
from typing import List

from .population import Population, CandidateSolution
from .boundary_conditions import enforce_BC
from .logger import Logger
from .toolbox import Density, Levy, hashable_array, weighted_lehmer_mean
from .sampling import sampling
from .evaluation import evaluate_cost_function

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

    Parameters
    ----------
    location : float
        Center of the Cauchy distribution.
    scale : float
        Scale parameter.
    lower : float, optional
        Lower bound.
    upper : float, optional
        Upper bound.
    max_attempts : int, optional
        Maximum attempts to sample a valid value.

    Returns
    -------
    float
        A sample within [lower, upper].
    """
    for _ in range(max_attempts):
        val = cauchy.rvs(loc=location, scale=scale)
        if val < lower:
            continue
        elif val > upper:
            return upper
        return val
    return np.clip(location, lower, upper)


class DEEM:
    """
    Differential Evolution with Elitism and Multi-populations (DEEM).

    Attributes:
        function: Objective function to minimize.
        LB, UB: Lower and upper bounds for each dimension.
        candidates: List of CandidateSolution objects.
        population: Population manager for subpopulations.
        ... (additional parameters)
    """

    def __init__(self, function, lower_bound, upper_bound, X0=None,
                 nparticles_max: int = 50, nparticles_min: int = 50, nswarm_max: int = 10, nswarm_min: int = 2,
                 maxiter: int = 1000, maxfev: int = 100000000, sampling_method: str = 'LHS',
                 nworkers: int = 1, tolerance: float = 1e-6, termination: str = 'iterations',
                 maxiter_below_tolerance: int = 30, log_interval: int = 1,
                 method_subswarm_reduction: str = 'sigmoid-3', method_boundary: str = 'damping',
                 method_subswarm_creation: str = 'equally-distributed',
                 method_reset: str = 'density', niter_reset_global: int = None, penalty: float = 1e22):
        """
        Initialize the DEEM optimizer and create the initial candidate solutions.
        """
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
        self.fev = 0  # function evaluations
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

        # Archive
        self.archive_elite: List[Position] = []
        self.archive: List[Position] = []
        self.archive_size = self.nparticles

        # Bounds
        self.method_boundary = method_boundary
        self.LB = np.array(lower_bound)
        self.UB = np.array(upper_bound)
        self.dist_ub_lb = self.UB - self.LB

        self.method_subswarm_reduction = method_subswarm_reduction
        self.method_subswarm_creation = method_subswarm_creation
        self.method_subswarm_forced_update = False
        self.method_reset = method_reset

        # Global best
        self.XBEST_history: List[np.ndarray] = []
        self.FBEST_history: List[float] = []

        # Population (merged swarm and candidate_solution)
        self.population = Population(nswarm_max, nswarm_min, method_subswarm_creation, method_subswarm_reduction)
        self.nswarms = nswarm_max
        self.nswarm_max = nswarm_max

        self.ndim = len(lower_bound)
        self.global_reset_condition = False

        # Density object for re-initialization
        self.Density = Density(LB=self.LB, UB=self.UB, num_bins=100 * self.ndim)

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
        # Create candidate solutions (cs)
        self.candidates: List[CandidateSolution] = [
            CandidateSolution(x0=samples[i, :], function=self.function)
            for i in range(self.nparticles)
        ]

        # Optional initial guess
        if X0 is not None:
            self.candidates[0].x = np.asarray(X0)

        print("... evaluate fitness of initial positions")
        self.candidates = evaluate_cost_function(self.candidates, nworkers=self.nworkers)

        # Sort and track best
        self.candidates.sort(key=lambda cs: cs.fbest)
        self.XBEST = self.candidates[0].xbest
        self.FBEST = self.candidates[0].fbest
        self.XBEST_history.append(self.XBEST)
        self.FBEST_history.append(self.FBEST)
        self.FBEST0 = self.FBEST

        print(f"... best cost: {self.FBEST}\n")

        # Show how many are penalized
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

        # Diversity metrics for each subpopulation
        self.DIV_GB_SUBPOP = [[] for _ in range(self.nswarms)]

        # Optionally plot initial distribution
        self.log.plot_initial_distribution(self.candidates)

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
        self.global_reset_condition = (self.niter_below_tolerance > self.niter_reset_global
                                       or self.DIV_NORM_GB < 1e-2)
        if self.global_reset_condition:
            if (self.iters - self.global_reset_iter) >= self.niter_reset_global:
                self.NPRESET = self.nparticles - 1
                self.global_reset_iter = self.iters
                self.reset_subswarms_iter = self.iters
                self.reset_subswarms = True
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

            nparticles_reset_this_iter = 0
            if self.DIV_GB > 0.666:
                n_exploit = int(np.ceil(0.5 * self.nparticles))
            elif self.DIV_GB > 0.333:
                n_exploit = int(np.ceil(0.25 * self.nparticles))
            else: 
                n_exploit = max(2, int(np.ceil(0.1 * self.nparticles)))
            n_explore = self.nparticles - n_exploit
            # Exploitation: Generate candidates near the global best (self.XBEST)
            # We add a normally distributed perturbation with a small standard deviation relative
            # to the overall search space (for example, 10% of the norm of (UB - LB)).
            for _ in range(n_exploit):
                #perturbation = np.random.normal(loc=0, scale=0.1 * np.linalg.norm(self.UB - self.LB), size=length)
                idx_choice = np.random.randint(0, max(1, len(self.archive_elite) // 5))
                xref = self.archive_elite[idx_choice].x
                lb_dist = [abs(xx - lbv) for xx, lbv in zip(xref, self.LB)]
                ub_dist = [abs(xx - ubv) for xx, ubv in zip(xref, self.UB)]
                dist = [min(ilb,iub) for ilb, iub in zip(lb_dist,ub_dist)]
                R = xref + dist*np.random.normal(loc=0, scale=0.1, size=self.ndim)
                R = np.clip(R, self.LB, self.UB)
                cs = CandidateSolution(x0=R, function=self.function, subpop_index=self.candidates[0].subpop_index)
                updated_candidates.append(cs)
                nparticles_reset_this_iter += 1
            # Exploration: Generate candidates using the density-based mechanism
            for _ in range(n_explore):
                R = self.Density.improved_least_visited_position()
                cs = CandidateSolution(x0=R, function=self.function, subpop_index=self.candidates[0].subpop_index)
                updated_candidates.append(cs)
                nparticles_reset_this_iter += 1
            self.nparticles = len(self.candidates)

        else:

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
    
                # Adaptive CR in the subpopulation, based on improved candidate solutions
                list_cr, list_f, list_f0 = [], [], []
                for cs in subpop:
                    if cs.improved:
                        list_cr.append(np.mean(cs.DE_CR))
                        list_f.append(cs.fbest)
                        list_f0.append(cs.fbest0)
                CR = weighted_lehmer_mean(np.array(list_cr), np.array(list_f) / (np.sum(np.abs(np.array(list_f) - np.array(list_f0))) + 1e-12)) if list_cr else 0.5
    
                # Update each candidate solution in this subpopulation
                for cs in subpop:
                    cs.x0 = cs.x.copy()
    
                    # DE_CR ~ Cauchy around CR
                    cs.DE_CR = bounded_cauchy_draw(location=CR, scale=0.2)
    
                    # cs.phi ~ Cauchy around 1.0
                    cs.phi = bounded_cauchy_draw(location=1.0, scale=0.2)
    
                    # Additional random factor phi2 ~ Cauchy around PHI
                    PHI = 0.5 + 0.5 * (1. - self.DIV_NORM_GB)
                    phi2 = bounded_cauchy_draw(location=PHI, scale=0.2)
    
                    if cs.randomize:
                        nparticles_reset_this_iter += 1
                        R = self.Density.least_visited_position()
                        cs = CandidateSolution(x0=R, function=cs.function, subpop_index=cs.subpop_index)
                    elif cs.elite:
                        if isubpop != 0:
                            r: List[Position] = []
                            r.append(Position(self.XBEST, self.FBEST))
                            max_tries = 10 * self.nparticles
                            distinct_positions: List[Position] = []
                            while len(distinct_positions) < 3 and max_tries > 0:
                                candidate = np.random.choice(unity_positions)
                                if (not np.array_equal(candidate.x, cs.xbest)
                                    and all(not np.array_equal(candidate.x, d.x) for d in distinct_positions)
                                    and not np.array_equal(candidate.x, r[0].x)):
                                    distinct_positions.append(candidate)
                                max_tries -= 1
                            r.extend(distinct_positions)
                            r.sort(key=lambda rp: rp.f)
                            A = (cs.phi * cs.xbest +
                                 (1. - cs.phi) * r[0].x +
                                 (r[1].x - r[2].x) * phi2) if len(r) >= 3 else cs.xbest
                            j_rand = np.random.randint(0, self.ndim)
                            mask = (np.random.rand(length) <= cs.DE_CR) | (np.arange(length) == j_rand)
                            cs.x = np.where(mask, A, self.XBEST)
                        else:
                            lb_dist = [abs(xx - lbv) for xx, lbv in zip(cs.xbest, self.LB)]
                            ub_dist = [abs(xx - ubv) for xx, ubv in zip(cs.xbest, self.UB)]
                            dx = np.ones(self.ndim)
                            lev = Levy(self.ndim, beta=1.99)
                            for i, ilev in enumerate(lev):
                                if ilev > 0:
                                    dx[i] = min(ilev * self.dist_ub_lb[i] / 50, ub_dist[i])
                                else:
                                    dx[i] = max(ilev * self.dist_ub_lb[i] / 50, -lb_dist[i])
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

        # Update the candidates list
        self.candidates = deepcopy(updated_candidates)
        self.nparticles_reset = nparticles_reset_this_iter

        # Expand population if a global reset was triggered
        if self.global_reset_condition and self.nparticles < self.nparticles_max:
            dn = self.nparticles_max - self.nparticles
            for _ in range(dn):
                if np.random.uniform() >= 0.5:
                    R = self.Density.least_visited_position()
                else:
                    idx_choice = np.random.randint(0, max(1, self.nparticles // 10))
                    xref = self.candidates[idx_choice].xbest
                    lb_dist = [abs(xx - lbv) for xx, lbv in zip(xref, self.LB)]
                    ub_dist = [abs(xx - ubv) for xx, ubv in zip(xref, self.UB)]
                    rr1 = np.random.uniform(0.0, 0.25, size=length)
                    rr2 = np.random.uniform(0.0, 0.25, size=length)
                    R = np.random.uniform(xref - rr1 * lb_dist, xref + rr2 * ub_dist)
                new_cs = CandidateSolution(x0=R, function=self.function, subpop_index=self.candidates[0].subpop_index)
                self.candidates.append(new_cs)
            self.nparticles = len(self.candidates)

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
        self.archive_elite = unique_candidates[:self.nparticles]

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
            unique_positions = unique_positions[:self.archive_size]
        self.archive = unique_positions

    def update(self) -> None:
        """
        Main DEEM optimization loop.
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
            self.fev += self.nparticles

            # Check for duplicate positions and apply Levy-based shift if needed
            visited_positions = set()
            for cs in self.candidates:
                pos_tuple = tuple(cs.x)
                if pos_tuple in visited_positions:
                    lb_dist = [abs(xx - lbv) for xx, lbv in zip(cs.x, self.LB)]
                    ub_dist = [abs(xx - ubv) for xx, ubv in zip(cs.x, self.UB)]
                    dx = np.ones(self.ndim)
                    lev = Levy(self.ndim, beta=1.99)
                    for i, ilev in enumerate(lev):
                        if ilev > 0:
                            dx[i] = min(ilev * self.dist_ub_lb[i] / 50, ub_dist[i])
                        else:
                            dx[i] = max(ilev * self.dist_ub_lb[i] / 50, -lb_dist[i])
                    cs.x += dx
                else:
                    visited_positions.add(pos_tuple)

            # Evaluate candidate solutions
            self.candidates = evaluate_cost_function(self.candidates, nworkers=self.nworkers)
            self.candidates.sort(key=lambda cs: cs.fbest)

            prev_best = self.FBEST
            if self.candidates[0].fbest < self.FBEST:
                self.XBEST_history.append(self.XBEST)
                self.FBEST_history.append(self.FBEST)
                self.XBEST = self.candidates[0].xbest
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

            # Population size shrinkage if needed
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
        self.log.plot_results()
        print("---------------------------------------------------------------")
        print(f"Found best position: {self.XBEST}")
        print(f"Execution time: {end_time - start_time:.3f} s")
        print("=======================================================================")
        print("")
        return
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
# DEEM - Differential Evolution with Elitism and Multi-populations
#
# For the theory behind it:
# Machaček, J., Siegel, S., & Zachert, H. (2025). 
# DEEM — Differential Evolution with Elitism and Multi-populations. 
# Swarm and Evolutionary Computation, 92, 101818. https://doi.org/10.1016/j.swevo.2024.101818
#
# This module merges the functionality of the former swarm and particle
# modules. It defines:
#   - CandidateSolution: Represents a single candidate solution.
#   - Population: Manages groups of candidate solutions (subpopulations)
#     for the DEEM algorithm.
# 
# Candidate solutions are abbreviated as 'cs' in the code.
# 
# History:
# 07.02.2025, J. Machacek - Initial version (merge of swarm.py and particles.py)
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""
#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from typing import List, Optional, Sequence
from .boundary_conditions import enforce_BC
from .toolbox import lehmer_mean


########################################################################
# CandidateSolution class (formerly Particle)
########################################################################

class CandidateSolution:
    """
    Represents a candidate solution in the search space.
    """
    def __init__(self, x0: np.ndarray, function, f0: Optional[float] = None, subpop_index: int = 0):
        self.function = function
        self.x = self.xbest = self.xbest0 = np.array(x0)
        if f0 is not None:
            self.f = f0
            self.fbest = f0
            self.fbest0 = f0
        else:
            self.f = self.fbest = self.fbest0 = 1e22
        self.improved = False
        self.randomize = False
        self.elite = False
        self.initialized = False
        self.niter_xbest = 0
        self.iiter_reset = 0
        self.subpop_index = subpop_index

        self.DE_CR = np.random.uniform()
        self.phi = np.random.uniform()

    def reset(self, x0: np.ndarray) -> None:
        """
        Reset the candidate solution with a new position.
        """
        self.x = np.array(x0)
        self.xbest = self.x.copy()
        self.xbest0 = self.x.copy()
        self.f = self.fbest = self.fbest0 = 1e22
        self.improved = False
        self.randomize = False
        self.elite = False
        self.initialized = False
        self.niter_xbest = 0

    def update_cost(self) -> None:
        """
        Evaluate the objective function at the current position.
        Update best-known position if improved.
        """
        self.improved = False
        self.f = self.function(self.x)
        if not self.initialized:
            self.fbest = self.fbest0 = self.f
            self.xbest = self.xbest0 = self.x.copy()
            self.improved = True
            self.initialized = True
        elif self.f < self.fbest:
            self.xbest0 = self.xbest.copy()
            self.fbest0 = self.fbest
            self.xbest = self.x.copy()
            self.fbest = self.f
            self.niter_xbest = 0
            self.improved = True
        self.niter_xbest += 1

    def enforce_BC(self, lb: np.ndarray, ub: np.ndarray, ref: np.ndarray, method: str = 'random') -> None:
        """
        Enforce boundary conditions on the candidate solution.
        """
        self.x = np.array(enforce_BC(self.x, lb, ub, ref, method))


########################################################################
# Population class (formerly Swarm)
########################################################################

class Population:
    """
    Manages the creation and assignment of candidate solutions to subpopulations.
    """
    def __init__(self, nswarms_max: int, nswarms_min: int, method_subpop_creation: str, method_subpop_reduction: str):
        self.niter_xbest = 0
        self.reset_subpops = False
        self.reset_subpops_iter = 0
        self.nswarms = 0
        self.nswarms_max = nswarms_max
        self.nswarms_min = nswarms_min
        self.method_subpop_creation = method_subpop_creation
        self.method_subpop_reduction = method_subpop_reduction

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Return Euclidean distance between two points.
        """
        return np.linalg.norm(a - b)

    @staticmethod
    def scale_coordinates(x: Sequence[float], LB: np.ndarray, UB: np.ndarray) -> List[float]:
        """
        Scale coordinates of x to [0, 1] based on LB and UB.
        """
        return [0 if (UB[i] == LB[i]) else (x[i] - LB[i]) / (UB[i] - LB[i]) for i in range(len(x))]

    @classmethod
    def euclidean_distance_scaled(cls, x1: np.ndarray, x2: np.ndarray, LB: np.ndarray, UB: np.ndarray) -> float:
        """
        Euclidean distance between x1 and x2 in a space scaled to [0,1].
        """
        x1_scaled = cls.scale_coordinates(x1, LB, UB)
        x2_scaled = cls.scale_coordinates(x2, LB, UB)
        return np.linalg.norm(np.array(x1_scaled) - np.array(x2_scaled))

    def create(self, candidates: List[CandidateSolution], LB: np.ndarray, UB: np.ndarray,
               iters: int, maxiter: int, reset: bool) -> List[List[CandidateSolution]]:
        """
        Create or update subpopulations from the candidate solutions.
        """
        iters_adj = iters - self.reset_subpops_iter
        maxiter_adj = max(1, maxiter - self.reset_subpops_iter)
        x_fraction = iters_adj / maxiter_adj
        self._update_number_of_subpops(x_fraction)
        sorted_candidates = sorted(candidates, key=lambda cs: cs.fbest)
        if reset:
            subpops = self._create_new_subpops(sorted_candidates, LB, UB)
        else:
            subpops = self._assign_existing_subpops(candidates)
        self._mark_elite_and_assign_indices(subpops)
        return subpops

    def _update_number_of_subpops(self, x_fraction: float) -> None:
        """
        Update the number of subpopulations based on x_fraction.
        """
        nmin, nmax = self.nswarms_min, self.nswarms_max
        method = self.method_subpop_reduction
        if nmin == nmax:
            self.nswarms = nmin
            return
        if method == 'sigmoid-2':
            val = 1. - (1. / (1. + (x_fraction / (1. - x_fraction)) ** -2))
            self.nswarms = int(round(nmin + (nmax - nmin) * val))
        elif method == 'sigmoid-3':
            val = 1. - (1. / (1. + (x_fraction / (1. - x_fraction)) ** -3))
            self.nswarms = int(round(nmin + (nmax - nmin) * val))
        elif method == 'linear':
            self.nswarms = int(round(nmin + (nmax - nmin) * (1. - x_fraction)))
        elif method == 'exponential':
            ratio = nmin / nmax if nmax != 0 else 1.0
            self.nswarms = int(round(nmax * (ratio ** x_fraction)))
        elif method == 'constant':
            self.nswarms = nmax

    def _create_new_subpops(self, sorted_candidates: List[CandidateSolution], LB: np.ndarray, UB: np.ndarray) -> List[List[CandidateSolution]]:
        """
        Create new subpopulations using the specified method.
        """
        if self.method_subpop_creation == 'equally-distributed':
            return self._create_equally_distributed_subpops(sorted_candidates)
        elif self.method_subpop_creation == 'fitness-focused':
            return self._create_fitness_focused_subpops(sorted_candidates, LB, UB)
        else:
            raise ValueError(f"Unknown method_subpop_creation: {self.method_subpop_creation}")

    def _assign_existing_subpops(self, candidates: List[CandidateSolution]) -> List[List[CandidateSolution]]:
        """
        Assign candidates to their previous subpopulations.
        """
        subpops = [[] for _ in range(self.nswarms)]
        for cs in candidates:
            subpops[cs.subpop_index].append(cs)
        alive_subpops = [sp for sp in subpops if sp]
        self.nswarms = len(alive_subpops)
        return alive_subpops

    def _mark_elite_and_assign_indices(self, subpops: List[List[CandidateSolution]]) -> None:
        """
        Mark the best candidate in each subpopulation as elite and assign the subpopulation index.
        """
        for i, subpop in enumerate(subpops):
            subpop.sort(key=lambda cs: cs.fbest)
            if subpop:
                subpop[0].elite = True
            for cs in subpop:
                cs.subpop_index = i

    def _create_equally_distributed_subpops(self, sorted_candidates: List[CandidateSolution]) -> List[List[CandidateSolution]]:
        """
        Create subpopulations by distributing the top candidates as leaders,
        then assign the remaining based on closeness (using vectorized distance computation).
        """
        leaders = [sorted_candidates.pop(0) for _ in range(self.nswarms)]
        subpops = [[leader] for leader in leaders]
        while sorted_candidates:
            for sp in subpops:
                if not sorted_candidates:
                    break
                leader = sp[0]
                # Vectorized distance computation from leader.xbest to all remaining candidates
                candidates_array = np.array([cs.xbest for cs in sorted_candidates])
                leader_arr = leader.xbest.reshape(1, -1)
                distances = np.linalg.norm(candidates_array - leader_arr, axis=1)
                idx = np.argmin(distances)
                sp.append(sorted_candidates.pop(idx))
        return subpops

    def _create_fitness_focused_subpops(self, sorted_candidates: List[CandidateSolution],
                                          LB: np.ndarray, UB: np.ndarray) -> List[List[CandidateSolution]]:
        """
        Create subpopulations by assigning each a best candidate then filling with the closest ones.
        Uses vectorized distance computation.
        """
        total = len(sorted_candidates)
        if self.nswarms <= 0:
            return []
        n_per_subpop = total // self.nswarms
        remainder = total % self.nswarms
        subpops = []
        for _ in range(self.nswarms):
            if not sorted_candidates:
                break
            subpop = []
            best_candidate = sorted_candidates.pop(0)
            subpop.append(best_candidate)
            extra = 1 if remainder > 0 else 0
            if remainder > 0:
                remainder -= 1
            # Vectorized distances from best_candidate.xbest to all remaining candidates
            if sorted_candidates:
                candidates_array = np.array([cs.xbest for cs in sorted_candidates])
                best_arr = best_candidate.xbest.reshape(1, -1)
                distances = np.linalg.norm(candidates_array - best_arr, axis=1)
                n_to_assign = n_per_subpop + extra - 1
                # Get indices of the n smallest distances
                if len(distances) > 0:
                    idxs = np.argsort(distances)[:n_to_assign]
                    # Append candidates in order and remove them
                    selected = [sorted_candidates[i] for i in sorted(idxs, reverse=True)]
                    for cs in selected:
                        subpop.insert(1, cs)
                        sorted_candidates.remove(cs)
            subpops.append(deepcopy(subpop))
        return subpops

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
# This module provides auxiliary functions and classes for DEEM, including 
# - contraction/expansion sequences
# - density-based sampling
# - diversity metrics
# - Levy flight step generation
# - and various means...
# 
# History:
# 05.02.2023, J. Machacek - Initial version
# 07.02.2025, J. Machacek - Refactored code, added docstrings
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""
#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from typing import Sequence, Tuple


def contraction_expansion(method: str, min_val: float, max_val: float, maxiter: int) -> np.ndarray:
    """
    Generate a sequence transitioning from 'max_val' to 'min_val' (or vice versa)
    according to the specified method.
    """
    iters = np.arange(maxiter + 1)
    x = iters / np.max(iters)  # normalized [0,1]

    if min_val == max_val:
        return np.full(maxiter + 1, min_val)

    if method == 'linear':
        f_linear = np.linspace(1.0, 0.0, maxiter + 1)
        return min_val + (max_val - min_val) * f_linear
    elif method == 'sigmoid-2':
        f_sigmoid2 = 1.0 - (1.0 / (1.0 + (x / (1.0 - x)) ** -2))
        return min_val + (max_val - min_val) * f_sigmoid2
    elif method == 'sigmoid-3':
        f_sigmoid3 = 1.0 - (1.0 / (1.0 + (x / (1.0 - x)) ** -3))
        return min_val + (max_val - min_val) * f_sigmoid3
    elif method == 'exponential':
        return max_val * ((min_val / max_val) ** x)
    elif method == 'gaussian':
        mu, sigma = 0.0, 1.0
        u = np.random.uniform(min_val, max_val, maxiter)
        cdf_value = norm.cdf(u, loc=mu, scale=sigma)
        return u * cdf_value
    else:
        raise ValueError(f"contraction_expansion: Invalid method '{method}'.")


class Density:
    """
    Density-based sampling class for tracking visit counts in the search space.
    """

    def __init__(self, LB: Sequence[float], UB: Sequence[float], num_bins: int = 10):
        self.LB = np.array(LB, dtype=float)
        self.UB = np.array(UB, dtype=float)
        self.num_bins = num_bins
        self.matrix = np.zeros((len(LB), num_bins), dtype=float)

    def update_density(self, positions: Sequence[np.ndarray]) -> None:
        """
        Update the density matrix based on candidate positions.
        This version uses vectorized bincounts for speed.
        """
        positions = np.array(positions, dtype=float)
        bins = np.floor((positions - self.LB) / (self.UB - self.LB + 1e-32) * self.num_bins).astype(int)
        bins = np.clip(bins, 0, self.num_bins - 1)
        # For each dimension d, count how many positions fall into each bin
        for d in range(positions.shape[1]):
            counts = np.bincount(bins[:, d], minlength=self.num_bins)
            self.matrix[d, :] += counts

    def least_visited_position(self) -> np.ndarray:
        """
        Return a point in the least-visited bin for each dimension.
        """
        bin_widths = (self.UB - self.LB) / self.num_bins
        min_bins_indices = np.argmin(self.matrix, axis=1)
        random_offsets = np.random.rand(len(self.LB)) * bin_widths
        new_position = self.LB + min_bins_indices * bin_widths + random_offsets
        self.update_density([new_position])
        return new_position


def shannon_entropy(particle_positions: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Shannon's entropy for the distribution of candidate positions.
    """
    if particle_positions.size == 0:
        return 0.0
    min_vals = np.min(particle_positions, axis=0)
    max_vals = np.max(particle_positions, axis=0)
    hist, _ = np.histogramdd(particle_positions, bins=num_bins, range=list(zip(min_vals, max_vals)))
    total_count = np.sum(hist)
    if total_count == 0:
        return 0.0
    prob_dist = hist / total_count
    prob_dist_nonzero = prob_dist[prob_dist > 0.0]
    entropy = -np.sum(prob_dist_nonzero * np.log(prob_dist_nonzero))
    return entropy


def compute_diversity(candidates: Sequence, method: str = "current-centroid") -> float:
    """
    Compute diversity of a set of candidate solutions.
    """
    def euclidean_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    if not candidates:
        return 0.0

    if method == "current-centroid":
        positions = np.array([cs.x for cs in candidates])
        centroid = np.mean(positions, axis=0)
        return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))
    elif method == "current-best":
        positions = np.array([cs.x for cs in candidates])
        best_current = min(candidates, key=lambda cs: cs.f).x
        return float(np.mean(np.linalg.norm(positions - best_current, axis=1)))
    elif method == "global-centroid":
        positions = np.array([cs.xbest for cs in candidates])
        centroid = np.mean(positions, axis=0)
        return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))
    elif method == "global-best":
        positions = np.array([cs.xbest for cs in candidates])
        best_global = min(candidates, key=lambda cs: cs.fbest).xbest
        return float(np.mean(np.linalg.norm(positions - best_global, axis=1)))
    elif method == "pair-wise":
        positions = [cs.x for cs in candidates]
        n = len(positions)
        if n < 2:
            return 0.0
        distances = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                distances += euclidean_distance(positions[i], positions[j])
                count += 1
        return float(distances * 2 / (n * (n - 1))) if count > 0 else 0.0
    else:
        raise ValueError(f"compute_diversity: Invalid method '{method}'.")


def lehmer_mean(values: np.ndarray) -> float:
    """
    Compute the Lehmer mean of an array.
    """
    total = np.sum(values)
    if total == 0.0:
        return 0.0
    return float(np.sum(values ** 2) / total)


def Levy(ndim: int, beta: float) -> np.ndarray:
    """
    Generate a Levy flight step vector.
    """
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2.0)
    denominator = gamma((1 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (numerator / denominator) ** (1.0 / beta)
    sigma_v = 1.0
    u = np.random.normal(0, sigma_u, ndim)
    v = np.random.normal(0, sigma_v, ndim)
    return u / (np.abs(v) ** (1.0 / beta))


def hashable_array(arr: np.ndarray) -> tuple:
    """
    Convert an array into a hashable tuple.
    """
    return tuple(arr.tolist())


def weighted_lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted Lehmer mean.
    """
    numerator = np.sum(weights * values ** 2)
    denominator = np.sum(weights * values)
    return float(numerator / denominator) if denominator != 0.0 else 0.5
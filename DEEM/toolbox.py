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
# 23.06.2026, J. Machacek - Added helpers for the improved restart and adaptation:
#                           - success_history_lehmer: SHADE-style Lehmer mean of
#                             successful control parameters, weighted by the
#                             achieved improvement (delta f), not by absolute
#                             fitness as before.
#                           - space_filling_sample: scrambled low-discrepancy
#                             (Sobol) fill of the search box for the restart.
#                           - covariance_seed: EDA-style Gaussian seeding around an
#                             elite using a shrinkage covariance, so that parameter
#                             interdependencies are respected during the restart.
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""
#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from typing import Sequence


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
        # Create a density matrix for each dimension and bin.
        self.matrix = np.zeros((len(LB), num_bins), dtype=float)

    def update_density(self, positions: Sequence[np.ndarray]) -> None:
        """
        Update the density matrix based on candidate positions.
        This version uses vectorized bincounts for speed.
        
        Parameters
        ----------
        positions : Sequence[np.ndarray]
            A sequence of positions (each an array of dimension values).
        """
        positions = np.array(positions, dtype=float)
        # Compute bin indices for each dimension:
        bins = np.floor((positions - self.LB) / (self.UB - self.LB + 1e-32) * self.num_bins).astype(int)
        bins = np.clip(bins, 0, self.num_bins - 1)
        # For each dimension d, count how many positions fall into each bin.
        for d in range(positions.shape[1]):
            counts = np.bincount(bins[:, d], minlength=self.num_bins)
            self.matrix[d, :] += counts

    def least_visited_position(self) -> np.ndarray:
        """
        Return a point in the least-visited bin for each dimension.
        This is the original method: for each dimension, it chooses the bin
        with the smallest visit count and samples uniformly within it.
        
        Returns
        -------
        np.ndarray
            A new candidate position.
        """
        bin_widths = (self.UB - self.LB) / self.num_bins
        # For each dimension, pick the bin index with the minimal count.
        min_bins_indices = np.argmin(self.matrix, axis=1)
        random_offsets = np.random.rand(len(self.LB)) * bin_widths
        new_position = self.LB + min_bins_indices * bin_widths + random_offsets
        self.update_density([new_position])
        return new_position

    def improved_least_visited_position(self) -> np.ndarray:
        """
        Return a candidate position sampled via weighted bin sampling.
        
        For each dimension, compute a weight for each bin that is inversely 
        proportional to the visit count (plus a small constant epsilon).
        Then sample a bin index according to these weights and sample uniformly
        within the chosen bin. Finally, update the density matrix with the new position.
        
        Returns
        -------
        np.ndarray
            A new candidate position.
        """
        epsilon = 1e-6
        bin_widths = (self.UB - self.LB) / self.num_bins
        new_position = []
        for d in range(len(self.LB)):
            # Get visit counts for dimension d.
            counts = self.matrix[d, :]
            # Compute weights: bins with fewer visits get higher weight.
            weights = 1.0 / (counts + epsilon)
            # Normalize weights to obtain a probability distribution.
            probabilities = weights / np.sum(weights)
            # Sample a bin index according to these probabilities.
            bin_index = np.random.choice(self.num_bins, p=probabilities)
            # Sample uniformly within the selected bin.
            offset = np.random.rand() * bin_widths[d]
            new_val = self.LB[d] + bin_index * bin_widths[d] + offset
            new_position.append(new_val)
        new_position = np.array(new_position)
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

def success_history_lehmer(values: np.ndarray, improvements: np.ndarray,
                           fallback: float = 0.5) -> float:
    """
    SHADE-style weighted Lehmer mean of successful control parameters.

    Each successful control value (e.g. CR or phi) is weighted by the *normalised
    improvement* it produced, w_k = df_k / sum_j df_j, with df_k = max(0, f_old - f_new).
    The weighted Lehmer mean (sum w v^2 / sum w v) biases the next generation
    towards the more productive values, exactly as in JADE/SHADE.

    Parameters
    ----------
    values : np.ndarray
        Control values of the successful candidates.
    improvements : np.ndarray
        Per-candidate improvement df_k (>= 0).
    fallback : float
        Returned when there is no usable success information.

    Returns
    -------
    float
        The adapted control value.
    """
    values = np.asarray(values, dtype=float)
    improvements = np.asarray(improvements, dtype=float)
    if values.size == 0:
        return fallback
    w = np.clip(improvements, 0.0, None)
    s = np.sum(w)
    if s <= 0.0:
        # successes without measurable improvement -> use the plain mean
        return float(np.mean(values))
    w = w / s
    denom = np.sum(w * values)
    if denom == 0.0:
        return fallback
    return float(np.sum(w * values ** 2) / denom)


def space_filling_sample(n: int, LB: np.ndarray, UB: np.ndarray) -> np.ndarray:
    """
    Draw n scrambled low-discrepancy (Sobol) points in the box [LB, UB].

    Falls back to uniform random sampling if scipy's QMC engine is unavailable.
    """
    LB = np.asarray(LB, dtype=float); UB = np.asarray(UB, dtype=float)
    if n <= 0:
        return np.empty((0, len(LB)))
    try:
        import warnings
        from scipy.stats.qmc import Sobol
        eng = Sobol(d=len(LB), scramble=True)
        with warnings.catch_warnings():
            # n is generally not a power of two here; the resulting (mild) loss of
            # balance is irrelevant for restart space-filling, so silence the notice.
            warnings.simplefilter("ignore")
            u = eng.random(n)
    except Exception:
        u = np.random.rand(n, len(LB))
    return LB + u * (UB - LB)


def covariance_seed(elite: np.ndarray, members: np.ndarray,
                    LB: np.ndarray, UB: np.ndarray, n: int,
                    shrink: float = 0.25) -> np.ndarray:
    """
    EDA-style Gaussian seeding around an elite position.

    A shrinkage covariance is estimated from the basin 'members' and used to draw
    n correlated samples centred on 'elite'. This respects parameter inter-
    dependencies (highlighted in the DEEM paper for hypoplastic calibration),
    unlike per-dimension-independent bin sampling.

    Parameters
    ----------
    elite : np.ndarray
        Centre of the basin (shape (ndim,)).
    members : np.ndarray
        Positions defining the basin shape (shape (m, ndim)); may be empty.
    LB, UB : np.ndarray
        Search-space bounds (used to scale the regulariser and to clip).
    n : int
        Number of samples to draw.
    shrink : float
        Shrinkage intensity towards a diagonal target in [0, 1].

    Returns
    -------
    np.ndarray
        Array of shape (n, ndim), clipped to [LB, UB].
    """
    LB = np.asarray(LB, dtype=float); UB = np.asarray(UB, dtype=float)
    elite = np.asarray(elite, dtype=float)
    ndim = len(elite)
    span = UB - LB
    span[span == 0.0] = 1.0
    if n <= 0:
        return np.empty((0, ndim))

    if members is not None and len(members) >= 2:
        cov = np.cov(np.asarray(members, dtype=float).T)
        cov = np.atleast_2d(cov)
    else:
        cov = np.zeros((ndim, ndim))

    # diagonal shrinkage target: a fraction of the box span
    diag_target = np.diag((0.05 * span) ** 2)
    cov = (1.0 - shrink) * cov + shrink * diag_target
    # tiny ridge for numerical stability
    cov += np.eye(ndim) * (1e-12 * np.mean(span) ** 2 + 1e-300)

    try:
        samples = np.random.multivariate_normal(elite, cov, size=n)
    except Exception:
        samples = elite + np.random.normal(size=(n, ndim)) * (0.05 * span)
    return np.clip(samples, LB, UB)

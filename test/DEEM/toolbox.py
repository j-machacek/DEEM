#!/usr/bin/env python3
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
# QuantumPSO - A quantum-behaved Particle Swarm Optimization (QPSO) toolbox
# (c) 2023 Jan Machacek
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~

import numpy as np
from scipy.stats import norm
from scipy.special import gamma


def contraction_expansion(method: str, min_val: float, max_val: float, maxiter: int) -> np.ndarray:
    """
    Generate a sequence that transitions (contracts or expands) from 'max_val' to 'min_val'
    or vice versa, according to a specified method.

    Parameters
    ----------
    method : str
        The method for contraction/expansion. One of {'linear', 'sigmoid-2', 'sigmoid-3',
        'exponential', 'gaussian'}.
    min_val : float
        The minimum value in the sequence.
    max_val : float
        The maximum value in the sequence.
    maxiter : int
        The number of iterations for which the sequence is generated.

    Returns
    -------
    seq : np.ndarray
        A NumPy array of length maxiter+1 containing the contraction/expansion values.

    Raises
    ------
    ValueError
        If 'method' is not one of the recognized options.
    """
    iters = np.arange(0, maxiter + 1, 1)
    x = iters / np.max(iters)  # normalized iteration index in [0,1]

    if min_val == max_val:
        return np.ones(maxiter + 1) * min_val

    if method == 'linear':
        # Linear transition from max_val to min_val or vice versa
        f_linear = np.linspace(1.0, 0.0, maxiter + 1)
        return min_val + (max_val - min_val) * f_linear

    elif method == 'sigmoid-2':
        # A type of sigmoid transition with exponent -2
        f_sigmoid2 = 1.0 - (1.0 / (1.0 + (x / (1.0 - x)) ** -2))
        return min_val + (max_val - min_val) * f_sigmoid2

    elif method == 'sigmoid-3':
        # A type of sigmoid transition with exponent -3
        f_sigmoid3 = 1.0 - (1.0 / (1.0 + (x / (1.0 - x)) ** -3))
        return min_val + (max_val - min_val) * f_sigmoid3

    elif method == 'exponential':
        # Exponential decay from max_val to min_val or vice versa
        # ratio = min_val / max_val if max_val != 0 else 1.0
        return max_val * ((min_val / max_val) ** x)

    elif method == 'gaussian':
        # Generate random values between min_val and max_val, then scale by a Gaussian CDF
        mu = 0.0
        sigma = 1.0
        u = np.random.uniform(min_val, max_val, maxiter)  # for maxiter steps
        cdf_value = norm.cdf(u, loc=mu, scale=sigma)
        return u * cdf_value

    else:
        raise ValueError(f"contraction_expansion: Invalid method: {method}.")


class Density:
    """
    A class for Density-Based Initialization.

    It helps exploration in optimization algorithms by keeping a record of the number of
    visits to each bin of the search space. When reinitializing particles, it can favor
    underexplored (low visitation count) areas.

    Attributes
    ----------
    LB : np.ndarray
        Lower bounds of the search space.
    UB : np.ndarray
        Upper bounds of the search space.
    num_bins : int
        Number of bins to divide each dimension into.
    matrix : np.ndarray
        2D array recording the number of visits per bin for each dimension.
    """

    def __init__(self, LB, UB, num_bins: int = 10):
        """
        Initialize the Density object.

        Parameters
        ----------
        LB : array-like
            Lower bounds for each dimension.
        UB : array-like
            Upper bounds for each dimension.
        num_bins : int, optional
            Number of bins per dimension. Default is 10.
        """
        self.LB = np.array(LB, dtype=float)
        self.UB = np.array(UB, dtype=float)
        self.num_bins = num_bins
        self.matrix = np.zeros((len(LB), num_bins), dtype=float)

    def update_density(self, positions):
        """
        Update the density matrix (visit counts) based on the current positions of particles.

        Parameters
        ----------
        positions : array-like
            A list or array of shape (n_particles, n_dims) containing particle positions.
        """
        positions = np.array(positions, dtype=float)

        # Calculate integer bin indices for each position
        # Clip to ensure indices are within [0, num_bins-1]
        bin_indices = np.floor(
            (positions - self.LB) / (self.UB - self.LB + 1e-32) * self.num_bins
        ).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Update visit counts
        for i, pos in enumerate(positions):
            # Filter out dimensions where LB==UB or position is invalid
            valid_dims = (self.UB - self.LB > 1e-32) & (~np.isnan(pos))
            for d in np.where(valid_dims)[0]:
                self.matrix[d, bin_indices[i, d]] += 1

    def least_visited_position(self) -> np.ndarray:
        """
        Return a point in the least-visited bin for each dimension, then increment its bin count.

        Returns
        -------
        new_position : np.ndarray
            A position in search space corresponding to underexplored (low count) bins.
        """
        bin_widths = (self.UB - self.LB) / np.maximum(self.num_bins, 1)

        # For each dimension, pick the bin with the smallest visit count
        min_bins_indices = np.argmin(self.matrix, axis=1)

        # Randomly place the new position within each chosen bin
        random_offsets = np.random.rand(len(self.LB)) * bin_widths
        new_position = self.LB + min_bins_indices * bin_widths + random_offsets

        # Update density with the newly created position
        self.update_density([new_position])
        return new_position


def shannon_entropy(particle_positions: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Shannon's entropy for a set of points in n-dimensional space.

    Parameters
    ----------
    particle_positions : np.ndarray
        A 2D array with shape (n_particles, n_dims).
    num_bins : int
        Number of bins for discretizing each dimension.

    Returns
    -------
    entropy : float
        The Shannon entropy of the distribution of particle positions.
    """
    if particle_positions.size == 0:
        return 0.0

    # Determine min/max for each dimension
    min_vals = np.min(particle_positions, axis=0)
    max_vals = np.max(particle_positions, axis=0)

    # Build histogram in n-dimensional space
    hist, _ = np.histogramdd(
        particle_positions,
        bins=num_bins,
        range=list(zip(min_vals, max_vals))
    )

    # Convert counts to probability distribution
    total_count = np.sum(hist)
    if total_count == 0:
        return 0.0
    prob_dist = hist / total_count

    # Filter out zero values to avoid log(0)
    prob_dist_nonzero = prob_dist[prob_dist > 0.0]
    # Shannon's entropy
    entropy = -np.sum(prob_dist_nonzero * np.log(prob_dist_nonzero))
    return entropy


def compute_diversity(particles, method: str = "current-centroid") -> float:
    """
    Compute the diversity of a swarm of particles.

    Parameters
    ----------
    particles : list
        A list of particle objects, each with at least attributes 'x', 'xbest', 'f', or 'fbest'.
    method : str
        Method used to compute diversity. Options:
          - "current-centroid": Distance of current positions from their centroid
          - "current-best": Distance of current positions from the best current position
          - "global-centroid": Distance of globally best positions from their centroid
          - "global-best": Distance of globally best positions from the global best
          - "pair-wise": Average pairwise distances among current positions

    Returns
    -------
    diversity : float
        The computed diversity measure.

    Raises
    ------
    ValueError
        If an unknown method is specified.
    """
    def euclidean_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    if not particles:
        return 0.0

    if method == "current-centroid":
        positions = np.array([p.x for p in particles])
        centroid = np.mean(positions, axis=0)
        return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

    elif method == "current-best":
        positions = np.array([p.x for p in particles])
        best_current = min(particles, key=lambda x: x.f).x
        return float(np.mean(np.linalg.norm(positions - best_current, axis=1)))

    elif method == "global-centroid":
        positions = np.array([p.xbest for p in particles])
        centroid = np.mean(positions, axis=0)
        return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

    elif method == "global-best":
        positions = np.array([p.xbest for p in particles])
        best_global = min(particles, key=lambda x: x.fbest).xbest
        return float(np.mean(np.linalg.norm(positions - best_global, axis=1)))

    elif method == "pair-wise":
        positions = [p.x for p in particles]
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
    Compute the Lehmer mean of an array of numbers.

    The Lehmer mean L of x_1, x_2, ..., x_n is:
        L = (x_1^2 + x_2^2 + ... + x_n^2) / (x_1 + x_2 + ... + x_n)

    Parameters
    ----------
    values : np.ndarray
        A 1D array of numeric values.

    Returns
    -------
    float
        The Lehmer mean. Returns 0.0 if the sum of values is zero.
    """
    total = np.sum(values)
    if total == 0.0:
        return 0.0
    return float(np.sum(values ** 2) / total)


def Levy(ndim: int, beta: float) -> np.ndarray:
    """
    Generate a Levy flight step vector of dimension 'ndim' using exponent 'beta'.

    Parameters
    ----------
    ndim : int
        Dimensionality of the step vector.
    beta : float
        Exponent controlling the step distribution.

    Returns
    -------
    step : np.ndarray
        A random step drawn from Levy distribution with the given beta.
    """
    # Precompute sigma for the u, v distribution
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2.0)
    denominator = gamma((1 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (numerator / denominator) ** (1.0 / beta)
    sigma_v = 1.0

    # Draw from normal distributions
    u = np.random.normal(0, sigma_u, ndim)
    v = np.random.normal(0, sigma_v, ndim)

    # Levy step
    return u / (np.abs(v) ** (1.0 / beta))


def hashable_array(arr: np.ndarray) -> tuple:
    """
    Convert a NumPy array into a hashable Python object (tuple) for set/dict operations.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array.

    Returns
    -------
    tuple
        A tuple representation of the input array.
    """
    return tuple(arr.tolist())


def weighted_lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute a weighted Lehmer mean for an array of values.

    For values x_i with corresponding weights w_i, the weighted Lehmer mean is:
        L_weighted = sum(w_i * x_i^2) / sum(w_i * x_i)

    Parameters
    ----------
    values : np.ndarray
        A 1D array of numeric values.
    weights : np.ndarray
        A 1D array of weights, same shape as 'values'.

    Returns
    -------
    float
        The weighted Lehmer mean. Returns 0.5 if the denominator is zero.
    """
    numerator = np.sum(weights * values ** 2)
    denominator = np.sum(weights * values)
    return float(numerator / denominator) if denominator != 0.0 else 0.5

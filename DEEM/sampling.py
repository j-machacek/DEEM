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
# Collection of different sampling methods:
# 1) Latin Hypercube Sampling
# 2) Random uniform sampling
# 3) Sobol sequences
# 4) Halton sequence
# 5) Regular grid sampling
#
# History
# 05.02.2023, J. Machacek - Initial version
# 07.02.2025, J. Machacek - Added Halton sampling and Grid sampling
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""
#!/usr/bin/env python3

import numpy as np
from typing import Union

def latin_hypercube_sampling(nparticles: int, dim: int, 
                             lower_bound: Union[np.ndarray, list], 
                             upper_bound: Union[np.ndarray, list]) -> np.ndarray:
    """
    Latin Hypercube Sampling

    Parameters
    ----------
    nparticles : int
        The number of samples.
    dim : int
        The number of dimensions.
    lower_bound : array-like
        Lower bounds for each dimension.
    upper_bound : array-like
        Upper bounds for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (nparticles, dim) containing the Latin Hypercube samples.
    """
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    # One sample per stratum: nparticles equal-width strata in [0, 1)
    cut = np.linspace(0, 1, nparticles + 1)
    a = cut[:nparticles]            # lower edge of each stratum, shape (nparticles,)
    b = cut[1:nparticles + 1]       # upper edge of each stratum, shape (nparticles,)
    u = np.random.rand(nparticles, dim)
    # Stratified positions, then permute EACH dimension independently so the
    # dimensions are uncorrelated (true Latin Hypercube). Permuting whole rows
    # instead would leave all dimensions perfectly correlated (points on the
    # diagonal) -- 16.06.2026, J. Machacek (bug fix).
    samples = u * (b - a)[:, np.newaxis] + a[:, np.newaxis]
    for j in range(dim):
        samples[:, j] = samples[np.random.permutation(nparticles), j]
    # Scale the samples to the provided bounds
    return lower_bound + samples * (upper_bound - lower_bound)


def random_uniform(nparticles: int, dim: int, 
                   lower_bound: Union[np.ndarray, list], 
                   upper_bound: Union[np.ndarray, list]) -> np.ndarray:
    """
    Generate samples using random uniform sampling.

    Parameters
    ----------
    nparticles : int
        The number of samples.
    dim : int
        The number of dimensions.
    lower_bound : array-like
        Lower bounds for each dimension.
    upper_bound : array-like
        Upper bounds for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (nparticles, dim) with uniformly sampled points.
    """
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    return lower_bound + np.random.uniform(size=(nparticles, dim)) * (upper_bound - lower_bound)


def initial_sampling_sobol(nparticles: int, dim: int, 
                           lower_bound: Union[np.ndarray, list], 
                           upper_bound: Union[np.ndarray, list]) -> np.ndarray:
    """
    Generate initial sampling based on Sobol sequences.

    Parameters
    ----------
    nparticles : int
        The number of samples.
    dim : int
        The number of dimensions.
    lower_bound : array-like
        Lower bounds for each dimension.
    upper_bound : array-like
        Upper bounds for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (nparticles, dim) containing Sobol sequence samples,
        scaled to the specified bounds.
    """
    from sobol_seq import i4_sobol_generate
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    samples = i4_sobol_generate(dim, nparticles)
    return lower_bound + samples * (upper_bound - lower_bound)


def halton_sampling(nparticles: int, dim: int, 
                    lower_bound: Union[np.ndarray, list], 
                    upper_bound: Union[np.ndarray, list]) -> np.ndarray:
    """
    Generate initial sampling based on the Halton sequence.

    Parameters
    ----------
    nparticles : int
        The number of samples.
    dim : int
        The number of dimensions.
    lower_bound : array-like
        Lower bounds for each dimension.
    upper_bound : array-like
        Upper bounds for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (nparticles, dim) containing Halton sequence samples,
        scaled to the specified bounds.
    """
    from scipy.stats.qmc import Halton
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    sampler = Halton(d=dim, scramble=True)
    samples = sampler.random(nparticles)
    return lower_bound + samples * (upper_bound - lower_bound)


def grid_sampling(nparticles: int, dim: int, 
                  lower_bound: Union[np.ndarray, list], 
                  upper_bound: Union[np.ndarray, list]) -> np.ndarray:
    """
    Generate samples using a regular grid sampling approach.

    Parameters
    ----------
    nparticles : int
        The desired number of samples.
    dim : int
        The number of dimensions.
    lower_bound : array-like
        Lower bounds for each dimension.
    upper_bound : array-like
        Upper bounds for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (nparticles, dim) containing grid-sampled points.
        If the full grid contains more points than needed, a random subset is returned.
    """
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    # Determine the number of grid points per dimension; note that in high dimensions,
    # grid sampling quickly becomes infeasible.
    num_points_per_dim = int(np.ceil(nparticles ** (1 / dim)))
    grids = [np.linspace(lower_bound[i], upper_bound[i], num_points_per_dim) for i in range(dim)]
    mesh = np.meshgrid(*grids)
    grid_points = np.stack([m.flatten() for m in mesh], axis=-1)
    total_points = grid_points.shape[0]
    if nparticles < total_points:
        indices = np.random.choice(total_points, size=nparticles, replace=False)
        return grid_points[indices]
    else:
        return grid_points[:nparticles]


def sampling(nparticles: int, dim: int, 
             lower_bound: Union[np.ndarray, list], 
             upper_bound: Union[np.ndarray, list], 
             method: str) -> np.ndarray:
    """
    Generate samples using the specified sampling method.

    Parameters
    ----------
    nparticles : int
        The number of samples.
    dim : int
        The number of dimensions.
    lower_bound : array-like
        Lower bounds for each dimension.
    upper_bound : array-like
        Upper bounds for each dimension.
    method : str
        Sampling method. Options include:
          - 'LHS' (Latin Hypercube Sampling)
          - 'Sobol' (Sobol sequence)
          - 'Random-Uniform' (uniform random sampling)
          - 'Halton' (Halton sequence)
          - 'Grid' (regular grid sampling)

    Returns
    -------
    np.ndarray
        Array of shape (nparticles, dim) containing the generated samples.

    Raises
    ------
    ValueError
        If an unknown sampling method is provided.
    """
    if method == 'LHS':
        return latin_hypercube_sampling(nparticles, dim, lower_bound, upper_bound)
    elif method == 'Random-Uniform':
        return random_uniform(nparticles, dim, lower_bound, upper_bound)
    elif method == 'Sobol':
        return initial_sampling_sobol(nparticles, dim, lower_bound, upper_bound)
    elif method == 'Halton':
        return halton_sampling(nparticles, dim, lower_bound, upper_bound)
    elif method == 'Grid':
        return grid_sampling(nparticles, dim, lower_bound, upper_bound)
    else:
        raise ValueError("Unknown sampling method. Use 'LHS', 'Sobol', 'Random-Uniform', 'Halton', or 'Grid'.")

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Jan Machaček
#
# This file is part of DEEM, released under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.

"""
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                            Copyright (C) 2023-2026 Jan Machacek  
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
# 07.09.2026, J. Machacek - Improve LHS sampling (Permute the strata independently in every dimension)
# 16.11.2026, J. Machacek - Sobol sampling uses SciPy's QMC engine instead of the
#                           undeclared sobol_seq package; the scrambled Halton engine
#                           draws its seed from the optimiser's RNG (scrambled_qmc_engine)
# 16.11.2026, J. Machacek - Every sampler takes an optional rng (numpy.random.RandomState
#                           interface; None = numpy's global RNG)
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""
#!/usr/bin/env python3

import numpy as np
from typing import Union


def _rng(rng):
    # 16.11.2026, J. Machacek - None selects numpy's global RNG (legacy behaviour)
    return np.random.mtrand._rand if rng is None else rng


def scrambled_qmc_engine(engine_class, dim: int, rng=None):
    """
    Create a scrambled SciPy QMC engine whose scrambling seed is drawn from rng
    (numpy's global RNG if None). DEEMI passes its own seeded RNG (``seed=``),
    so a seeded run is reproducible; an unseeded run stays stochastic.
    """
    generator = np.random.default_rng(int(_rng(rng).randint(0, 2**31 - 1)))
    try:
        return engine_class(d=dim, scramble=True, rng=generator)
    except TypeError:
        # SciPy < 1.15 names the argument ``seed``.
        return engine_class(d=dim, scramble=True, seed=generator)


def latin_hypercube_sampling(nparticles: int, dim: int, 
                             lower_bound: Union[np.ndarray, list], 
                             upper_bound: Union[np.ndarray, list],
                             rng=None) -> np.ndarray:
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
    rng = _rng(rng)
    u = rng.rand(nparticles, dim)
    # Stratified positions, then permute EACH dimension independently so the
    # dimensions are uncorrelated (true Latin Hypercube). Permuting whole rows
    # instead would leave all dimensions perfectly correlated (points on the
    # diagonal) -- 16.06.2026, J. Machacek (bug fix).
    samples = u * (b - a)[:, np.newaxis] + a[:, np.newaxis]
    for j in range(dim):
        samples[:, j] = samples[rng.permutation(nparticles), j]
    # Scale the samples to the provided bounds
    return lower_bound + samples * (upper_bound - lower_bound)


def random_uniform(nparticles: int, dim: int, 
                   lower_bound: Union[np.ndarray, list], 
                   upper_bound: Union[np.ndarray, list],
                   rng=None) -> np.ndarray:
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
    return lower_bound + _rng(rng).uniform(size=(nparticles, dim)) * (upper_bound - lower_bound)


def initial_sampling_sobol(nparticles: int, dim: int, 
                           lower_bound: Union[np.ndarray, list], 
                           upper_bound: Union[np.ndarray, list],
                           rng=None) -> np.ndarray:
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
    # SciPy's QMC engine replaces the historical, undeclared ``sobol_seq``
    # package. Generate a power-of-two block and discard the all-zero first
    # point (as sobol_seq did), which also avoids SciPy's balance warning for
    # arbitrary population sizes. The sequence is deterministic (unscrambled).
    from scipy.stats.qmc import Sobol

    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    if nparticles < 0:
        raise ValueError("nparticles must be non-negative.")
    if nparticles == 0:
        return np.empty((0, dim), dtype=float)
    exponent = int(np.ceil(np.log2(nparticles + 1)))
    samples = Sobol(d=dim, scramble=False).random_base2(m=exponent)
    samples = samples[1:nparticles + 1]
    return lower_bound + samples * (upper_bound - lower_bound)


def halton_sampling(nparticles: int, dim: int, 
                    lower_bound: Union[np.ndarray, list], 
                    upper_bound: Union[np.ndarray, list],
                    rng=None) -> np.ndarray:
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
    sampler = scrambled_qmc_engine(Halton, dim, rng=rng)
    samples = sampler.random(nparticles)
    return lower_bound + samples * (upper_bound - lower_bound)


def grid_sampling(nparticles: int, dim: int, 
                  lower_bound: Union[np.ndarray, list], 
                  upper_bound: Union[np.ndarray, list],
                  rng=None) -> np.ndarray:
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
        indices = _rng(rng).choice(total_points, size=nparticles, replace=False)
        return grid_points[indices]
    else:
        return grid_points[:nparticles]


def sampling(nparticles: int, dim: int, 
             lower_bound: Union[np.ndarray, list], 
             upper_bound: Union[np.ndarray, list], 
             method: str, rng=None) -> np.ndarray:
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
    rng : numpy.random.RandomState or None
        Random number generator (None: numpy's global RNG).
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
        return latin_hypercube_sampling(nparticles, dim, lower_bound, upper_bound, rng=rng)
    elif method == 'Random-Uniform':
        return random_uniform(nparticles, dim, lower_bound, upper_bound, rng=rng)
    elif method == 'Sobol':
        return initial_sampling_sobol(nparticles, dim, lower_bound, upper_bound, rng=rng)
    elif method == 'Halton':
        return halton_sampling(nparticles, dim, lower_bound, upper_bound, rng=rng)
    elif method == 'Grid':
        return grid_sampling(nparticles, dim, lower_bound, upper_bound, rng=rng)
    else:
        raise ValueError("Unknown sampling method. Use 'LHS', 'Sobol', 'Random-Uniform', 'Halton', or 'Grid'.")

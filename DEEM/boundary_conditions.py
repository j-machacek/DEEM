#!/usr/bin/env python3
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
# History:
# 05.02.2023, J. Machacek - Initial version
# 07.02.2025, J. Machacek - Improved code by adding A), B) and C)
# 16.11.2026, J. Machacek - A degenerate interval (ub <= lb, a fixed parameter) returns lb
#                           instead of dividing by zero in the periodic map
# 16.11.2026, J. Machacek - Optional rng argument (numpy.random.RandomState interface;
#                           None = numpy's global RNG) for all randomised methods
#
# Improvements:
# A) Fixed typos (e.g., 'radom' -> 'random')
# B) Refactored boundary enforcement to use a mapping via a helper function.
# C) Added type annotations and enhanced docstrings.
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import numpy as np


def _rng(rng):
    # 16.11.2026, J. Machacek - None selects numpy's global RNG (legacy behaviour)
    return np.random.mtrand._rand if rng is None else rng


def enforce_BC_random(x: float, lb: float, ub: float, ref: float, max_iter: int = 100, rng=None) -> float:
    """
    Enforce boundary conditions using a randomized reinitialization strategy.
    
    If x is out of bounds, iteratively adjust x toward the reference value ref until it
    falls within [lb, ub] or until max_iter iterations are reached.

    Parameters
    ----------
    x : float
        The coordinate value to adjust.
    lb : float
        The lower bound.
    ub : float
        The upper bound.
    ref : float
        The reference coordinate for adjustment.
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    float
        The adjusted coordinate within [lb, ub].
    """
    if x < lb or x > ub:
        counter = 0
        while (x < lb or x > ub) and counter < max_iter:
            x = ref + (x - ref) / (1.0 + _rng(rng).random())
            counter += 1
        x = np.clip(x, lb, ub)
    return x


def enforce_BC_damping(x: float, lb: float, ub: float, rng=None) -> float:
    """
    Enforce boundary conditions using a damping strategy.
    
    If x is out of bounds, it is adjusted toward the boundary by a fraction determined
    by a random factor.

    Parameters
    ----------
    x : float
        The coordinate value to adjust.
    lb : float
        The lower bound.
    ub : float
        The upper bound.

    Returns
    -------
    float
        The adjusted coordinate within [lb, ub].
    """
    r = _rng(rng).random()
    max_dist = ub - lb
    if x < lb:
        dist = min(lb - x, max_dist)
        x = lb + r * dist
    if x > ub:
        dist = min(x - ub, max_dist)
        x = ub - r * dist
    return x


def enforce_BC_periodic(x: float, lb: float, ub: float, rng=None) -> float:
    """
    Enforce boundary conditions using a periodic (wrap-around) strategy.
    
    When x falls outside [lb, ub], it is wrapped around to the other side.

    Parameters
    ----------
    x : float
        The coordinate value to adjust.
    lb : float
        The lower bound.
    ub : float
        The upper bound.

    Returns
    -------
    float
        The adjusted coordinate within [lb, ub].
    """
    r = _rng(rng).random()
    max_dist = ub - lb
    if x < lb:
        dist = lb - x
        dx = (dist / max_dist) % 1 * max_dist
        x = ub - r * dx
    if x > ub:
        dist = x - ub
        dx = (dist / max_dist) % 1 * max_dist
        x = lb + r * dx
    return x


def _apply_method(x: float, lb: float, ub: float, ref: float, method: str, rng=None) -> float:
    """
    Helper function to apply the chosen boundary condition enforcement method to a single coordinate.

    Parameters
    ----------
    x : float
        The coordinate value.
    lb : float
        The lower bound.
    ub : float
        The upper bound.
    ref : float
        The reference coordinate (used by some methods).
    method : str
        The enforcement method. Options:
        'clip', 'random', 'damping', 'periodic', 'damping-periodic',
        'damping-periodic-random', 'damping-periodic-clip'.

    Returns
    -------
    float
        The adjusted coordinate.
    
    Raises
    ------
    ValueError
        If an unknown method is provided.
    """
    if ub <= lb:
        # Degenerate interval: the periodic map would divide by ub - lb = 0.
        return float(lb)
    if method == 'clip':
        return float(np.clip(x, lb, ub))
    elif method == 'random':
        return enforce_BC_random(x, lb, ub, ref, rng=rng)
    elif method == 'damping':
        return enforce_BC_damping(x, lb, ub, rng=rng)
    elif method == 'periodic':
        return enforce_BC_periodic(x, lb, ub, rng=rng)
    elif method == 'damping-periodic':
        return (enforce_BC_damping(x, lb, ub, rng=rng) if _rng(rng).random() >= 0.5
                else enforce_BC_periodic(x, lb, ub, rng=rng))
    elif method == 'damping-periodic-random':
        r = _rng(rng).random()
        if r <= 0.33:
            return enforce_BC_damping(x, lb, ub, rng=rng)
        elif r <= 0.66:
            return enforce_BC_periodic(x, lb, ub, rng=rng)
        else:
            return enforce_BC_random(x, lb, ub, ref, rng=rng)
    elif method == 'damping-periodic-clip':
        r = _rng(rng).random()
        if r <= 0.33:
            return enforce_BC_damping(x, lb, ub, rng=rng)
        elif r <= 0.66:
            return enforce_BC_periodic(x, lb, ub, rng=rng)
        else:
            return float(np.clip(x, lb, ub))
    else:
        raise ValueError(
            "enforce_BC: unknown method '{}'. Use one of: "
            "clip, random, damping, periodic, damping-periodic, "
            "damping-periodic-random, damping-periodic-clip".format(method)
        )


def enforce_BC(x, lb, ub, ref, method: str = 'random', rng=None):
    """
    Enforce boundary conditions on a vector x using the specified method.
    
    Each coordinate x[i] is adjusted individually to lie within [lb[i], ub[i]] using
    the selected enforcement strategy. The parameter 'ref' provides reference values used
    in some methods.

    Parameters
    ----------
    x : array-like of float
        The vector of coordinates to adjust.
    lb : array-like of float
        Lower bounds for each coordinate.
    ub : array-like of float
        Upper bounds for each coordinate.
    ref : array-like of float
        Reference coordinates for each dimension.
    method : str, optional
        The enforcement method. Options:
            'clip', 'random', 'damping', 'periodic',
            'damping-periodic', 'damping-periodic-random', 'damping-periodic-clip'.
        Default is 'random'.
    rng : numpy.random.RandomState or None
        Random number generator (None: numpy's global RNG).

    Returns
    -------
    list of float
        The vector x after enforcing boundary conditions.
    """
    # Ensure inputs are lists for coordinate-wise processing
    x = list(x)
    lb = list(lb)
    ub = list(ub)
    ref = list(ref)

    for i in range(len(x)):
        x[i] = _apply_method(x[i], lb[i], ub[i], ref[i], method, rng=rng)
    return x

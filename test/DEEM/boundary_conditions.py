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
# History:
# 05.02.2023, J. Machacek - Initial version
# 07.02.2025, J. Machacek - Improved code by adding A), B) and C)
#
# Improvements:
# A) Fixed typos (e.g., 'radom' -> 'random')
# B) Refactored boundary enforcement to use a mapping via a helper function.
# C) Added type annotations and enhanced docstrings.
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import numpy as np


def enforce_BC_random(x: float, lb: float, ub: float, ref: float, max_iter: int = 100) -> float:
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
            x = ref + (x - ref) / (1.0 + np.random.random())
            counter += 1
        x = np.clip(x, lb, ub)
    return x


def enforce_BC_damping(x: float, lb: float, ub: float) -> float:
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
    r = np.random.random()
    max_dist = ub - lb
    if x < lb:
        dist = min(lb - x, max_dist)
        x = lb + r * dist
    if x > ub:
        dist = min(x - ub, max_dist)
        x = ub - r * dist
    return x


def enforce_BC_periodic(x: float, lb: float, ub: float) -> float:
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
    r = np.random.random()
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


def _apply_method(x: float, lb: float, ub: float, ref: float, method: str) -> float:
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
    if method == 'clip':
        return float(np.clip(x, lb, ub))
    elif method == 'random':
        return enforce_BC_random(x, lb, ub, ref)
    elif method == 'damping':
        return enforce_BC_damping(x, lb, ub)
    elif method == 'periodic':
        return enforce_BC_periodic(x, lb, ub)
    elif method == 'damping-periodic':
        return enforce_BC_damping(x, lb, ub) if np.random.random() >= 0.5 else enforce_BC_periodic(x, lb, ub)
    elif method == 'damping-periodic-random':
        r = np.random.random()
        if r <= 0.33:
            return enforce_BC_damping(x, lb, ub)
        elif r <= 0.66:
            return enforce_BC_periodic(x, lb, ub)
        else:
            return enforce_BC_random(x, lb, ub, ref)
    elif method == 'damping-periodic-clip':
        r = np.random.random()
        if r <= 0.33:
            return enforce_BC_damping(x, lb, ub)
        elif r <= 0.66:
            return enforce_BC_periodic(x, lb, ub)
        else:
            return float(np.clip(x, lb, ub))
    else:
        raise ValueError(
            "enforce_BC: unknown method '{}'. Use one of: "
            "clip, random, damping, periodic, damping-periodic, "
            "damping-periodic-random, damping-periodic-clip".format(method)
        )


def enforce_BC(x, lb, ub, ref, method: str = 'random'):
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
        x[i] = _apply_method(x[i], lb[i], ub[i], ref[i], method)
    return x

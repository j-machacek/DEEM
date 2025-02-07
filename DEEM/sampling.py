#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                               Copyright (C) 2023 Jan Machacek  
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 05.02.2023
#
# Collection of different sampling methods
#
# History
# 05.02.2023, J. Machacek - Initial version
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#!/usr/bin/env python3
import numpy as np


def latin_hypercube_sampling(nparticles, dim, lower_bound, upper_bound):
    """
    Latin Hypercube Sampling

    Parameters
    ----------
    dim (int): The number of dimensions.
    nparticles (int): The number of samples.
    lower_bound (numpy array): A numpy array of size (dim) representing the lower bound for each dimension.
    upper_bound (numpy array): A numpy array of size (dim) representing the upper bound for each dimension.

    Returns
    -------
    samples : array-like of shape (nparticles, dim)
        The Latin Hypercube samples.

    """
    # Generate n + 1 cuts between 0 and 1
    cut = np.linspace(0, 1, nparticles + 1)

    # Generate random numbers in the unit interval [0, 1]
    u = np.random.rand(nparticles, dim)

    # Get the lower and upper bound of the intervals
    a = cut[:nparticles, np.newaxis]
    b = cut[1:nparticles + 1, np.newaxis]

    # Generate random points in each interval
    samples = u * (b - a) + a

    # Make the random pairings using numpy's permutation function
    permuted_indices = np.random.permutation(np.arange(nparticles))
    permuted_samples = np.zeros_like(samples)
    for i in range(nparticles):
        permuted_samples[i,:] = samples[permuted_indices[i],:]

    # Scale the points to respect the lower and upper bounds
    return lower_bound + permuted_samples * (upper_bound - lower_bound)


def random_uniform(nparticles, dim, lower_bound ,upper_bound):
    """
    This function creates a numpy array of size (nparticles, dim) with random uniform sampling.
    
    Parameters:
    -----------
        nparticles (int): The number of particles to generate.
        dim (int): The number of dimensions for each particle.
        lower_bound (numpy array): A numpy array of size (dim) representing the lower bound for each dimension.
        upper_bound (numpy array): A numpy array of size (dim) representing the upper bound for each dimension.
    
    Returns:
    --------
        numpy array: An array of size (nparticles, dim) with random uniform sampling between lb and ub.
    """
    # samples = np.random.uniform(low=lower_bound, high=upper_bound, size=(nparticles, dim))
    samples = lower_bound + np.random.uniform(size=(nparticles, dim))*(upper_bound-lower_bound)
    return samples


def initial_sampling_sobol(nparticles, dim, lower_bound, upper_bound):
    """
    Generate initial sampling based on Sobol sequences using numpy.
    
    Parameters
    ----------
        size (int): The number of dimensions
        n (int): The number of samples
        lower_bound (int/float): lower bound of the samples
        upper_bound (int/float): upper bound of the samples
        
    Returns
    -------
        numpy array: The Sobol sequence points in `size` dimensions, scaled to respect `lower_bound` and `upper_bound`
    """
    from sobol_seq import i4_sobol_generate
    samples = i4_sobol_generate(dim, nparticles)
    return lower_bound + samples * (upper_bound - lower_bound)


def sampling(nparticles, dim, lower_bound ,upper_bound, method):
    """
    Function to generate samples using a specified method.
    
    Parameters:
    -----------
        nparticles (int): number of particles/samples
        dim (int): number of dimensions
        lower_bound (int/float): lower bound of the samples
        upper_bound (int/float): upper bound of the samples
        method (str): sampling method, either 'LHS' (Latin Hypercube Sampling) 'Sobol' or 'Random-Uniform'
    
    Returns:
    --------
        numpy array: generated samples
    
    """
    if method == 'LHS':
        samples = latin_hypercube_sampling(nparticles, dim, lower_bound, upper_bound)
    elif method == 'Random-Uniform':
        samples = random_uniform(nparticles, dim, lower_bound, upper_bound)
    elif method == 'Sobol':
        samples = initial_sampling_sobol(nparticles, dim, lower_bound, upper_bound)
    else:
        raise NameError('The function `sampling` was calles with an unkonwn sampling method.')
    return samples



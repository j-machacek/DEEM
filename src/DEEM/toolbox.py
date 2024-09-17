#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#                                          QuantumPSO
#                               Copyright (C) 2023 Jan Machacek  
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 05.02.2023
#
# Quantum-behaved Particle Swarm Optimization (QPSO) algorithm.
#
# History
# 05.02.2023, J. Machacek - Initial version
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
from scipy.special import gamma


def contraction_expansion(method,min,max,maxiter):

    iters = np.arange(0,maxiter+1,1)
    x = iters/np.max(iters)

    if min == max:
        return np.ones(maxiter+1) * min

    if method == 'linear':
        f_linear = np.linspace(1, 0, maxiter+1)
        return min + (max-min) * f_linear
    elif method == 'sigmoid-2':
        f_sigmoid2 = 1. - (1. / (1.+(x/(1.-x))**(-2)))
        return min + (max-min) * f_sigmoid2
    elif method == 'sigmoid-3':
        f_sigmoid3 = 1. - (1. / (1.+(x/(1.-x))**(-3)))
        return min + (max-min) * f_sigmoid3
    elif method == 'exponential':
        return max*(min/max)**(x)
    elif method =='gaussian':
        mu = 0
        sigma = 1
        u = np.random.uniform(min,max,maxiter)
        cdf_value = norm.cdf(u, mu, sigma)
        return u*cdf_value
    else:
        raise ValueError(f"contraction_expansion: Invalid method: {method}.")


class Density:
    """
    A class for Density-Based Initialization.
    
    It facilitates the process of exploration in optimization algorithms 
    by keeping a record of the number of visits to each region (or bin) of the 
    search space. During particle reinitialization, it favors the less explored regions.
    
    Attributes:
        LB (ndarray): Lower bounds of the search space for each dimension.
        UB (ndarray): Upper bounds of the search space for each dimension.
        num_bins (int): Number of bins to divide each dimension into.
        matrix (ndarray): Matrix to record the number of visits to each bin.
    """

    def __init__(self, LB, UB, num_bins=10):
        """
        Initializes the Density object.
        
        Args:
            LB (list or ndarray): Lower bounds of the search space.
            UB (list or ndarray): Upper bounds of the search space.
            num_bins (int, optional): Number of bins to divide each dimension. Default is 10.
        """
        self.LB = np.array(LB)
        self.UB = np.array(UB)
        self.num_bins = num_bins
        self.matrix = np.zeros([len(LB), num_bins])

    def update_density(self, positions):
        """
        Updates the density matrix based on the current positions of particles.
        
        Args:
            positions (list or ndarray): List of current positions of the particles in the search space.
        """
        positions = np.array(positions)
        
        # Calculate bin indices for positions
        bin_indices = np.clip(((positions - self.LB) / (self.UB - self.LB) * self.num_bins).astype(int), 0, self.num_bins - 1)
        
        # Check for valid positions (not NaN and valid bounds)
        valid_positions = ~np.isnan(positions) & ~np.isnan(self.LB) & ~np.isnan(self.UB) & (self.UB - self.LB != 0)

        # Update the matrix with the visit counts
        for i, position in enumerate(positions):
            self.matrix[np.arange(len(self.LB))[valid_positions[i]], bin_indices[i][valid_positions[i]]] += 1

    def least_visited_position(self):
        """
        Determines a position in the least visited region of the search space.
        
        Returns:
            ndarray: Position in the least explored region of the search space.
        """
        # Compute bin width for each dimension
        bin_width = (self.UB - self.LB) / self.num_bins
        
        # Get the index of the least visited bin for each dimension
        least_visited_bin_indices = np.argmin(self.matrix, axis=1)
        
        # Calculate position within the least visited bin
        position = self.LB + bin_width * least_visited_bin_indices + np.random.rand(*self.LB.shape) * bin_width
        
        # Update density with the newly determined position
        self.update_density(positions=[position])

        return position
    

def shannon_entropy(particles_positions, num_bins=10):
    """
    Compute the Shannon's entropy for the given particle positions in an n-dimensional space.

    Parameters:
    - particles_positions: A 2D numpy array where each row is a particle and columns represent dimensions.
    - num_bins: Number of bins to discretize each dimension of the search space.

    Returns:
    - entropy: Shannon's entropy value.
    """
    # Determine the minimum and maximum values for each dimension
    min_vals = np.min(particles_positions, axis=0)
    max_vals = np.max(particles_positions, axis=0)
    
    # Create histogram for n-dimensional data
    hist = np.histogramdd(particles_positions, bins=num_bins, range=list(zip(min_vals, max_vals)))[0]
    
    # Normalize histogram to create a probability distribution
    prob_dist = hist / np.sum(hist)
    
    # Filter out zero values to avoid log(0)
    prob_dist = prob_dist[prob_dist > 0]
    
    # Compute Shannon's entropy
    entropy = -np.sum(prob_dist * np.log(prob_dist))
    
    return entropy


def compute_diversity(particles, method="current-centroid"):
    """
    Computes the diversity of a swarm of particles.
    
    Parameters:
    - particles: List of particles. Each particle should have a 'position' attribute.
    - method: The method used to compute diversity. Can be: "current-centroid", "current-best", "global-centroid", "global-best" or "pair-wise".
    
    Returns:
    - Diversity value.
    """
    
    def euclidean_distance(a, b):
        """Calculate the Euclidean distance between two points."""
        return np.linalg.norm(np.array(a) - np.array(b))
       
    # Compute diversity based on the method
    if method == "current-centroid":
        positions = np.array([p.x for p in particles])
        centroid = np.mean(positions, axis=0)
        diversity = np.mean(np.linalg.norm(positions - centroid, axis=1))
        
    elif method == "current-best":
        positions = np.array([p.x for p in particles])
        global_best_position = min(particles, key=lambda x: x.f).x
        diversity = np.mean(np.linalg.norm(positions - global_best_position, axis=1))

    elif method == "global-centroid":
        positions = np.array([p.xbest for p in particles])
        centroid = np.mean(positions, axis=0)
        diversity = np.mean(np.linalg.norm(positions - centroid, axis=1))

    elif method == "global-best":
        positions = np.array([p.xbest for p in particles])
        global_best_position = min(particles, key=lambda x: x.fbest).xbest
        diversity = np.mean(np.linalg.norm(positions - global_best_position, axis=1))

    elif method == 'pair-wise':
        N = len(particles)
        total_distance = sum(euclidean_distance(particles[i].x, particles[j].x) for i in range(N) for j in range(i+1, N))
        diversity = 2 * total_distance / (N * (N - 1))
        
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'centroid' or 'global_best'.")
    
    return diversity



def lehmer_mean(list_objects: np.ndarray) -> float:
    """
    Compute the Lehmer mean of a numpy array of numbers.
    The Lehmer mean, for a list of numbers x_1, x_2, ..., x_n, is defined as:
    L = (x_1^2 + x_2^2 + ... + x_n^2) / (x_1 + x_2 + ... + x_n)
    
    It provides a mean value that gives more weight to larger numbers in the array.
    Parameters:
    - list_objects (np.ndarray): A numpy array of numbers for which the Lehmer mean is to be computed.
    Returns:
    - float: The Lehmer mean of the provided array of numbers. 
             Returns 0 if the sum of numbers in the array is 0 to avoid division by zero.
    """
    
    temp = np.sum(list_objects)

    return 0 if temp == 0 else np.sum(list_objects**2) / temp


def Levy(ndim,beta):
    num = gamma(1+beta)*np.sin(np.pi*beta/2)
    den = gamma((1+beta)/2)*beta*(2**((beta-1)/2))
    sigmaU = (num/den)**(1/beta)
    sigmaV = 1
    u = np.random.normal(0, sigmaU, ndim)
    v = np.random.normal(0, sigmaV, ndim)
    l = u/(np.abs(v)**(1/beta))
    return l


def hashable_array(arr):
    return tuple(arr.tolist())


def weighted_lehmer_mean(list_objects, list_weights):
    up = np.sum(list_weights * list_objects ** 2)
    down = np.sum(list_weights * list_objects)
    return up / down if down != 0 else 0.5
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                               Copyright (C) 2023 Jan Machacek  
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 05.02.2023
#
# DEEM - Differential Evolution with Elitism and Multi-populations
#
# History
# 05.02.2023, J. Machacek - Initial version
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#!/usr/bin/env python3
import numpy as np
from scipy.stats import cauchy
from .boundary_conditions import enforce_BC
from .toolbox import lehmer_mean
import multiprocessing as mp

#
#
# PARTICLE
#
#
class Particle:
    """
    Single particle in the swarm optimization algorithm

    Attributes:
    -----------
        x (np.ndarray): The current position of the particle in the search space.
        best (np.ndarray): The best position found by the particle so far.
        fbest (float): The cost associated with the best position.

    """

    def __init__(self, x0, function, f0=None, iswarm0=None):
        """
        Initialize the particle with a random position within the specified bounds.

        Args:
        -----
            bounds (np.ndarray): A 2D array that defines the lower and upper bounds
                of the search space in each dimension.

        """
        self.function = function
        self.x = self.xbest = self.xbest0 = x0
        if f0:
            self.f = f0
            self.fbest = f0
            self.fbest0 = f0
        else:
            self.f = 1e22
            self.fbest = 1e22
            self.fbest0 = 1e22
        self.improved = False
        self.randomize = False
        self.elite = False
        self.initialized = False
        self.niter_xbest = 0
        self.iiter_reset = 0
        if iswarm0:
            self.iswarm = iswarm0
        else:
            self.iswarm = 0

        self.DE_CR = np.random.uniform()
        self.phi = np.random.uniform()


    def reset(self,x0):
        self.x = x0
        self.xbest = self.x.copy()
        self.xbest0 = self.x.copy()
        self.f = 1e22
        self.fbest = 1e22
        self.fbest0 = 1e22
        self.improved = False
        self.randomize = False
        self.elite = False
        self.initialized = False
        self.niter_xbest = 0


    def update_cost(self):
        """
        Evaluate the objective function at the current position.
        If the new position is better, update the best position and the archive.
        Adapt F and CR parameters based on success or failure.
        """
        self.improved = False

        self.f = self.function(self.x)

        if not self.initialized:
            self.fbest = self.fbest0 = self.f
            self.xbest = self.xbest0 = self.x.copy()
            self.improved = True
            self.initialized = True

        elif self.f < self.fbest:
            self.xbest0 = self.xbest
            self.fbest0 = self.fbest
            self.xbest = self.x.copy()
            self.fbest = self.f
            self.niter_xbest = 0
            self.improved = True

        self.niter_xbest += 1


    def enforce_BC(self,lb,ub,ref,method='random'):
        self.x = enforce_BC(self.x,lb,ub,ref,method)

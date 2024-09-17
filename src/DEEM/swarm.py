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
from copy import deepcopy

#
#
# SWARM
#
#
class Swarm:

    def __init__(self, nswarms_max, nswarms_min, method_subswarm_creation, method_subswarm_reduction):

        self.niter_xbest = 0
        self.reset_swarms = False
        self.reset_swarms_iter = 0
        self.nswarms = 0
        self.nswarms_max = nswarms_max
        self.nswarms_min = nswarms_min
        self.method_subswarm_creation = method_subswarm_creation
        self.method_subswarm_reduction = method_subswarm_reduction
    
    def create(self, particles, LB, UB, iters, maxiter, reset):

        def euclidean_distance(a, b):
            """Calculate the Euclidean distance between two points."""
            return np.linalg.norm(np.array(a) - np.array(b))
        
        def scale_coordinates(x, LB, UB):
            """Scale the coordinates according to the domain bounds."""
            return [(x[i] - LB[i]) / (UB[i] - LB[i]) if UB[i] != LB[i] else 0 for i in range(len(x))]
                    
        def euclidean_distance_scaled(x1, x2, LB, UB):
            """Calculate the Euclidean distance in the scaled space."""
            x1_scaled = scale_coordinates(x1, LB, UB)
            x2_scaled = scale_coordinates(x2, LB, UB)
            return sum((a - b) ** 2 for a, b in zip(x1_scaled, x2_scaled)) ** 0.5

        nswarms0 = self.nswarms

        iters = iters - self.reset_swarms_iter
        maxiter = max(1,maxiter - self.reset_swarms_iter)
        x = iters/maxiter

        nmax = self.nswarms_max
        nmin = self.nswarms_min
        if nmin == nmax:
            self.nswarms = nmin
        elif self.method_subswarm_reduction == 'sigmoid-2':
            self.nswarms = int( round(nmin + (nmax-nmin)*(1.-(1./(1.+(x/(1.-x))**(-2)))), 0) )
        elif self.method_subswarm_reduction == 'sigmoid-3':
            self.nswarms = int(round(nmin + (nmax-nmin)*(1.-(1./(1.+(x/(1.-x))**(-3)))), 0))
        elif self.method_subswarm_reduction == 'linear':
            self.nswarms = int(round(nmin + (nmax-nmin)*(1.-x), 0))
        elif self.method_subswarm_reduction == 'exponential':
            self.nswarms = int(round(nmax*(nmin/nmax)**(x), 0))
        elif self.method_subswarm_reduction == 'constant':
            self.nswarms = nmax
        
        sorted_particles = sorted(particles, key=lambda p: p.fbest)

        #
        # Create swarms new
        #
        if self.nswarms != nswarms0 or reset:
            
            if self.method_subswarm_creation == 'equally-distributed':
                best_particless = [sorted_particles.pop(0) for _ in range(self.nswarms)]
                swarms = [[best_particles] for best_particles in best_particless]
                while sorted_particles:
                    for sublist in self.swarms:
                        if sorted_particles:
                            elite = sublist[0]
                            distances = [(euclidean_distance(elite.xbest, p.xbest), p) for p in sorted_particles]
                            distances.sort(key=lambda x: x[0])
                            _, closest_particle = distances.pop(0)
                            sorted_particles.remove(closest_particle)
                            sublist.append(closest_particle)
                        else:
                            break

            elif self.method_subswarm_creation == 'fitness-focused':

                total_particles = len(sorted_particles)
                NPS = total_particles // self.nswarms         
                remainder = total_particles % self.nswarms
                swarms = []

                for _ in range(self.nswarms):
                    sublist = []
                    best_particles = sorted_particles.pop(0)
                    sublist.append(best_particles)
                    extra_particles = 1 if remainder > 0 else 0
                    if extra_particles:
                        remainder -= 1
                    distances = [(euclidean_distance_scaled(best_particles.xbest, p.xbest, LB, UB), p) for p in sorted_particles]
                    distances.sort(key=lambda x: x[0])
                    for _ in range(NPS + extra_particles - 1):
                        if distances:
                            _, closest_particle = distances.pop(0)
                            sorted_particles.remove(closest_particle)
                            sublist.append(closest_particle)
                        else:
                            break
                    swarms.append(deepcopy(sublist))

            iswarm = 0
            for swarm in swarms:
                swarm.sort(key=lambda p: p.fbest)
                swarm[0].elite = True
                for p in swarm:
                    p.iswarm = iswarm
                iswarm += 1

        #
        # Assign particles to existing swarms
        #
        else:

            swarms = []
            for _ in range(self.nswarms):
                swarms.append([])
            for p in particles:
                swarms[p.iswarm].append(p)
            for iswarm in range(self.nswarms):
                if len(swarms[iswarm]) == 0:
                    self.swarms.pop(iswarm)
                    self.nswarms = len(swarms)
            for swarm in swarms:
                swarm.sort(key=lambda p: p.fbest)
                swarm[0].elite = True

        return swarms
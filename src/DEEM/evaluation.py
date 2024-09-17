
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
from concurrent import futures


def update_particles_group(particle_group,updated_particles):

    for p in particle_group:
        p.update_cost()

    updated_particles.extend(particle_group)


def update_particle(particle):
    particle.update_cost()
    return particle


def evaluate_cost_function(particles, nworkers=1, mode='process'):

    if nworkers == 1:

        for p in particles:
            p.update_cost()

    else:

        updated_particles = []

        if mode == "thread":
            with futures.ThreadPoolExecutor(nworkers) as executor:
                list_executors = [executor.submit(update_particle, p) for p in particles]
                for f in futures.as_completed(list_executors):
                    updated_particles.append(f.result())

        elif mode == "process":
            with futures.ProcessPoolExecutor(nworkers) as executor:
                list_executors = [executor.submit(update_particle, p) for p in particles]
                for f in futures.as_completed(list_executors):
                    updated_particles.append(f.result())

        else:
            updated_particles = [update_particle(p) for p in particles]

        return updated_particles
        
    return particles
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
import sys

#
#
# Enforce Boundary Conditions
#
#

def enforce_BC_random(x,lb,ub,ref):
    if x < lb or x > ub:
        counter = 0
        while x < lb or x > ub:
            x = ref + (x - ref) / (1.+np.random.random())
            counter += 1
            if counter == 100:
                if x < lb:
                    x = lb
                if x > ub:
                    x = ub
                break  
    return x


def enforce_BC_damping(x,lb,ub):
    r = np.random.random()
    max_dist = ub-lb
    if x < lb:
        dist = min(lb-x,max_dist)
        x = lb + r*dist
    if x > ub:
        dist = min(x-ub,max_dist)
        x = ub - r*dist
    return x


def enforce_BC_periodic(x,lb,ub):
    r = np.random.random()
    max_dist = ub-lb
    if x < lb:
        dist = lb-x
        dx = (dist/max_dist)%1 * max_dist
        x = ub - r*dx
    if x > ub:
        dist = x-ub
        dx = (dist/max_dist)%1 * max_dist
        x = lb + r*dx 
    return x


def enforce_BC(x,lb,ub,ref,method='random'):

    if method == 'clip':
        for i in range(len(x)):
            x[i] = np.clip(x[i],lb[i],ub[i])
    elif method == 'radom':
        for i in range(len(x)):
            x[i] = enforce_BC_random(x[i],lb[i],ub[i],ref[i])
    elif method == 'damping':
        for i in range(len(x)):
            x[i] = enforce_BC_damping(x[i],lb[i],ub[i])
    elif method == 'periodic':
        for i in range(len(x)):
            x[i] = enforce_BC_periodic(x[i],lb[i],ub[i])
    elif method == 'damping-periodic':
        for i in range(len(x)):
            if np.random.uniform(low=0,high=1,size=1) >= 0.5:
                x[i] = enforce_BC_damping(x[i],lb[i],ub[i])
            else:
                x[i] = enforce_BC_periodic(x[i],lb[i],ub[i])
    elif method == 'damping-periodic-random':
        for i in range(len(x)):
            r = np.random.uniform(low=0,high=1,size=1)
            if r <= 0.33:
                x[i] = enforce_BC_damping(x[i],lb[i],ub[i])
            elif r > 0.33 and r <= 0.66:
                x[i] = enforce_BC_periodic(x[i],lb[i],ub[i])
            else:
                x[i] = enforce_BC_random(x[i],lb[i],ub[i],ref[i])
    elif method == 'damping-periodic-clip':
        for i in range(len(x)):
            r = np.random.uniform(low=0,high=1,size=1)
            if r <= 0.33:
                x[i] = enforce_BC_damping(x[i],lb[i],ub[i])
            elif r > 0.33 and r <= 0.66:
                x[i] = enforce_BC_periodic(x[i],lb[i],ub[i])
            else:
                x[i] = np.clip(x[i],lb[i],ub[i])
    else:
        print('enforce_BC: unknown method, use one of the followin: ' 
              + '(1) radom, (2) damping, (3) periodic, ' 
              + '(4) damping-periodic or (5) damping-periodic-random')
        raise

    return x
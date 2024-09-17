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

import numpy as np
import time
from scipy.stats import norm

from .swarm import Swarm
from .boundary_conditions import enforce_BC
from .logger import Logger
from .toolbox import Density, Levy, hashable_array, weighted_lehmer_mean
from .particle import Particle
from .sampling import sampling
from .evaluation import evaluate_cost_function

from copy import deepcopy
from scipy.stats import cauchy

class position:
    def __init__(self,x: np.ndarray,f: np.ndarray):
        self.x = x
        self.f = f

#
#
# DEEM - Differential Evolution with Elitism and Multi-populations
#
#
class DEEM():

    def __init__(self, function, lower_bound, upper_bound, X0=None, nparticles_max=50, nparticles_min=50, nswarm_max=10, nswarm_min=2,\
            maxiter=1000, maxfev=100000000, sampling_method='LHS', nworkers=1, tolerance=1e-6, termination='iterations', maxiter_below_tolerance=30,\
            log_interval=1, method_subswarm_reduction='sigmoid-3', method_boundary='damping', method_subswarm_creation='equally-distributed',\
            method_reset='density', niter_reset_global=None, penalty = 1e22):

        if nparticles_min > nparticles_max:
            self.nparticles_min = nparticles_max
        else:
            self.nparticles_min = nparticles_min
        self.nparticles_max = nparticles_max
        self.nparticles = self.nparticles_max
        nparticles = self.nparticles
        self.nparticles_reset = 0
        self.nworkers = nworkers
        
        self.PENALTY = penalty

        # store settings
        self.function = function
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.fev = 0
        self.iters = 0
        self.sampling_method = sampling_method

        self.niter_below_tolerance = 0
        self.niter_above_tolerance = 0
        if niter_reset_global:
            self.niter_reset_global = niter_reset_global
        else:
            self.niter_reset_global = self.maxiter//10
        self.global_reset_iter = 0

        self.tolerance = tolerance
        self.maxiter_below_tolerance = maxiter_below_tolerance
        self.log_interval = log_interval
        
        self.termination = termination

        # Archive
        self.archive_elite = []
        self.archive = []
        self.archive_size = self.nparticles

        # Bounds
        self.method_boundary = method_boundary
        self.LB = np.array(lower_bound)
        self.UB = np.array(upper_bound)
        self.dist_ub_lb = self.UB - self.LB

        self.method_subswarm_reduction = method_subswarm_reduction
        self.method_subswarm_creation = method_subswarm_creation
        self.method_subswarm_forced_update = False
        self.method_reset = method_reset

        # global best
        self.XBEST_history = []
        self.FBEST_history = []

        # swarms
        self.swarms = Swarm(nswarm_max, nswarm_min, method_subswarm_creation, method_subswarm_reduction)
        self.nswarms = nswarm_max
        self.nswarm_max = nswarm_max

        self.ndim = len(lower_bound)

        self.global_reset_condition = False

        self.Density = Density(LB=self.LB, UB=self.UB, num_bins=100*self.ndim)

        # initialize logger
        self.log = Logger(path='./', lower_bound=self.LB, upper_bound=self.UB)

        #
        # write the header
        #

        print("")
        print("=======================================================================")
        print("")
        print("DEEM - Differential Evolution with Elitism and Multi-populations")
        print("")
        print("(c) Jan Machacek, jan-machacek@outlook.com")
        print("")
        print("=======================================================================")
        print("")

        #
        # initialize particles
        #

        # create initial sampling
        print("... generate initial positions: " + self.sampling_method)
        samples = sampling(self.nparticles, self.ndim, self.LB, self.UB, self.sampling_method)

        # create a set of particles and assign the initial position
        self.particles = [Particle(samples[i,:], self.function) for i in range(0, nparticles)]

        # in case an initial guess is provided
        if X0:
            self.particles[0].x = np.asarray(X0)

        # calculate initial cost function value for each particle
        # for p in self.particles:
        #     for i in range(len(p.x)):
        #         p.x[i] = np.round(p.x[i],4)

        print("... evaluate fitness of initial positions")
        self.particles = evaluate_cost_function(self.particles, nworkers=self.nworkers)

        self.particles.sort(key=lambda x: x.fbest)
        self.XBEST = self.particles[0].xbest
        self.FBEST = self.particles[0].fbest
        self.XBEST_history.append(self.XBEST)
        self.FBEST_history.append(self.FBEST)
        self.FBEST0 = self.FBEST
        print("... best cost:" + str(self.FBEST) + '\n')

        tmp = [1 for p in self.particles if p.f == self.PENALTY]
        print('penalties = ' + str(np.sum(tmp)))

        self.update_archive()

        pos_matrix = np.array([p.xbest for p in self.particles])
        self.DIV_GB0 = np.mean(np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0), axis=0)
        pos_matrix = np.array([p.x for p in self.particles])
        self.DIV_CB0 = np.mean(np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0), axis=0)
        # self.DIV_GB0 = compute_diversity(self.particles, method="global-best")
        # self.DIV_CB0 = compute_diversity(self.particles, method="current-best")
        self.DIV_NORM_GB = 1.0
        self.DIV_NORM_CB = 1.0

        self.DIV_GB_SWARM = []
        for _ in range(self.nswarm_max):
            self.DIV_GB_SWARM.append([])

        self.log.plot_initial_distribution(self.particles)


    def positioning(self):
        """
        updates the position of each particle in the swarm. 
        It uses a combination of the particle's best position, the global best position, 
        and random numbers to update the position of each particle.

        Arguments:
            **kwargs: Optional keyword arguments that can be used to provide additional parameters to the method if necessary.
    
        ReturnElite:
            None. Updates the position of each particle in the swarm.
        """
                  
        length = len(self.particles[0].x)
        
        updated_particles = []

        #
        # Check if reinitialization is required
        #

        self.NPRESET = 0 #self.nparticles//5
        self.particles.sort(key=lambda x: x.fbest)
        
        self.reset_subswarms = False
        self.global_reset_condition = self.niter_below_tolerance > self.niter_reset_global or self.DIV_NORM_GB < 1e-2
        if self.global_reset_condition:
            if (self.iters-self.global_reset_iter) >= self.niter_reset_global:
                self.NPRESET = self.nparticles - 1
                self.global_reset_iter = self.reset_subswarms_iter = self.iters
                self.reset_subswarms = True
            else:
                self.global_reset_condition = False     

        ip = 0
        for p in self.particles:
            p.randomize = False ; p.elite = False
            if ip >= (self.nparticles-self.NPRESET):
                if self.global_reset_condition:
                    p.randomize = True
                    p.iiter_reset = self.iters
            ip += 1

        self.reset_subswarms = True
        swarms = self.swarms.create(self.particles, self.LB, self.UB, self.iters-self.global_reset_iter, self.maxiter-self.global_reset_iter, self.reset_subswarms)
        self.nswarms = len(swarms)

        #
        # Global best/mean, elite mean and archives for DE operation
        #

        unity = self.archive_elite + [position(x=p.xbest,f=p.fbest) for p in self.particles[0:self.nparticles] if p.fbest != self.PENALTY]

        #
        # 
        #
        for iswarm in self.DIV_GB_SWARM:
            iswarm.append(0)

        #
        # Update particles positions
        #

        iswarm = ip = self.nparticles_reset = 0
        for swarm in swarms:

            pos_matrix = np.array([p.xbest for p in swarm])
            self.DIV_GB_SWARM[iswarm][-1]=(np.mean(np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0), axis=0))
            DIV_GB_SWARM_NORM = max(0., min(1., self.DIV_GB_SWARM[iswarm][-1]/self.DIV_GB_SWARM[iswarm][0]))

            SB = min(swarm, key=lambda x: x.fbest).xbest

            list_cr = []
            list_f = []
            list_f0 = []
            for p in swarm:
                if p.improved:
                    list_cr.append(np.mean(p.DE_CR))
                    list_f.append(p.fbest)
                    list_f0.append(p.fbest0)
            if not list_cr:
                CR = 0.5
            else:
                df = np.abs(np.array(list_f) - np.array(list_f0))
                total_fit = np.sum(df)
                list_weights = 0 if total_fit == 0 else df / total_fit
                CR = weighted_lehmer_mean(np.array(list_cr), list_weights)

            ip_swarm = 0
            for p in swarm:

                p.x0 = p.x.copy()

                std = 0.2 
                while True:
                    mutation = cauchy.rvs(CR, std)
                    if mutation < 0.:
                        continue
                    elif mutation > 1.0:
                        mutation = 1.0
                    break
                p.DE_CR = mutation

                PHI = 1.0
                std = 0.2
                while True:
                    phi = cauchy.rvs(PHI, std)
                    if phi < 0.:
                        continue
                    elif phi > 1.:
                        phi = 1.
                    break
                p.phi = phi    

                #PHI = 0.1*(2.-DIV_GB_SWARM_NORM)
                #std = 0.2*(2.-DIV_GB_SWARM_NORM)
                PHI = 0.5+0.5*(1.-self.DIV_NORM_GB)
                std = 0.2
                while True:
                    phi = cauchy.rvs(PHI, std)
                    if phi < 0.:
                        phi = 0.
                    elif phi > 1.:
                        phi = 1.
                    break
                phi2 = phi 

                if p.randomize:

                    self.nparticles_reset += 1
                    R = self.Density.least_visited_position()
                    p = Particle(x0=R, function=p.function, iswarm0=p.iswarm)


                elif p.elite:

                    r = []
                    if iswarm != 0:
                        tmp = position(x=self.XBEST,f=self.FBEST)
                        r.append(tmp)

                        for i in range(10*self.nparticles):
                            idx = np.random.randint(len(unity))
                            tmp = unity[idx]
                            if not np.array_equal(tmp.x,p.xbest) and not np.array_equal(r[0].x,tmp.x):
                                break
                        r.append(tmp)

                        for i in range(10*self.nparticles):
                            idx = np.random.randint(len(unity))
                            tmp = unity[idx]
                            if not np.array_equal(tmp.x,p.xbest) and not np.array_equal(r[0].x,tmp.x) and not np.array_equal(r[1].x,tmp.x):
                                break
                        r.append(tmp)

                        r.sort(key=lambda x: x.f)

                        A = p.phi*p.xbest + (1.-p.phi)*r[0].x + (r[1].x-r[2].x)*phi2
                        j_rand = np.random.randint(0, self.ndim)
                        mask = (np.random.rand(length) <= p.DE_CR) | (np.arange(length) == j_rand)
                        p.x = np.where(mask, A, self.XBEST)
                    else:
                        lb = [] ; ub = []
                        for idim, ilb, iub in zip(p.xbest, self.LB, self.UB):
                            lb.append( np.abs(idim-ilb) )
                            ub.append( np.abs(idim-iub) )
                        dx = np.ones(self.ndim)
                        lev = Levy(self.ndim, beta=1.99)
                        for i, ilev in enumerate(lev):
                            if ilev > 0:
                                dx[i] = min( ilev*self.dist_ub_lb[i]/50, ub[i])
                            else:
                                dx[i] = max( ilev*self.dist_ub_lb[i]/50, -lb[i])
                        p.x = p.xbest.copy() + dx 


                else:

                    r = []

                    for i in range(10*self.nparticles):
                        idx = np.random.randint(1,len(swarm))
                        tmp = position(x=swarm[idx].xbest,f=swarm[idx].fbest)
                        if not np.array_equal(tmp.x,p.xbest):
                            break
                    r.append(tmp)

                    for i in range(10*self.nparticles):
                        idx = np.random.randint(1,len(swarm))
                        tmp = position(x=swarm[idx].xbest,f=swarm[idx].fbest)
                        if not np.array_equal(tmp.x,p.xbest) and not np.array_equal(r[0].x,tmp.x):
                            break
                    r.append(tmp)

                    r.sort(key=lambda x: x.f)

                    A = p.phi*p.xbest + (1.-p.phi)*SB + (r[0].x-r[1].x)*phi2

                    j_rand = np.random.randint(0, self.ndim)
                    mask = (np.random.rand(length) <= p.DE_CR) | (np.arange(length) == j_rand)
                    p.x = np.where(mask, A, SB)

                p.enforce_BC(lb=self.LB, ub=self.UB, ref=SB, method=self.method_boundary)

                updated_particles.append(p)

                ip_swarm += 1
                ip += 1

            iswarm += 1

        self.particles = deepcopy(updated_particles)

        if self.global_reset_condition:
            nparticles = self.nparticles_max
            if nparticles > self.nparticles:
                dn = nparticles - self.nparticles
                for _ in range(dn):
                    if np.random.uniform() >= 0.5:
                        R = self.Density.least_visited_position()
                    else:
                        lb = [] ; ub = []
                        idx = np.random.randint(0, self.nparticles//10)
                        xref = self.particles[idx].xbest
                        for idim, ilb, iub in zip(xref, self.LB, self.UB):
                            lb.append( np.abs(idim-ilb) )
                            ub.append( np.abs(idim-iub) )
                        rr1 = np.random.uniform(low=0.0,high=0.25,size=length)
                        rr2 = np.random.uniform(low=0.0,high=0.25,size=length)
                        R = np.random.uniform(xref-rr1*lb,xref+rr2*ub)
                    p = Particle(x0=R, function=p.function, iswarm0=p.iswarm)
                    self.particles.append(p)
                self.nparticles = len(self.particles)


    def update_archive(self):

        #
        # Elite archive
        #

        unique_xbest = set()
        unique_particles = []

        if self.iters > 0:
            tmp = [position(x=p.xbest,f=p.fbest) for p in self.particles[0:self.nparticles] if p.fbest != self.PENALTY]
        else:
            tmp = [position(x=p.xbest0,f=p.fbest0) for p in self.particles[0:self.nparticles] if p.fbest0 != self.PENALTY]

        for p in (self.archive_elite + tmp):
            xbest_tuple = hashable_array(p.x)
            if xbest_tuple not in unique_xbest:
                unique_particles.append(deepcopy(p))
                unique_xbest.add(xbest_tuple)

        self.archive_elite = unique_particles

        sorted(self.archive_elite, key=lambda p: p.f)
        if len(self.archive_elite) > self.nparticles:
            self.archive_elite = self.archive_elite[0:self.nparticles]

        #
        # General archive
        #

        unique_x = set()
        unique_positions = []

        if self.iters > 0:
            tmp = [p for p in self.particles if (not p.improved and p.f != self.PENALTY)]
        else:
            tmp = [p for p in self.particles if (p.f != self.PENALTY)]

        for p in (self.archive + tmp):
            x_tuple = hashable_array(p.x)
            if x_tuple not in unique_x:
                unique_positions.append(position(x=p.x,f=p.f))
                unique_x.add(x_tuple)

        self.archive = unique_positions

        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[0:self.archive_size]


    def update(self):
        """
        Main loop of the DEEM algorithm. 
        It updates the kernel and the best position of each particle in the swarm, 
        until the maximum number of iterationElite has been reached or the tolerance criteria is met. 
        If a callback function is provided, it will be called at a specified interval.
        """
        
        print("--------------------------------------------------------------------------")
        print("{0: >5}  {1: >12}  {2: >14}  {3: >13}  {4: >5}  {5: >5}  {6: >10}  {7: >5}  {8: >5}"
              .format("Iters.", "Best f(x_t)", "f(x_t)-f(x_t0)", "Early stoppage", "Time / s", "FEV", "Particles/Swarms", "COV", "DIV-GB"))
        print("--------------------------------------------------------------------------")

        # set counter to zero
        self.niter_below_tolerance = 0

        start = time.time()

        while self.iters <= self.maxiter:

            start_iter = time.time()

            self.positioning()
            self.fev += self.nparticles

            visited_positions = set()
            for p in self.particles:
                pos_tuple = tuple(p.x)
                if pos_tuple in visited_positions:
                    lb = [] ; ub = []
                    for idim, ilb, iub in zip(p.x, self.LB, self.UB):
                        lb.append( np.abs(idim-ilb) )
                        ub.append( np.abs(idim-iub) )
                    dx = np.ones(self.ndim)
                    lev = Levy(self.ndim, beta=1.99)
                    for i, ilev in enumerate(lev):
                        if ilev > 0:
                            dx[i] = min( ilev*self.dist_ub_lb[i]/50, ub[i])
                        else:
                            dx[i] = max( ilev*self.dist_ub_lb[i]/50, -lb[i])
                    p.x += + dx 
                else:
                    visited_positions.add(pos_tuple)

            self.particles = evaluate_cost_function(self.particles, nworkers=self.nworkers)

            self.particles.sort(key=lambda x: x.fbest)
                   
            FBEST0 = self.FBEST
            if self.particles[0].fbest < self.FBEST:
                self.XBEST_history.append(self.XBEST)
                self.FBEST_history.append(self.FBEST)
                self.XBEST = self.particles[0].xbest
                self.FBEST = self.particles[0].fbest

            self.update_archive()

            self.Density.update_density(positions=[p.x for p in self.particles])

            pos_matrix = np.array([p.xbest for p in self.particles])
            self.DIV_GB = np.mean(np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0), axis=0)
            pos_matrix = np.array([p.x for p in self.particles])
            self.DIV_CB = np.mean(np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0), axis=0)
            # self.DIV_GB = compute_diversity(self.particles, method="global-best")
            # self.DIV_CB = compute_diversity(self.particles, method="current-best")
            self.DIV_NORM_GB = min(self.DIV_GB / self.DIV_GB0, 1)
            self.DIV_NORM_CB = min(self.DIV_CB / self.DIV_CB0, 1)

            end_iter = time.time()

            if self.nparticles_min != self.nparticles_max:
                iters = self.iters - self.reset_subswarms_iter
                maxiter = max(1,self.maxiter - self.reset_subswarms_iter)
                x = iters/maxiter
                nparticles = int(self.nparticles_max*(self.nparticles_min/self.nparticles_max)**(x))
                if self.nparticles > nparticles:
                    dn = self.nparticles - nparticles
                    self.particles = self.particles[:-dn]
                    self.nparticles = len(self.particles)

            # Check if the change in the best cost of the swarm is below the tolerance
            df = abs(FBEST0-self.FBEST)
            if df <= self.tolerance:
                self.niter_below_tolerance += 1
                self.niter_above_tolerance = 0
            else:
                self.niter_above_tolerance += 1
                self.niter_below_tolerance = 0

            self.log.save(self.iters,self.FBEST,self.XBEST,self.particles,self.nswarms,self.nparticles_reset,self.DIV_CB, self.DIV_GB, self.DIV_GB_SWARM)

            if self.iters % self.log_interval == 0:
                if self.global_reset_condition:
                    print("{0: >5}  {1: >12.5E}  {2: >14.5E}  {3: >6} {4} {5: <6}  {6: >9.3E}  {7: >5}  {8: >5} {9} {10: <5}  {11: <5}  {12: >9.3E} {13: >5}"
                          .format(self.iters, self.FBEST, df, self.niter_below_tolerance, '/', self.maxiter_below_tolerance, 
                                  (end_iter-start_iter), self.fev, self.nparticles, '/', self.nswarms, self.nparticles_reset, np.round(self.DIV_NORM_GB,4), '<- RESET'))
                else:
                    print("{0: >5}  {1: >12.5E}  {2: >14.5E}  {3: >6} {4} {5: <6}  {6: >9.3E}  {7: >5}  {8: >5} {9} {10: <5}  {11: <5}  {12: >9.3E}"
                          .format(self.iters, self.FBEST, df, self.niter_below_tolerance, '/', self.maxiter_below_tolerance, 
                                  (end_iter-start_iter), self.fev, self.nparticles, '/', self.nswarms, self.nparticles_reset, np.round(self.DIV_NORM_GB,4)))
                    
            # Check if the change in the best cost of the swarm has been below the tolerance 
            # for more than the maximum allowed iteration
            if (self.termination == 'tolerance' and self.niter_below_tolerance >= self.maxiter_below_tolerance) \
                or (self.fev >= self.maxfev):
                break
    
            self.FBEST0 = self.FBEST

            self.iters += 1

        end = time.time()       
        self.log.plot_results()     
        print("---------------------------------------------------------------")
        print("Found best position: {0}".format(self.XBEST))
        print("Execution time: ", end - start)
        print("=======================================================================")
        print("")       
        return
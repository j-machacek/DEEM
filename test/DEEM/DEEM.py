import numpy as np
import time
from scipy.stats import cauchy
from copy import deepcopy

from .swarm import Swarm
from .boundary_conditions import enforce_BC
from .logger import Logger
from .toolbox import Density, Levy, hashable_array, weighted_lehmer_mean
from .particle import Particle
from .sampling import sampling
from .evaluation import evaluate_cost_function


class Position:
    """
    A lightweight container for storing (x, f) pairs.
    """

    def __init__(self, x: np.ndarray, f: float):
        self.x = x
        self.f = f


def bounded_cauchy_draw(location: float, scale: float,
                        lower: float = 0.0, upper: float = 1.0) -> float:
    """
    Draw a sample from a Cauchy distribution centered at 'location' with standard 'scale',
    repeatedly until it falls in [lower, upper].

    :param location: Center of the Cauchy distribution.
    :param scale: Scale (width) parameter of the Cauchy distribution.
    :param lower: Lower bound for the returned value.
    :param upper: Upper bound for the returned value.
    :return: A float in the interval [lower, upper].
    """
    while True:
        val = cauchy.rvs(loc=location, scale=scale)
        if val < lower:
            continue
        elif val > upper:
            val = upper
        return val


class DEEM:
    """
    Differential Evolution with Elitism and Multi-populations (DEEM).

    Attributes:
        function: The objective function to minimize.
        LB, UB:   The lower and upper bound arrays for the search space.
        ...
    """

    def __init__(self, function, lower_bound, upper_bound, X0=None,
                 nparticles_max=50, nparticles_min=50, nswarm_max=10, nswarm_min=2,
                 maxiter=1000, maxfev=100000000, sampling_method='LHS',
                 nworkers=1, tolerance=1e-6, termination='iterations',
                 maxiter_below_tolerance=30, log_interval=1,
                 method_subswarm_reduction='sigmoid-3', method_boundary='damping',
                 method_subswarm_creation='equally-distributed',
                 method_reset='density', niter_reset_global=None, penalty=1e22):
        """
        Initialize the DEEM optimizer and create an initial population of particles.
        """

        # Handle min/max particles
        if nparticles_min > nparticles_max:
            self.nparticles_min = nparticles_max
        else:
            self.nparticles_min = nparticles_min
        self.nparticles_max = nparticles_max
        self.nparticles = self.nparticles_max
        self.nparticles_reset = 0
        self.nworkers = nworkers

        # Penalty for infeasible or penalized solutions
        self.PENALTY = penalty

        # Store settings
        self.function = function
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.fev = 0  # function evaluations
        self.iters = 0
        self.sampling_method = sampling_method

        # Tolerance tracking
        self.niter_below_tolerance = 0
        self.niter_above_tolerance = 0
        if niter_reset_global is not None:
            self.niter_reset_global = niter_reset_global
        else:
            self.niter_reset_global = self.maxiter // 10
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

        # Global best
        self.XBEST_history = []
        self.FBEST_history = []

        # Swarms
        self.swarms = Swarm(nswarm_max, nswarm_min,
                            method_subswarm_creation,
                            method_subswarm_reduction)
        self.nswarms = nswarm_max
        self.nswarm_max = nswarm_max

        self.ndim = len(lower_bound)
        self.global_reset_condition = False

        # Density object for re-initialization
        self.Density = Density(LB=self.LB, UB=self.UB, num_bins=100*self.ndim)

        # Initialize logger
        self.log = Logger(path='./', lower_bound=self.LB, upper_bound=self.UB)

        # -- HEADER INFO --
        print("")
        print("=======================================================================")
        print("DEEM - Differential Evolution with Elitism and Multi-populations")
        print("(c) Jan Machacek, jan-machacek@outlook.com")
        print("=======================================================================")
        print("")

        # -- Initialize particles --
        print(f"... generate initial positions: {self.sampling_method}")
        samples = sampling(self.nparticles, self.ndim, self.LB, self.UB,
                           self.sampling_method)
        self.particles = [Particle(samples[i, :], self.function)
                          for i in range(self.nparticles)]

        # Optional initial guess
        if X0 is not None:
            self.particles[0].x = np.asarray(X0)

        print("... evaluate fitness of initial positions")
        self.particles = evaluate_cost_function(self.particles, nworkers=self.nworkers)

        # Sort and track best
        self.particles.sort(key=lambda x: x.fbest)
        self.XBEST = self.particles[0].xbest
        self.FBEST = self.particles[0].fbest
        self.XBEST_history.append(self.XBEST)
        self.FBEST_history.append(self.FBEST)
        self.FBEST0 = self.FBEST

        print(f"... best cost: {self.FBEST}\n")

        # Show how many are penalized
        penalties_count = sum(1 for p in self.particles if p.f == self.PENALTY)
        print(f"penalties = {penalties_count}")

        self.update_archive()

        # Compute initial diversity stats
        pos_matrix_gbest = np.array([p.xbest for p in self.particles])
        self.DIV_GB0 = np.mean(np.mean(np.abs(np.median(pos_matrix_gbest, axis=0)
                                              - pos_matrix_gbest), axis=0), axis=0)

        pos_matrix_curr = np.array([p.x for p in self.particles])
        self.DIV_CB0 = np.mean(np.mean(np.abs(np.median(pos_matrix_curr, axis=0)
                                              - pos_matrix_curr), axis=0), axis=0)

        self.DIV_NORM_GB = 1.0
        self.DIV_NORM_CB = 1.0

        # Diversity metrics for each swarm
        self.DIV_GB_SWARM = [[] for _ in range(self.nswarm_max)]

        # Optionally plot initial distribution
        self.log.plot_initial_distribution(self.particles)

    def positioning(self) -> None:
        """
        Update the position of each particle in the swarm(s).
        Uses a combination of each particle's best position, the global best,
        swarm-based best, and random (Cauchy/Levy) perturbations.
        """

        length = self.ndim
        updated_particles = []

        # Sort by fbest
        self.particles.sort(key=lambda x: x.fbest)

        # Check if reinitialization is required
        self.NPRESET = 0
        self.reset_subswarms = False
        self.global_reset_condition = (self.niter_below_tolerance > self.niter_reset_global
                                       or self.DIV_NORM_GB < 1e-2)

        if self.global_reset_condition:
            # Enough iterations since last reset?
            if (self.iters - self.global_reset_iter) >= self.niter_reset_global:
                self.NPRESET = self.nparticles - 1
                self.global_reset_iter = self.iters
                self.reset_subswarms_iter = self.iters
                self.reset_subswarms = True
            else:
                self.global_reset_condition = False

        # Mark which particles get re-randomized
        for idx, particle in enumerate(self.particles):
            particle.randomize = False
            particle.elite = False
            if idx >= (self.nparticles - self.NPRESET) and self.global_reset_condition:
                particle.randomize = True
                particle.iiter_reset = self.iters

        # Create or update swarms
        self.reset_subswarms = True
        swarms = self.swarms.create(
            self.particles, self.LB, self.UB,
            (self.iters - self.global_reset_iter),
            (self.maxiter - self.global_reset_iter),
            self.reset_subswarms
        )
        self.nswarms = len(swarms)

        # Merge archive elites and current best solutions
        unity_positions = (self.archive_elite
                           + [Position(p.xbest, p.fbest)
                              for p in self.particles[:self.nparticles]
                              if p.fbest != self.PENALTY])

        # Track swarm-level diversity
        for swarm_div_list in self.DIV_GB_SWARM:
            swarm_div_list.append(0)

        # Go swarm by swarm
        nparticles_reset_this_iter = 0
        for iswarm, swarm in enumerate(swarms):

            # Compute swarm diversity
            pos_matrix = np.array([p.xbest for p in swarm])
            current_div = np.mean(np.mean(np.abs(np.median(pos_matrix, axis=0)
                                                 - pos_matrix), axis=0), axis=0)
            self.DIV_GB_SWARM[iswarm][-1] = current_div

            # Protect from division by zero
            base_div = max(self.DIV_GB_SWARM[iswarm][0], 1e-12)
            div_swarm_norm = max(0., min(1., current_div / base_div))

            # Swarm best (SB)
            SB = min(swarm, key=lambda x: x.fbest).xbest

            # Adaptive CR in the swarm, based on improved particles
            list_cr, list_f, list_f0 = [], [], []
            for p in swarm:
                if p.improved:
                    list_cr.append(np.mean(p.DE_CR))
                    list_f.append(p.fbest)
                    list_f0.append(p.fbest0)

            if not list_cr:
                CR = 0.5
            else:
                df_arr = np.abs(np.array(list_f) - np.array(list_f0))
                total_fit = df_arr.sum()
                weights = (df_arr / total_fit) if total_fit > 0 else 0
                CR = weighted_lehmer_mean(np.array(list_cr), weights)

            # Update each particle in this swarm
            for p in swarm:
                p.x0 = p.x.copy()

                # DE_CR ~ Cauchy around CR
                p.DE_CR = bounded_cauchy_draw(location=CR, scale=0.2)

                # p.phi ~ Cauchy around 1.0
                p.phi = bounded_cauchy_draw(location=1.0, scale=0.2)

                # Additional random factor phi2 ~ Cauchy around PHI
                PHI = 0.5 + 0.5 * (1. - self.DIV_NORM_GB)
                phi2 = bounded_cauchy_draw(location=PHI, scale=0.2)

                if p.randomize:
                    # Reinitialize from least visited position
                    nparticles_reset_this_iter += 1
                    R = self.Density.least_visited_position()
                    p = Particle(x0=R, function=p.function, iswarm0=p.iswarm)

                elif p.elite:
                    # Elitist update in non-primary swarm
                    if iswarm != 0:
                        r = []
                        # Add global best first
                        r.append(Position(self.XBEST, self.FBEST))

                        # Try to pick three distinct positions from unity that differ from p.xbest
                        max_tries = 10 * self.nparticles
                        distinct_positions = []
                        while len(distinct_positions) < 3 and max_tries > 0:
                            candidate = np.random.choice(unity_positions)
                            if (not np.array_equal(candidate.x, p.xbest)
                               and all(not np.array_equal(candidate.x, d.x) for d in distinct_positions)
                               and not np.array_equal(candidate.x, r[0].x)):
                                distinct_positions.append(candidate)
                            max_tries -= 1

                        r.extend(distinct_positions)
                        r.sort(key=lambda rp: rp.f)

                        # DE-like update
                        A = (p.phi * p.xbest
                             + (1. - p.phi) * r[0].x
                             + (r[1].x - r[2].x) * phi2) if len(r) >= 3 else p.xbest

                        j_rand = np.random.randint(0, self.ndim)
                        mask = (np.random.rand(length) <= p.DE_CR) | (np.arange(length) == j_rand)
                        p.x = np.where(mask, A, self.XBEST)

                    else:
                        # If swarm 0, use Levy-based random shift
                        lb_dist = []
                        ub_dist = []
                        for xx, lbv, ubv in zip(p.xbest, self.LB, self.UB):
                            lb_dist.append(np.abs(xx - lbv))
                            ub_dist.append(np.abs(xx - ubv))

                        dx = np.ones(self.ndim)
                        lev = Levy(self.ndim, beta=1.99)
                        for i, ilev in enumerate(lev):
                            if ilev > 0:
                                dx[i] = min(ilev * self.dist_ub_lb[i] / 50, ub_dist[i])
                            else:
                                dx[i] = max(ilev * self.dist_ub_lb[i] / 50, -lb_dist[i])
                        p.x = p.xbest.copy() + dx

                else:
                    # Regular update
                    r = []
                    # Attempt to pick two distinct different xbest from the same swarm
                    max_tries = 10 * self.nparticles
                    while len(r) < 2 and max_tries > 0:
                        idx_choice = np.random.randint(1, len(swarm))
                        candidate = swarm[idx_choice]
                        if not np.array_equal(candidate.xbest, p.xbest) \
                           and all(not np.array_equal(candidate.xbest, rr.x) for rr in r):
                            r.append(Position(candidate.xbest, candidate.fbest))
                        max_tries -= 1

                    r.sort(key=lambda rp: rp.f)
                    if len(r) < 2:
                        # fallback if we didn't find enough distinct positions
                        A = p.xbest
                    else:
                        A = (p.phi * p.xbest
                             + (1. - p.phi) * SB
                             + (r[0].x - r[1].x) * phi2)

                    j_rand = np.random.randint(0, self.ndim)
                    mask = (np.random.rand(length) <= p.DE_CR) | (np.arange(length) == j_rand)
                    p.x = np.where(mask, A, SB)

                # Enforce boundary conditions
                p.enforce_BC(lb=self.LB, ub=self.UB, ref=SB, method=self.method_boundary)
                updated_particles.append(p)

        # Update class-level
        self.particles = deepcopy(updated_particles)
        self.nparticles_reset = nparticles_reset_this_iter

        # Expand population to nparticles_max if we triggered a global reset
        if self.global_reset_condition:
            desired_nparticles = self.nparticles_max
            if desired_nparticles > self.nparticles:
                dn = desired_nparticles - self.nparticles
                for _ in range(dn):
                    # Reinit with 50% chance from density or random local
                    if np.random.uniform() >= 0.5:
                        R = self.Density.least_visited_position()
                    else:
                        # Local random search around a top-10% best
                        idx_choice = np.random.randint(0, max(1, self.nparticles // 10))
                        xref = self.particles[idx_choice].xbest
                        lb_dist = []
                        ub_dist = []
                        for xx, lbv, ubv in zip(xref, self.LB, self.UB):
                            lb_dist.append(np.abs(xx - lbv))
                            ub_dist.append(np.abs(xx - ubv))
                        # Random scaling
                        rr1 = np.random.uniform(low=0.0, high=0.25, size=length)
                        rr2 = np.random.uniform(low=0.0, high=0.25, size=length)
                        R = np.random.uniform(xref - rr1 * lb_dist,
                                              xref + rr2 * ub_dist)
                    # Add new particle
                    new_particle = Particle(x0=R, function=self.function,
                                            iswarm0=self.particles[0].iswarm)
                    self.particles.append(new_particle)
                self.nparticles = len(self.particles)

    def update_archive(self) -> None:
        """
        Update both the elite archive (archive_elite) and the general archive (archive).
        """

        unique_xbest = set()
        unique_particles = []

        # Build a candidate list of xbest from current or initial best (depending on iters)
        if self.iters > 0:
            tmp = [Position(x=p.xbest, f=p.fbest)
                   for p in self.particles[:self.nparticles]
                   if p.fbest != self.PENALTY]
        else:
            tmp = [Position(x=p.xbest0, f=p.fbest0)
                   for p in self.particles[:self.nparticles]
                   if p.fbest0 != self.PENALTY]

        # Merge with existing archive_elite
        for pos_obj in (self.archive_elite + tmp):
            xbest_tuple = hashable_array(pos_obj.x)
            if xbest_tuple not in unique_xbest:
                unique_particles.append(deepcopy(pos_obj))
                unique_xbest.add(xbest_tuple)

        # Sort by fitness and truncate to self.nparticles
        unique_particles.sort(key=lambda a: a.f)
        self.archive_elite = unique_particles[:self.nparticles]

        # Update the general archive with non-improved or newly encountered positions
        unique_x = set()
        unique_positions = []
        if self.iters > 0:
            tmp2 = [p for p in self.particles if (not p.improved and p.f != self.PENALTY)]
        else:
            tmp2 = [p for p in self.particles if p.f != self.PENALTY]

        for pp in (self.archive + tmp2):
            x_tuple = hashable_array(pp.x)
            if x_tuple not in unique_x:
                unique_positions.append(Position(pp.x, pp.f))
                unique_x.add(x_tuple)

        # Truncate if exceeds capacity
        if len(unique_positions) > self.archive_size:
            np.random.shuffle(unique_positions)
            unique_positions = unique_positions[:self.archive_size]

        self.archive = unique_positions

    def update(self) -> None:
        """
        Main loop of DEEM.
        Continues until max iteration or tolerance-based termination is met.
        """

        print("--------------------------------------------------------------------------")
        print("{0: >5}  {1: >12}  {2: >14}  {3: >13}  {4: >5}  {5: >5}  {6: >10}  {7: >5}  {8: >5}"
              .format("Iters.", "Best f(x_t)", "f(x_t)-f(x_t0)", "Early stoppage",
                      "Time / s", "FEV", "Particles/Swarms", "COV", "DIV-GB"))
        print("--------------------------------------------------------------------------")

        # Reset iteration-based counters
        self.niter_below_tolerance = 0
        start_time = time.time()

        while self.iters <= self.maxiter:
            iter_start_time = time.time()

            # 1) Move the population
            self.positioning()
            self.fev += self.nparticles

            # 2) If repeated positions occurred, do Levy-based random shift
            visited_positions = set()
            for p in self.particles:
                pos_tuple = tuple(p.x)
                if pos_tuple in visited_positions:
                    lb_dist = []
                    ub_dist = []
                    for xx, lbv, ubv in zip(p.x, self.LB, self.UB):
                        lb_dist.append(abs(xx - lbv))
                        ub_dist.append(abs(xx - ubv))
                    dx = np.ones(self.ndim)
                    lev = Levy(self.ndim, beta=1.99)
                    for i, ilev in enumerate(lev):
                        if ilev > 0:
                            dx[i] = min(ilev * self.dist_ub_lb[i] / 50, ub_dist[i])
                        else:
                            dx[i] = max(ilev * self.dist_ub_lb[i] / 50, -lb_dist[i])
                    p.x += dx
                else:
                    visited_positions.add(pos_tuple)

            # 3) Evaluate the population
            self.particles = evaluate_cost_function(self.particles, nworkers=self.nworkers)
            self.particles.sort(key=lambda x: x.fbest)

            # 4) Update global best if improved
            prev_best = self.FBEST
            if self.particles[0].fbest < self.FBEST:
                self.XBEST_history.append(self.XBEST)
                self.FBEST_history.append(self.FBEST)
                self.XBEST = self.particles[0].xbest
                self.FBEST = self.particles[0].fbest

            # 5) Update archives
            self.update_archive()

            # 6) Update density (for re-initialization logic)
            self.Density.update_density(positions=[p.x for p in self.particles])

            # 7) Recompute diversity
            pos_matrix_gbest = np.array([p.xbest for p in self.particles])
            self.DIV_GB = np.mean(np.mean(np.abs(np.median(pos_matrix_gbest, axis=0)
                                                 - pos_matrix_gbest), axis=0), axis=0)
            pos_matrix_curr = np.array([p.x for p in self.particles])
            self.DIV_CB = np.mean(np.mean(np.abs(np.median(pos_matrix_curr, axis=0)
                                                 - pos_matrix_curr), axis=0), axis=0)

            self.DIV_NORM_GB = min(self.DIV_GB / max(self.DIV_GB0, 1e-12), 1.0)
            self.DIV_NORM_CB = min(self.DIV_CB / max(self.DIV_CB0, 1e-12), 1.0)

            # 8) Timing
            iter_end_time = time.time()

            # 9) Possibly shrink the population if needed
            if self.nparticles_min != self.nparticles_max:
                iters_local = (self.iters - getattr(self, 'reset_subswarms_iter', 0))
                total_local = max(1, (self.maxiter - getattr(self, 'reset_subswarms_iter', 0)))
                x_frac = iters_local / total_local
                target_nparticles = int(self.nparticles_max * (self.nparticles_min / self.nparticles_max) ** x_frac)
                if self.nparticles > target_nparticles:
                    remove_count = self.nparticles - target_nparticles
                    self.particles = self.particles[:-remove_count]
                    self.nparticles = len(self.particles)

            # 10) Check if best cost improvement is below tolerance
            df = abs(prev_best - self.FBEST)
            if df <= self.tolerance:
                self.niter_below_tolerance += 1
                self.niter_above_tolerance = 0
            else:
                self.niter_below_tolerance = 0
                self.niter_above_tolerance += 1

            # 11) Log iteration
            self.log.save(self.iters, self.FBEST, self.XBEST,
                          self.particles, self.nswarms,
                          self.nparticles_reset, self.DIV_CB,
                          self.DIV_GB, self.DIV_GB_SWARM)

            if self.iters % self.log_interval == 0:
                # Print iteration info
                extra_reset_flag = "<- RESET" if self.global_reset_condition else ""
                print("{0: >5}  {1: >12.5E}  {2: >14.5E}  {3: >6} / {4: <6}  {5: >9.3E}  {6: >5}  {7: >5} / {8: <5}  {9: <5}  {10: >9.3E} {11: >5}"
                      .format(self.iters, self.FBEST, df,
                              self.niter_below_tolerance, self.maxiter_below_tolerance,
                              (iter_end_time - iter_start_time),
                              self.fev, self.nparticles, self.nswarms,
                              self.nparticles_reset, np.round(self.DIV_NORM_GB, 4), extra_reset_flag))

            # 12) Termination conditions
            if ((self.termination == 'tolerance'
                 and self.niter_below_tolerance >= self.maxiter_below_tolerance)
                or (self.fev >= self.maxfev)):
                break

            self.FBEST0 = self.FBEST
            self.iters += 1

        # Final logs
        end_time = time.time()
        self.log.plot_results()
        print("---------------------------------------------------------------")
        print(f"Found best position: {self.XBEST}")
        print(f"Execution time: {end_time - start_time:.3f} s")
        print("=======================================================================")
        print("")

        return

import numpy as np
from copy import deepcopy

class Swarm:
    """
    Manages the creation or update of multiple subswarms in a DEEM-like optimization.
    """

    def __init__(self, nswarms_max, nswarms_min, method_subswarm_creation, method_subswarm_reduction):
        self.niter_xbest = 0
        self.reset_swarms = False
        self.reset_swarms_iter = 0
        self.nswarms = 0
        self.nswarms_max = nswarms_max
        self.nswarms_min = nswarms_min
        self.method_subswarm_creation = method_subswarm_creation
        self.method_subswarm_reduction = method_subswarm_reduction

    @staticmethod
    def euclidean_distance(a, b):
        """Return Euclidean distance between points a and b."""
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def scale_coordinates(x, LB, UB):
        """
        Scale coordinates of x to [0, 1] based on the search space [LB, UB].
        If UB[i] == LB[i], we define that dimension’s scaled value as 0 to avoid division by zero.
        """
        return [
            0 if (UB[i] == LB[i]) else (x[i] - LB[i]) / (UB[i] - LB[i])
            for i in range(len(x))
        ]

    @classmethod
    def euclidean_distance_scaled(cls, x1, x2, LB, UB):
        """Euclidean distance between x1 and x2 in a space scaled to [0,1]."""
        x1_scaled = cls.scale_coordinates(x1, LB, UB)
        x2_scaled = cls.scale_coordinates(x2, LB, UB)
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(x1_scaled, x2_scaled)))

    def create(self, particles, LB, UB, iters, maxiter, reset):
        """
        Creates or updates subswarms based on the number of swarms and a chosen creation method.

        Parameters
        ----------
        particles : list
            List of Particle objects.
        LB : ndarray
            Lower bounds of the search space.
        UB : ndarray
            Upper bounds of the search space.
        iters : int
            Current iteration count (adjusted for resets).
        maxiter : int
            Maximum number of iterations (adjusted for resets).
        reset : bool
            Whether to force new swarm creation logic.

        Returns
        -------
        swarms : list of lists
            A list where each sublist is a swarm of Particle objects.
        """
        # Adjust iteration counters to skip the phase before the last reset
        iters_adj = iters - self.reset_swarms_iter
        maxiter_adj = max(1, maxiter - self.reset_swarms_iter)
        x_fraction = iters_adj / maxiter_adj

        # Decide how many swarms we should have
        self._update_number_of_swarms(x_fraction)

        # Sort particles by their best fitness
        sorted_particles = sorted(particles, key=lambda p: p.fbest)

        # If the number of swarms changed OR we have an explicit reset, create new swarms
        nswarms_before = self.nswarms
        if self.nswarms != nswarms_before or reset:
            swarms = self._create_new_swarms(sorted_particles, LB, UB)
        else:
            swarms = self._assign_existing_swarms(particles)

        # Mark the best in each swarm as elite
        self._mark_elite_and_assign_indices(swarms)

        return swarms

    def _update_number_of_swarms(self, x_fraction):
        """
        Update self.nswarms based on the chosen `method_subswarm_reduction` strategy.
        """
        nmin, nmax = self.nswarms_min, self.nswarms_max
        method = self.method_subswarm_reduction

        if nmin == nmax:
            self.nswarms = nmin
            return

        if method == 'sigmoid-2':
            # sigmoid shape with exponent -2
            val = 1. - (1. / (1. + (x_fraction / (1. - x_fraction)) ** -2))
            self.nswarms = int(round(nmin + (nmax - nmin) * val))
        elif method == 'sigmoid-3':
            # sigmoid shape with exponent -3
            val = 1. - (1. / (1. + (x_fraction / (1. - x_fraction)) ** -3))
            self.nswarms = int(round(nmin + (nmax - nmin) * val))
        elif method == 'linear':
            self.nswarms = int(round(nmin + (nmax - nmin) * (1. - x_fraction)))
        elif method == 'exponential':
            # exponential decay from nmax to nmin
            ratio = nmin / nmax if nmax != 0 else 1.0
            self.nswarms = int(round(nmax * (ratio ** x_fraction)))
        elif method == 'constant':
            self.nswarms = nmax

    def _create_new_swarms(self, sorted_particles, LB, UB):
        """
        Create brand-new swarms based on the chosen `method_subswarm_creation`.
        This discards any previous assignment.
        """
        if self.method_subswarm_creation == 'equally-distributed':
            return self._create_equally_distributed_swarms(sorted_particles)
        elif self.method_subswarm_creation == 'fitness-focused':
            return self._create_fitness_focused_swarms(sorted_particles, LB, UB)
        else:
            raise ValueError(f"Unknown method_subswarm_creation: {self.method_subswarm_creation}")

    def _assign_existing_swarms(self, particles):
        """
        Assign particles to the swarms they had before.
        If a swarm becomes empty, it is removed.
        """
        # Build empty swarm lists
        swarms = [[] for _ in range(self.nswarms)]
        for p in particles:
            swarms[p.iswarm].append(p)

        # Remove any empty swarms
        alive_swarms = []
        for iswarm, swarm in enumerate(swarms):
            if len(swarm) == 0:
                # Remove from self.swarms to keep internal consistency
                if iswarm < len(self.swarms):
                    self.swarms.pop(iswarm)
            else:
                alive_swarms.append(swarm)

        # Update the actual number of swarms
        self.nswarms = len(alive_swarms)
        return alive_swarms

    def _mark_elite_and_assign_indices(self, swarms):
        """
        Within each swarm, sort by best fitness, mark the top as `elite`,
        and assign the swarm index to each particle.
        """
        for iswarm, swarm in enumerate(swarms):
            swarm.sort(key=lambda p: p.fbest)
            if len(swarm) > 0:
                swarm[0].elite = True
            for particle in swarm:
                particle.iswarm = iswarm

    def _create_equally_distributed_swarms(self, sorted_particles):
        """
        Create subswarms by distributing the top nswarms particles as swarm leaders,
        then assign the remaining based on which swarm's leader is geographically closer.
        (Original code attempts to find the closest particle for each leader in a round-robin fashion.)
        """
        # Initialize each swarm with one top particle
        swarm_leaders = [sorted_particles.pop(0) for _ in range(self.nswarms)]
        swarms = [[leader] for leader in swarm_leaders]

        # Round-robin assignment to each sublist based on distance to its first member
        while sorted_particles:
            for sublist in swarms:
                if not sorted_particles:
                    break
                elite_particle = sublist[0]
                # Compute distances from the sublist's leader to all remaining
                distances = [
                    (self.euclidean_distance(elite_particle.xbest, candidate.xbest), candidate)
                    for candidate in sorted_particles
                ]
                distances.sort(key=lambda pair: pair[0])
                # Take the closest particle
                _, closest_particle = distances[0]
                sublist.append(closest_particle)
                sorted_particles.remove(closest_particle)

        return swarms

    def _create_fitness_focused_swarms(self, sorted_particles, LB, UB):
        """
        Create subswarms by assigning each swarm 1 best particle, then fill each swarm
        with the particles closest (in scaled space) to that best particle.
        """
        total_particles = len(sorted_particles)
        if self.nswarms <= 0:
            return []

        # Number of particles per swarm
        n_per_swarm = total_particles // self.nswarms
        remainder = total_particles % self.nswarms
        swarms = []

        for _ in range(self.nswarms):
            if not sorted_particles:
                break
            swarm = []
            # Take the top best particle for this swarm
            best_particle = sorted_particles.pop(0)
            swarm.append(best_particle)

            if remainder > 0:
                extra = 1
                remainder -= 1
            else:
                extra = 0

            # Build distance list relative to best_particle
            distances = [
                (self.euclidean_distance_scaled(best_particle.xbest, p.xbest, LB, UB), p)
                for p in sorted_particles
            ]
            distances.sort(key=lambda pair: pair[0])

            # Assign the next (n_per_swarm + extra - 1) closest
            # (minus 1 because we already used best_particle)
            n_to_assign = n_per_swarm + extra - 1
            for _ in range(n_to_assign):
                if not distances:
                    break
                _, closest = distances.pop(0)
                swarm.append(closest)
                sorted_particles.remove(closest)

            swarms.append(deepcopy(swarm))

        return swarms

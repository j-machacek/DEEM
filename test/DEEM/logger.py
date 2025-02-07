#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
import matplotlib as mlp

def cm2inch(*dimensions):
    """
    Convert a dimension or tuple of dimensions from centimeters to inches.

    Parameters
    ----------
    dimensions : float or tuple
        One or more values in centimeters.

    Returns
    -------
    tuple of float
        The converted dimensions in inches.
    """
    inch = 2.54
    if isinstance(dimensions[0], tuple):
        return tuple(value / inch for value in dimensions[0])
    else:
        return tuple(value / inch for value in dimensions)


class Logger:
    """
    Handles logging and plotting of data during DEEM optimization runs.

    Attributes
    ----------
    path : str
        Directory where figures and data are stored.
    iteration : list of int
        List of iteration indices over the run.
    fbest : list of float
        Best cost values for each iteration.
    best_position : list of array-like
        Best solution(s) found during the run.
    generations : list of list
        History of particles (or solutions) at each iteration.
    np_reset : list of int
        Number of reset particles at each iteration.
    nElite : list of int
        Number of swarms or elite count (depending on usage).
    diversity_best : list of float
        Diversity measure (e.g., position-based) for best solutions each iteration.
    diversity_gbest : list of float
        Diversity measure for global best solutions each iteration.
    diversity_gbest_swarms : list of list
        Per-swarm diversity tracking, each sublist is appended each iteration.
    upper_bound : array-like
        Upper bound(s) of the search space.
    lower_bound : array-like
        Lower bound(s) of the search space.
    FigWidth : float
        Width of figures in centimeters.
    FigHeight : float
        Height of figures in centimeters.
    my_dpi : int
        Resolution for saved figures (dots per inch).
    myfontsize : int
        Font size for all plots.

    Methods
    -------
    save(...)
        Save iteration data into the logger’s lists.
    plot_initial_distribution(particles)
        Optional function to visualize initial distribution of particles.
    plot_results()
        Generate and save summary plots of cost, diversity, and convergence history.
    """

    def __init__(self, path, lower_bound, upper_bound):
        self.path = path

        self.iteration = []
        self.fbest = []
        self.best_position = []
        self.generations = []
        self.np_reset = []
        self.convergence_history = []

        self.nElite = []
        self.diversity_best = []
        self.diversity_gbest = []
        self.diversity_gbest_swarms = []

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        # Plotting configuration
        self.FigWidth = 8.0   # Width of figures in centimeters
        self.FigHeight = 5.0  # Height of figures in centimeters
        self.my_dpi = 600
        self.myfontsize = 9

    def save(self, iteration, fbest, best_position, particles,
             nElite, np_reset, DIV_best, DIV_gbest, DIV_swarms):
        """
        Save data for a single iteration into the logger.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        fbest : float
            Best cost found at this iteration.
        best_position : array-like
            Current best position (solution) at this iteration.
        particles : list
            List of particles (or solutions) at this iteration.
        nElite : int
            Number of swarms or elite count for this iteration.
        np_reset : int
            Number of reset particles during this iteration.
        DIV_best : float
            Diversity measure for current best solutions.
        DIV_gbest : float
            Diversity measure for global best solutions.
        DIV_swarms : list
            Diversity measure(s) per swarm, appended each iteration.
        """
        self.iteration.append(iteration)
        self.fbest.append(fbest)
        self.best_position.append(best_position)
        self.generations.append(particles)
        self.nElite.append(nElite)
        self.np_reset.append(np_reset)

        self.diversity_best.append(DIV_best)
        self.diversity_gbest.append(DIV_gbest)

        # Overwrite with the most recent swarm-level diversities
        self.diversity_gbest_swarms = DIV_swarms

    def plot_initial_distribution(self, particles):
        """
        Plot the initial distribution of particles in the search space.
        Currently commented out for demonstration; you can enable and adapt
        for your own problem or dimensionality.

        Parameters
        ----------
        particles : list
            Particles (or solutions) to be plotted.
        """
        mlp.use('Agg')  # Use a non-GUI backend suitable for batch mode.
        ndim = len(self.upper_bound)

        # Example code snippet (commented)
        #
        # positions = np.zeros((len(particles), ndim))
        # costs = np.zeros(len(particles))
        # for i, p in enumerate(particles):
        #     positions[i, :] = p.x
        #     costs[i] = p.fbest
        #
        # fig, sp = plt.subplots(ndim + 1, 1, figsize=(self.FigWidth, ndim * self.FigHeight / 5))
        # # Additional plotting logic goes here...
        #
        # plt.tight_layout()
        # plt.savefig(self.path + 'initial_parameter.pdf', bbox_inches='tight', dpi=self.my_dpi)
        # plt.close(fig)

    def plot_results(self):
        """
        Generate and save summary plots: 
          1) Cost vs. iteration
          2) Diversity vs. iteration
          3) Number of reset particles vs. iteration
          4) Per-swarm diversity vs. iteration
        Also saves a simple 'history.dat' file listing the best cost per iteration.
        """
        from matplotlib import gridspec

        mlp.use('Agg')  # Use a non-GUI backend suitable for batch mode.
        mlp.rcParams['font.size'] = self.myfontsize
        mlp.rcParams['font.family'] = 'serif'
        mlp.rcParams['font.weight'] = 'light'

        # Main figure
        fig = plt.figure(figsize=(2.0 * self.FigWidth, 2.0 * self.FigHeight))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        # 1) Cost vs Iteration (linear or semilog)
        plt.subplot(gs[0])
        ax = fig.gca()
        cost_range = max(self.fbest) - min(self.fbest)
        if cost_range < 100:
            ax.plot(self.iteration, self.fbest, lw=0.75)
        else:
            ax.semilogy(self.iteration, self.fbest, lw=0.75)
        ax.set_xlabel(r'Iteration no.')
        ax.set_ylabel(r'Best cost $f(x)$')
        ax.tick_params(direction='in', which='both')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 2) Diversity vs Iteration
        plt.subplot(gs[1])
        ax = fig.gca()
        ax.plot(self.iteration, self.diversity_best, lw=0.75, label='current best position')
        ax.plot(self.iteration, self.diversity_gbest, lw=0.75, label='global best position')
        ax.set_xlabel(r'Iteration no.')
        ax.set_ylabel(r'Diversity')
        plt.legend(loc='upper right')
        ax.tick_params(direction='in', which='both')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 3) Number of reset particles
        plt.subplot(gs[2])
        ax = fig.gca()
        ax.plot(self.iteration, self.np_reset, lw=0.75)
        ax.set_xlabel(r'Iteration no.')
        ax.set_ylabel(r'Number of reset particles')
        ax.tick_params(direction='in', which='both')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 4) Per-swarm diversity
        plt.subplot(gs[3])
        ax = fig.gca()
        for iswarm, swarm_div in enumerate(self.diversity_gbest_swarms):
            ax.plot(self.iteration, swarm_div, lw=0.75, label=f'swarm = {iswarm}')
        ax.set_xlabel(r'Iteration no.')
        ax.set_ylabel(r'Diversity')
        plt.legend(loc='upper right')
        ax.tick_params(direction='in', which='both')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout(w_pad=1.1)
        plt.savefig(self.path + 'DEEM-summary.pdf', bbox_inches='tight', dpi=self.my_dpi)
        plt.close(fig)

        # Save convergence history
        with open(self.path + 'history.dat', 'w') as f:
            for val in self.fbest:
                f.write(f"{val}\n")

        # Additional plotting of best-position evolution (commented out)
        # ...
        # Example commented code for final parameter distribution
        #
        # ndim = len(self.upper_bound)
        # fig, sp = plt.subplots(ndim + 1, 1, figsize=(self.FigWidth, ndim * self.FigHeight / 5))
        # # Additional logic goes here...
        # plt.tight_layout()
        # plt.savefig(self.path + 'final_parameter.pdf', bbox_inches='tight', dpi=self.my_dpi)
        # plt.close(fig)

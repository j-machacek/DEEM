#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Jan Machaček
#
# This file is part of DEEM, released under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.

"""
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                            Copyright (C) 2023-2025 Jan Machacek  
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 05.02.2023
#
# DEEM - Differential Evolution with Elitism and Multi-populations
#
# Handles logging and plotting of data during DEEM optimization runs.
# Also writes a summary file reporting the best fitness value and corresponding 
# parameters for each iteration.
# 
# History:
# 05.02.2023, J. Machacek - Initial version
# 07.02.2025, J. Machacek - Refactored code
# 23.06.2026, J. Machacek - A) save() no longer retains the full candidate objects
#                              every iteration (memory grew with maxiter*nparticles
#                              and pinned the objective); only the population size is
#                              kept. B) plot_results() guards against a changing
#                              number of subpopulations (ragged diversity series) and
#                              against empty histories, so logging can never abort a run.
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Any, Sequence, Union

def cm2inch(*dimensions: Union[float, Sequence[float]]) -> tuple:
    """
    Convert one or more dimensions from centimeters to inches.

    Parameters
    ----------
    dimensions : float or sequence of float
        One or more values in centimeters.

    Returns
    -------
    tuple of float
        The converted dimensions in inches.
    """
    inch = 2.54
    if isinstance(dimensions[0], (list, tuple, np.ndarray)):
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
    iteration : List[int]
        List of iteration indices.
    fbest : List[float]
        Best cost values for each iteration.
    best_position : List[Any]
        Best solution(s) found at each iteration.
    generations : List[List[Any]]
        History of candidate solutions at each iteration.
    np_reset : List[int]
        Number of reset candidate solutions at each iteration.
    nElite : List[int]
        Number of elite candidates (or swarms) at each iteration.
    diversity_best : List[float]
        Diversity measure for the current best candidates.
    diversity_gbest : List[float]
        Diversity measure for the global best candidates.
    diversity_gbest_swarms : List[List[float]]
        Per-subpopulation diversity tracking.
    upper_bound : np.ndarray
        Upper bounds of the search space.
    lower_bound : np.ndarray
        Lower bounds of the search space.
    FigWidth : float
        Width of figures in centimeters.
    FigHeight : float
        Height of figures in centimeters.
    my_dpi : int
        Resolution for saved figures (dots per inch).
    myfontsize : int
        Font size for plots.
    """

    def __init__(self, path: str, lower_bound: Sequence[float], upper_bound: Sequence[float]) -> None:
        self.path: str = path
        self.iteration: List[int] = []
        self.fbest: List[float] = []
        self.best_position: List[Any] = []
        self.generations: List[List[Any]] = []
        self.np_reset: List[int] = []
        self.convergence_history: List[Any] = []

        self.nElite: List[int] = []
        self.diversity_best: List[float] = []
        self.diversity_gbest: List[float] = []
        self.diversity_gbest_swarms: List[List[float]] = []

        self.lower_bound: np.ndarray = np.array(lower_bound)
        self.upper_bound: np.ndarray = np.array(upper_bound)

        # Plotting configuration (dimensions in centimeters)
        self.FigWidth: float = 8.0
        self.FigHeight: float = 5.0
        self.my_dpi: int = 600
        self.myfontsize: int = 9

    def save(self, iteration: int, fbest: float, best_position: Any, particles: List[Any],
             nElite: int, np_reset: int, DIV_best: float, DIV_gbest: float,
             DIV_swarms: List[float]) -> None:
        """
        Save data for a single iteration into the logger.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        fbest : float
            Best cost found at this iteration.
        best_position : Any
            Best solution (position) at this iteration.
        particles : List[Any]
            List of candidate solutions at this iteration.
        nElite : int
            Number of elite candidates (or swarms) for this iteration.
        np_reset : int
            Number of reset candidate solutions during this iteration.
        DIV_best : float
            Diversity measure for current best candidates.
        DIV_gbest : float
            Diversity measure for global best candidates.
        DIV_swarms : List[float]
            Diversity measure(s) per subpopulation.
        """
        self.iteration.append(iteration)
        self.fbest.append(fbest)
        self.best_position.append(best_position)
        # 23.06.2026, J. Machacek - keep only the population size, not the heavy
        # candidate objects (avoids unbounded memory growth and pinning the objective).
        self.generations.append(len(particles) if particles is not None else 0)
        self.nElite.append(nElite)
        self.np_reset.append(np_reset)
        self.diversity_best.append(DIV_best)
        self.diversity_gbest.append(DIV_gbest)
        self.diversity_gbest_swarms = DIV_swarms

    def plot_initial_distribution(self, particles: List[Any]) -> None:
        """
        Plot the initial distribution of candidate solutions in the search space.
        This function is provided as a template and is currently commented out.
        Adapt the code as needed for your specific problem.

        Parameters
        ----------
        particles : List[Any]
            List of candidate solutions to be plotted.
        """
        mpl.use('Agg')  # Use a non-GUI backend for batch mode.
        ndim = len(self.upper_bound)
        # Example (uncomment and modify as needed):
        #
        # positions = np.array([p.x for p in particles])
        # fig, ax = plt.subplots(figsize=cm2inch(self.FigWidth, ndim * self.FigHeight / 5))
        # ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=10)
        # ax.set_title("Initial Distribution")
        # plt.tight_layout()
        # plt.savefig(self.path + 'initial_distribution.pdf', dpi=self.my_dpi, bbox_inches='tight')
        # plt.close(fig)
        pass

    def plot_results(self) -> None:
        """
        Generate and save summary plots:
          1) Cost vs. iteration
          2) Diversity vs. iteration
          3) Number of reset candidate solutions vs. iteration
          4) Per-subpopulation diversity vs. iteration

        Also saves two files:
          - 'history.dat': a simple text file with the best cost per iteration.
          - 'iteration_summary.dat': a file listing the iteration number, best fitness, 
            and the corresponding best parameters (comma-separated).
        """
        from matplotlib import gridspec
        mpl.use('Agg')
        mpl.rcParams['font.size'] = self.myfontsize
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.weight'] = 'light'

        fig = plt.figure(figsize=(2.0 * self.FigWidth, 2.0 * self.FigHeight))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        # 1) Cost vs. Iteration (linear or semilog)
        ax1 = fig.add_subplot(gs[0])
        if self.fbest:
            cost_range = max(self.fbest) - min(self.fbest)
            if cost_range < 100:
                ax1.plot(self.iteration, self.fbest, lw=0.75)
            else:
                ax1.semilogy(self.iteration, self.fbest, lw=0.75)
        ax1.set_xlabel('Iteration no.')
        ax1.set_ylabel('Best cost f(x)')
        ax1.tick_params(direction='in')
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # 2) Diversity vs. Iteration
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.iteration, self.diversity_best, lw=0.75, label='current best')
        ax2.plot(self.iteration, self.diversity_gbest, lw=0.75, label='global best')
        ax2.set_xlabel('Iteration no.')
        ax2.set_ylabel('Diversity')
        ax2.legend(loc='upper right')
        ax2.tick_params(direction='in')
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # 3) Number of reset candidate solutions vs. Iteration
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(self.iteration, self.np_reset, lw=0.75)
        ax3.set_xlabel('Iteration no.')
        ax3.set_ylabel('Reset candidates')
        ax3.tick_params(direction='in')
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # 4) Per-subpopulation diversity vs. Iteration
        ax4 = fig.add_subplot(gs[3])
        # 23.06.2026, J. Machacek - the number of subpopulations may change over the
        # run, so each series can be shorter than the iteration axis. Plot only the
        # matching tail to avoid a length-mismatch error.
        it = np.asarray(self.iteration)
        for i, div in enumerate(self.diversity_gbest_swarms):
            div = np.asarray(div)
            m = min(len(div), len(it))
            if m > 0:
                ax4.plot(it[-m:], div[-m:], lw=0.75, label=f'subpop {i}')
        ax4.set_xlabel('Iteration no.')
        ax4.set_ylabel('Diversity')
        if self.diversity_gbest_swarms:
            ax4.legend(loc='upper right')
        ax4.tick_params(direction='in')
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        plt.tight_layout(w_pad=1.1)
        plt.savefig(self.path + 'DEEM-summary.pdf', bbox_inches='tight', dpi=self.my_dpi)
        plt.close(fig)

        # Save convergence history (only best cost per iteration)
        with open(self.path + 'history.dat', 'w') as f:
            for val in self.fbest:
                f.write(f"{val}\n")

        # Save iteration summary (iteration number, best fitness, best parameters)
        with open(self.path + 'iteration_summary.dat', 'w') as f:
            header = "Iteration\tBestFitness\tBestParameters\n"
            f.write(header)
            for iter_num, fitness, best_pos in zip(self.iteration, self.fbest, self.best_position):
                # Convert best_pos (an array) to a comma-separated string
                pos_str = ",".join(f"{x:.6g}" for x in best_pos)
                f.write(f"{iter_num}\t{fitness:.6g}\t{pos_str}\n")

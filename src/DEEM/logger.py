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
import matplotlib.pyplot as plt
import matplotlib as mlp

def cm2inch(*tupl):
    '''
    Transform values (v1,v2,v3,...) given in cm to inch
    '''
    # History:
    # 26.10.2021 - J. Machacek, Initial version

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


class Logger:

    def __init__(self,path,lower_bound,upper_bound):

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

        self.FigWidth = 8.  # width of figures in cm
        self.FigHeight = 5. # height of figures in cm
        self.my_dpi = 600
        self.myfontsize = 9


    def save(self,iteration,fbest,best_position,particles,nElite,np_reset,DIV_best,DIV_gbest,DIV_swarms):
        self.iteration.append(iteration)
        self.fbest.append(fbest)
        self.best_position.append(best_position)
        self.generations.append(particles)
        self.nElite.append(nElite)
        self.np_reset.append(np_reset)

        self.diversity_best.append(DIV_best)
        self.diversity_gbest.append(DIV_gbest)

        self.diversity_gbest_swarms = DIV_swarms


    def plot_initial_distribution(self,particles):

        mlp.use('Agg')
        ndim = len(self.upper_bound)

        # positions = np.zeros((len(particles),ndim)) ; cost = np.zeros(len(particles))
        # counter = 0
        # for p in particles:
        #     positions[counter,:] = p.x ; cost[counter] = p.fbest ; counter += 1

        # fig, sp = plt.subplots(ndim+1, 1, figsize=(self.FigWidth,ndim*self.FigHeight/5))

        # #fig.suptitle('Initial position of particles $\mathbf{x}=(x_1,...,x_n)$, with $n=$ %d' % (ndim), fontsize=self.myfontsize, y=1.3)

        # # make a first empty plot to create space for the color bar
        # sp[0].spines["bottom"].set_visible(False)
        # sp[0].spines["left"].set_visible(False)
        # sp[0].spines["top"].set_visible(False)
        # sp[0].spines["right"].set_visible(False)
        # sp[0].get_yaxis().set_ticks([])
        # sp[0].get_xaxis().set_ticks([])

        # for idim in range(0,ndim):
            
        #     sc = sp[idim+1].scatter(positions[:,idim], positions[:,idim]*0., c=cost, s=10, cmap='viridis')
        #     sp[idim+1].vlines((self.lower_bound[idim],self.upper_bound[idim]), -1, 1, color='r', lw=1.5)
        #     sp[idim+1].set_xlim((self.lower_bound[idim],self.upper_bound[idim]))
        #     sp[idim+1].set_ylim((-0.01,0.01))
        #     sp[idim+1].get_yaxis().set_ticks([])
        #     sp[idim+1].set_xlabel(r'Component $x_{{{:2d}}}$'.format(idim+1))
        #     sp[idim+1].tick_params(direction='in', which='both') 
        #     sp[idim+1].spines['bottom'].set_position('center')
        #     sp[idim+1].spines["left"].set_visible(False)
        #     sp[idim+1].spines["top"].set_visible(False)
        #     sp[idim+1].spines["right"].set_visible(False)

        # cbar = fig.colorbar(sc, ax=sp, location='top', orientation='horizontal', pad=-0.2, aspect=80)
        # cbar.ax.set_title('Initial position $\mathbf{x}=(x_1,...,x_n)$ and objective function value $\epsilon$ of particles (color bar)', fontsize=self.myfontsize)

        # plt.tight_layout(h_pad=1.25)
        # plt.savefig(self.path + 'initial_parameter.pdf', bbox_inches='tight', dpi=self.my_dpi)
        # plt.close(fig)


    def plot_results(self):

        from matplotlib import gridspec

        mlp.use('Agg')
        mlp.rcParams['font.size'] = self.myfontsize
        mlp.rcParams['font.family'] = 'serif'
        mlp.rcParams['font.weight'] = 'light'

        #
        # COST VS ITERATION
        #

        fig = plt.figure(figsize=(2.*self.FigWidth,2.*self.FigHeight))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1]*2, width_ratios=[1]*2)

        plt.subplot(gs[0])
        if (max(self.fbest)-min(self.fbest)) < 100:
            ax = fig.gca()
            ax.plot(self.iteration, self.fbest, lw=0.75)
            ax.set_xlabel(r'Iteration no.')
            ax.set_ylabel(r'Best cost $f(x)$')
            ax.tick_params(direction='in', which='both') 
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax = fig.gca()
            ax.semilogy(self.iteration, self.fbest, lw=0.75)
            ax.set_xlabel(r'Iteration no.')
            ax.set_ylabel(r'Best cost $f(x)$')
            ax.tick_params(direction='in', which='both') 
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

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

        plt.subplot(gs[2])
        ax = fig.gca()
        ax.plot(self.iteration, self.np_reset, lw=0.75)
        ax.set_xlabel(r'Iteration no.')
        ax.set_ylabel(r'Number of reset particles')
        ax.tick_params(direction='in', which='both') 
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.subplot(gs[3])
        ax = fig.gca()
        for iswarm, swarm in enumerate(self.diversity_gbest_swarms):
            ax.plot(self.iteration, swarm, lw=0.75, label='swarm = ' + str(iswarm))
        ax.set_xlabel(r'Iteration no.')
        ax.set_ylabel(r'Diversity')
        plt.legend(loc='upper right')
        ax.tick_params(direction='in', which='both') 
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout(w_pad=1.1)
        plt.savefig(self.path + 'DEEM-summary.pdf', bbox_inches='tight', dpi=self.my_dpi)
        plt.close(fig)
        
        #
        # CONVERGENCE HISTORY
        #
        
        with open(self.path + r'history.dat', 'w') as f:
          for item in self.fbest:
              f.write("%s\n" % item)


        # #
        # # COST VS ITERATION
        # #

        # ndim = len(self.upper_bound)

        # fig, sp = plt.subplots(ndim+1, 1, figsize=(self.FigWidth,ndim*self.FigHeight/5))

        # fig.suptitle('Evolution of best particle position $\mathbf{x}=(x_1,...,x_n)$, with $n=$ %d' % (ndim), fontsize=self.myfontsize)

        # # make a first empty plot to create space for the color bar
        # sp[0].spines["bottom"].set_visible(False)
        # sp[0].spines["left"].set_visible(False)
        # sp[0].spines["top"].set_visible(False)
        # sp[0].spines["right"].set_visible(False)
        # sp[0].get_yaxis().set_ticks([])
        # sp[0].get_xaxis().set_ticks([])

        # for idim in range(0,ndim):
        #     sp[idim+1].plot(self.best_position[:][idim], np.zeros(len(self.best_position[:][idim])), ls='', marker='o', markersize=7, color='gray')
        #     sp[idim+1].plot(self.best_position[-1][idim], 0., ls='', marker='o', markersize=10, color='C0')
        #     sp[idim+1].vlines((self.lower_bound[idim],self.upper_bound[idim]), 0., 0.001, color='r', lw=1.5)

        #     sp[idim+1].annotate("{:.2f}".format(self.best_position[-1][idim]),
        #          (self.best_position[-1][idim],0.), 
        #          textcoords="offset points", 
        #          xytext=(0,10),
        #          ha='center')

        #     sp[idim+1].set_xlim((self.lower_bound[idim],self.upper_bound[idim]))
        #     sp[idim+1].set_ylim((-0.001,0.001))
        #     sp[idim+1].get_yaxis().set_ticks([])
        #     sp[idim+1].set_xlabel(r'Component $x_{{{:2d}}}$'.format(idim+1))
        #     sp[idim+1].tick_params(direction='in', which='both') 
        #     sp[idim+1].spines['bottom'].set_position('center')
        #     sp[idim+1].spines["left"].set_visible(False)
        #     sp[idim+1].spines["top"].set_visible(False)
        #     sp[idim+1].spines["right"].set_visible(False)

        # plt.tight_layout(h_pad=1.25)
        # plt.savefig(self.path + 'final_parameter.pdf', bbox_inches='tight', dpi=self.my_dpi)
        # plt.close(fig)

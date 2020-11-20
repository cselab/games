#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys

import pretty_errors

n = 33
Ns = 60
N_epoch = 20000

beta = np.linspace(3,0,Ns)

G = gr.lattice_von_neumann(n)

N_nodes = G.number_of_nodes()
N_tags = 1

LHM = np.zeros((3,Ns,N_tags))

fig, ax = plt.subplots()


for k in range(Ns):
    print(f'Running {k} out of {Ns} betas...')

    game = bargain(G, beta=beta[k], J0=[ 4, 4, 4 ], N_tags=N_tags)

    game.plot_statistics_init()

    game.play(N_epoch)

    game.plot_statistics()

    LHM[0,k,:] = game.statistics['per_L'][-1]/N_nodes
    LHM[1,k,:] = game.statistics['per_M'][-1]/N_nodes
    LHM[2,k,:] = game.statistics['per_H'][-1]/N_nodes

    G = game.G
    plt.figure(fig.number)
    ax.clear()
    ax.plot(beta, LHM[0,:,:], 'ro-', label='Low')
    ax.plot(beta, LHM[1,:,:], 'go-', label='Med')
    ax.plot(beta, LHM[2,:,:], 'bo-', label='Hig')

    ax.set_xlabel('beta')
    ax.set_ylabel('percentages of actions ')
    plt.pause(0.005)
    plt.show(block=False)


plt.show()

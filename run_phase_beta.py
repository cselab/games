#!/usr/bin/env python3
from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys

import pretty_errors

n = 10
Ns = 60

beta = np.linspace(3, 0, Ns)

# G = gr.lattice_von_neumann(n)
G = gr.off_lattice(n*n)

N_nodes = G.number_of_nodes()
N_tags = 1

LHM = np.zeros((3, Ns, N_tags))

fig, ax = plt.subplots()

for k in range(Ns):
    print(f'Running {k} out of {Ns} betas...')

    game = bargain(G, beta=beta[k], J0=[ 4, 4, 4 ], N_tags=N_tags)

    game.play(N_epochs=10,N_per_epoch=100)

    game.plot_statistics()

    game.copy_data_to_graph()
    G = game.G

    LHM[0, k, :] = np.sum( game.actions==0 ) / N_nodes
    LHM[1, k, :] = np.sum( game.actions==1 ) / N_nodes
    LHM[2, k, :] = np.sum( game.actions==2 ) / N_nodes

    plt.figure(fig.number)
    ax.clear()
    ax.plot(beta, LHM[0, :, :], 'ro-', label='Low', markersize=3)
    ax.plot(beta, LHM[1, :, :], 'go-', label='Med', markersize=3)
    ax.plot(beta, LHM[2, :, :], 'bo-', label='Hig', markersize=3)

    ax.set_xlabel('beta')
    ax.set_ylabel('percentages of actions ')
    plt.pause(0.005)
    plt.show(block=False)

for tag in range(N_tags):
    name ='phase_beta_tag_' + str(tag) + '.txt'
    np.savetxt(name, np.squeeze(LHM[tag,:,:]), delimiter=' ')

plt.show()

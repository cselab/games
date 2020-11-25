#!/usr/bin/env python3
from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys

import pretty_errors

# n = 33
# Ns = 60
# beta0 = 0.
# beta1 = 3.
# N_tags = 1
# J0 = [ 4, 4, 4 ]
# G = gr.lattice_von_neumann(n)

n = 33
Ns = 60
beta = 2.
lamda0 = 0.05
lamda1 = 0.55
gamma = 0.1
N_tags = 1
J0 = [ 4, 4, 4 ]
G = gr.lattice_von_neumann(n)

lamdas = np.linspace(lamda0, lamda1, Ns)

N_nodes = G.number_of_nodes()
LHM = np.zeros((3, Ns, N_tags))

fig, ax = plt.subplots()

N_epochs = 200

for k in range(Ns):
    print(f'Running {k} out of {Ns} lamdas...')

    game = bargain(G, beta=beta, lamda=lamdas[k], gamma=gamma, J0=J0, N_tags=N_tags)

    game.play(N_epochs=N_epochs)

    game.plot_statistics()

    LHM[0, k, :] = np.sum(game.actions == 0) / N_nodes
    LHM[1, k, :] = np.sum(game.actions == 1) / N_nodes
    LHM[2, k, :] = np.sum(game.actions == 2) / N_nodes

    plt.figure(fig.number)
    ax.clear()
    ax.plot(lamdas, LHM[0, :, :], 'ro-', label='Low', markersize=3)
    ax.plot(lamdas, LHM[1, :, :], 'go-', label='Med', markersize=3)
    ax.plot(lamdas, LHM[2, :, :], 'bo-', label='Hig', markersize=3)

    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('percentages of actions ')
    # plt.pause(0.005)
    # plt.show(block=False)

plt.savefig('phase_lamda.eps', dpi=150)

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
# beta = np.linspace(beta1, beta0, Ns)

n = 33
beta0 = 0.
beta1 = 2.
N_tags = 1
# J0 = [ 1, 10, 1 ]
J0 = [ 6, 0, 3 ]
G = gr.off_lattice(n * n)
# Ns = 60
# beta = np.linspace(beta1, beta0, Ns)
Ns = 14 + 20 + 20
x1 = np.linspace(beta1, 1.5, 10)
x2 = np.linspace(1.5, 1.25, 20)
x3 = np.linspace(1.25, beta0, 20)
beta = np.concatenate((x1, x2, x3))
Ns = beta.shape[0]

N_nodes = G.number_of_nodes()
LHM = np.zeros((3, Ns, N_tags))

fig, ax = plt.subplots()

for k in range(Ns):
    print(f'Running {k} out of {Ns} betas...')

    game = bargain(G, beta=beta[k], J0=J0, N_tags=N_tags)

    if k == 0:
        N_per_epoch = int(5e5)
    else:
        N_per_epoch = int(1e5)

    game.play(N_epochs=100, N_per_epoch=N_per_epoch)

    game.plot_statistics()

    game.copy_data_to_graph()
    G = game.G

    LHM[0, k, :] = np.sum(game.actions == 0) / N_nodes
    LHM[1, k, :] = np.sum(game.actions == 1) / N_nodes
    LHM[2, k, :] = np.sum(game.actions == 2) / N_nodes

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
    name = 'phase_beta_tag_' + str(tag) + '.txt'
    z = (beta[:, np.newaxis], np.squeeze(LHM[:, :, tag]).T)
    z = np.concatenate(z, axis=1)
    np.savetxt(name, z, delimiter=' ')

plt.savefig('phase_beta.eps', dpi=150)

plt.show()

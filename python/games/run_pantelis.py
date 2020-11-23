#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

########################
# Figure 1
########################
# 1000 iterations per agent
n = 33
beta = 0.3
gamma = 0.1
N_tags = 1
J0 = [ 4, 4, 4 ]

lattice = "lattice_von_neumann"


# n = 6
# beta = 0.3
# beta = 40.0

node_size = 100

# J0 = [ 1, 4, 1 ]
# J0 = [ 4, 1, 4 ]
# J0 = [ 4, 4, 4 ]

# lattice = "off_lattice"


if lattice == "off_lattice":
    G = gr.off_lattice(n * n)
elif lattice == "lattice_von_neumann":
    G = gr.lattice_von_neumann(n)

run_name = "_results_n{:}_beta{:}_gamma{:}_J0{:}_lattice{:}_Ν{:}".format(n, beta, gamma, J0, lattice, N_tags)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name, N_tags=N_tags)

for k in range(1):
    game.play(N_epochs=1000)
    game.plot_graph(node_size=node_size)
    game.plot_statistics()

# game.plot_graph()
# game.plot_graph(node_size=node_size)

# plt.show()


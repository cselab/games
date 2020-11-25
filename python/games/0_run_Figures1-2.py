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
# Figure 1a (upper left)
n = 33
beta = 0.3
gamma = 0.1
N_tags = 1
J0 = [ 4, 4, 4 ]
N_epochs = 2000
num_runs = 1
lattice = "lattice_von_neumann"
node_size = 100

# Figure 1b (upper right)
n = 33
beta = 1.0
gamma = 0.1
N_tags = 1
J0 = [ 4, 4, 4 ]
N_epochs = 2000
num_runs = 1
lattice = "lattice_von_neumann"
node_size = 100

# Figure 1c (lower left)
n = 33
beta = 2.0
gamma = 0.1
N_tags = 1
J0 = [ 4, 1, 4 ]
N_epochs = 2000
num_runs = 1
lattice = "lattice_von_neumann"
node_size = 100

# Figure 1d (lower right)
n = 33
beta = 2.0
gamma = 0.1
N_tags = 1
J0 = [ 4, 4, 4 ]
N_epochs = 2000
num_runs = 1
lattice = "lattice_von_neumann"
node_size = 100

# Figure 2a
n = 33
beta = 2.0
gamma = 0.1
N_tags = 1
J0 = [ 1, 4, 1 ]
N_epochs = 2000
num_runs = 1
lattice = "lattice_von_neumann"
node_size = 100

if lattice == "off_lattice":
    G = gr.off_lattice(n * n)
elif lattice == "lattice_von_neumann":
    G = gr.lattice_von_neumann(n)

run_name = "_results_n={:}_beta={:}_gamma={:}_J0={:}_lattice={:}_Ν={:}".format(n, beta, gamma, J0, lattice, N_tags)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name, N_tags=N_tags)

for k in range(num_runs):
    game.play(N_epochs=N_epochs)
    # game.plot_graph(node_size=node_size)
    # game.plot_statistics()

game.plot_graph(node_size=node_size)
game.plot_statistics()

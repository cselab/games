#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

n           = 36
beta        = 1.2
gamma       = 0.1
N_tags      = 2
J0          = [ 4, 1, 4 ]
# N_epochs    = 600
N_epochs    = 2000
num_runs    = 1
lattice     = "off_lattice"
node_size   = 1000
seed        = 50

# n           = 36
# beta        = 1.0
# gamma       = 0.1
# N_tags      = 2
# J0          = [ 4, 1, 4 ]
# N_epochs    = 500
# num_runs    = 4
# lattice     = "off_lattice"
# node_size   = 1000
# seed        = 1


if lattice == "off_lattice":
    G = gr.off_lattice(n * n)
elif lattice == "lattice_von_neumann":
    G = gr.lattice_von_neumann(n)

run_name = "_results_n={:}_beta={:}_gamma={:}_J0={:}_lattice={:}_Ν={:}".format(n, beta, gamma, J0, lattice, N_tags)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name, N_tags=N_tags, seed=seed)

for k in range(num_runs):
    game.play(N_epochs=N_epochs)
    # game.plot_graph(node_size=node_size)
    # game.plot_statistics()
    # game.plot_simplex()

game.plot_statistics()
game.plot_simplex()
# game.plot_graph(node_size=node_size)


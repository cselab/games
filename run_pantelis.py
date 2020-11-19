#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

n = 6
# n = 33
beta = 2.0
# beta = 0.3
# beta = 40.0
gamma = 0.1

# J0 = [ 1, 4, 1 ]
# J0 = [ 4, 1, 4 ]
J0 = [ 4, 4, 4 ]

# lattice = "off_lattice"
lattice = "lattice_von_neumann"

if lattice == "off_lattice":
    G = gr.off_lattice(n * n)
elif lattice == "lattice_von_neumann":
    G = gr.lattice_von_neumann(n)

run_name = "_results_{:}_{:}_{:}_{:}_{:}".format(n, beta, gamma, J0, lattice)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name)

num_epochs = 200
game.play(num_epochs)
game.plot_statistics()

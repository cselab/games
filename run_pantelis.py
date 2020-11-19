#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# n = 6
n = 33
beta = 2.0
gamma = 0.1

# J0 = [ 1, 4, 1 ]
J0 = [ 4, 1, 4 ]
# J0 = [ 4, 4, 4 ]
run_name = "_results"

G = gr.lattice_von_neumann(n)
# G = gr.off_lattice(n * n)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name)

num_epochs = 1
game.play(num_epochs)
game.plot_statistics()

game.plot_init()
game.plot()
plt.show()

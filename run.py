#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import pretty_errors

n = 20
beta = 4
gamma = 0.1

node_size = 200

# G = gr.lattice_von_neumann(n)
# G = gr.lattice_moore(n)
G = gr.barabasi_albert(n * n, k=2)
# G = gr.path(n*n)
# G = gr.off_lattice(n * n)

game = bargain(G, beta=2., J0=[ 4, 4, 4 ], N_tags=4)

game.plot_init()

game.plot(node_size=node_size)

for k in range(10):
    game.play(100)
    game.plot(node_size=node_size, silent=True)
    game.plot_statistics()

plt.show()

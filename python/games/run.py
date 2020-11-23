#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import pretty_errors

n = 33
node_size = 100

# G = gr.lattice_von_neumann(n)
# G = gr.lattice_moore(n)
# G = gr.barabasi_albert(n * n, k=2)
# G = gr.path(n*n)
G = gr.off_lattice(n * n)

game = bargain(G, beta=0.25, J0=[ 0, 10, 0 ], N_tags=1)

# game.plot_graph(node_size=node_size)

for k in range(1):
    game.play(N_epochs=100, N_per_epoch=10000)
    game.plot_graph(node_size=node_size)
    game.plot_statistics()

plt.show()

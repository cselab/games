#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import pretty_errors

n = 10
node_size = 100

G = gr.lattice_von_neumann(n)
# G = gr.lattice_moore(n)
# G = gr.barabasi_albert(n * n, k=2)
# G = gr.path(n*n)
# G = gr.off_lattice(n * n)

game = bargain(G, beta=2., J0=[ 4, 1, 4 ], N_tags=1)

# game.plot_graph(node_size=node_size)

for k in range(5):
    game.play(N_epochs=1, N_per_epoch=100)
    # game.plot_graph(node_size=node_size)
    game.plot_statistics()

plt.show()

#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

n = 10
beta = 40
gamma = 0.1

G = gr.lattice_von_neumann(n)
# G = gr.path(n)
# G = gr.off_lattice(n * n)

game = bargain(G, beta=2., J0=[ 4, 1, 4 ])

game.plot_init()

game.plot(with_labels=False)

for k in range(50):
    game.play(1000)
    game.plot()
    game.plot_statistics()


plt.show()

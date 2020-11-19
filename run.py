#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np

n = 10
beta = 4
gamma = 0.1

G = gr.lattice_von_neumann(n)
# G = gr.path(n)

game = bargain(G, beta=2.)

game.plot_init()
game.plot(with_labels=False)

for k in range(30):
    game.play(100)
    game.plot()

plt.show()

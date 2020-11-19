#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt

n = 33
beta = 4
gamma = 0.1

G = gr.lattice_von_neumann(n)

game = bargain(G,beta=2.)

game.plot_init()
game.plot()

for k in range(100):
    game.play(100)
    game.plot()

plt.show()

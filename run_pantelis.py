#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

run_name = "example"
n = 6

beta = 4.0
gamma = 0.1

J0 = [ 4, 1, 4 ]

G = gr.lattice_von_neumann(n)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name)

num_epochs = 10
game.play(num_epochs)
game.plot_statistics()


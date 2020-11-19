#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

n = 6
beta = 2.0
gamma = 0.1

J0 = [ 4, 4, 4 ]
run_name = "example"

G = gr.lattice_von_neumann(n)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name)

num_epochs = 2000
game.play(num_epochs)
game.plot_statistics()


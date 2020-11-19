#!/usr/bin/env python3

from games import bargain
import graphs as gr
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

run_name = "example"
n = 33

beta = 4.0
gamma = 0.1

J0 = [ 4, 4, 4 ]

G = gr.lattice_von_neumann(n)

game = bargain(G, beta=beta, gamma=gamma, J0=J0, folder=run_name)

num_epochs = 10
game.play(num_epochs)
game.plot_statistics()



# p = self.statistics["p_all"]
# p = np.array(p)
# print(np.shape(p))

# den = np.sum(p, axis=2)

# # (J2 + J3/2)/(J1+J2+J3)
# p_x = (p[:,:,1] + p[:,:,2]/2.0) / den
# # \sqrt(3) * J3/2 /(J1+J2+J3)
# p_y = np.sqrt(3) * p[:,:,2] / 2.0 / den

# fig, ax = plt.subplots(figsize=fig_size)
# fig_path = self.results_folder + "/simplex"
# for particle in range(np.shape(p_x)[1]):
#     x = p_x[:,particle]
#     y = p_y[:,particle]
#     ax.plot(x, y)
# plt.savefig(fig_path)

# print(np.shape(p_y))



#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx

n = 5
# G = nx.erdos_renyi_graph(100, 0.15)
# G = nx.triangular_lattice_graph(10,10)
G = nx.grid_2d_graph(n, n)
# G = nx.hexagonal_lattice_graph(4,4)
# G = nx.path_graph(11)
# G = nx.watts_strogatz_graph(20, 4, 0.4)
G = nx.barabasi_albert_graph(50, 1)

pos = nx.circular_layout(G)
pos = nx.spring_layout(G, iterations=100)
# pos = nx.kamada_kawai_layout(G, pos=pos)

# G = nx.complete_multipartite_graph(8, 4, 8)
# pos = nx.multipartite_layout(G)

nx.draw(G, pos=pos, with_labels=True, node_size=300)

plt.show()

# from games import random_choice
# import time
#
# p = np.array([0.1,0.4, 0.1, 0.2, 0.2])
# N = 100000
#
# start = time.time()
# Q = np.random.uniform(low=0.0, high=1.0, size=(N,))
# for k in range(N):
#     x = random_choice(p, Q[k])
# end = time.time()
# print(end - start)
#
# for k in range(N):
#      x = np.random.choice(5, size=1, p=p )
# end = time.time()
# print(end - start)

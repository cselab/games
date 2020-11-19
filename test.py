#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx

# G = nx.erdos_renyi_graph(100, 0.15)
# G = nx.triangular_lattice_graph(10,10)
# G = nx.grid_2d_graph(5,5)
# G = nx.hexagonal_lattice_graph(4,4)
G = nx.path_graph(11)


pos =  nx.spring_layout(G, iterations=100)
pos = nx.kamada_kawai_layout(G, pos=pos)


# G = nx.complete_multipartite_graph(8, 4, 8)
# pos = nx.multipartite_layout(G)


nx.draw( G, pos=pos, with_labels=True, node_size=300 )

plt.show()

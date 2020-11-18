#!/usr/bin/env python3
import networkx as nx




def grid_von_neumann(n):
    G = nx.grid_2d_graph(n, n)
    return G

    # labels = {}
    # for k,x in enumerate(G):
    #     labels[x] = k
    # G = nx.relabel_nodes(G, labels)

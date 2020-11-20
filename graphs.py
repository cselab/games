#!/usr/bin/env python3
import networkx as nx


def lattice_von_neumann(n):
    G = nx.grid_2d_graph(n, n)
    G = relabel(G)
    return G


def lattice_moore(n):
    G = nx.grid_2d_graph(n, n)
    for i in range(0, n - 1):
        for j in range(1, n - 1):
            G.add_edge((i, j), (i + 1, j + 1))
            G.add_edge((i, j), (i + 1, j - 1))

    for i in range(0, n - 1):
        G.add_edge((i, 0), (i + 1, 1))
        G.add_edge((i, n - 1), (i + 1, n - 2))
    G = relabel(G)
    return G


def lattice_hexagonal(n):
    G = nx.hexagonal_lattice_graph(n, n)
    G = relabel(G)
    return G


def lattice_triangular(n):
    G = nx.triangular_lattice_graph(n, n)
    G = relabel(G)
    return G


def watts_strogatz(n, k=4, p=0.1):
    G = nx.watts_strogatz_graph(n, k, p)
    G = relabel(G)
    return G


def barabasi_albert(n, k=1):
    G = nx.barabasi_albert_graph(n, k)
    G = relabel(G)
    return G


def path(n):
    G = nx.path_graph(n)
    G = relabel(G)
    return G


def off_lattice(n):
    G = nx.empty_graph(n)
    G = relabel(G)
    return G


def relabel(G):
    labels = {}
    for k, x in enumerate(G):
        labels[x] = k
    G = nx.relabel_nodes(G, labels)
    return G

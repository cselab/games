#!/usr/bin/env python3
import networkx as nx


def lattice_von_neumann(n):
    G = nx.grid_2d_graph(n, n)
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


def path(n):
    G = nx.path_graph(n)
    G = relabel(G)
    return G


def relabel(G):
    labels = {}
    for k, x in enumerate(G):
        labels[x] = k
    G = nx.relabel_nodes(G, labels)
    return G

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys


def random_choice(p, r):
    return np.argmax(np.cumsum(p) > r)


class bargain:

    def __init__(self, G, beta=1., gamma=0.1, seed=None, J0=[ 4, 4, 4 ], payoff=None):

        self.G = G
        self.beta = beta
        self.gamma = gamma

        np.random.seed(seed)

        self.N_nodes = G.number_of_nodes()

        self.positions = None

        J0 = np.array(J0)
        for node in G:
            G.nodes[node]['J'] = np.random.randint(0, J0 + 1).astype('float64')
            G.nodes[node]['P'] = np.exp(self.beta * G.nodes[node]['J'])
            G.nodes[node]['P'] /= np.sum(G.nodes[node]['P'])

        if payoff == None:
            self.payoff = np.zeros((3, 3))
            self.payoff[0] = [ 0.3, 0.3, 0.3 ]
            self.payoff[1] = [ 0.5, 0.5, 0.0 ]
            self.payoff[2] = [ 0.7, 0.0, 0.0 ]
            self.gamma_payoff = self.gamma * self.payoff

        self.N_per_epoch = np.ceil(self.N_nodes / 2).astype(int)

        self.nodes = list(self.G.nodes)
        self.neighbors = [list(self.G.neighbors(k)) for k in self.G.nodes]

        self.fig = None
        self.ax = None

    def play(self, N_epochs=10):

        for e in range(N_epochs):
            R = np.random.randint(0, self.N_nodes, size=self.N_per_epoch)
            Q = np.random.uniform(low=0.0, high=1.0, size=(self.N_nodes, 2))

            for i in range(self.N_per_epoch):
                node1 = self.nodes[R[i]]
                s = np.random.randint(0, self.G.degree(node1))
                node2 = self.neighbors[R[i]][s]

                node1_data = self.G.nodes[node1]
                node2_data = self.G.nodes[node2]

                action1 = random_choice(node1_data['P'], Q[i, 0])
                action2 = random_choice(node2_data['P'], Q[i, 1])

                J1_old = node1_data['J'][action1]
                J2_old = node2_data['J'][action2]

                node1_data['J'] *= self.gamma
                node2_data['J'] *= self.gamma
                node1_data['J'][action1] = J1_old - node1_data['J'][action1] + self.payoff[action1][action2]
                node2_data['J'][action2] = J2_old - node2_data['J'][action2] + self.payoff[action2][action1]

                node1_data['P'] = np.exp(self.beta * node1_data['J'])
                node1_data['P'] /= np.sum(node1_data['P'])
                node2_data['P'] = np.exp(self.beta * node2_data['J'])
                node2_data['P'] /= np.sum(node2_data['P'])

    def plot_init(self, fig_size=(10, 10)):
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        if self.positions == None:
            pos = nx.spring_layout(self.G, iterations=100)
            self.positions = nx.kamada_kawai_layout(self.G, pos=pos)

    def plot(self, with_labels=False, node_size=500, pause=0.05):
        if self.positions == None:
            # sys.exit('Error!')
            print('Run plot_init() before plot()')
            return

        node_color = [list(self.G.nodes[node]['J']) for node in self.G]
        node_color = np.vstack(node_color)
        node_color = node_color / np.amax(node_color)
        nx.draw(self.G,
                ax=self.ax,
                pos=self.positions,
                with_labels=with_labels,
                node_size=node_size,
                node_color=node_color)

        plt.pause(0.5)
        plt.show(block=False)

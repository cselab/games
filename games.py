#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

import os
from tqdm import tqdm


def random_choice(p, r):
    return np.argmax(np.cumsum(p) > r)


def size_of_nodes(x):
    x = np.sqrt(x)
    a = 9.7981
    b = -0.2964
    c = 7.1286
    d = -0.0781
    return np.exp(b*x + a) + np.exp(d*x + c)


class bargain:

    def __init__(self, G, beta=1., gamma=0.1, seed=None, J0=[ 4, 4, 4 ], payoff=None, folder="_results"):

        self.G = G
        self.beta = beta
        self.gamma = gamma

        np.random.seed(seed)

        self.N_nodes = G.number_of_nodes()

        self.positions = None

        J0 = np.array(J0)
        for node in G:
            G.nodes[node]['J'] = np.random.uniform(low=0, high=J0).astype('float64')
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

        self.results_folder = folder

        os.makedirs(self.results_folder, exist_ok=True)

        self.statistics = self._initialize_statistics()

    def play(self, N_epochs=10):
        print(f'[games] Simulating {N_epochs} epochs...')

        with tqdm(total=N_epochs,
                  desc="[games] Running for {:} epochs".format(N_epochs),
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            self.iter = 0
            for e in range(N_epochs):
                R = np.random.randint(0, self.N_nodes, size=self.N_per_epoch)
                Q = np.random.uniform(low=0.0, high=1.0, size=(self.N_nodes, 2))

                for i in range(self.N_per_epoch):
                    node1 = self.nodes[R[i]]
                    if self.neighbors[R[i]] != []:
                        s = np.random.randint(0, self.G.degree(node1))
                        node2 = self.neighbors[R[i]][s]
                    else:
                        node2 = np.random.randint(0, self.N_nodes)

                    node1_data = self.G.nodes[node1]
                    node2_data = self.G.nodes[node2]

                    action1 = random_choice(node1_data['P'], Q[i, 0])
                    action2 = random_choice(node2_data['P'], Q[i, 1])

                    node1_data['J'] *= (1.0 - self.gamma)
                    node2_data['J'] *= (1.0 - self.gamma)
                    node1_data['J'][action1] += self.payoff[action1][action2]
                    node2_data['J'][action2] += self.payoff[action2][action1]

                    node1_data['P'] = np.exp(self.beta * node1_data['J'])
                    node1_data['P'] /= np.sum(node1_data['P'])
                    node2_data['P'] = np.exp(self.beta * node2_data['J'])
                    node2_data['P'] /= np.sum(node2_data['P'])
                    self.iter += 1

                statistics_epoch = self._get_epoch_statistics()
                self._update_statistics(statistics_epoch)
                pbar.update(1)

        print("[games] Total iterations = {:}".format(self.iter))
        print("[games] Average iterations per agent = {:.2f}".format(self.iter / self.N_nodes))

    def plot_init(self, fig_size=(10, 10), position_function=None, *args):
        print(f'[games] Calculating nodes positions...')
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        if self.positions == None:
            if position_function == None:
                pos = nx.spring_layout(self.G, iterations=100)
                self.positions = nx.kamada_kawai_layout(self.G, pos=pos)
            else:
                self.positions = position_function(self.G, *args)

    def plot(self, with_labels=False, node_size=None, node_shape='o'):
        if self.positions == None:
            # sys.exit('Error!')
            print('Run plot_init() before plot()')
            return

        if node_size == None:
            node_size = size_of_nodes(self.N_nodes)

        node_color = [list(self.G.nodes[node]['J']) for node in self.G]
        node_color = np.vstack(node_color)
        node_color = node_color / np.amax(node_color)

        self.ax.clear()

        nx.draw(self.G,
                ax=self.ax,
                pos=self.positions,
                with_labels=with_labels,
                node_size=node_size,
                node_color=node_color,
                node_shape=node_shape)

        plt.pause(0.5)
        plt.show(block=False)

    def _update_statistics(self, statistics_epoch):
        for key in self.statistics:
            self.statistics[key].append(statistics_epoch[key])

    def _initialize_statistics(self):
        statistics = self._get_epoch_statistics()
        for key in statistics:
            statistics[key] = [statistics[key]]
        return statistics

    def _get_epoch_statistics(self):
        p_all = []
        j_all = []
        for node in self.G.nodes:
            node_data = self.G.nodes[node]
            p_all.append(node_data['P'])
            j_all.append(node_data['J'])

        p_all = np.array(p_all)
        p_avg = np.mean(p_all, axis=0)

        j_all = np.array(j_all)
        j_avg = np.mean(j_all, axis=0)

        statistics = {
            "p_all": p_all,
            "j_all": j_all,
            "p_low": p_avg[0],
            "p_med": p_avg[1],
            "p_high": p_avg[2],
            "j_low": j_avg[0],
            "j_med": j_avg[1],
            "j_high": j_avg[2],
        }
        return statistics

    def get_vertex_positions(self, data):
        assert (len(np.shape(data)) == 3)
        # data of the form [T, N, 3]

        # (J1+J2+J3)
        den = np.sum(data, axis=2)

        # (J2 + J3/2)/(J1+J2+J3)
        data_x = (data[:, :, 1] + data[:, :, 2] / 2.0) / den

        # \sqrt(3) * J3/2 /(J1+J2+J3)
        data_y = np.sqrt(3) * data[:, :, 2] / 2.0 / den

        return data_x, data_y

    def plot_statistics(self, fig_size=(10, 10)):

        # Final figure
        fig, ax = plt.subplots(figsize=fig_size)
        fig_path = self.results_folder + "/graph_final"
        node_color = np.array(self.statistics["j_all"])[-1]
        node_color = node_color / np.amax(node_color)
        node_size = 50
        pos = nx.spring_layout(self.G, iterations=100)
        positions = nx.kamada_kawai_layout(self.G, pos=pos)
        nx.draw(self.G, ax=self.ax, pos=positions, node_size=node_size, node_color=node_color)
        plt.savefig(fig_path)
        plt.close()

        plot_every = 10

        p = np.array(self.statistics["p_all"])
        p_x, p_y = self.get_vertex_positions(p)
        fig, ax = plt.subplots(figsize=fig_size)
        fig_path = self.results_folder + "/simplex_P"
        for particle in range(np.shape(p_x)[1]):
            x = p_x[::plot_every, particle]
            y = p_y[::plot_every, particle]
            ax.plot(x, y)
        plt.savefig(fig_path)
        plt.close()

        j = np.array(self.statistics["j_all"])
        j_x, j_y = self.get_vertex_positions(j)
        fig, ax = plt.subplots(figsize=fig_size)
        fig_path = self.results_folder + "/simplex_J"
        for particle in range(np.shape(p_x)[1]):
            x = j_x[::plot_every, particle]
            y = j_y[::plot_every, particle]
            ax.plot(x, y)
        plt.savefig(fig_path)
        plt.close()

        fig, ax = plt.subplots(figsize=fig_size)
        fig_path = self.results_folder + "/statistics_P"
        for key in [ "p_low", "p_med", "p_high"]:
            data = self.statistics[key]
            ax.plot(np.arange(len(data)), data, label=key)
        ax.legend()
        plt.savefig(fig_path)
        plt.close()

        fig, ax = plt.subplots(figsize=fig_size)
        fig_path = self.results_folder + "/statistics_J"
        for key in [ "j_low", "j_med", "j_high"]:
            data = self.statistics[key]
            ax.plot(np.arange(len(data)), data, label=key)
        ax.legend()
        plt.savefig(fig_path)
        plt.close()

        # Plotting attractors in the J and P space:
        for keys in [
            [ "j_low", "j_med"],
            [ "j_low", "j_high"],
            [ "j_med", "j_high"],
            [ "p_low", "p_med"],
            [ "p_low", "p_high"],
            [ "p_med", "p_high"],
        ]:
            key1, key2 = keys

            fig, ax = plt.subplots(figsize=fig_size)
            fig_path = self.results_folder + "/statistics_{:}-{:}".format(key1, key2)
            data1 = self.statistics[key1]
            data2 = self.statistics[key2]
            ax.plot(data1, data2)
            ax.set_xlabel(key1)
            ax.set_ylabel(key2)
            plt.savefig(fig_path)
            plt.close()

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

    def __init__(
        self,
        G,
        beta=1.,
        gamma=0.1,
        lamda=0.2,
        seed=None,
        J0=[ 4, 4, 4 ],
        payoff=None,
        N_tags=1,
        folder='_results',
    ):

        self.G = G
        self.beta = beta
        self.gamma = gamma
        self.N_tags = N_tags
        self.results_folder = folder
        np.random.seed(seed)

        self.N_nodes = G.number_of_nodes()

        self.positions = None

        J0 = np.array(J0)
        self.nodes_with_tag = [[] for _ in range(self.N_tags)]
        for node in G:
            node_data = G.nodes[node]

            if 'J' not in node_data.keys():
                node_data['J'] = np.random.uniform(low=0, high=J0, size=(N_tags, 3)).astype('float64')

            node_data['P'] = self._energy(node_data['J'])
            node_data['P'] /= np.sum(node_data['P'])

            if 'tag' not in node_data.keys():
                tag = np.random.randint(0, self.N_tags)
                node_data['tag'] = tag
            self.nodes_with_tag[node_data['tag']].append(node)

            node_data['last action'] = None

        if payoff == None:
            self.payoff = np.zeros((3, 3))
            p1 = 0.5 - lamda
            p2 = 0.5 + lamda
            self.payoff[0] = [ p1, p1, p1 ]
            self.payoff[1] = [ 0.5, 0.5, 0.0 ]
            self.payoff[2] = [ p2, 0.0, 0.0 ]
            self.gamma_payoff = self.gamma * self.payoff

        self.nodes = list(self.G.nodes)
        self.neighbors = [list(self.G.neighbors(k)) for k in self.G.nodes]

        self.node_shapes = 'so^>v<dph8'

        self.fig_graph = None
        self.ax_graph = None

        self.fig_stats = None
        self.ax_stats = None

        os.makedirs(self.results_folder, exist_ok=True)

        self.statistics = None
        self.iter = 0

    def __del__(self):
        if self.fig_stats != None:
            for fig in self.fig_stats:
                plt.close(fig)
        if self.fig_graph != None:
            for fig in self.fig_graph:
                plt.close(fig)

    def _energy(self, x):
        return np.exp(self.beta * x)

    def play(self, N_epochs=10, N_per_epoch=None):
        print(f'[games] Simulating {N_epochs} epochs...')

        if N_per_epoch == None:
            N_per_epoch = np.ceil(self.N_nodes / 2).astype(int)

        with tqdm(total=N_epochs,
                  desc='[games] Running for {:} epochs'.format(N_epochs),
                  bar_format='{l_bar}{bar} [ time left: {remaining} ]') as pbar:

            self._update_statistics()

            for e in range(N_epochs):
                R = np.random.randint(0, self.N_nodes, size=N_per_epoch)
                Q = np.random.uniform(low=0.0, high=1.0, size=(N_per_epoch, 2))

                for i in range(N_per_epoch):
                    node1 = self.nodes[R[i]]
                    if self.neighbors[R[i]] != []:
                        s = np.random.randint(0, self.G.degree(node1))
                        node2 = self.neighbors[R[i]][s]
                    else:
                        node2 = np.random.randint(0, self.N_nodes)

                    node1_data = self.G.nodes[node1]
                    node2_data = self.G.nodes[node2]

                    tag1 = node1_data['tag']
                    tag2 = node2_data['tag']
                    action1 = random_choice(node1_data['P'][tag2], Q[i, 0])
                    action2 = random_choice(node2_data['P'][tag1], Q[i, 1])

                    node1_data['last action'] = action1
                    node2_data['last action'] = action2

                    node1_data['J'][tag2] *= (1.0 - self.gamma)
                    node2_data['J'][tag1] *= (1.0 - self.gamma)
                    node1_data['J'][tag2][action1] += self.payoff[action1][action2]
                    node2_data['J'][tag1][action2] += self.payoff[action2][action1]

                    node1_data['P'][tag2] = self._energy(node1_data['J'][tag2])
                    node1_data['P'][tag2] /= np.sum(node1_data['P'][tag2])
                    node2_data['P'][tag1] = self._energy(node2_data['J'][tag1])
                    node2_data['P'][tag1] /= np.sum(node2_data['P'][tag1])

                    self.iter += 1

                self._update_statistics()
                pbar.update(1)

        print('[games] Total iterations = {:}'.format(self.iter))
        print('[games] Average iterations per agent = {:.2f}'.format(self.iter / self.N_nodes))

    def plot_graph(self, with_labels=False, fig_size=(10, 10), node_size=None, position_function=None):

        if self.fig_graph == None:
            print(f'[games] Initializing plotting graph:')
            self.fig_graph = []
            self.ax_graph = []
            for i in range(self.N_tags):
                fig, ax = plt.subplots(figsize=fig_size)
                self.fig_graph.append(fig)
                self.ax_graph.append(ax)

        if self.positions == None:
            print(f'[games] Calculating nodes positions...')
            if position_function == None:
                pos = nx.spring_layout(self.G, iterations=100)
                self.positions = nx.kamada_kawai_layout(self.G, pos=pos)
            else:
                self.positions = position_function(self.G, *args)

        if node_size == None:
            node_size = size_of_nodes(self.N_nodes)

        for k in range(self.N_tags):

            node_color = [list(self.G.nodes[node]['P'][k]) for node in self.G]
            node_color = np.vstack(node_color)
            node_color = node_color / np.amax(node_color)

            plt.figure(self.fig_graph[k].number)
            self.ax_graph[k].clear()

            for l in range(self.N_tags):
                nx.draw_networkx_nodes(self.G,
                                       ax=self.ax_graph[k],
                                       pos=self.positions,
                                       node_size=node_size,
                                       node_color=node_color[self.nodes_with_tag[l]],
                                       node_shape=self.node_shapes[l],
                                       nodelist=self.nodes_with_tag[l])

            nx.draw_networkx_edges(self.G, ax=self.ax_graph[k], pos=self.positions, alpha=0.2)

            self.ax_graph[k].axis('off')

            plt.text(0,
                     1,
                     f'tag = {k}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=self.ax_graph[k].transAxes)

            fig_path = self.results_folder + '/graph_' + str(k)
            plt.savefig(fig_path)

            plt.pause(0.005)
            plt.show(block=False)

    def _update_statistics(self):
        p_all = np.zeros((self.N_nodes, self.N_tags, 3))
        j_all = np.zeros((self.N_nodes, self.N_tags, 3))
        for i in range(self.N_nodes):
            node_data = self.G.nodes[i]
            p_all[i] = node_data['P']
            j_all[i] = node_data['J']

        # find the percentage of nodes that pick a specific probability per tag
        epsilon = 0.01
        per_L = np.sum(p_all[:, :, 0] > 1.0 - epsilon, axis=0)[:, np.newaxis].T
        per_M = np.sum(p_all[:, :, 1] > 1.0 - epsilon, axis=0)[:, np.newaxis].T
        per_H = np.sum(p_all[:, :, 2] > 1.0 - epsilon, axis=0)[:, np.newaxis].T

        statistics = {
            'p_all': p_all,
            'j_all': j_all,
            'per_L': per_L,
            'per_M': per_M,
            'per_H': per_H,
        }

        if self.statistics == None:
            self.statistics = {}
            for key in statistics:
                self.statistics[key] = statistics[key]
        else:
            for key in statistics:
                self.statistics[key] = np.append(self.statistics[key], statistics[key], axis=0)

    def _get_vertex_positions(self, data):
        assert (len(np.shape(data)) == 3)
        # data of the form [T, N, 3]
        # (J1+J2+J3)
        den = np.sum(data, axis=2)
        # (J2 + J3/2)/(J1+J2+J3)
        data_x = (data[:, :, 1] + data[:, :, 2] / 2.0) / den
        # \sqrt(3) * J3/2 /(J1+J2+J3)
        data_y = np.sqrt(3) * data[:, :, 2] / 2.0 / den
        return data_x, data_y

    def plot_statistics(self, fig_size=(10, 10),):
        if self.fig_stats == None:
            print(f'[games] Initializing plotting statistics:')
            self.N_stats = 1  # number of stats plots
            self.fig_stats = []
            self.ax_stats = []
            for i in range(self.N_stats):
                fig, ax = plt.subplots(figsize=fig_size)
                self.fig_stats.append(fig)
                self.ax_stats.append(ax)

        level_keys = [ 'L', 'M', 'H']
        level_colors = {
            'L': 'tab:red',
            'M': 'tab:green',
            'H': 'tab:blue',
        }
        linewidth = 2

        # Plot the percentages of each action
        k = 0
        plt.figure(self.fig_stats[k].number)
        self.ax_stats[k].clear()

        for level_key in level_keys:
            key_data = 'per_' + level_key
            data = self.statistics[key_data]

            for l in range(self.N_tags):
                self.ax_stats[k].plot(
                    np.arange(len(data)),
                    data[:, l],
                    label=level_key + ' of tag ' + str(l),
                    color=level_colors[level_key],
                    marker=self.node_shapes[l],
                    linewidth=linewidth,
                )
        self.ax_stats[k].legend()
        self.ax_stats[k].set_xlabel('Epoch')
        self.ax_stats[k].set_ylabel('Number of nodes with p>0.99')
        fig_path = self.results_folder + '/number_of_nodes_P'
        plt.savefig(fig_path)
        plt.pause(0.005)
        plt.show(block=False)

        # p = np.array(self.statistics['p_all'])
        # for level in range(len(level_keys)):
        #     level_key = level_keys[level]
        #     for particle in range(np.shape(p)[1]):
        #         data = p[:, particle, level]
        #         ax.plot(
        #             np.arange(len(data)),
        #             data,
        #             label=level_key,
        #             color=level_colors[level_key],
        #             linewidth=linewidth,
        #         )
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('P_i for all particles')
        # fig_path = self.results_folder + '/evolution_P'
        # plt.savefig(fig_path)
        # plt.pause(0.005)
        # plt.show(block=False)

        #
        # plot_every = 1
        #
        # p = np.array(self.statistics['p_all'])
        # p_x, p_y = self._get_vertex_positions(p)
        # fig, ax = plt.subplots(figsize=fig_size)
        # fig_path = self.results_folder + '/simplex_P'
        # for particle in range(np.shape(p_x)[1]):
        #     x = p_x[::plot_every, particle]
        #     y = p_y[::plot_every, particle]
        #     ax.plot(x, y, linewidth=linewidth)
        # plt.axis('off')
        # plt.savefig(fig_path)
        # plt.close()
        #
        # j = np.array(self.statistics['j_all'])
        # j_x, j_y = self._get_vertex_positions(j)
        # fig, ax = plt.subplots(figsize=fig_size)
        # fig_path = self.results_folder + '/simplex_J'
        # for particle in range(np.shape(p_x)[1]):
        #     x = j_x[::plot_every, particle]
        #     y = j_y[::plot_every, particle]
        #     ax.plot(x, y, linewidth=linewidth)
        # plt.axis('off')
        # plt.savefig(fig_path)
        # plt.close()
        #
        # fig, ax = plt.subplots(figsize=fig_size)
        # fig_path = self.results_folder + '/statistics_P'
        # for level_key in level_keys:
        #     key_data = 'p_' + level_key
        #     data = self.statistics[key_data]
        #     ax.plot(
        #         np.arange(len(data)),
        #         data,
        #         label=level_key,
        #         color=level_colors[level_key],
        #         linewidth=linewidth,
        #     )
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Average P_i')
        # ax.legend()
        # plt.savefig(fig_path)
        # plt.close()
        #
        # fig, ax = plt.subplots(figsize=fig_size)
        # fig_path = self.results_folder + '/statistics_J'
        # for level_key in level_keys:
        #     key_data = 'j_' + level_key
        #     data = self.statistics[key_data]
        #     ax.plot(
        #         np.arange(len(data)),
        #         data,
        #         label=level_key,
        #         color=level_colors[level_key],
        #         linewidth=linewidth,
        #     )
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Average J_i')
        # ax.legend()
        # plt.savefig(fig_path)
        # plt.close()
        #
        # # Plotting attractors in the J and P space:
        # for level_keys in [
        #     [ 'j_L', 'j_M'],
        #     [ 'j_L', 'j_H'],
        #     [ 'j_M', 'j_H'],
        #     [ 'p_L', 'p_M'],
        #     [ 'p_L', 'p_H'],
        #     [ 'p_M', 'p_H'],
        # ]:
        #     key1, key2 = level_keys
        #
        #     fig, ax = plt.subplots(figsize=fig_size)
        #     fig_path = self.results_folder + '/statistics_{:}-{:}'.format(key1, key2)
        #     data1 = self.statistics[key1]
        #     data2 = self.statistics[key2]
        #     ax.plot(
        #         data1,
        #         data2,
        #         color='tab:blue',
        #         linewidth=linewidth,
        #     )
        #     ax.set_xlabel(key1)
        #     ax.set_ylabel(key2)
        #     plt.savefig(fig_path)
        #     plt.close()

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
from numba import jit
import os
from tqdm import tqdm


@jit(nopython=True)
def random_choice(p, r):
    return np.argmax(np.cumsum(p) > r)


@jit(nopython=True)
def size_of_nodes(x):
    x = np.sqrt(x)
    a = 9.7981
    b = -0.2964
    c = 7.1286
    d = -0.0781
    return np.exp(b*x + a) + np.exp(d*x + c)


@jit(nopython=True)
def _probalility_of_action(x, beta):
    p = np.exp(beta * x)
    return (p.T / p.sum(axis=-1)).T


@jit
def _iterate_graph(R, Q, S, N_per_epoch, N_nodes, P, J, actions, tags, nodes, neighbors_flat, neighbors_offset, payoff,
                   beta, gamma):

    for i in range(N_per_epoch):
        k = R[i]
        node1 = nodes[k]

        begin = neighbors_offset[k]
        end = neighbors_offset[k + 1]
        if begin != end:
            s = int(S[i] * (end-begin))
            node2 = neighbors_flat[begin + s]
        else:
            node2 = int(S[i] * N_nodes)

        tag1 = tags[node1]
        tag2 = tags[node2]

        action1 = random_choice(P[node1][tag2], Q[i, 0])
        action2 = random_choice(P[node2][tag1], Q[i, 1])

        actions[node1] = action1
        actions[node2] = action2

        J[node1][tag2] *= (1.0 - gamma)
        J[node2][tag1] *= (1.0 - gamma)
        J[node1][tag2][action1] += payoff[action1][action2]
        J[node2][tag1][action2] += payoff[action2][action1]

        P[node1][tag2] = _probalility_of_action(J[node1][tag2], beta)
        P[node2][tag1] = _probalility_of_action(J[node2][tag1], beta)


def iterate_graph(N_per_epoch, N_nodes, *args, **kwargs):
    R = np.random.randint(0, N_nodes, size=N_per_epoch)
    Q = np.random.uniform(low=0.0, high=1.0, size=(N_per_epoch, 2))
    S = np.random.uniform(low=0.0, high=1.0, size=(N_per_epoch, ))

    _iterate_graph(R, Q, S, N_per_epoch, N_nodes, *args, **kwargs)


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
        self.payoff = payoff
        self.lamda = lamda
        self.N_tags = N_tags
        self.results_folder = folder
        np.random.seed(seed)

        self.N_nodes = G.number_of_nodes()

        self.positions = None

        self.J0 = np.array(J0)

        self._initialize_game()

        self.node_shapes = 'so^>v<dph8'

        self.fig_graph = None
        self.ax_graph = None
        self.fig_stats = None
        self.ax_stats = None
        self.fig_simplex = None
        self.ax_simplex = None

        os.makedirs(self.results_folder, exist_ok=True)

        self.statistics = None
        self.iter = 0

        self._update_statistics()

    def __del__(self):
        if self.fig_stats != None:
            for fig in self.fig_stats:
                plt.close(fig)
        if self.fig_graph != None:
            for fig in self.fig_graph:
                plt.close(fig)

    def _initialize_game(self):
        print('[games] Initializing the game...')
        self.J = np.zeros((self.N_nodes, self.N_tags, 3))
        self.P = np.zeros((self.N_nodes, self.N_tags, 3))
        self.tags = np.zeros((self.N_nodes, ), dtype=int)
        self.actions = np.zeros((self.N_nodes, ), dtype=int)
        self.nodes_with_tag = [[] for _ in range(self.N_tags)]

        # Graph has already 'J' and 'tag' data
        if (('has data' in self.G.graph.keys() and self.G.graph['has data'] != True)
                or 'has data' not in self.G.graph.keys()):
            for node in self.G:
                node_data = self.G.nodes[node]
                node_data['J'] = np.random.uniform(low=0, high=self.J0, size=(self.N_tags, 3)).astype('float64')
                node_data['tag'] = np.random.randint(0, self.N_tags)
            self.G.graph['has data'] = True

        for node in self.G:
            node_data = self.G.nodes[node]
            self.J[node] = node_data['J']

            self.P[node] = _probalility_of_action(self.J[node], self.beta)
            node_data['P'] = self.P[node]

            self.tags[node] = node_data['tag']
            self.nodes_with_tag[node_data['tag']].append(node)
            node_data['last action'] = None

        if self.payoff == None:
            self.payoff = np.zeros((3, 3))
            p1 = 0.5 - self.lamda
            p2 = 0.5 + self.lamda
            self.payoff[0] = [ p1, p1, p1 ]
            self.payoff[1] = [ 0.5, 0.5, 0.0 ]
            self.payoff[2] = [ p2, 0.0, 0.0 ]
            self.gamma_payoff = self.gamma * self.payoff

        self.nodes = np.array(self.G.nodes)
        self.neighbors = [list(self.G.neighbors(k)) for k in self.G.nodes]
        total_neighbors = sum(len(l) for l in self.neighbors)
        self.neighbors_flat = np.zeros((total_neighbors, ), dtype=np.int32)
        self.neighbors_offsets = np.zeros((len(self.G.nodes) + 1, ), dtype=np.int32)
        offset = 0
        for i, l in enumerate(self.neighbors):
            self.neighbors_flat[offset:offset + len(l)] = np.array(l)
            self.neighbors_offsets[i + 1] = offset + len(l)
            offset += len(l)

    def copy_data_to_graph(self):
        for node in self.G:
            node_data = self.G.nodes[node]
            node_data['J'] = self.J[node]
            node_data['P'] = self.P[node]

    def play(self, N_epochs=10, N_per_epoch=None):
        print(f'[games] Simulating {N_epochs} epochs...')

        if N_per_epoch == None:
            N_per_epoch = np.ceil(self.N_nodes / 2).astype(int)

        with tqdm(total=N_epochs,
                  desc='[games] Running for {:} epochs'.format(N_epochs),
                  bar_format='{l_bar}{bar} [ time left: {remaining} ]') as pbar:
            for e in range(N_epochs):
                iterate_graph(N_per_epoch, self.N_nodes, self.P, self.J, self.actions, self.tags, self.nodes,
                              self.neighbors_flat, self.neighbors_offsets, self.payoff, self.beta, self.gamma)
                self.iter += N_per_epoch
                self._update_statistics()
                pbar.update(1)

        print('[games] Total iterations = {:}'.format(self.iter))
        print('[games] Average iterations per agent = {:.2f}'.format(self.iter / self.N_nodes))

    def plot_graph(self, with_labels=False, fig_size=(10, 10), node_size=None, position_function=None):

        if self.fig_graph == None:
            print(f'[games] Initializing plotting graph...')
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

        node_color = self.P

        for k in range(self.N_tags):

            plt.figure(self.fig_graph[k].number)
            self.ax_graph[k].clear()

            for l in range(self.N_tags):
                nx.draw_networkx_nodes(self.G,
                                       ax=self.ax_graph[k],
                                       pos=self.positions,
                                       node_size=node_size,
                                       node_color=np.squeeze(node_color[self.nodes_with_tag[l]][:, k, :]),
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
        # initialize the statistics dictionary, only once
        if self.statistics == None:
            self.statistics = {}
            self.statistics['p_all'] = np.empty((0, self.N_nodes, self.N_tags, 3))
            self.statistics['j_all'] = np.empty((0, self.N_nodes, self.N_tags, 3))

            self.statistics['per_L'] = np.empty((0, self.N_tags))
            self.statistics['per_M'] = np.empty((0, self.N_tags))
            self.statistics['per_H'] = np.empty((0, self.N_tags))

        # find the percentage of nodes that pick a specific probability per tag
        epsilon = 0.01
        per_L = np.sum(self.P[:, :, 0] > 1.0 - epsilon, axis=0)[:, np.newaxis].T
        per_M = np.sum(self.P[:, :, 1] > 1.0 - epsilon, axis=0)[:, np.newaxis].T
        per_H = np.sum(self.P[:, :, 2] > 1.0 - epsilon, axis=0)[:, np.newaxis].T

        self.statistics['p_all'] = np.append(self.statistics['p_all'], self.P[np.newaxis], axis=0)
        self.statistics['j_all'] = np.append(self.statistics['j_all'], self.J[np.newaxis], axis=0)

        self.statistics['per_L'] = np.append(self.statistics['per_L'], per_L, axis=0)
        self.statistics['per_M'] = np.append(self.statistics['per_M'], per_M, axis=0)
        self.statistics['per_H'] = np.append(self.statistics['per_H'], per_H, axis=0)

    def plot_statistics(self, fig_size=(10, 10)):
        if self.fig_stats == None:
            print(f'[games] Initializing plotting statistics:')
            self.N_plot_stats = 1  # number of stats plots
            self.fig_stats = []
            self.ax_stats = []
            for i in range(self.N_plot_stats):
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

        # ------------------------------------
        # Plot the percentages of each action
        # ------------------------------------
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
                    markersize=4,
                    linewidth=linewidth,
                )
        self.ax_stats[k].legend()
        self.ax_stats[k].set_xlabel('Epoch')
        self.ax_stats[k].set_ylabel('Number of nodes with p>0.99')
        fig_path = self.results_folder + '/number_of_nodes_P'
        plt.savefig(fig_path)
        plt.pause(0.005)
        plt.show(block=False)

    def _get_vertex_positions(self, data, prob_axis=2):
        assert (prob_axis in [ 2, 3 ])
        assert (np.shape(data)[prob_axis] == 3)
        # data of the form [T, N, 3]
        # (JL + JM + JH)
        den = np.sum(data, axis=prob_axis)

        # Vertices JL, JM, JH
        if prob_axis == 2:

            # (JL + JM/2)/(JL+JM+JH)
            data_x = (data[:, :, 0] + data[:, :, 1] / 2.0) / den

            # \sqrt(3) * JM/2 /(JL+JM+JH)
            data_y = np.sqrt(3) * data[:, :, 1] / 2.0 / den

        elif prob_axis == 3:
            data_x = (data[:, :, :, 0] + data[:, :, :, 1] / 2.0) / den
            data_y = np.sqrt(3) * data[:, :, :, 1] / 2.0 / den

        return data_x, data_y

    def _add_triangle(self, ax):
        temp = np.sin(60. * np.pi / 180.)
        edges = [[ 0, 0 ], [ 0.5, temp ], [ 1, 0 ]]
        ax.add_patch(plt.Polygon(edges, color="k", fill=False, linewidth=4, alpha=0.8))
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        margin = 0.05
        ax.text(-0.02, 0 + margin, "H", fontsize=16, fontweight='bold')
        ax.text(1, 0 + margin, "L", fontsize=16, fontweight='bold')
        ax.text(0.5, temp + margin, "M", fontsize=16, fontweight='bold')

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis("off")

    def plot_simplex(self, fig_size=(10, 10)):
        if self.fig_simplex == None:
            print(f'[games] Initializing plotting statistics:')
            self.N_plot_stats = 2  # number of stats plots
            self.fig_simplex = []
            self.ax_simplex = []
            for i in range(self.N_plot_stats):
                fig, ax = plt.subplots(figsize=fig_size)
                self.fig_simplex.append(fig)
                self.ax_simplex.append(ax)

        level_keys = [ 'L', 'M', 'H']
        level_colors = {
            'L': 'tab:red',
            'M': 'tab:green',
            'H': 'tab:blue',
        }
        linewidth = 2

        if self.N_tags == 1:
            # ------------------------------------
            # Plot the simplex in J (evolution)
            # ------------------------------------
            k = 0
            plt.figure(self.fig_simplex[k].number)
            self.ax_simplex[k].clear()
            self._add_triangle(self.ax_simplex[k])

            # Last element of p_all (in time)
            data = self.statistics['j_all']
            p_x, p_y = self._get_vertex_positions(data, prob_axis=3)
            # print(np.shape(p_x))
            # print(np.shape(p_y))
            for agent in range(self.N_nodes):
                for l in range(self.N_tags):
                    p_x_tag = p_x[:, agent, l]
                    p_y_tag = p_y[:, agent, l]
                    self.ax_simplex[k].plot(
                        p_x_tag,
                        p_y_tag,
                        markersize=8,
                        linewidth=linewidth,
                        marker=self.node_shapes[l],
                    )

            fig_path = self.results_folder + '/simplex_J_evolution'
            plt.savefig(fig_path)
            plt.pause(0.005)
            plt.show(block=False)

            # ------------------------------------
            # Plot the simplex in J (final state)
            # ------------------------------------
            k = 1
            plt.figure(self.fig_simplex[k].number)
            self.ax_simplex[k].clear()
            self._add_triangle(self.ax_simplex[k])

            # Last element of p_all (in time)
            data = self.statistics['j_all'][-1]
            p_x, p_y = self._get_vertex_positions(data, prob_axis=2)
            # print(np.shape(p_x))
            # print(np.shape(p_y))
            for l in range(self.N_tags):
                p_x_tag = p_x[:, l]
                p_y_tag = p_y[:, l]
                self.ax_simplex[k].plot(
                    p_x_tag + 0.01 * np.random.randn(*np.shape(p_x_tag)),
                    p_y_tag + 0.01 * np.random.randn(*np.shape(p_y_tag)),
                    markersize=8,
                    linewidth=0,
                    marker=self.node_shapes[l],
                )

            fig_path = self.results_folder + '/simplex_J_final_all'
            plt.savefig(fig_path)
            plt.pause(0.005)
            plt.show(block=False)

        # Only plotting the inter-type equity (between Tags)
        # when there are more than 1 tag.
        # Function for N_tags > 2 not implemented.
        if self.N_tags == 2:

            # ------------------------------------------------
            # Plot the simplex in J
            # Plotting the intra-type equity (Tag versus own tag)
            # ------------------------------------------------
            k = 0
            plt.figure(self.fig_simplex[k].number)
            self.ax_simplex[k].clear()
            self._add_triangle(self.ax_simplex[k])

            # Last element of p_all (in time)
            data = self.statistics['j_all'][-1]
            p_x, p_y = self._get_vertex_positions(data)
            for l in range(self.N_tags):
                tag_own = l
                tag_oponent = l
                idx_tag = np.where(self.tags == tag_own)[0]
                p_x_tag = p_x[idx_tag, tag_oponent]
                p_y_tag = p_y[idx_tag, tag_oponent]
                self.ax_simplex[k].plot(
                    p_x_tag + 0.01 * np.random.randn(*np.shape(p_x_tag)),
                    p_y_tag + 0.01 * np.random.randn(*np.shape(p_y_tag)),
                    markersize=8,
                    linewidth=0,
                    marker=self.node_shapes[l],
                    label="Tag {:} against tag {:}".format(tag_own, tag_oponent),
                )

            self.ax_simplex[k].legend()
            fig_path = self.results_folder + '/simplex_J_intra_within'
            plt.savefig(fig_path)
            plt.pause(0.005)
            plt.show(block=False)

            # ------------------------------------------------
            # Plot the simplex in J
            # Plotting the inter-type equity (between Tags)
            # ------------------------------------------------
            k = 1
            plt.figure(self.fig_simplex[k].number)
            self.ax_simplex[k].clear()
            self._add_triangle(self.ax_simplex[k])
            # Last element of p_all (in time)
            data = self.statistics['j_all'][-1]
            p_x, p_y = self._get_vertex_positions(data)

            for l in range(self.N_tags):
                tag_own = l
                tag_oponents = set(range(self.N_tags))
                tag_oponents = tag_oponents.difference(set([l]))
                assert (len(tag_oponents) == 1)
                for tag_oponent in tag_oponents:
                    idx_tag = np.where(self.tags == tag_own)[0]
                    p_x_tag = p_x[idx_tag, tag_oponent]
                    p_y_tag = p_y[idx_tag, tag_oponent]
                    self.ax_simplex[k].plot(
                        p_x_tag + 0.01 * np.random.randn(*np.shape(p_x_tag)),
                        p_y_tag + 0.01 * np.random.randn(*np.shape(p_y_tag)),
                        markersize=6,
                        linewidth=0,
                        marker=self.node_shapes[l],
                        label="Tag {:} against tag {:}".format(tag_own, tag_oponent),
                    )
            self.ax_simplex[k].legend()
            fig_path = self.results_folder + '/simplex_J_inter_between'
            plt.savefig(fig_path)
            plt.pause(0.005)
            plt.show(block=False)

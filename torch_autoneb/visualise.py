from typing import List, Union

import torch_autoneb as ta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
import networkx as nx
import numpy as np


def draw_connectivity_graph(graph, value_key, weight_key, pos=None):
    # Determine value range
    nodelist = list(graph)
    edgelist = list(graph.edges)

    cmap = plt.get_cmap("plasma")

    # Minima
    if pos is None:
        pos = nx.circular_layout(graph)
    if value_key is None:
        node_colour = "k"
        vmin = None
        vmax = None
    else:
        node_colour = [graph.nodes[m][value_key] for m in nodelist]
        vmin = min(node_colour)
        vmax = max(node_colour)
    nx.draw_networkx_nodes(graph, pos, nodelist, node_color=node_colour,
                           cmap=cmap, vmin=vmin, vmax=vmax)
    nx.draw_networkx_labels(graph, pos, font_color="w")

    # Edges
    if len(edgelist) > 0:
        edgecolours = [graph.get_edge_data(*e)[weight_key] for e in edgelist]
        vmin = min(edgecolours)
        vmax = max(edgecolours)
        nx.draw_networkx_edges(graph, pos, edgelist, 4, edgecolours,
                               edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax)

        # Colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        a = plt.colorbar(sm)

    plt.axis("equal")
    plt.axis("off")


def plot_dense(dense_data, pivot_distances, normed_length=False):
    x = x_for_dense(dense_data, pivot_distances, normed_length)
    line, = plt.plot(x.numpy(), dense_data.numpy())

    chain_count = pivot_distances.shape[0] + 1
    total_count = dense_data.shape[0]
    sub_image_count = (total_count - 1) // (chain_count - 1) - 1
    pivot_slice = slice(None, None, sub_image_count + 1)
    plt.plot(x.numpy()[pivot_slice], dense_data.numpy()[pivot_slice], "d", c=line.get_color())

    plt.xlabel("Position on path")
    plt.xticks([0] + list(pivot_distances.cumsum(0)), ["$ \\theta_1 $"] + ([""] * (pivot_distances.shape[0] - 1)) + ["$ \\theta_2 $"])


def x_for_dense(data, distances, normed_length=False):
    cumsum = distances.cumsum(0)
    normed = distances.new(distances.shape[0] + 1).zero_()
    normed[1:] = cumsum  # / cumsum[-1]
    chain_count = distances.shape[0] + 1
    total_count = data.shape[0]
    assert (total_count - 1) % (chain_count - 1) == 0, "Cannot compute sub-image-count"
    sub_image_count = (total_count - 1) // (chain_count - 1) - 1
    sub_normed = distances.new(total_count).zero_()
    # Fill each offset with interpolation
    for j in range(sub_image_count + 1):
        alpha = j / (sub_image_count + 1)
        sub_normed[j:-1:sub_image_count + 1] = (1 - alpha) * normed[0:-1] + alpha * normed[1:]
    sub_normed[-1] = normed[-1]
    if normed_length:
        sub_normed /= sub_normed[-1]
    return sub_normed


class Leaf:
    def __init__(self, m_idx, value):
        self.m_idx = m_idx
        self.value = value
        self.width = 1

    def plot(self, center):
        plt.plot(center, self.value, ".k", zorder=3)


class Cluster:
    def __init__(self, cluster_a: Union["Cluster", Leaf], cluster_b: Union["Cluster", Leaf], value):
        self.cluster_a = cluster_a
        self.cluster_b = cluster_b
        self.value = value
        self.width = cluster_a.width + cluster_b.width + 1

    def plot(self, center):
        width_a = self.cluster_a.width
        width_b = self.cluster_b.width

        center_a = center - self.width / 2 + width_a / 2
        center_b = center + self.width / 2 - width_b / 2

        self.cluster_a.plot(center_a)
        self.cluster_b.plot(center_b)

        value_a = self.cluster_a.value
        value_b = self.cluster_b.value

        trajectory = np.array([
            [center_a, value_a],
            [center_a, self.value],
            [center_b, self.value],
            [center_b, value_b]
        ])
        plt.plot(trajectory[:, 0], trajectory[:, 1], "k-", lw=1)


def draw_disconnectivity_graph(graph, value_key, weight_key):
    simple_graph = ta.to_simple_graph(graph, weight_key)
    mst = nx.minimum_spanning_tree(simple_graph, weight_key)

    dist = np.zeros([len(graph.nodes)] * 2)
    for i, mi in enumerate(mst.nodes):
        for j, mj in enumerate(mst.nodes):
            dist[i, j] = ta.topographic_distance(mst, mi, mj, weight_key)
    pdist = squareform(dist, checks=False)

    link_info = linkage(pdist, optimal_ordering=True)
    clusters: List[Union[Leaf, Cluster]] = [Leaf(m, mst.nodes[m][value_key]) for m in mst]
    for c1, c2, d12, _ in link_info:
        clusters.append(Cluster(clusters[int(c1)], clusters[int(c2)], d12))
    clusters[-1].plot(0)
    plt.yscale("log")

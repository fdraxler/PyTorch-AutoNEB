try:
    import matplotlib.pyplot as plt
except:
    pass
import networkx as nx

def draw_connectivity_graph(graph, value_key, weight_key, pos=None):
    # Determine value range
    nodelist = list(graph)
    edgelist = list(graph.edges)

    # Minima
    if pos is None:
        pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, nodelist, node_color="k")
    nx.draw_networkx_labels(graph, pos, font_color="w")

    # Edges
    if len(edgelist) > 0:
        edgecolours = [graph.get_edge_data(*e)[weight_key] for e in edgelist]
        vmin = min(edgecolours)
        vmax = max(edgecolours)
        cmap = plt.get_cmap("plasma")
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
    normed[1:] = cumsum# / cumsum[-1]
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
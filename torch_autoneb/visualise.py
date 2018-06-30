try:
    import matplotlib.pyplot as plt
except:
    pass
import networkx as nx

def draw_connectivity_graph(graph, value_key, weight_key):
    # Determine value range
    nodelist = list(graph)
    nodecolours = [graph.node[node][value_key] for node in nodelist]
    vmin = min(nodecolours)
    vmax = max(nodecolours)

    edgelist = list(graph.edges)
    edgecolours = [graph.get_edge_data(*e)[weight_key] for e in edgelist]
    if len(edgelist) > 0:
        vmin = min(vmin, min(edgecolours))
        vmin = max(vmax, max(edgecolours))

    cmap = plt.get_cmap("plasma")

    # Minima
    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, nodelist, node_color=nodecolours,
                           cmap=cmap, vmin=vmin, vmax=vmax)
    nx.draw_networkx_labels(graph, pos, font_color="w")

    # Edges
    nx.draw_networkx_edges(graph, pos, edgelist, 4, edgecolours,
                           edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    a = plt.colorbar(sm)

    plt.axis("equal")
    plt.axis("off")
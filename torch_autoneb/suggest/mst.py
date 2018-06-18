from logging import Logger

from networkx import minimum_spanning_tree, connected_components, MultiGraph

from torch_autoneb import to_simple_graph


def mst_suggest(graph: MultiGraph, value_key, weight_key, logger: Logger):
    """
    Replace an existing edge with a "shortcut" via another node.
    """

    # Reduce the connections to the best for each pair
    assert len(list(connected_components(graph))) == 1
    simple_graph = to_simple_graph(graph, weight_key)
    assert len(list(connected_components(simple_graph))) == 1

    # Find the highest connection in the mst
    # The mst contains the best connection to each minimum
    mst = minimum_spanning_tree(simple_graph, weight_key)  # type: MultiGraph
    highest_edges = sorted(mst.edges, key=lambda e: simple_graph[e[0]][e[1]][weight_key], reverse=True)

    # Try removing edges one by one (highest first) and find an alternative
    for highest_edge in highest_edges:
        mst.remove_edge(*highest_edge)
        cluster1, cluster2 = connected_components(mst)
        for source_minimum in sorted(cluster1, key=lambda x: simple_graph.nodes[x][value_key]):
            for target_minimum in sorted(cluster2, key=lambda x: simple_graph.nodes[x][value_key]):
                if (source_minimum, target_minimum) not in simple_graph.edges and source_minimum is not target_minimum:
                    return source_minimum, target_minimum
        mst.add_edge(*highest_edge)

    return None, None

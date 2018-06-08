from networkx import Graph, minimum_spanning_tree, connected_components


def mst_suggest(graph):
    """
    Replace an existing edge with a "shortcut" via another node.

    Parameters
    ----------
    graph : Graph
    """
    assert len(list(connected_components(graph))) == 1

    # Find the highest connection in the mst
    # The mst contains the best connection to each minimum
    mst = minimum_spanning_tree(graph, "weight")  # type: Graph
    highest_edges = sorted(mst.edges, key=lambda e: graph[e[0]][e[1]]["weight"], reverse=True)

    # Try removing edges one by one (highest first) and find an alternative
    for highest_edge in highest_edges:
        mst.remove_edge(*highest_edge)
        cluster1, cluster2 = connected_components(mst)
        for source_minimum in sorted(cluster1, key=lambda x: graph.nodes[x]["value"]):
            for target_minimum in sorted(cluster2, key=lambda x: graph.nodes[x]["value"]):
                if (source_minimum, target_minimum) not in graph.edges and source_minimum is not target_minimum:
                    return source_minimum, target_minimum
        mst.add_edge(*highest_edge)

    return None, None

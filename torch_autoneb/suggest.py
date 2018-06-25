from logging import getLogger

from networkx import MultiGraph, connected_components, minimum_spanning_tree, Graph

import torch_autoneb

logger = getLogger(__name__)


def unfinished(graph: MultiGraph, config):
    """
    Find unfinished AutoNEB runs.
    """
    collected_unfinished = []
    cycle_count = config.auto_neb_config.cycle_count
    for m1 in graph.nodes:
        for m2 in graph[m1]:
            # Check if final cycle is there
            cycles = set(graph[m1][m2].keys())
            if cycle_count not in cycles:
                collected_unfinished.append((m1, m2))

            # Warn about missing intermediate cycles (should never occur)
            if logger is not None:
                intermediate = set(range(1, max(cycles) + 1))
                if cycles != intermediate:
                    logger.warning(f"Minima {m1} and {m2} are missing intermediate cycles {intermediate.difference(cycles)}.")

    if len(collected_unfinished) > 0:
        return collected_unfinished[0]
    return None, None


def disconnected(graph: MultiGraph, config):
    """
    Find minima that are not connected to the global minimum.
    """
    # Find the connected components
    subgraphs = list(connected_components(graph))
    value_key = config.value_key
    if len(subgraphs) > 1:
        # Find the global minimum in the graph
        minimum = min(graph, key=lambda x: graph.nodes[x][value_key])

        # Find the cluster of the global minimum
        minimum_subgraph = None
        for subgraph in subgraphs:
            if minimum in subgraph:
                minimum_subgraph = subgraph
                break
        assert minimum_subgraph is not None

        # Return the smallest element from all other clusters
        smallest_unconnected = None
        smallest_value = float("inf")
        for subgraph in subgraphs:
            if subgraph is not minimum_subgraph:
                submin = min(subgraph, key=lambda x: graph.nodes[x][value_key])
                submin_value = graph.nodes[submin][value_key]
                if submin_value < smallest_value:
                    smallest_unconnected = submin
                    smallest_value = submin_value

        return minimum, smallest_unconnected
    return None, None


def mst(graph: MultiGraph, config):
    """
    Replace an existing edge with a "shortcut" via another node.
    """

    # Reduce the connections to the best for each pair
    assert len(list(connected_components(graph))) == 1
    weight_key = config.weight_key
    value_key = config.value_key
    simple_graph = torch_autoneb.to_simple_graph(graph, weight_key)
    assert len(list(connected_components(simple_graph))) == 1

    # Find the highest connection in the mst
    # The mst contains the best connection to each minimum
    mst = minimum_spanning_tree(simple_graph, weight_key)  # type: Graph
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


def lowest(graph: MultiGraph, config):
    """
    Add all missing edges in the graph, prioritising minima with lower loss/energy.
    """
    value_key = config.value_key
    for source_minimum in sorted(graph, key=lambda x: graph.nodes[x][value_key]):
        for target_minimum in sorted(graph, key=lambda x: graph.nodes[x][value_key]):
            if (source_minimum, target_minimum) not in graph.edges and source_minimum is not target_minimum:
                return source_minimum, target_minimum

    return None, None

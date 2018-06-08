from networkx import Graph
from networkx import connected_components


def disconnected_suggest(graph):
    """
    Parameters
    ----------
    graph : Graph
    """
    # Find the connected components
    subgraphs = list(connected_components(graph))
    if len(subgraphs) > 1:
        # Find the global minimum in the graph
        minimum = min(graph, key=lambda x: graph.nodes[x]["value"])

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
                submin = min(subgraph, key=lambda x: graph.nodes[x]["value"])
                submin_value = graph.nodes[submin]["value"]
                if submin_value < smallest_value:
                    smallest_unconnected = submin
                    smallest_value = submin_value

        return minimum, smallest_unconnected
    return None, None

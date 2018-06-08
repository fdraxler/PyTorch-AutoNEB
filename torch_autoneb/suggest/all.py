from networkx import Graph


def all_suggest(graph):
    """
    Add all missing edges in the graph.

    Parameters
    ----------
    graph : Graph
    """
    for source_minimum in sorted(graph, key=lambda x: graph.nodes[x]["value"]):
        for target_minimum in sorted(graph, key=lambda x: graph.nodes[x]["value"]):
            if (source_minimum, target_minimum) not in graph.edges and source_minimum is not target_minimum:
                return source_minimum, target_minimum

    return None, None

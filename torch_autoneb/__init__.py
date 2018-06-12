import pickle

from networkx import MultiGraph

from torch_autoneb.suggest import suggest_pair
try:
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:
    def _tqdm(iterable, *args, **kwargs):
        yield from iterable


def find_minimum():
    return {
        "coords": None,
        "value": 42,
    }


def neb(minimum1, minimum2, previous_cycle_data):
    return {
        "path_coords": None,
        "weight": 42 + 3.14,
    }


def auto_neb(cycle_count, graph, suggest_engines):
    while True:
        # Suggest new pair based on current graph
        m1, m2 = suggest_pair(graph, *suggest_engines)
        if m1 is None or m2 is None:
            break

        # Continue existing cycles or start from scratch
        existing_edges = graph[m1][m2]
        if len(existing_edges) > 0:
            previous_cycle_idx = max(existing_edges[m1][m2])
            previous_cycle_data = existing_edges[m1][m2][previous_cycle_idx]
            this_cycle_idx = previous_cycle_idx + 1
        else:
            previous_cycle_data = None
            this_cycle_idx = 1

        # Run NEB and add to graph
        assert this_cycle_idx <= cycle_count
        connection_data = neb(m1, m2, previous_cycle_data)
        graph.add_edge(m1, m2, key=this_cycle_idx, **connection_data)


def load_pickle_graph(graph_file_name):
    with open(graph_file_name, "rb") as file:
        graph = pickle.load(file)

        # Check file structure
        if not isinstance(graph, MultiGraph):
            raise ValueError(f"{graph_file_name} does not contain a nx.MultiGraph")
        for node in graph.nodes:
            if not "value" in graph.nodes[node]:
                raise ValueError(f"{graph_file_name} does not contain a 'value' property for node {node}.")
        for edge in graph.edges:
            if not "weight" in graph.get_edge_data(*edge):
                raise ValueError(f"{graph_file_name} does not contain a 'weight' property for edge from {edge[0]} to {edge[1]} (cycle {edge[2]}).")
    return graph

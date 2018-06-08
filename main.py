import os

from networkx import MultiGraph

from torch_autoneb import auto_neb, load_pickle_graph, find_minimum
from torch_autoneb.suggest.all import all_suggest
from torch_autoneb.suggest.disconnected import disconnected_suggest
from torch_autoneb.suggest.mst import mst_suggest
from torch_autoneb.suggest.unfinished import create_unfinished_suggest

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, *args, **kwargs):
        yield from iterable


def main():
    minima_count = 10
    cycle_count = 14
    graph_file_name = "bliblablu.p"
    suggest_engines = [create_unfinished_suggest(cycle_count), disconnected_suggest, mst_suggest, all_suggest]

    # === Create graph ===
    if os.path.exists(graph_file_name):
        graph = load_pickle_graph(graph_file_name)
    else:
        graph = MultiGraph()

    # === Ensure the specified number of minima ===
    for _ in tqdm(range(len(graph.nodes), minima_count)):
        minimum_data = find_minimum()
        graph.add_node(max(graph.nodes) + 1, **minimum_data)

    # === Connect minima ordered by suggestion algorithm ===
    auto_neb(cycle_count, graph, suggest_engines)

    pass


if __name__ == '__main__':
    main()

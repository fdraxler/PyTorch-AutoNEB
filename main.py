import os

from networkx import MultiGraph

from torch_autoneb import auto_neb, load_pickle_graph, find_minimum, _tqdm
from torch_autoneb.hyperparameters import OptimHyperparameters, NEBHyperparameters
from torch_autoneb.models import ModelWrapper
from torch_autoneb.suggest.all import all_suggest
from torch_autoneb.suggest.disconnected import disconnected_suggest
from torch_autoneb.suggest.mst import mst_suggest
from torch_autoneb.suggest.unfinished import create_unfinished_suggest


def main():
    minima_count = 10
    cycle_count = 14
    graph_file_name = "bliblablu.p"
    suggest_engines = [create_unfinished_suggest(cycle_count), disconnected_suggest, mst_suggest, all_suggest]
    minimisation_hyperparameters = OptimHyperparameters()
    auto_neb_hyperparameters = NEBHyperparameters(cycle_count)
    model = ModelWrapper(None)

    # === Create graph ===
    if os.path.exists(graph_file_name):
        graph = load_pickle_graph(graph_file_name)
    else:
        graph = MultiGraph()

    # === Ensure the specified number of minima ===
    for _ in _tqdm(range(len(graph.nodes), minima_count)):
        minimum_data = find_minimum(model, minimisation_hyperparameters)
        graph.add_node(max(graph.nodes) + 1 if len(graph.nodes) > 0 else 1, **minimum_data)

    # === Connect minima ordered by suggestion algorithm ===
    auto_neb(model, auto_neb_hyperparameters, graph, suggest_engines)


if __name__ == '__main__':
    main()

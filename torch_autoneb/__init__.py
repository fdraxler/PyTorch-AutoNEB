import pickle

import torch
from networkx import MultiGraph
from torch import optim

from torch_autoneb.hyperparameters import NEBHyperparameters, OptimHyperparameters, AutoNEBHyperparameters, LandscapeExplorationHyperparameters
from torch_autoneb.models import ModelWrapper
from torch_autoneb.suggest import suggest_pair

try:
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:
    def _tqdm(iterable, *args, **kwargs):
        yield from iterable


def find_minimum(model: ModelWrapper, config: OptimHyperparameters) -> dict:
    optimiser = getattr(optim, config.optim_name)(model.model.parameters(), **config.optim_args)  # type: optim.Optimizer

    # Initialise
    model.initialise_randomly()

    # Optimise
    with torch.set_grad_enabled(True):
        for iteration in range(config.nsteps):
            optimiser.zero_grad()
            model.forward()
            optimiser.step()

    # Analyse
    with torch.set_grad_enabled(False):
        pass

    return {
        "coords": None,
        "value": 42,
    }


def neb(m1, m2, previous_cycle_data, model: ModelWrapper, config: NEBHyperparameters) -> dict:
    # Initialise

    # Optimise

    # Analyse

    return {
        "path_coords": None,
        "weight": 42 + 3.14,
    }


def auto_neb(m1, m2, graph: MultiGraph, model: ModelWrapper, config: AutoNEBHyperparameters):
    # Continue existing cycles or start from scratch
    existing_edges = graph[m1][m2]
    if len(existing_edges) > 0:
        previous_cycle_idx = max(existing_edges[m1][m2])
        previous_cycle_data = existing_edges[m1][m2][previous_cycle_idx]
        start_cycle_idx = previous_cycle_idx + 1
    else:
        previous_cycle_data = None
        start_cycle_idx = 1
    assert start_cycle_idx <= config.cycle_count

    # Run NEB and add to graph
    for cycle_idx in range(start_cycle_idx, config.cycle_count + 1):
        cycle_config = config.hyperparameter_sets[start_cycle_idx - 1]
        connection_data = neb(m1, m2, previous_cycle_data, model, cycle_config)
        graph.add_edge(m1, m2, key=cycle_idx, **connection_data)


def landscape_exploration(graph: MultiGraph, model: ModelWrapper, config: LandscapeExplorationHyperparameters):
    while True:
        # Suggest new pair based on current graph
        m1, m2 = suggest_pair(graph, *config.suggest_engines)
        if m1 is None or m2 is None:
            break
        auto_neb(m1, m2, graph, model, config.auto_neb_config)


def load_pickle_graph(graph_file_name) -> MultiGraph:
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

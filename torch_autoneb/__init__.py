import pickle
from logging import getLogger

import torch
from networkx import MultiGraph, Graph, minimum_spanning_tree
from torch import optim

from torch_autoneb.helpers import pbar
from torch_autoneb.hyperparameters import NEBConfig, OptimConfig, AutoNEBConfig, LandscapeExplorationConfig
from torch_autoneb.models import ModelWrapper
from torch_autoneb.neb import NEB
from torch_autoneb.suggest import suggest_pair

__all__ = ["find_minimum", "neb", "auto_neb", "landscape_exploration", "load_pickle_graph"]

logger = getLogger(__name__)


def find_minimum(model: ModelWrapper, config: OptimConfig) -> dict:
    optimiser = config.optim_type(model.parameters(), **config.optim_args)  # type: optim.Optimizer

    # Wrap in scheduler
    if config.scheduler_type is not None:
        optimiser = config.scheduler_type(optimiser, **config.scheduler_args)

    # Initialise
    model.initialise_randomly()
    model.adapt_to_config(config.eval_config)

    # Optimise
    for _ in pbar(range(config.nsteps), "Find mimimum"):
        optimiser.zero_grad()
        model.apply(gradient=True)
        optimiser.step()
    result = {
        "coords": model.get_coords(),
    }

    # Analyse
    analysis = model.analyse()
    logger.debug(f"Found minimum: {analysis}.")
    result.update(analysis)
    return result


def neb(previous_cycle_data, model: ModelWrapper, config: NEBConfig) -> dict:
    # Initialise chain by inserting pivots
    start_path, target_distances = config.insert_method(previous_cycle_data, **config.insert_args)

    # Model and optimiser
    neb_model = NEB(model, start_path, target_distances)
    optim_config = config.optim_config
    optimiser = optim_config.optim_type(neb_model.parameters(), **optim_config.optim_args)  # type: optim.Optimizer

    # Wrap in scheduler
    if optim_config.scheduler_type is not None:
        optimiser = optim_config.scheduler_type(optimiser, **optim_config.scheduler_args)

    # Optimise
    for _ in pbar(range(optim_config.nsteps), "NEB"):
        # optimiser.zero_grad()  # has no effect, is overwritten anyway
        neb_model.apply(gradient=True)
        optimiser.step()
    result = {
        "path_coords": neb_model.path_coords.detach().clone(),
        "target_distances": target_distances
    }

    # Analyse
    analysis = neb_model.analyse(config.subsample_pivot_count)
    saddle_analysis = {key: value for key, value in analysis.items() if "saddle_" in key}
    logger.debug(f"Found saddle: {saddle_analysis}.")
    result.update(analysis)
    return result


def auto_neb(m1, m2, graph: MultiGraph, model: ModelWrapper, config: AutoNEBConfig):
    # Continue existing cycles or start from scratch
    if m2 in graph[m1]:
        existing_edges = graph[m1][m2]
        previous_cycle_idx = max(existing_edges[m1][m2])
        previous_cycle_data = existing_edges[m1][m2][previous_cycle_idx]
        start_cycle_idx = previous_cycle_idx + 1
    else:
        previous_cycle_data = {
            "path_coords": torch.cat([graph.nodes[m]["coords"].view(1, -1) for m in (m1, m2)]),
            "target_distances": torch.ones(1)
        }
        start_cycle_idx = 1
    assert start_cycle_idx <= config.cycle_count

    # Run NEB and add to graph
    for cycle_idx in pbar(range(start_cycle_idx, config.cycle_count + 1), "AutoNEB"):
        cycle_config = config.neb_configs[cycle_idx - 1]
        connection_data = neb(previous_cycle_data, model, cycle_config)
        graph.add_edge(m1, m2, key=cycle_idx, **connection_data)


def landscape_exploration(graph: MultiGraph, model: ModelWrapper, config: LandscapeExplorationConfig):
    try:
        with pbar(desc="Landscape Exploration") as bar:
            while True:
                # Suggest new pair based on current graph
                m1, m2 = suggest_pair(graph, config.value_key, config.weight_key, *config.suggest_methods)
                if m1 is None or m2 is None:
                    break
                auto_neb(m1, m2, graph, model, config.auto_neb_config)
                bar.update()

                # Analyse new saddle
                simple_graph = to_simple_graph(graph, config.weight_key)
                best_saddle = simple_graph[m1][m2][config.weight_key]
                in_mst_str = "included" if minimum_spanning_tree(simple_graph, config.weight_key) else "not included"
                logger.info(f"Saddle loss between {m1} and {m2} is {best_saddle}, {in_mst_str} in MST.")
    except KeyboardInterrupt:
        raise
    finally:
        simple_graph = to_simple_graph(graph, config.weight_key)
        mst_graph = minimum_spanning_tree(simple_graph, config.weight_key)
        mean_saddle_loss = sum(simple_graph.get_edge_data(*edge)[config.weight_key] for edge in mst_graph.edges) / len(mst_graph.edges)
        logger.info(f"Average loss in MST: {mean_saddle_loss}.")


def to_simple_graph(graph: MultiGraph, weight_key: str) -> Graph:
    """
    Reduce the MultiGraph to a simple graph by reducing each multi-edge
    to its lowest container.
    """
    simple_graph = Graph()
    for node in graph:
        simple_graph.add_node(node, **graph.nodes[node])

    for m1 in graph:
        for m2 in graph[m1]:
            best_edge_key = min(graph[m1][m2], key=lambda key: graph[m1][m2][key][weight_key])
            best_edge_data = graph[m1][m2][best_edge_key]
            best_edge_data["cycle_idx"] = best_edge_key
            simple_graph.add_edge(m1, m2, **best_edge_data)

    return simple_graph


def load_pickle_graph(graph_file_name) -> MultiGraph:
    with open(graph_file_name, "rb") as file:
        graph = pickle.load(file)

        # Check file structure
        if not isinstance(graph, MultiGraph):
            raise ValueError(f"{graph_file_name} does not contain a nx.MultiGraph")
    return graph

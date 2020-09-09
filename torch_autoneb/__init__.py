import gzip
import os
import pickle
from logging import getLogger

import torch
from networkx import MultiGraph, Graph, minimum_spanning_tree
from torch import optim

import torch_autoneb.config as config
import torch_autoneb.fill as fill
import torch_autoneb.helpers as helper
import torch_autoneb.models as models
import torch_autoneb.neb_model as neb_model
import torch_autoneb.visualise as visualise

__all__ = ["find_minimum", "neb", "auto_neb", "landscape_exploration", "load_pickle_graph", "config", "helper", "models", "neb_model", "fill", "visualise"]

logger = getLogger(__name__)


def find_minimum(model: models.ModelWrapper, optim_config: config.OptimConfig) -> dict:
    optimiser = optim_config.algorithm_type(model.parameters(), **optim_config.algorithm_args)  # type: optim.Optimizer

    # Scheduler
    if optim_config.scheduler_type is not None:
        scheduler = optim_config.scheduler_type(optimiser, **optim_config.scheduler_args)
    else:
        scheduler = None

    # Initialise
    model.initialise_randomly()
    if optim_config.eval_config is not None:
        model.adapt_to_config(optim_config.eval_config)

    # Optimise
    for _ in helper.pbar(range(optim_config.nsteps), "Find mimimum"):
        model.apply(gradient=True)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        # todo tensorboard logging or similar
    result = {
        "coords": model.get_coords().to("cpu"),
    }

    # Analyse
    analysis = model.analyse()
    logger.debug(f"Found minimum: {analysis}.")
    result.update(analysis)
    return result


def neb(previous_cycle_data, model: models.ModelWrapper, neb_config: config.NEBConfig) -> dict:
    # Initialise chain by inserting pivots
    start_path, target_distances = neb_config.insert_method(previous_cycle_data, **neb_config.insert_args)

    # Model
    neb_mod = neb_model.NEB(model, start_path, target_distances)
    neb_mod.adapt_to_config(neb_config)

    # Load optimiser
    optim_config = neb_config.optim_config
    # HACK: Optimisers only like parameters registered to autograd -> proper solution would keep several model instances as path and nudge their gradients after backward.
    neb_mod.path_coords.requires_grad_(True)
    optimiser = optim_config.algorithm_type(neb_mod.parameters(), **optim_config.algorithm_args)  # type: optim.Optimizer
    # HACK END: We don't want autograd to mingle with our computations
    neb_mod.path_coords.requires_grad_(False)
    if "weight_decay" in optimiser.defaults:
        assert optimiser.defaults["weight_decay"] == 0, "NEB is not compatible with weight decay on the optimiser. Set weight decay on NEB instead."

    # Scheduler
    if optim_config.scheduler_type is not None:
        scheduler = optim_config.scheduler_type(optimiser, **optim_config.scheduler_args)
    else:
        scheduler = None

    # Optimise
    for _ in helper.pbar(range(optim_config.nsteps), "NEB"):
        neb_mod.apply(gradient=True)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
    result = {
        "path_coords": neb_mod.path_coords.clone().to("cpu"),
        "target_distances": target_distances.to("cpu")
    }

    # Analyse
    analysis = neb_mod.analyse(neb_config.subsample_pivot_count)
    saddle_analysis = {key: value for key, value in analysis.items() if "saddle_" in key}
    logger.debug(f"Found saddle: {saddle_analysis}.")
    result.update(analysis)
    return result


def auto_neb(m1, m2, graph: MultiGraph, model: models.ModelWrapper, config: config.AutoNEBConfig, callback: callable = None):
    # Continue existing cycles or start from scratch
    if m2 in graph[m1]:
        existing_edges = graph[m1][m2]
        previous_cycle_idx = max(existing_edges)
        connection_data = graph[m1][m2][previous_cycle_idx]
        start_cycle_idx = previous_cycle_idx + 1
    else:
        connection_data = {
            "path_coords": torch.cat([graph.nodes[m]["coords"].view(1, -1) for m in (m1, m2)]),
            "target_distances": torch.ones(1)
        }
        start_cycle_idx = 1
    assert start_cycle_idx <= config.cycle_count

    # Run NEB and add to graph
    for cycle_idx in helper.pbar(range(start_cycle_idx, config.cycle_count + 1), "AutoNEB"):
        cycle_config = config.neb_configs[cycle_idx - 1]
        connection_data = neb(connection_data, model, cycle_config)
        graph.add_edge(m1, m2, key=cycle_idx, **helper.move_to(connection_data, "cpu"))
        if callback is not None:
            callback()


def suggest_pair(graph: MultiGraph, config: config.LandscapeExplorationConfig):
    engines_args = config.suggest_args
    if engines_args is None:
        engines_args = [{}] * len(config.suggest_methods)

    for engine, engines_args in zip(config.suggest_methods, engines_args):
        m1, m2 = engine(graph, config, **engines_args)
        if m1 is None or m2 is None:
            assert m1 is None and m2 is None
        else:
            if logger is not None:
                logger.info("Connecting %d <-> %d based on %s." % (m1, m2, engine.__name__))
            assert m1 is not m2
            return m1, m2
    return None, None


def landscape_exploration(graph: MultiGraph, model: models.ModelWrapper, lex_config: config.LandscapeExplorationConfig, callback: callable = None):
    weight_key = lex_config.weight_key
    try:
        with helper.pbar(desc="Landscape Exploration") as bar:
            while True:
                # Suggest new pair based on current graph
                m1, m2 = suggest_pair(graph, lex_config)
                if m1 is None or m2 is None:
                    break
                auto_neb(m1, m2, graph, model, lex_config.auto_neb_config, callback=callback)
                bar.update()

                # Analyse new saddle
                simple_graph = to_simple_graph(graph, weight_key)
                best_saddle = simple_graph[m1][m2][weight_key]
                in_mst_str = "included" if minimum_spanning_tree(simple_graph, weight_key) else "not included"
                logger.info(f"Saddle loss between {m1} and {m2} is {best_saddle}, {in_mst_str} in MST.")
    finally:
        if len(graph.edges) > 0:
            simple_graph = to_simple_graph(graph, weight_key)
            mst_graph = minimum_spanning_tree(simple_graph, weight_key)
            mean_saddle_loss = sum(simple_graph.get_edge_data(*edge)[weight_key] for edge in mst_graph.edges) / len(mst_graph.edges)
            logger.info(f"Average loss in MST: {mean_saddle_loss}.")


def to_simple_graph(graph: MultiGraph, weight_key: str, cycle_count: int = None) -> Graph:
    """
    Reduce the MultiGraph to a simple graph by reducing each multi-edge
    to its lowest container.
    """
    simple_graph = Graph()
    for node in graph:
        simple_graph.add_node(node, **graph.nodes[node])

    for m1 in graph:
        for m2 in graph[m1]:
            if cycle_count is not None and len(graph[m1][m2]) < cycle_count:
                continue

            best_edge_key = min(graph[m1][m2], key=lambda key: graph[m1][m2][key][weight_key])
            best_edge_data = graph[m1][m2][best_edge_key]
            best_edge_data["cycle_idx"] = best_edge_key
            simple_graph.add_edge(m1, m2, **best_edge_data)

    return simple_graph


def load_pickle_graph(graph_file_name: str) -> MultiGraph:
    open_fn = gzip.open if graph_file_name.endswith(".gz") else open
    with open_fn(graph_file_name, "rb") as file:
        graph = pickle.load(file)

    # Check file content
    if not isinstance(graph, MultiGraph):
        raise ValueError(f"{graph_file_name} does not contain a nx.MultiGraph")

    return graph


def store_pickle_graph(graph: MultiGraph, graph_file_name: str):
    # In case the writing is interrupted, first write to a temporary file
    open_fn = gzip.open if graph_file_name.endswith(".gz") else open
    directory, filename = os.path.split(graph_file_name)
    with open_fn(os.path.join(directory, "~" + filename), "wb") as file:
        pickle.dump(graph, file)

    # Then overwrite the old file
    os.rename(os.path.join(directory, "~" + filename), graph_file_name)

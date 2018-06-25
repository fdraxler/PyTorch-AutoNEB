from argparse import ArgumentParser
from logging import getLogger
from os import makedirs
from shutil import copyfile

from networkx import MultiGraph
from os.path import isdir, isfile, join
from yaml import safe_load

from torch_autoneb import load_pickle_graph, find_minimum, landscape_exploration, models
from torch_autoneb.config import replace_instanciation, LandscapeExplorationConfig, OptimConfig
from torch_autoneb.datasets import load_dataset
from torch_autoneb.helpers import pbar
from torch_autoneb.models import ModelWrapper

logger = getLogger(__name__)


def read_config_file(config_file: str):
    with open(config_file, "r") as file:
        config = safe_load(file)

    architecture, arguments = replace_instanciation(config["architecture"], models)
    if "dataset" in config:
        dataset, input_size, output_size = load_dataset(config["dataset"])
        arguments["input_size"], arguments["output_size"] = input_size, output_size
    model = architecture(**arguments)
    model = ModelWrapper(model)

    minima_count = int(config["minima_count"])
    min_config = OptimConfig.from_dict(config["minimum"])
    lex_config = LandscapeExplorationConfig.from_dict(config["exploration"])

    return model, minima_count, min_config, lex_config


def main():
    parser = ArgumentParser()
    parser.add_argument("project_directory", nargs=1)
    parser.add_argument("config_file", nargs=1)
    args = parser.parse_args()

    project_directory = args.project_directory[0]
    config_file = args.config_file[0]
    graph_path, project_config_path = setup_project(config_file, project_directory)

    model, minima_count, min_config, lex_config = read_config_file(project_config_path)

    # === Create/load graph ===
    if isfile(graph_path):
        graph = load_pickle_graph(graph_path)
    else:
        graph = MultiGraph()

    # === Ensure the specified number of minima ===
    for _ in pbar(range(len(graph.nodes), minima_count), "Finding minima"):
        minimum_data = find_minimum(model, min_config)
        graph.add_node(max(graph.nodes) + 1 if len(graph.nodes) > 0 else 1, **minimum_data)

    # === Connect minima ordered by suggestion algorithm ===
    landscape_exploration(graph, model, lex_config)


def setup_project(config_file, project_directory):
    if not isfile(config_file):
        raise ValueError(f"Config file {config_file} not found.")

    # Create project directory
    if not isdir(project_directory):
        makedirs(project_directory)

    project_config_path = join(project_directory, "config.yaml")
    graph_path = join(project_directory, "graph.p")

    if not isfile(project_config_path):
        # Copy the config to the project
        if isfile(graph_path):
            logger.warning("Graph file graph.p exists, but no config file 'config.yaml' was not found in project directory.")
        copyfile(config_file, project_config_path)
    else:
        # Make sure that the config file has not been modified
        with open(project_config_path, "r") as file:
            project_config = safe_load(file)
        with open(config_file, "r") as file:
            original_config = safe_load(file)
        if project_config != original_config:
            raise ValueError(f"Config file 'config.yaml' in project directory is structurally different from original config '{config_file}'")
    return graph_path, project_config_path


if __name__ == '__main__':
    main()

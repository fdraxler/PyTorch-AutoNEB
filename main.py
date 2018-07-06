import sys
from argparse import ArgumentParser
from logging import getLogger, StreamHandler, FileHandler, INFO
from os import makedirs
from shutil import copyfile
from time import strftime

from networkx import MultiGraph
from os.path import isdir, isfile, join
from torch.nn import NLLLoss
from yaml import safe_load

from torch_autoneb import load_pickle_graph, find_minimum, landscape_exploration, models, store_pickle_graph
from torch_autoneb.config import replace_instanciation, LandscapeExplorationConfig, OptimConfig, EvalConfig
from torch_autoneb.datasets import load_dataset
from torch_autoneb.helpers import pbar, move_to
from torch_autoneb.models import ModelWrapper, DataModel, CompareModel

logger = getLogger(__name__)


def read_config_file(config_file: str, move_to_device: bool = True):
    with open(config_file, "r") as file:
        config = safe_load(file)

    architecture, arguments = replace_instanciation(config["architecture"], models)
    if "dataset" in config:
        datasets, input_size, output_size = load_dataset(config["dataset"])
        arguments["input_size"], arguments["output_size"] = input_size, output_size
    else:
        datasets = None
    model = architecture(**arguments)
    if datasets is not None:
        model = DataModel(CompareModel(model, NLLLoss()), datasets)
    model = ModelWrapper(model)
    if move_to_device:
        model.to(config["device"])

    minima_count = int(config["minima_count"])
    min_config = OptimConfig.from_dict(config["minimum"])
    lex_config = LandscapeExplorationConfig.from_dict(config["exploration"])

    return model, minima_count, min_config, lex_config


def repair_graph(graph, model):
    model.adapt_to_config(EvalConfig(1024))
    for m in graph.nodes:
        found_none = False
        for value in graph.nodes[m].values():
            if value is None:
                found_none = True
        if found_none:
            graph.nodes[m].update(model.analyse())
    return graph


def main():
    parser = ArgumentParser()
    parser.add_argument("project_directory", nargs=1)
    parser.add_argument("config_file", nargs=1)
    parser.add_argument("--no-backup", default=False, action="store_true")
    args = parser.parse_args()

    project_directory = args.project_directory[0]
    config_file = args.config_file[0]
    graph_path, project_config_path = setup_project(config_file, project_directory)

    model, minima_count, min_config, lex_config = read_config_file(project_config_path)

    # Setup Logger
    root_logger = getLogger()
    root_logger.setLevel(INFO)
    root_logger.addHandler(StreamHandler(sys.stdout))
    root_logger.addHandler(FileHandler(join(project_directory, "exploration.log")))

    # === Create/load graph ===
    if isfile(graph_path):
        if not args.no_backup:
            root_logger.info("Copying current 'graph.p' to backup file.")
            copyfile(graph_path, graph_path.replace(".p", f"_bak{strftime('%Y%m%d-%H%M')}.p"))
        else:
            root_logger.info("Not creating a backup of 'graph.p' because of user request.")
        graph = repair_graph(load_pickle_graph(graph_path), model)
    else:
        graph = MultiGraph()

    # Call this after every optmisiation
    def save_callback():
        store_pickle_graph(graph, graph_path)

    # === Ensure the specified number of minima ===
    for _ in pbar(range(len(graph.nodes), minima_count), "Finding minima"):
        minimum_data = find_minimum(model, min_config)
        graph.add_node(max(graph.nodes) + 1 if len(graph.nodes) > 0 else 1, **move_to(minimum_data, "cpu"))
        save_callback()

    # === Connect minima ===
    landscape_exploration(graph, model, lex_config, callback=save_callback)


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
            raise ValueError(f"Config file 'config.yaml' in project directory is structurally different from original config '{config_file}'.")
    return graph_path, project_config_path


if __name__ == '__main__':
    main()

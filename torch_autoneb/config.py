from copy import deepcopy

from torch import optim
from torch.optim import lr_scheduler

import torch_autoneb
import torch_autoneb.suggest as suggest


def replace_instanciation(config, package):
    if isinstance(config, dict):
        # Name + args
        name = config["name"]
        del config["name"]
        return getattr(package, name), config
    elif config is None:
        return None, None
    elif isinstance(config, str):
        # Just the name
        return getattr(package, config), {}
    return config, {}


def _deep_update(source: dict, target: dict):
    for key, value in target.items():
        if key in source and isinstance(value, dict) and key != "args":
            _deep_update(source[key], value)
        else:
            source[key] = value

class BaseConfig:
    def value_string(self, level=0):
        value_string = self.__class__.__name__ + "["
        for key, value in self.__dict__.items():
            value_string += "\n" + "  " * (level + 1) + key + ": "
            if isinstance(value, BaseConfig):
                value_string += value.value_string(level + 1)
            elif isinstance(value, (list, tuple)):
                value_string += "["
                for sub_value in value:
                    value_string += "\n" + "  " * (level + 2)
                    if isinstance(sub_value, BaseConfig):
                        value_string += sub_value.value_string(level + 2)
                    else:
                        value_string += str(sub_value)
                    value_string += ","
                value_string += "\n" + "  " * (level + 1) + "]"
            else:
                value_string += str(value)
        value_string += "\n" + "  " * level + "]"
        return value_string
    
    def __repr__(self):
        return self.value_string()


class EvalConfig(BaseConfig):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @staticmethod
    def from_dict(config_dict: dict):
        return EvalConfig(**deepcopy(config_dict))


class OptimConfig(BaseConfig):
    def __init__(self, nsteps: int, algorithm_type, algorithm_args: dict, scheduler_type, scheduler_args: dict, eval_config: EvalConfig):
        self.nsteps = nsteps
        self.algorithm_type = algorithm_type
        self.algorithm_args = algorithm_args
        self.scheduler_type = scheduler_type
        self.scheduler_args = scheduler_args
        self.eval_config = eval_config

    @staticmethod
    def from_dict(config_dict: dict):
        config_dict = deepcopy(config_dict)
        config_dict["algorithm_type"], config_dict["algorithm_args"] = replace_instanciation(config_dict["algorithm"], optim)
        del config_dict["algorithm"]
        if "scheduler" in config_dict:
            config_dict["scheduler_type"], config_dict["scheduler_args"] = replace_instanciation(config_dict["scheduler"], lr_scheduler)
            del config_dict["scheduler"]
        else:
            config_dict["scheduler_type"], config_dict["scheduler_args"] = None, None
        if "eval" in config_dict:
            config_dict["eval_config"] = EvalConfig.from_dict(config_dict["eval"])
            del config_dict["eval"]
        else:
            config_dict["eval_config"] = None
        return OptimConfig(**config_dict)


class NEBConfig(BaseConfig):
    def __init__(self, spring_constant: float, weight_decay: float, insert_method: callable, insert_args: dict, subsample_pivot_count: int, optim_config: OptimConfig):
        self.spring_constant = spring_constant
        self.weight_decay = weight_decay
        self.insert_method = insert_method
        self.insert_args = insert_args
        self.subsample_pivot_count = subsample_pivot_count
        self.optim_config = optim_config

    @staticmethod
    def from_dict(config_dict: dict):
        config_dict = deepcopy(config_dict)
        config_dict["insert_method"], config_dict["insert_args"] = replace_instanciation(config_dict["insert"], torch_autoneb.fill)
        config_dict["spring_constant"] = float(config_dict["spring_constant"])
        del config_dict["insert"]
        config_dict["optim_config"] = OptimConfig.from_dict(config_dict["optim"])
        del config_dict["optim"]
        if "weight_decay" not in config_dict:
            config_dict["weight_decay"] = 0
        return NEBConfig(**config_dict)


class AutoNEBConfig(BaseConfig):
    def __init__(self, neb_configs: list):
        self.cycle_count = len(neb_configs)
        self.neb_configs = neb_configs

    @staticmethod
    def from_list(configs_list: iter):
        current_state = {}
        cycles = []
        for cycle in configs_list:
            _deep_update(current_state, deepcopy(cycle))
            cycles.append(NEBConfig.from_dict(deepcopy(current_state)))
        return AutoNEBConfig(cycles)


class LandscapeExplorationConfig(BaseConfig):
    def __init__(self, value_key: str, weight_key: str, suggest_methods: list, suggest_args: list, auto_neb_config: AutoNEBConfig):
        self.value_key = value_key
        self.weight_key = weight_key
        self.suggest_methods = suggest_methods
        self.suggest_args = suggest_args
        self.auto_neb_config = auto_neb_config

    @staticmethod
    def from_dict(config_dict: dict):
        config_dict = deepcopy(config_dict)
        config_dict["suggest_methods"], config_dict["suggest_args"] = zip(*[replace_instanciation(engine, suggest) for engine in config_dict["suggest"]])
        del config_dict["suggest"]
        config_dict["auto_neb_config"] = AutoNEBConfig.from_list(config_dict["autoneb"])
        del config_dict["autoneb"]
        return LandscapeExplorationConfig(**config_dict)

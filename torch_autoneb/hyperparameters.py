from torch_autoneb.fill import Fill


class EvalHyperparameters:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size


class OptimHyperparameters:
    def __init__(self, nsteps: int, optim_name: str, optim_args: dict, eval_config: EvalHyperparameters):
        self.nsteps = nsteps
        self.optim_args = optim_args
        self.optim_name = optim_name
        self.eval_config = eval_config


class NEBHyperparameters:
    def __init__(self, spring_constant: float, fill_method: Fill, optim_config: OptimHyperparameters, insert_count: int, subsample_pivot_count: int):
        self.insert_count = insert_count
        self.fill_method = fill_method
        self.spring_constant = spring_constant
        self.optim_config = optim_config
        self.subsample_pivot_count = subsample_pivot_count


class AutoNEBHyperparameters:
    def __init__(self, neb_configs: list):
        self.cycle_count = len(neb_configs)
        self.neb_configs = neb_configs


class LandscapeExplorationHyperparameters:
    def __init__(self, value_key: str, weight_key: str, suggest_engines: list, auto_neb_config: AutoNEBHyperparameters):
        self.value_key = value_key
        self.weight_key = weight_key
        self.suggest_engines = suggest_engines
        self.auto_neb_config = auto_neb_config

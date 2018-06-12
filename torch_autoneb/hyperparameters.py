class Hyperparameters:
    optim_name = None


class CycleHyperparameters:
    def __init__(self, cycle_count):
        self.cycle_count = cycle_count
        self.hyperparameter_sets = [Hyperparameters() for _ in range(cycle_count)]

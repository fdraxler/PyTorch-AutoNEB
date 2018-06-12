import torch

from torch_autoneb.fill import Fill


class FillEqual(Fill):
    """
    Fill in the specified amound of pivots between each pair of existing pivots.
    """

    def fill(self, path_coords, insert_count, weights, transition_data):
        return PyTorchNEB.fill_chain(path_coords, [torch.linspace(0, 1, insert_count + 2)[1:-1]] * (path_coords.shape[0] - 1), weights)

    def __repr__(self):
        return "equal"

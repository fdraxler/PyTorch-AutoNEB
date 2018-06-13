from itertools import chain

from torch import Tensor

from torch_autoneb.hyperparameters import NEBHyperparameters
from torch_autoneb.helpers import fast_inter_distance
from torch_autoneb.models import ModelWrapper


class NEB:
    def __init__(self, model: ModelWrapper, path_coords: Tensor):
        self.model = model
        self.path_coords = path_coords

    def adapt_to_config(self, config: NEBHyperparameters):
        """
        Adapts the model to hyperparameters, if supported.

        :param config: The hyperparameters relevant for evaluating the model.
        """
        self.model.adapt_to_config(config.optim_config.eval_config)

    def forward(self, gradient=False):
        raise NotImplementedError


def fill_chain(existing_chain: Tensor, insert_alphass: list, relative_lengths: Tensor = None):
    """
    Extend a chain of coordinates by inserting additional items (through linear interpolation).

    :param existing_chain: The current chain
    :param insert_alphass: A list of float values in range (0, 1) specifying the relative position between each pair of pivots.
    :param relative_lengths: The current relative lengths between pivots. New relative lengths are only returned when a tensor is passed.
    :return:
    """
    existing_count = existing_chain.shape[0]
    assert len(insert_alphass) == existing_count - 1, "For each connection in line, a number of new items must be specified!"

    new_count = sum(map(len, insert_alphass)) + existing_count
    new_chain = existing_chain.new(new_count, *existing_chain.shape[1:])
    if relative_lengths is not None:
        # noinspection PyUnboundLocalVariable
        new_relative_lengths = relative_lengths.new(new_count - 1)

    # Fill first position
    new_chain[0] = existing_chain[0]
    offset = 1
    # Fill in missing positions
    for i, insert_alphas in enumerate(insert_alphass):
        start = existing_chain[i]
        stop = existing_chain[i + 1]
        if relative_lengths is not None:
            # noinspection PyUnboundLocalVariable
            section_weight = relative_lengths[i]

        last_alpha = 0
        for alpha in chain(insert_alphas, [1]):
            assert alpha > last_alpha
            # Position is linear interpolation
            new_chain[offset] = (1 - alpha) * start + alpha * stop
            if relative_lengths is not None:
                # noinspection PyUnboundLocalVariable
                new_relative_lengths[offset - 1] = (alpha - last_alpha) * section_weight
            last_alpha = alpha
            offset += 1

    if relative_lengths is not None:
        return new_chain, new_relative_lengths / sum(new_relative_lengths)
    else:
        return new_chain, None


def distribute_by_weights(path, nimages, path_target=None, weights=None, climbing_pivots=None):
    """
    Redistribute the pivots on the path so that they are spaced as given by the weights.
    """
    # Ensure storage for coordinates
    if path_target is None:
        path_target = path.new(nimages, path.shape[1])
    else:
        assert path_target is not path, "Source must be unequal to target for redistribution"
        assert path_target.shape[0] == nimages
    # Ensure weights
    if weights is None:
        weights = path.new(nimages - 1).fill_(1)
    else:
        assert len(weights.shape) == 1
        assert weights.shape[0] == nimages - 1

    # In climbing mode, reinterpolate only between the climbing images
    if climbing_pivots is not None:
        assert path.shape[0] == nimages, "Cannot change number of items when reinterpolating with respect to climbing images."
        assert len(climbing_pivots) == nimages
        assert all(isinstance(b, bool) for b in climbing_pivots), "Image must be climbing or not."
        start = 0
        for i, is_climbing in enumerate(climbing_pivots):
            if is_climbing or i == nimages - 1:
                distribute_by_weights(path[start:i + 1], i + 1 - start, path_target[start:i + 1], weights[start:i])
                start = i
        return path_target

    if path is path_target:
        # For the computation the original path is necessary
        path_source = path.clone()
    else:
        path_source = path

    # The current distances between elements on chain
    current_distances = fast_inter_distance(path)
    target_positions = (weights / weights.sum()).cumsum(0) * current_distances.sum()  # Target positions of elements (spaced by weights)
    # Put each new item spaced by weights (measured along line) on the line
    last_idx = 0  # Index of previous pivot
    pos_prev = 0.  # Position of previous pivot on chain
    pos_next = current_distances[last_idx]  # Position of next pivot on chain
    path_target[0] = path_source[0]
    for i in range(1, nimages - 1):
        position = target_positions[i - 1]
        while position > pos_next:
            last_idx += 1
            pos_prev = pos_next
            pos_next += current_distances[last_idx]

        t = (position - pos_prev) / (pos_next - pos_prev)
        path_target[i] = (t * path_source[last_idx + 1] + (1 - t) * path_source[last_idx])
    path_target[nimages - 1] = path_source[-1]

    return path_target

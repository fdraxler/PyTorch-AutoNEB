from itertools import chain

from torch import Tensor, linspace

from torch_autoneb import pbar
from torch_autoneb.helpers import fast_inter_distance
from torch_autoneb.hyperparameters import NEBHyperparameters
from torch_autoneb.models import ModelWrapper, ModelInterface


class NEB(ModelInterface):
    def __init__(self, model: ModelWrapper, path_coords: Tensor, target_distances: Tensor = None):
        self.model = model
        self.path_coords = path_coords.clone()
        self.path_coords.requires_grad_()
        self.path_coords.grad = path_coords.clone().zero_()
        self.target_distances = target_distances
        self.spring_constant = 0

    def get_device(self):
        return self.path_coords.device

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self._check_device()

    def _check_device(self):
        new_device = self.model.stored_parameters[0].device
        if new_device != self.path_coords.device:
            self.path_coords = self.path_coords.to(new_device)

    def parameters(self):
        return [self.path_coords]

    def adapt_to_config(self, config: NEBHyperparameters):
        """
        Adapts the model to hyperparameters, if supported.

        :param config: The hyperparameters relevant for evaluating the model.
        """
        self.model.adapt_to_config(config.optim_config.eval_config)
        self.spring_constant = config.spring_constant

    def apply(self, gradient=False, **kwargs):
        npivots = self.path_coords.shape[0]
        losses = self.path_coords.new(npivots)

        # Redistribute if spring_constant == inf
        assert self.target_distances is not None or not gradient, "Cannot compute gradient if target distances are unavailable"
        if gradient and self.spring_constant == float("inf"):
            self.path_coords[:] = distribute_by_weights(self.path_coords, self.path_coords.shape[0], weights=self.target_distances)

        # Compute losses (and gradients)
        for i in range(npivots):
            self.model.set_coords_no_grad(self.path_coords[i])
            losses[i] = self.model.apply(gradient and (0 < i < npivots))
            if gradient and (0 < i < npivots):
                # If the coordinates were modified, move them back to the cache
                self.path_coords[i] = self.model.get_coords(update_cache=True)
                self.path_coords.grad[i] = self.model.get_grad(update_cache=False)
            else:
                # Make sure no gradient is there
                self.path_coords.grad[i].zero_()

        # Compute NEB gradients as in (Henkelmann & Jonsson, 2000)
        if gradient:
            distances = fast_inter_distance(self.path_coords)
            for i in range(1, npivots - 1):
                d_prev, d_next = distances[i - 1:i + 1]
                td_prev, td_next = self.target_distances[i - 1:i + 1]
                l_prev, loss, l_next = losses[i - 1:i + 2]

                # Compute tangent
                tangent = self.compute_tangent(d_next, d_prev, i, l_next, l_prev, loss)

                # Project gradients perpendicular to tangent
                self.path_coords.grad[i] -= self.path_coords.grad[i].dot(tangent) * tangent

                if self.spring_constant < float("inf"):
                    # Spring force parallel to tangent
                    self.path_coords.grad[i] += (d_prev - td_prev) - (d_next - td_next) * self.spring_constant * tangent

        return losses.max()

    def compute_tangent(self, d_next, d_prev, i, l_next, l_prev, loss):
        if l_prev < loss > l_next or l_prev > loss < l_next:
            # Interpolate tangent at maxima/minima to make convergence smooth
            t_prev = (self.path_coords[i] - self.path_coords[i - 1]) / d_prev
            t_next = (self.path_coords[i + 1] - self.path_coords[i]) / d_next
            l_max = max(abs(loss - l_prev), abs(loss - l_next))
            l_min = min(abs(loss - l_prev), abs(loss - l_next))
            l_max /= l_max + l_min
            l_min /= l_max + l_min
            if l_prev > l_next:
                return l_min * t_prev + l_max * t_next
            else:
                return l_max * t_prev + l_min * t_next
        elif l_prev > l_next:
            # Tangent to the previous
            return (self.path_coords[i] - self.path_coords[i - 1]) / d_prev
        else:
            # Tangent to the next
            return (self.path_coords[i + 1] - self.path_coords[i]) / d_next

    def analyse(self, sub_pivot_count=9):
        # Collect stats here
        analysis = {}

        dense_pivot_count = (self.path_coords.shape[0] - 1) * (sub_pivot_count + 1) + 1
        alphas = linspace(0, 1, sub_pivot_count + 2)[:-1]
        for i in pbar(range(dense_pivot_count), "Saddle analysis"):
            base_pivot = i // (sub_pivot_count + 1)
            sub_pivot = i % (sub_pivot_count + 1)

            if sub_pivot == 0:
                # Coords of pivot
                coords = self.path_coords[base_pivot]
            else:
                # Or interpolation between pivots
                alpha = alphas[sub_pivot]
                coords = self.path_coords[base_pivot] * (1 - alpha) + self.path_coords[base_pivot + 1] * alpha

            # Retrieve values from model analysis
            self.model.set_coords_no_grad(coords)
            point_analysis = self.model.analyse()
            for key, value in point_analysis.items():
                dense_key = "dense_" + key
                if dense_key not in analysis:
                    analysis[dense_key] = value.new(dense_pivot_count, *value.shape)
                analysis[dense_key][i] = value

        # Compute saddle values
        for key, value in list(analysis.items()):
            if len(value.shape) == 1 or value.shape[1] == 1:
                analysis[key.replace("dense_", "saddle_")] = value.max()
            else:
                print(key)

        return analysis


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
        new_relative_lengths = relative_lengths.new(new_count - 1)

    # Fill first position
    new_chain[0] = existing_chain[0]
    offset = 1
    # Fill in missing positions
    for i, insert_alphas in enumerate(insert_alphass):
        start = existing_chain[i]
        stop = existing_chain[i + 1]
        if relative_lengths is not None:
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


def distribute_by_weights(path: Tensor, nimages: int, path_target: Tensor = None, weights: Tensor = None, climbing_pivots: list = None):
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

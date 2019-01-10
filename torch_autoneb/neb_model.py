from torch import Tensor, linspace

import torch_autoneb.config as config
import torch_autoneb.helpers as helpers
import torch_autoneb.models as models


class NEB(models.ModelInterface):
    def __init__(self, model: models.ModelWrapper, path_coords: Tensor, target_distances: Tensor = None):
        """
        Creates a NEB instance that is prepared for evaluating the band.

        For computing gradients, adapt_to_config has to be called with valid values.
        """
        self.model = model
        self.path_coords = path_coords.clone()
        self.target_distances = target_distances
        self._check_device()

        # This will raise an exception if gradients are computed without adapt_to_config() being called
        self.spring_constant = -1
        self.weight_decay = -1

    def get_device(self):
        return self.path_coords.device

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self._check_device()

    def _check_device(self):
        new_device = self.model.stored_parameters[0].device
        if new_device != self.path_coords.device:
            previous = self.path_coords
            self.path_coords = self.path_coords.to(new_device)
            if self.path_coords.grad is not None:
                self.path_coords.grad = previous.grad.to(new_device)
            if self.target_distances is not None:
                self.target_distances = self.target_distances.to(new_device)

    def _assert_grad(self):
        if self.path_coords.grad is None:
            self.path_coords.grad = self.path_coords.new(self.path_coords.shape).zero_()

    def parameters(self):
        return [self.path_coords]

    def adapt_to_config(self, neb_config: config.NEBConfig):
        """
        Adapts the model to hyperparameters, if supported.

        :param neb_config: The hyperparameters relevant for evaluating the model.
        """
        if neb_config.optim_config.eval_config is not None:
            self.model.adapt_to_config(neb_config.optim_config.eval_config)
        self.spring_constant = neb_config.spring_constant
        self.weight_decay = neb_config.weight_decay

    def apply(self, gradient=False, **kwargs):
        npivots = self.path_coords.shape[0]
        losses = self.path_coords.new(npivots)

        # Redistribute if spring_constant == inf
        assert self.target_distances is not None or not gradient, "Cannot compute gradient if target distances are unavailable"
        if gradient and self.spring_constant == float("inf"):
            self.path_coords[:] = distribute_by_weights(self.path_coords, self.path_coords.shape[0], weights=self.target_distances).data

        # Assert gradient storage is available
        if gradient:
            self._assert_grad()

        # Compute losses (and gradients)
        for i in range(npivots):
            self.model.set_coords_no_grad(self.path_coords[i])
            losses[i] = self.model.apply(gradient and (0 < i < npivots - 1))
            if gradient and (0 < i < npivots):
                # If the coordinates were modified, move them back to the cache
                self.model.get_coords(update_cache=True, target=self.path_coords[i])
                self.model.get_grad(update_cache=True, target=self.path_coords.grad[i])

                assert self.weight_decay >= 0
                if self.weight_decay > 0:
                    self.path_coords.grad[i] += self.weight_decay * self.path_coords[i]
            elif gradient:
                # Make sure no gradient is there on the endpoints
                self.path_coords.grad[i].zero_()

        # Compute NEB gradients as in (Henkelmann & Jonsson, 2000)
        if gradient:
            distances = (self.path_coords[:-1] - self.path_coords[1:]).norm(2, 1)
            for i in range(1, npivots - 1):
                d_prev, d_next = distances[i - 1].item(), distances[i].item()
                td_prev, td_next = self.target_distances[i - 1].item(), self.target_distances[i].item()
                l_prev, loss, l_next = losses[i - 1].item(), losses[i].item(), losses[i + 1].item()

                # Compute tangent
                tangent = self.compute_tangent(d_next, d_prev, i, l_next, l_prev, loss)

                # Project gradients perpendicular to tangent
                self.path_coords.grad[i] -= self.path_coords.grad[i].dot(tangent) * tangent

                assert self.spring_constant > 0
                if self.spring_constant < float("inf"):
                    # Spring force parallel to tangent
                    self.path_coords.grad[i] += (d_prev - td_prev) - (d_next - td_next) * self.spring_constant * tangent

        return losses.max().item()

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
        alphas = linspace(0, 1, sub_pivot_count + 2)[:-1].to(self.path_coords.device)
        for i in helpers.pbar(range(dense_pivot_count), "Saddle analysis"):
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
                if not isinstance(value, Tensor):
                    value = Tensor([value]).squeeze()
                if dense_key not in analysis:
                    analysis[dense_key] = value.new(dense_pivot_count, *value.shape)
                analysis[dense_key][i] = value

        # Compute saddle values
        for key, value in list(analysis.items()):
            if len(value.shape) == 1 or value.shape[1] == 1:
                analysis[key.replace("dense_", "saddle_")] = value.max().item()
            else:
                print(key)

        # Compute lengths
        end_to_end_distance = (self.path_coords[-1] - self.path_coords[0]).norm(2)
        analysis["lengths"] = (self.path_coords[1:] - self.path_coords[:-1]).norm(2, 1) / end_to_end_distance
        analysis["length"] = end_to_end_distance

        return analysis


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
    current_distances = (path_source[:-1] - path_source[1:]).norm(2, 1)
    target_positions = (weights / weights.sum()).cumsum(0) * current_distances.sum()  # Target positions of elements (spaced by weights)

    # Put each new item spaced by weights (measured along line) on the line
    last_idx = 0  # Index of previous pivot
    pos_prev = 0.  # Position of previous pivot on chain
    pos_next = current_distances[last_idx].item()  # Position of next pivot on chain
    path_target[0] = path_source[0]
    for i in range(1, nimages - 1):
        position = target_positions[i - 1].item()
        while position > pos_next:
            last_idx += 1
            pos_prev = pos_next
            pos_next += current_distances[last_idx].item()

        t = (position - pos_prev) / (pos_next - pos_prev)
        path_target[i] = (t * path_source[last_idx + 1] + (1 - t) * path_source[last_idx])
    path_target[nimages - 1] = path_source[-1]

    return path_target

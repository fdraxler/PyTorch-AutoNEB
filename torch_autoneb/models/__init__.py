import operator

import torch
from functools import reduce

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torch_autoneb.hyperparameters import EvalHyperparameters


class ModelInterface:
    def get_device(self):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, gradient=False, **kwargs):
        raise NotImplementedError


class ModelWrapper(ModelInterface):
    """
    Wraps around models which should return a loss for a given input batch.
    """

    def __init__(self, model: Module):
        super().__init__()
        self.model = model
        self.stored_parameters = self.model.parameters()
        # noinspection PyProtectedMember
        self.stored_buffers = self.model._all_buffers()
        number_of_coords = sum(size for _, _, size, _ in self.iterate_params_buffers())
        self.device = self.stored_parameters[0].device
        self.coords = torch.empty(number_of_coords, dtype=torch.float32).to(self.device).zero_()
        self.coords.grad = self.coords.copy().zero_()

    def get_device(self):
        return self.device

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def iterate_params_buffers(self):
        offset = 0
        for param in self.stored_parameters:
            size = reduce(operator.mul, param.data.shape)
            data = param
            yield offset, data, size, False
            offset += size
        for buffer in self.stored_buffers:
            size = reduce(operator.mul, buffer.shape)
            yield offset, buffer, size, True
            offset += size

    def _check_device(self):
        self.device = self.stored_parameters[0].device
        if self.device != self.coords.device:
            self.coords = self.coords.to(self.device)

    def _coords_to_model(self):
        self._check_device()

        final = 0
        for offset, data, size, is_buffer in self.iterate_params_buffers():
            # Copy coordinates
            data[:] = self.coords[offset:offset + size]

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def _coords_to_cache(self):
        self._check_device()

        final = 0
        for offset, data, size, is_buffer in self.iterate_params_buffers():
            # Copy coordinates
            self.coords[offset:offset + size] = data.detach().view(-1)

            # Copy gradient
            if data.grad is None:
                self.coords.grad[offset:offset + size] = 0
            else:
                self.coords.grad[offset:offset + size] = data.grad.detach().view(-1)

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def get_coords(self, target: Tensor = None, copy: bool = True, update_cache: bool = True) -> Tensor:
        """
        Retrieve the coordinates of the current model.

        :param target: If given, copy the data to this destination.
        :param copy: Copy the data before returning it.
        :param update_cache: Before copying, retrieve the current coordinates from the model. Set `False` if you are sure that they have been retrieved before.
        :return: A tensor holding the coordinates.
        """
        assert target is None or not copy, "Must copy if target is specified"

        if update_cache:
            self._coords_to_cache()

        if target is None:
            if copy:
                return self.coords.copy()
            else:
                return self.coords.detach()
        else:
            target[:] = self.coords.to(target.device)
            return target

    def get_grad(self, target: Tensor = None, copy: bool = True, update_cache: bool = True):
        """
        Retrieve the gradient of the current model.

        :param target:
        :param copy:
        :param update_cache:
        :return:
        """
        assert target is None or not copy, "Must copy if target is specified"

        if update_cache:
            self._coords_to_cache()

        if target is None:
            if copy:
                return self.coords.grad.copy()
            else:
                return self.coords.grad.detach()
        else:
            target[:] = self.coords.grad.to(target.device)
            return target

    def set_coords_no_grad(self, coords, copy=True, update_model=True):
        self._check_device()
        coords = coords.to(self.device)

        if copy:
            self.coords[:] = coords
        else:
            self.coords = coords

        if update_model:
            self._coords_to_model()

    def adapt_to_config(self, config: EvalHyperparameters):
        """
        Adapts the model to hyperparameters, if supported.

        :param config: The hyperparameters relevant for evaluating the model.
        """
        if hasattr(self.model, "adapt_to_config"):
            self.model.adapt_to_config(config)

    def forward(self, gradient=False, **kwargs):
        # Forward data -> loss
        if gradient:
            self.model.train()
        else:
            self.model.eval()
        with torch.set_grad_enabled(gradient):
            loss = self.model(**kwargs)

        # Backpropation
        if gradient:
            loss.backward()


class DataModel(Module):
    def __init__(self, model: Module, datasets: dict):
        super().__init__()

        self.batch_size = None
        self.model = model

        self.datasets = datasets
        self.dataset_loaders = {}
        self.dataset_iters = {}

    def adapt_to_config(self, config: EvalHyperparameters):
        self.batch_size = config.batch_size

    def forward(self, dataset="train", **kwargs):
        # Retrieve batch
        while True:
            # Make sure that there is a non-empty iterator
            if dataset not in self.dataset_iters:
                if dataset not in self.dataset_loaders:
                    # todo multi-threaded batch loading
                    self.dataset_loaders[dataset] = DataLoader(self.datasets[dataset], self.batch_size, True)
                loader = self.dataset_loaders[dataset]
                self.dataset_iters[dataset] = iter(loader)
            iterator = self.dataset_iters[dataset]

            try:
                batch = next(iterator)
                break
            except StopIteration:
                del self.dataset_iters[dataset]

        # Apply model on batch and use returned loss
        data, target = batch
        return self.model(data, target, **kwargs)


class LossModel(Module):
    def __init__(self, model: Module, loss: Module):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, data, target, **kwargs):
        soft_pred = self.model(data, target)
        return self.loss(soft_pred, target)

import operator

import torch
from functools import reduce

from torch.nn import Module
from torch.utils.data import DataLoader

from torch_autoneb.hyperparameters import EvalHyperparameters


class ModelWrapper:
    """
    Wraps around models which should return a loss for a given input batch.
    """

    def __init__(self, model: Module):
        super().__init__()
        self.model = model
        self.stored_parameters = self.model.parameters()
        self.stored_buffers = self.model._all_buffers()
        number_of_coords = sum(size for _, _, size, _ in self.iterate_params_buffers())
        self.device = self.stored_parameters[0].device
        self.coords = torch.FloatTensor(number_of_coords).to(self.device).zero_()
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

    def get_coords(self, target=None, copy=True, update_cache=True):
        assert target is None or not copy, "Must copy if target is specified"

        if update_cache:
            self._coords_to_cache()

        if target is None:
            if copy:
                return self.coords.copy()
            else:
                return self.coords
        else:
            target[:] = self.coords.to(target.device)
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


class DataModel:
    def __init__(self, model: Module, datasets: dict):
        self.batch_size = None
        self.model = model

        self.datasets = datasets
        self.dataset_loaders = {}
        self.dataset_iters = {}

    def adapt_to_config(self, config: EvalHyperparameters):
        self.batch_size = config.batch_size

    def forward(self, dataset="train", **kwargs):
        while True:
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

        data, target = batch
        return self.model(data, target, **kwargs)

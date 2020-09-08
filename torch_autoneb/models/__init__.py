import operator
from functools import reduce
from math import sqrt
from typing import Iterable

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader

from torch_autoneb.config import EvalConfig
from torch_autoneb.models.cnn import CNN
from torch_autoneb.models.densenet import DenseNet
from torch_autoneb.models.mlp import MLP
from torch_autoneb.models.resnet import ResNet
from torch_autoneb.models.simple import Eggcarton, CurvyValley, Flat, Linear


class ModelInterface:
    def get_device(self):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, gradient=False, **kwargs):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def analyse(self):
        raise NotImplementedError


def param_init(mod: Module):
    if isinstance(mod, nn.Linear):
        n = mod.in_features
        mod.weight.data.normal_(0, 1. / sqrt(n))
        if mod.bias is not None:
            mod.bias.data.zero_()
    elif isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        n = mod.in_channels
        for k in mod.kernel_size:
            n *= k
        mod.weight.data.normal_(0, 1. / sqrt(n))
        if mod.bias is not None:
            mod.bias.data.zero_()
    elif isinstance(mod, (nn.Conv3d, nn.ConvTranspose3d)):
        n = mod.in_channels
        for k in mod.kernel_size:
            n *= k
        mod.weight.data.normal_(0, 1. / sqrt(n))
        if mod.bias is not None:
            mod.bias.data.zero_()
    elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        mod.reset_parameters()
    elif hasattr(mod, "initialise_randomly"):
        mod.initialise_randomly()
    elif len(mod._parameters) == 0:
        # Module has no parameters on its own
        pass
    else:
        print("Don't know how to initialise %s" % mod.__class__.__name__)


class ModelWrapper(ModelInterface):
    """
    Wrapper around model. Inner model should handle data loading and return a value to be minimized.
    """

    def __init__(self, model: Module, parameters: Iterable[Parameter] = None, buffers: Iterable[Tensor] = None):
        super().__init__()
        self.model = model
        if parameters is not None:
            parameters = list(parameters)
        self.stored_parameters = parameters
        if buffers is not None:
            buffers = list(buffers)
        self.stored_buffers = buffers
        self.number_of_dimensions = sum(size for _, _, size, _ in self.iterate_params_buffers())
        device = self.stored_parameters[0].device
        self.coords = torch.empty(self.number_of_dimensions, dtype=torch.float32).to(device).zero_()
        self.coords.grad = self.coords.clone().zero_()

    def get_device(self):
        return self.coords.device

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        # noinspection PyProtectedMember
        self.stored_buffers = list(self.model.buffers())
        self._check_device()

    def parameters(self):
        return self.stored_parameters

    def initialise_randomly(self):
        self.model.apply(param_init)

    def iterate_params_buffers(self, gradient=False):
        offset = 0
        for param in self.stored_parameters:
            size = reduce(operator.mul, param.data.shape)
            data = param
            yield offset, data.data if not gradient else data.grad.data, size, False
            offset += size
        for buffer in self.stored_buffers:
            size = reduce(operator.mul, buffer.shape, 1)
            yield offset, buffer if not gradient else None, size, True
            offset += size

    def _check_device(self):
        new_device = self.stored_parameters[0].device
        if self.coords.device != self.coords.device:
            self.coords.to(new_device)

    def _coords_to_model(self):
        self._check_device()

        final = 0
        for offset, data, size, is_buffer in self.iterate_params_buffers():
            # Copy coordinates
            if len(data.shape) == 0:
                data[0] = self.coords[offset:offset + size].item()
            else:
                data[:] = self.coords[offset:offset + size].reshape(data.shape)

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def _coords_to_cache(self):
        self._check_device()

        final = 0
        for offset, tensor, size, is_buffer in self.iterate_params_buffers():
            # Copy coordinates
            self.coords[offset:offset + size] = tensor.data.view(-1)

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def _grad_to_cache(self):
        self._check_device()

        final = 0
        for offset, tensor, size, is_buffer in self.iterate_params_buffers(True):
            # Copy gradient
            if tensor is None:
                self.coords.grad[offset:offset + size] = 0
            else:
                self.coords.grad[offset:offset + size] = tensor.data.view(-1)

            # Size consistency check
            final = final + size
        assert final == self.coords.shape[0]

    def get_coords(self, target: Tensor = None, copy: bool = True, update_cache: bool = True) -> Tensor:
        """
        Retrieve the coordinates of the current model.

        :param target: If given, copy the data to this destination.
        :param copy: Copy the data before returning it.
        :param update_cache: Before copying, retrieve the current coordinates from the model. Set `False` only if you are sure that they have been retrieved before.
        :return: A tensor holding the coordinates.
        """
        assert target is None or copy, "Must copy if target is specified"

        if update_cache:
            self._coords_to_cache()

        if target is None:
            if copy:
                return self.coords.clone()
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
        :param update_cache: Before copying, retrieve the current gradient from the model. Set `False` only if you are sure that it has been retrieved before.
        :return:
        """
        assert target is None or copy, "Must copy if target is specified"

        if update_cache:
            self._grad_to_cache()

        if target is None:
            if copy:
                return self.coords.grad.clone()
            else:
                return self.coords.grad.detach()
        else:
            target[:] = self.coords.grad.to(target.device)
            return target

    def set_coords_no_grad(self, coords, update_model=True):
        self._check_device()
        self.coords[:] = coords.to(self.coords.device)

        if update_model:
            self._coords_to_model()

    def adapt_to_config(self, config: EvalConfig):
        """
        Adapts the model to hyperparameters, if supported.

        :param config: The hyperparameters relevant for evaluating the model.
        """
        if hasattr(self.model, "adapt_to_config"):
            self.model.adapt_to_config(config)

    def apply(self, gradient=False, **kwargs):
        # Forward data -> loss
        if gradient:
            self.model.zero_grad()
            self.model.train()
        else:
            self.model.eval()
        with torch.set_grad_enabled(gradient):
            loss = self.model(**kwargs)

        # Backpropation
        if gradient:
            loss.backward()
        return loss.item()

    def analyse(self):
        self.model.eval()
        with torch.set_grad_enabled(False):
            return self.model.analyse()


class DataModel(Module):
    def __init__(self, model: Module, datasets: dict):
        super().__init__()

        self.batch_size = None
        self.model = model

        self.datasets = datasets
        self.dataset_loaders = {}
        self.dataset_iters = {}

    def adapt_to_config(self, config: EvalConfig):
        self.batch_size = config.batch_size
        self.dataset_iters = {}
        if hasattr(self.model, "adapt_to_config"):
            self.model.adapt_to_config(config)

    def forward(self, dataset="train", **kwargs):
        # Retrieve batch
        while True:
            # Make sure that there is a non-empty iterator
            if dataset not in self.dataset_iters:
                if dataset not in self.dataset_loaders:
                    self.dataset_loaders[dataset] = DataLoader(self.datasets[dataset], self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
                loader = self.dataset_loaders[dataset]
                self.dataset_iters[dataset] = iter(loader)
            iterator = self.dataset_iters[dataset]

            try:
                batch = next(iterator)
                break
            except StopIteration:
                del self.dataset_iters[dataset]

        # Apply model on batch and use returned loss
        device_batch = self.batch_to_device(batch)
        return self.model(*device_batch, **kwargs)

    def batch_to_device(self, batch):
        device = list(self.model.parameters())[0].device
        return [item.to(device) for item in batch]

    def analyse(self):
        # Go through all data points and accumulate stats
        analysis = {}
        for ds_name, dataset in self.datasets.items():
            ds_length = len(dataset)
            for batch in DataLoader(dataset, self.batch_size):
                batch = self.batch_to_device(batch)
                result = self.model.analyse(*batch)
                for key, value in result.items():
                    ds_key = f"{ds_name}_{key}"
                    if ds_key not in analysis:
                        analysis[ds_key] = 0
                    analysis[ds_key] += value * batch[0].shape[0] / ds_length
        return analysis


class CompareModel(Module):
    ERROR = "error"
    ERROR_5 = "error_5"
    LOSS = "loss"

    def __init__(self, model: Module, loss: Module):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, data, target, **kwargs):
        soft_pred = self.model(data, **kwargs)
        return self.loss(soft_pred, target)

    def analyse(self, data, target):
        # Compute some statistics over the given batch
        soft_pred = self.model(data)
        hard_pred = soft_pred.data.sort(1, True)[1]

        hard_pred_correct = hard_pred[:].eq(target.data.view(-1, 1)).cumsum(1)
        error = 1 - hard_pred_correct[:, 0].float().mean().item()
        loss = self.loss(soft_pred, target).item()
        return {
            CompareModel.ERROR: error,
            CompareModel.LOSS: loss,
        }

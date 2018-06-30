import operator
from collections import OrderedDict
from functools import reduce

from torch import Tensor, int64
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, Module, Conv2d, MaxPool2d, BatchNorm2d, Dropout2d
from torch.nn.functional import log_softmax

from torch_autoneb.helpers import ntuple


class CNN(Module):
    def __init__(self, conv_depth, num_filters, filter_size, padding_size, pool_size, dense_depth, num_dense_units, input_size, output_size, pool_every=1, batch_norm=True, dropout=0):
        super().__init__()

        # Need to keep track of dimensions to know which linear layer we need later
        in_channels = int(input_size[0])
        dims = Tensor(tuple(input_size[1:])).to(dtype=int64)

        # Expand definitions
        pool_size = ntuple(pool_size, 2)
        filter_size = ntuple(filter_size, 2)
        padding_size = ntuple(padding_size, 2)
        num_filters = ntuple(num_filters, conv_depth)
        num_dense_units = ntuple(num_dense_units, dense_depth)
        non_lin = ReLU()

        # Build up convolutional layers
        conv_layers = OrderedDict()
        for i in range(conv_depth):
            layer = OrderedDict()

            layer["conv"] = Conv2d(in_channels, num_filters[i], kernel_size=filter_size, padding=padding_size)
            dims -= Tensor(filter_size).to(dtype=int64) - 1 - Tensor(padding_size).to(int64) * 2
            in_channels = num_filters[i]
            # print(dims, in_channels)

            if i % pool_every == pool_every - 1:
                layer["maxpool"] = MaxPool2d(pool_size)
                dims /= Tensor(pool_size).to(int64)

            if batch_norm:
                layer["batchnorm"] = BatchNorm2d(num_filters[i])
            layer["nonlin"] = non_lin
            layer["dropout"] = Dropout2d(dropout)

            conv_layers[f"conv_{i}"] = (Sequential(layer))

        # Fully connected layers
        previous_size = in_channels * reduce(operator.mul, dims)
        dense_layers = OrderedDict()
        for i in range(dense_depth):
            layer = OrderedDict()
            layer["linear"] = Linear(previous_size, num_dense_units[i])
            previous_size = num_dense_units[i]
            if batch_norm:
                layer["batch_norm"] = BatchNorm1d(num_dense_units[i])
            layer["non_lin"] = non_lin
            if dropout > 0:
                layer["dropout"] = Dropout(float(dropout))
            dense_layers[f"fc_{i}"] = Sequential(layer)

        self.conv = Sequential(conv_layers)
        self.fc = Sequential(dense_layers)
        self.final = Linear(previous_size, output_size)

    def forward(self, data):
        data = self.conv(data)
        data = data.reshape(data.shape[0], -1)
        data = self.fc(data)
        data = self.final(data)
        data = log_softmax(data, 1)
        return data

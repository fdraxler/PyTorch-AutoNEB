import operator
from collections import OrderedDict
from typing import Iterable
from functools import reduce

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, Module
from torch.nn.functional import log_softmax

from torch_autoneb.helpers import ntuple


class MLP(Module):
    def __init__(self, depth, widths, input_size, output_size, batch_norm=True, dropout=0):
        super().__init__()
        widths = ntuple(widths, depth)

        layers = OrderedDict()
        previous_size = reduce(operator.mul, input_size) if isinstance(input_size, Iterable) else input_size
        non_lin = ReLU()
        for i, width in enumerate(widths):
            layer = OrderedDict()
            layer["linear"] = Linear(previous_size, width)
            previous_size = width
            if batch_norm:
                layer["batch_norm"] = BatchNorm1d(width)
            layer["non_lin"] = non_lin
            if dropout > 0:
                layer["dropout"] = Dropout(float(dropout))
            layers["layer_{i}".format(i=i)] = Sequential(layer)
        self.body = Sequential(layers)
        self.final = Linear(previous_size, output_size)

    def forward(self, data):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        data = self.body(data)
        data = self.final(data)
        data = log_softmax(data, 1)
        return data

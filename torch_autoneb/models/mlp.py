import operator

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, Softmax, LogSoftmax

from torch_autoneb.helpers import ntuple
from functools import reduce


class MLP:
    def __init__(self, depth, widths, input_size, batch_norm=True, dropout=0):
        widths = ntuple(widths, depth)

        layers = []
        previous_size = reduce(operator.mul, input_size)
        non_lin = ReLU()
        for i, width in enumerate(widths):
            layer = [Linear(previous_size, width)]
            previous_size = width
            if batch_norm:
                layer.append(BatchNorm1d(width))
            layer.append(non_lin)
            if dropout > 0:
                layer.append(Dropout(float(dropout)))
            layers.append(Sequential(layer))
        self.body = Sequential(layers)
        self.softmax = LogSoftmax()

    def forward(self, data):
        return self.softmax(self.body(data))

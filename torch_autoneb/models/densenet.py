import math

import torch
from torch.nn import functional as F, BatchNorm2d, Conv2d, Module, Linear, Sequential
from torch.nn.functional import log_softmax

"""
Based on https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
"""


class Bottleneck(Module):
    def __init__(self, n_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = BatchNorm2d(n_channels)
        self.conv1 = Conv2d(n_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(inter_channels)
        self.conv2 = Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(Module):
    def __init__(self, n_channels, growth_rate):
        super(SingleLayer, self).__init__()
        self.bn1 = BatchNorm2d(n_channels)
        self.conv1 = Conv2d(n_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(Module):
    def __init__(self, n_channels, n_out_channels):
        super(Transition, self).__init__()
        self.bn1 = BatchNorm2d(n_channels)
        self.conv1 = Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(Module):
    def __init__(self, growth_rate, depth, reduction, bottleneck, input_size, output_size):
        super(DenseNet, self).__init__()

        if bottleneck:
            n_dense_blocks = int((depth - 4) / 6)
        else:
            n_dense_blocks = int((depth - 4) / 3)

        # Assumes an images size of 32x32

        n_channels = 2 * growth_rate
        self.conv1 = Conv2d(input_size[0], n_channels, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate

        self.bn1 = BatchNorm2d(n_channels)
        self.linear_out = Linear(n_channels, output_size)

    def _make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck):
        layers = []
        for i in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(Bottleneck(n_channels, growth_rate))
            else:
                layers.append(SingleLayer(n_channels, growth_rate))
            n_channels += growth_rate
        return Sequential(*layers)

    def forward(self, out):
        out = self.conv1(out)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.linear_out(out)
        out = log_softmax(out, 1)
        return out

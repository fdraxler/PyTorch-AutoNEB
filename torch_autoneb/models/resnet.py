import operator
from functools import reduce

from torch.nn import Module, BatchNorm2d, ReLU, AvgPool2d, Conv2d, Linear, Sequential
from torch.nn.functional import log_softmax

"""
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py for the CIFAR10/CIFAR100 setting described in https://arxiv.org/abs/1512.03385.
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, non_linearity=ReLU()):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.nonlin1 = non_linearity
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlin1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin1(out)

        return out

    def reset_self(self):
        pass


class ResNet(Module):
    def __init__(self, depth, input_size, output_size, dropout=0.0, batch_norm=True):
        super().__init__()

        non_lin = ReLU()

        assert dropout == 0, "Dropout not allowed for ResNet"
        assert batch_norm

        # Configure ResNet
        # number of layers per block
        if (depth - 2) % 6 != 0:
            raise ValueError("Only ResNet-X with X=6N-2 allowed where N natural number.")

        layers = (depth - 2) // 6
        # number of filters for each block
        filters = [16, 32, 64]
        # Subsampling at first layer in each block
        strides = [1, 2, 2]

        # Build ResNet
        self.conv1 = conv3x3(int(input_size[0]), filters[0])
        self.inplanes = filters[0]
        self.bn1 = BatchNorm2d(filters[0])
        self.non_lin1 = non_lin

        # Each layer makes the input smaller using stride, then adds `layers` layers of two layers each with a skip
        self.layer1 = self._make_layer(filters[0], layers, strides[0], non_lin)
        self.layer2 = self._make_layer(filters[1], layers, strides[1], non_lin)
        self.layer3 = self._make_layer(filters[2], layers, strides[2], non_lin)

        # Average each filter
        self.avgpool = AvgPool2d(int(input_size[1]) // reduce(operator.mul, strides, 1))
        self.linear_out = Linear(filters[-1], output_size)

        self.kind = depth
        self.non_linearity = non_lin.__class__
        self.batch_norm = batch_norm

    def _make_layer(self, planes, blocks, stride, non_linearity):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Sequential(
                Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes)
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample, non_linearity=non_linearity)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, non_linearity=non_linearity))

        return Sequential(*layers)

    def forward(self, data):
        data = self.conv1(data)
        data = self.bn1(data)
        data = self.non_lin1(data)

        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)

        data = self.avgpool(data)
        data = data.reshape(data.shape[0], -1)
        data = self.linear_out(data)

        data = log_softmax(data, 1)

        return data

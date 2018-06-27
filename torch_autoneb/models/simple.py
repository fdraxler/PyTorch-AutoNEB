from math import pi

import torch
from torch.nn import Module, Parameter


class SimpleEnergy(Module):
    def __init__(self, dims=2):
        super().__init__()
        self.location = Parameter(torch.zeros(dims))

    def forward(self):
        raise NotImplementedError

    def analyse(self):
        return {"loss": self.forward().item()}


class Eggcarton(SimpleEnergy):
    def forward(self):
        return (self.location * (2 * pi)).cos().mean()

    def initialise_randomly(self):
        self.location.data.uniform_(-1, 1)

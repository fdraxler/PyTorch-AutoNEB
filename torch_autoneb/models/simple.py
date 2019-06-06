from math import pi

import torch
from torch.nn import Module, Parameter


class SimpleEnergy(Module):
    def __init__(self, dims=2):
        super().__init__()
        self.location = Parameter(torch.zeros(dims))

    def forward(self):
        raise NotImplementedError

    def initialise_randomly(self):
        raise NotImplementedError

    def analyse(self):
        return {"loss": self.forward().item()}

    def sample_on_grid(self, x_space, y_space):
        z = torch.zeros(y_space.shape[0], x_space.shape[0])
        for i, xi in enumerate(x_space):
            for j, yj in enumerate(y_space):
                self.location.data[:] = torch.Tensor((xi, yj))
                z[j, i] = self.forward().item()
        z -= z.min()
        return z


class Eggcarton(SimpleEnergy):
    def forward(self):
        return (self.location * (2 * pi)).cos().sum()

    def initialise_randomly(self):
        self.location.data.uniform_(-2, 2)


class CurvyValley(SimpleEnergy):
    def __init__(self):
        super().__init__()

        self.alpha = 1
        self.x0 = (5 / 2 * pi)
        self.beta = self.alpha / (2 * self.x0 ** 2)

        self.random_idx = -1

    def forward(self):
        # Reset the location to a box around the center
        bound = 15
        self.location.data[self.location.data > bound] = bound
        self.location.data[self.location.data < -bound] = -bound

        x = self.location[0]
        y = self.location[1]
        curvy = (y - x.sin()) ** 2
        bounding = (self.beta * x ** 4 - self.alpha * x ** 2) / (self.alpha ** 2 / (4 * self.beta)) + 1

        return curvy + bounding

    def initialise_randomly(self):
        self.random_idx += 1
        self.random_idx %= 2
        self.location.data[:] = torch.rand(2) * 2
        if self.random_idx == 0:
            self.location.data[0] -= self.x0
        else:
            self.location.data[0] += self.x0


class Flat(SimpleEnergy):
    def __init__(self):
        super().__init__()

        self.register_buffer("iterated_buffer", torch.tensor(0.0))
        self.random_idx = -1

    def forward(self):
        if self.training:
            self.iterated_buffer += 1
        return (self.location.sum()) * 0

    def initialise_randomly(self):
        self.random_idx += 1
        self.random_idx %= 2
        self.location.data[:] = torch.tensor([0, self.random_idx])
        self.iterated_buffer[0] = 0


class Linear(SimpleEnergy):
    def __init__(self):
        super().__init__()
        self.random_idx = -1

    def forward(self):
        x = self.location[0]
        y = self.location[1]
        return y

    def initialise_randomly(self):
        self.random_idx += 1
        self.random_idx %= 2
        self.location.data[:] = torch.tensor([self.random_idx, 0])

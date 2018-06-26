import operator
from functools import reduce
from itertools import repeat

import collections
from torch import Tensor

try:
    from tqdm import tqdm as pbar
except ModuleNotFoundError:
    class pbar:
        def __init__(self, iterable=None, desc=None, total=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            yield from self.iterable

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def update(self, N=None):
            pass


def ntuple(x, n):
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) != 1:
            assert len(x) == n, "Expected length %d, but found %d" % (n, len(x))
            return x
        else:
            x = x[0]
    return tuple(repeat(x, n))


def basic_dist_op(a, b):
    return (a - b).norm(2, 1)


def move_to(something, target):
    if isinstance(something, list):
        return [move_to(item, target) for item in something]
    elif isinstance(something, dict):
        return {key: move_to(value, target) for key, value in something.items()}
    elif isinstance(something, Tensor):
        return something.to(target)


def fast_inter_distance(tensor, mode=None):
    assert len(tensor.shape) == 2

    if not tensor.is_cuda:
        tot_size = reduce(operator.mul, tensor.shape)
        if (tot_size > 1e7 and mode is None) or mode == "sep":
            # Move pieces to GPU and compute the distance
            last = tensor[0:1].cuda()
            norms = tensor.new(tensor.shape[0] - 1)
            for i in range(1, tensor.shape[0]):
                current = tensor[i:i + 1].cuda()
                norms[i - 1] = basic_dist_op(current, last).cpu()[0]
                last = current
            return norms
        elif (tot_size > 1e5 and mode is None) or mode == "gpu":
            # Move as whole to GPU and compute distances
            tensor_cuda = tensor.cuda()
            return basic_dist_op(tensor_cuda[:-1], tensor_cuda[1:]).cpu()
    return basic_dist_op(tensor[:-1], tensor[1:])

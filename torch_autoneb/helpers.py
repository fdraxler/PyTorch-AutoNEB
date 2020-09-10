import collections
import logging
from itertools import repeat

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

logger = logging.getLogger(__name__)


def ntuple(x, n):
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) != 1:
            assert len(x) == n, "Expected length %d, but found %d" % (n, len(x))
            return x
        else:
            x = x[0]
    return tuple(repeat(x, n))


def move_to(something, target):
    if isinstance(something, list):
        return [move_to(item, target) for item in something]
    elif isinstance(something, dict):
        return {key: move_to(value, target) for key, value in something.items()}
    elif isinstance(something, Tensor):
        return something.to(target)
    else:
        return something

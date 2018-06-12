from itertools import repeat

import collections


def ntuple(x, n):
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) != 1:
            assert len(x) == n, "Expected length %d, but found %d" % (n, len(x))
            return x
        else:
            x = x[0]
    return tuple(repeat(x, n))

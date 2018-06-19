from logging import getLogger

from networkx import MultiGraph

logger = getLogger(__name__)


def suggest_pair(graph: MultiGraph, value_key: str, weight_key: str, engines: iter, engines_args: iter = None):
    if engines_args is None:
        engines_args = [{}] * len(engines)

    for engine, engines_args in zip(engines, engines_args):
        m1, m2 = engine(graph, value_key, weight_key, logger, **engines_args)
        if m1 is None or m2 is None:
            assert m1 is None and m2 is None
        else:
            if logger is not None:
                logger.info("Connecting %d <-> %d based on %s." % (m1, m2, engine.__name__))
            assert m1 is not m2
            return m1, m2
    return None, None

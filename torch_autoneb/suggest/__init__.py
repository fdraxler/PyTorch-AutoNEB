from networkx import MultiGraph


def suggest_pair(graph: MultiGraph, *engines, logger=None):
    for engine in engines:
        m1, m2 = engine(graph)
        if m1 is None or m2 is None:
            assert m1 is None and m2 is None
        else:
            if logger is not None:
                logger.info("Connecting %d <-> %d based on %s." % (m1, m2, engine.__name__))
            assert m1 is not m2
            return m1, m2
    return None, None

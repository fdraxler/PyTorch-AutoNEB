from logging import Logger

from networkx import MultiGraph


def create_unfinished_suggest(cycle_count):
    def unfinished_suggest(graph: MultiGraph, value_key, weight_key, logger: Logger):
        """
        Find unfinished AutoNEB runs
        """
        collected_unfinished = []
        for m1 in graph.nodes:
            for m2 in graph[m1]:
                # Check if final cycle is there
                cycles = set(graph[m1][m2].keys())
                if cycle_count not in cycles:
                    collected_unfinished.append((m1, m2))

                # Warn about missing intermediate cycles (should never occur)
                if logger is not None:
                    intermediate = set(range(1, max(cycles) + 1))
                    if cycles != intermediate:
                        logger.warning(f"Minima {m1} and {m2} are missing intermediate cycles {intermediate.difference(cycles)}.")

        if len(collected_unfinished) > 0:
            return collected_unfinished[0]
        return None, None

    return unfinished_suggest

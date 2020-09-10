from itertools import chain

import torch
from torch import Tensor


def equal(previous_cycle_data, count):
    """
    Insert `count` pivots between each existing pair of pivots.

    :param previous_cycle_data: The information on the previous path.
    :param count: The number of pivots to insert between each pair.
    :return:
    """
    path_coords = previous_cycle_data["path_coords"]
    weights = previous_cycle_data["target_distances"]
    return fill_chain(path_coords, [torch.linspace(0, 1, count + 2)[1:-1]] * (path_coords.shape[0] - 1), weights)


def highest(previous_cycle_data: dict, count: int, key: str, threshold=0.1):
    """
    Insert pivots where the linear interpolation of the value
    associated to `key` between existing pivots deviates more
    than threshold from the true values at those positions.

    :param previous_cycle_data: The information on the previous path.
    :param count: The maximum number of pivots to insert.
    :param key: The key to retrieve dense information from.
    :param threshold: The threshold after which a pivot should be inserted.
    :return:
    """
    path_coords = previous_cycle_data["path_coords"]
    weights = previous_cycle_data["target_distances"]
    dense_data = previous_cycle_data[key]
    
    if count == 0:
        return path_coords, weights
    
    assert (dense_data.shape[0] - 1) % (path_coords.shape[0] - 1) == 0, f"Bad shape of dense data {dense_data.shape}."
    assert float("nan") not in dense_data

    interpolate_count = int((dense_data.shape[0] - 1) / (path_coords.shape[0] - 1) - 1)
    scores = path_coords.new(path_coords.shape[0] - 1, interpolate_count).zero_()

    alphas = torch.linspace(0, 1, interpolate_count + 2)[1:-1]
    chain_min = 1e12
    chain_max = -1e12

    for i in range(path_coords.shape[0] - 1):
        # Pivot values
        idx_a = i * (interpolate_count + 1)
        score_a = dense_data[idx_a]
        score_b = dense_data[idx_a + interpolate_count + 1]

        # Update min/max of the chain
        chain_min = min(chain_min, min(score_a, score_b))
        chain_max = max(chain_max, max(score_a, score_b))

        # Compare intermediate values to linear interpolation between pivots
        for j, alpha in enumerate(alphas):
            scores[i, j] = dense_data[idx_a + j + 1] - (score_a * (1 - alpha) + score_b * alpha)

    # These casts go from numpy -> float
    # scores -= float(chain_min)
    scores /= float(chain_max - chain_min) + 1e-12
    values, order = scores.sort()

    max_values = values[:, -1]
    max_alphas = alphas[order[:, -1]]

    _, gap_order = max_values.sort()
    gaps_to_fill = gap_order[-count:]

    fill = []
    for gap_idx in range(path_coords.shape[0] - 1):
        if gap_idx in gaps_to_fill and max_values[gap_idx] > threshold:
            fill.append([max_alphas[gap_idx]])
        else:
            fill.append([])

    a, b = fill_chain(path_coords, fill, weights)
    return a, b


def leave(previous_cycle_data: dict, **kwargs):
    return previous_cycle_data["path_coords"], previous_cycle_data["target_distances"]


def fill_chain(existing_chain: Tensor, insert_alphass: list, relative_lengths: Tensor = None):
    """
    Extend a chain of coordinates by inserting additional items (through linear interpolation).

    :param existing_chain: The current chain
    :param insert_alphass: A list of float values in range (0, 1) specifying the relative position between each pair of pivots.
    :param relative_lengths: The current relative lengths between pivots. New relative lengths are only returned when a tensor is passed.
    :return:
    """
    existing_count = existing_chain.shape[0]
    assert len(insert_alphass) == existing_count - 1, "For each connection in line, a number of new items must be specified!"

    new_count = sum(map(len, insert_alphass)) + existing_count
    new_chain = existing_chain.new(new_count, *existing_chain.shape[1:])
    if relative_lengths is not None:
        new_relative_lengths = relative_lengths.new(new_count - 1)

    # Fill first position
    new_chain[0] = existing_chain[0]
    offset = 1
    # Fill in missing positions
    for i, insert_alphas in enumerate(insert_alphass):
        start = existing_chain[i]
        stop = existing_chain[i + 1]
        if relative_lengths is not None:
            section_weight = relative_lengths[i]

        last_alpha = 0
        for alpha in chain(insert_alphas, [1]):
            assert alpha > last_alpha
            # Position is linear interpolation
            new_chain[offset] = (1 - alpha) * start + alpha * stop
            if relative_lengths is not None:
                # noinspection PyUnboundLocalVariable
                new_relative_lengths[offset - 1] = (alpha - last_alpha) * section_weight
            last_alpha = alpha
            offset += 1

    if relative_lengths is not None:
        return new_chain, new_relative_lengths / sum(new_relative_lengths)
    else:
        return new_chain, None

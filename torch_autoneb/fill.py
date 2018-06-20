import torch

from torch_autoneb.neb import fill_chain


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

    return fill_chain(path_coords, fill, weights)

"""Ways of making a layer's activity sparse."""

import numpy as np


def n_active(n_units, sparsity):
    """Number of units allowed to be active, given the fraction active."""
    return max(1, round(sparsity * n_units))


def k_winners_take_all(activations, k):
    """Keep the k most active units in each pattern and set the rest to 0.

    A simple stand-in for lateral inhibition. Works on one pattern or a
    batch of patterns (one per row).
    """
    # Indices of the k largest activations in each pattern.
    winners = np.argsort(activations, axis=-1)[..., -k:]

    # 1 for winners, 0 for everyone else.
    mask = np.zeros_like(activations)
    np.put_along_axis(mask, winners, 1, axis=-1)

    return activations * mask


if __name__ == "__main__":
    example = np.array([0.2, 0.9, 0.5, 0.1, 0.7])
    print("activations: ", example)
    print("top 2 kept:  ", k_winners_take_all(example, 2))

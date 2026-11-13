"""Ways of making a layer's activity sparse."""

import numpy as np


def n_active(n_units, sparsity):
    """Number of units allowed to be active, given the fraction active."""
    if not isinstance(n_units, (int, np.integer)) or n_units < 1:
        raise ValueError("Number of units must be a positive integer")
    if not 0 <= sparsity <= 1:
        raise ValueError("Sparsity fraction must be between 0 and 1")
    return max(1, round(sparsity * n_units))


def k_winners_take_all(activations, k, return_mask=False):
    """Keep the k most active units in each pattern and set the rest to 0.

    A simple stand-in for lateral inhibition. Works on one pattern or a
    batch of patterns (one per row). Optionally return the mask for backprop.
    """
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= activations.shape[-1]:
        raise ValueError("k must be between 1 and the number of hidden units")
    if k == activations.shape[-1]:
        mask = np.ones_like(activations)
    else:
        # Retain the k largest activations in each pattern.
        winners = np.argsort(activations, axis=-1)[..., -k:]
        mask = np.zeros_like(activations)
        np.put_along_axis(mask, winners, 1, axis=-1)
    sparse = activations * mask
    if return_mask:
        return sparse, mask
    return sparse


if __name__ == "__main__":
    example = np.array([0.2, 0.9, 0.5, 0.1, 0.7])
    print("activations: ", example)
    print("top 2 kept:  ", k_winners_take_all(example, 2))

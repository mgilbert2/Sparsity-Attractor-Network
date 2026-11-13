"""Continuous Hopfield dynamics for the autoencoder's hidden layer."""

import numpy as np


def sigmoid(z):
    """Squash values into 0 to 1 without overflowing for large inputs."""
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


class Hopfield:
    """Settle hidden activations using symmetric recurrent connections."""

    def __init__(self, n_units, steps=3, seed=0):
        """Initialize recurrent weights and the number of settling sweeps."""
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("Hopfield steps must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        weights = rng.normal(0, 0.1, (n_units, n_units))
        self.W_R = (weights + weights.T) / 2
        # A Hopfield neuron receives input from other neurons, not itself.
        np.fill_diagonal(self.W_R, 0)
        self.steps = steps

    def forward(self, drive, return_cache=False):
        """Run fixed sweeps with the encoder's input held throughout recall."""
        initial_state = sigmoid(drive)
        state = initial_state.copy()
        updates = []
        for _ in range(self.steps):
            for neuron in range(len(self.W_R)):
                if return_cache:
                    previous_state = state.copy()
                # Update one neuron at a time so the Hopfield energy decreases.
                field = drive[..., neuron] + state @ self.W_R[:, neuron]
                state[..., neuron] = sigmoid(field)
                if return_cache:
                    updates.append((neuron, previous_state,
                                    state[..., neuron].copy()))
        if return_cache:
            return state, (initial_state, updates)
        return state

    def backward(self, gradient_state, cache):
        """Backpropagate through the sweeps to the encoder and recurrent weights."""
        initial_state, updates = cache
        gradient_state = gradient_state.copy()
        gradient_drive = np.zeros_like(initial_state)
        gradient_weights = np.zeros_like(self.W_R)
        for neuron, previous_state, activation in reversed(updates):
            gradient_field = (gradient_state[..., neuron]
                              * activation * (1 - activation))
            # The encoder drives this neuron on every sweep, not only initially.
            gradient_drive[..., neuron] += gradient_field
            previous_rows = previous_state.reshape(-1, len(self.W_R))
            gradient_weights[:, neuron] += (previous_rows.T
                                           @ gradient_field.reshape(-1))
            # The updated activation replaces this neuron's previous value.
            gradient_state[..., neuron] = 0
            gradient_state += gradient_field[..., None] * self.W_R[:, neuron]

        gradient_drive += gradient_state * initial_state * (1 - initial_state)
        # Project the update onto symmetric connections with no self-connections.
        gradient_weights = (gradient_weights + gradient_weights.T) / 2
        np.fill_diagonal(gradient_weights, 0)
        return gradient_drive, gradient_weights

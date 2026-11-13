"""Autoencoder with a recurrent Hopfield network as its hidden layer."""

import numpy as np

from src.hopfield import Hopfield, sigmoid


class Autoencoder:
    """Encode an input, settle the hidden Hopfield network, then decode."""

    def __init__(self, n_in, n_hidden, lr=0.2, seed=0, hopfield_steps=3,
                 hidden_target=None, sparsity_weight=1.0):
        """Initialize the encoder, decoder, and hidden recurrent connections."""
        if hidden_target is not None and not 0 <= hidden_target <= 1:
            raise ValueError("Hidden activity target must be between 0 and 1")
        if not np.isfinite(sparsity_weight) or sparsity_weight < 0:
            raise ValueError("Sparsity weight must be finite and nonnegative")
        rng = np.random.default_rng(seed)
        # Input to hidden weights (W in the diagram) and hidden biases.
        self.W = rng.normal(0, 0.1, (n_in, n_hidden))
        self.b = np.zeros(n_hidden)
        # Hidden to output weights (W' in the diagram) and output biases.
        self.W_out = rng.normal(0, 0.1, (n_hidden, n_in))
        self.b_out = np.zeros(n_in)
        self.hopfield = Hopfield(n_hidden, steps=hopfield_steps, seed=rng)
        self.hidden_target = hidden_target
        self.sparsity_weight = sparsity_weight
        self.lr = lr  # learning rate

    def encode(self, x):
        """Input pattern to the external drive of the hidden Hopfield neurons."""
        return x @ self.W + self.b

    def decode(self, h):
        """Hidden activations to output pattern."""
        return sigmoid(h @ self.W_out + self.b_out)

    def forward(self, x):
        """Input pattern to output pattern."""
        h = self.hopfield.forward(self.encode(x))
        return self.decode(h)

    def activity_penalty(self, h):
        """Return the population activity penalty and its hidden-state gradient."""
        if self.hidden_target is None or self.sparsity_weight == 0:
            return 0.0, np.zeros_like(h)
        # Target the mean across neurons within each pattern, after settling.
        difference = h.mean(axis=-1, keepdims=True) - self.hidden_target
        loss = 0.5 * self.sparsity_weight * h.shape[-1] * np.sum(difference ** 2)
        gradient = np.broadcast_to(self.sparsity_weight * difference, h.shape)
        return float(loss), gradient

    def train_step(self, x, target):
        """One backprop step on cross-entropy plus optional hidden sparsity loss."""
        # Forward pass.
        h, cache = self.hopfield.forward(self.encode(x), return_cache=True)
        y = self.decode(h)

        # Error at the output, then through the hidden settling dynamics.
        d_out = y - target
        _, d_activity = self.activity_penalty(h)
        d_hidden = d_out @ self.W_out.T + d_activity
        d_drive, d_recurrent = self.hopfield.backward(d_hidden, cache)

        # Move each weight a small step to reduce the error.
        self.W_out -= self.lr * np.outer(h, d_out)
        self.b_out -= self.lr * d_out
        self.W -= self.lr * np.outer(x, d_drive)
        self.b -= self.lr * d_drive
        self.hopfield.W_R -= self.lr * d_recurrent

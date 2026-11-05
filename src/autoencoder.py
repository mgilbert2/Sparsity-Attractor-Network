"""Three-layer autoencoder: input -W-> hidden -W'-> output.

encode and decode are kept separate so a recurrent hidden stage
(the Hopfield net, W_R) can later sit between them.
"""

import numpy as np

from src.sparsity import k_winners_take_all, n_active


def sigmoid(z):
    """Squash any value into the range 0 to 1."""
    return 1 / (1 + np.exp(-z))


class Autoencoder:
    def __init__(self, n_in, n_hidden, lr=0.2, sparsity=1.0, seed=0):
        rng = np.random.default_rng(seed)
        # Input to hidden weights (W in the diagram) and hidden biases.
        self.W = rng.normal(0, 0.1, (n_in, n_hidden))
        self.b = np.zeros(n_hidden)
        # Hidden to output weights (W' in the diagram) and output biases.
        self.W_out = rng.normal(0, 0.1, (n_hidden, n_in))
        self.b_out = np.zeros(n_in)
        self.lr = lr  # learning rate
        # How many hidden units may be active on each pattern.
        # sparsity is the fraction active, so 1.0 means no constraint.
        self.k = n_active(n_hidden, sparsity)

    def encode(self, x):
        """Input pattern to hidden activations, with only k units active."""
        h = sigmoid(x @ self.W + self.b)
        return k_winners_take_all(h, self.k)

    def decode(self, h):
        """Hidden activations to output pattern."""
        return sigmoid(h @ self.W_out + self.b_out)

    def forward(self, x):
        """Input pattern to output pattern."""
        return self.decode(self.encode(x))

    def train_step(self, x, target):
        """One backprop step on a single pattern, cross-entropy loss."""
        # Forward pass.
        h = self.encode(x)
        y = self.decode(h)

        # Error at the output, then passed back to the hidden layer.
        d_out = y - target
        # Losing hidden units have h = 0, so h * (1 - h) is 0 for them
        # and only the winners' incoming weights change.
        d_hid = (d_out @ self.W_out.T) * h * (1 - h)

        # Move each weight a small step to reduce the error.
        self.W_out -= self.lr * np.outer(h, d_out)
        self.b_out -= self.lr * d_out
        self.W -= self.lr * np.outer(x, d_hid)
        self.b -= self.lr * d_hid

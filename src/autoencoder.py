"""Three-layer autoencoder: input -W-> hidden -W'-> output.

encode and decode are kept separate so a recurrent hidden stage
(the Hopfield net, W_R) can later sit between them.
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Autoencoder:
    def __init__(self, n_in, n_hidden, lr=0.2, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (n_in, n_hidden))
        self.b = np.zeros(n_hidden)
        self.W_out = rng.normal(0, 0.1, (n_hidden, n_in))
        self.b_out = np.zeros(n_in)
        self.lr = lr

    def encode(self, x):
        return sigmoid(x @ self.W + self.b)

    def decode(self, h):
        return sigmoid(h @ self.W_out + self.b_out)

    def forward(self, x):
        return self.decode(self.encode(x))

    def train_step(self, x, target):
        """One backprop step on a single pattern, cross-entropy loss."""
        h = self.encode(x)
        y = self.decode(h)
        d_out = y - target
        d_hid = (d_out @ self.W_out.T) * h * (1 - h)
        self.W_out -= self.lr * np.outer(h, d_out)
        self.b_out -= self.lr * d_out
        self.W -= self.lr * np.outer(x, d_hid)
        self.b -= self.lr * d_hid

"""AB/AC paired-associate patterns laid out as [A | B/C | L]."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    n_a: int = 20
    n_bc: int = 20
    n_l: int = 10
    k_active: int = 4  # active units per A, B, or C item

    @property
    def size(self):
        return self.n_a + self.n_bc + self.n_l

    @property
    def a(self):
        return slice(0, self.n_a)

    @property
    def bc(self):
        return slice(self.n_a, self.n_a + self.n_bc)

    @property
    def l(self):
        return slice(self.n_a + self.n_bc, self.size)


@dataclass
class Task:
    layout: Layout
    A: np.ndarray  # (n_pairs, n_a)
    B: np.ndarray  # (n_pairs, n_bc)
    C: np.ndarray  # (n_pairs, n_bc)
    L: dict  # list name -> context vector (n_l,)

    @property
    def n_pairs(self):
        return len(self.A)

    def targets(self, list_name):
        return self.B if list_name == "AB" else self.C

    def full(self, list_name):
        """Complete patterns A, B/C, L for one list."""
        n = self.n_pairs
        L = np.tile(self.L[list_name], (n, 1))
        return np.hstack([self.A, self.targets(list_name), L])

    def probe(self, list_name):
        """Recall probes: A and L present, B/C slot blank."""
        x = self.full(list_name)
        x[:, self.layout.bc] = 0
        return x


def sparse(rng, n_items, n_units, k):
    x = np.zeros((n_items, n_units))
    for row in x:
        row[rng.choice(n_units, k, replace=False)] = 1
    return x


def make_task(n_pairs=8, layout=Layout(), seed=0):
    rng = np.random.default_rng(seed)
    half = layout.n_l // 2
    L_ab = np.zeros(layout.n_l)
    L_ab[:half] = 1
    L_ac = 1 - L_ab
    return Task(
        layout=layout,
        A=sparse(rng, n_pairs, layout.n_a, layout.k_active),
        B=sparse(rng, n_pairs, layout.n_bc, layout.k_active),
        C=sparse(rng, n_pairs, layout.n_bc, layout.k_active),
        L={"AB": L_ab, "AC": L_ac},
    )

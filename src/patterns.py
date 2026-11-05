"""AB/AC paired-associate patterns laid out as [A | B/C | L]."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    """Sizes of the three parts of the input and output layers."""

    n_a: int = 20  # units for the cue item A
    n_bc: int = 20  # units for the associate item B or C
    n_l: int = 10  # units for the list context
    k_active: int = 4  # active units per A, B, or C item

    @property
    def size(self):
        """Total number of units in the layer."""
        return self.n_a + self.n_bc + self.n_l

    # The slices below pick out each part of a full pattern vector.

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
    """The items for one AB/AC experiment. Row i of A, B, and C is pair i."""

    layout: Layout
    A: np.ndarray  # (n_pairs, n_a)
    B: np.ndarray  # (n_pairs, n_bc)
    C: np.ndarray  # (n_pairs, n_bc)
    L: dict  # list name -> context vector (n_l,)

    @property
    def n_pairs(self):
        return len(self.A)

    def targets(self, list_name):
        """The associates for a list: B for AB, C for AC."""
        return self.B if list_name == "AB" else self.C

    def full(self, list_name):
        """Complete patterns A, B/C, L for one list."""
        n = self.n_pairs
        # Every pair on a list shares the same context vector.
        L = np.tile(self.L[list_name], (n, 1))
        return np.hstack([self.A, self.targets(list_name), L])

    def probe(self, list_name):
        """Recall probes: A and L present, B/C slot blank."""
        x = self.full(list_name)
        x[:, self.layout.bc] = 0
        return x


def sparse(rng, n_items, n_units, k):
    """Random binary patterns, each with exactly k of n_units active."""
    x = np.zeros((n_items, n_units))
    for row in x:
        # Turn on k distinct units in this pattern.
        row[rng.choice(n_units, k, replace=False)] = 1
    return x


def make_task(n_pairs=8, layout=Layout(), seed=0):
    """Random A, B, and C items plus two list contexts.

    The AB context is the first half of the L units on, and the AC
    context is the second half on, so the two do not overlap.
    """
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


def show_task(task):
    """Print each list's patterns, one row per pair, as A | B/C | L."""

    def bits(vector):
        # Turn a 0/1 vector into a string like "0010...".
        return "".join(str(int(value)) for value in vector)

    layout = task.layout
    for list_name in ("AB", "AC"):
        print(f"{list_name} list (A | {list_name[1]} | L)")
        for pattern in task.full(list_name):
            a = bits(pattern[layout.a])
            bc = bits(pattern[layout.bc])
            l = bits(pattern[layout.l])
            print(f"  {a} | {bc} | {l}")
        print()


if __name__ == "__main__":
    show_task(make_task())

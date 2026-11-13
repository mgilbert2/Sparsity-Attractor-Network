"""Checks for hard sparsity and its integration with Hopfield settling."""

import unittest

import numpy as np

from src.autoencoder import Autoencoder
from src.sparsity import k_winners_take_all, n_active
from src.sweep import sweep


class SparsityTests(unittest.TestCase):
    """Check retained positions, input validation, and measured experiment output."""

    def test_winners_and_mask_for_vectors_and_batches(self):
        """Retain the largest activations independently in each pattern."""
        activations = np.array([[0.2, 0.9, 0.5, 0.1], [0.8, 0.2, 0.7, 0.3]])
        original = activations.copy()
        expected = np.array([[0, 0.9, 0.5, 0], [0.8, 0, 0.7, 0]])
        sparse, mask = k_winners_take_all(activations, 2, return_mask=True)
        np.testing.assert_array_equal(sparse, expected)
        np.testing.assert_array_equal(mask, expected > 0)
        np.testing.assert_array_equal(activations, original)
        for row, wanted in zip(activations, expected):
            np.testing.assert_array_equal(k_winners_take_all(row, 2), wanted)

    def test_counts_and_invalid_controls(self):
        """Retained counts must stay valid, including fractions near zero."""
        self.assertEqual(n_active(40, 0.1), 4)
        self.assertEqual(n_active(40, 0), 1)
        self.assertEqual(n_active(40, 1), 40)
        for fraction in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                n_active(40, fraction)
        for k in (0, 5, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                k_winners_take_all(np.ones(4), k)

    def test_gate_applies_after_settling(self):
        """The decoder must receive k winners while recurrent states stay continuous."""
        inputs = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        for steps in (0, 3):
            net = Autoencoder(3, 5, sparsity=0.4, hopfield_steps=steps)
            settled = net.hopfield.forward(net.encode(inputs))
            self.assertTrue(np.all(settled > 0))
            code = net.hidden_state(inputs)
            np.testing.assert_array_equal((code > 0).sum(axis=-1), [2, 2])
            np.testing.assert_allclose(code, k_winners_take_all(settled, 2))
            np.testing.assert_allclose(net.forward(inputs), net.decode(code))

    def test_sweep_records_hard_sparsity_and_gated_activity(self):
        """Experiment metadata and activity must describe the code used for recall."""
        rows = list(sweep(targets=(None,), seeds=(0,), hopfield_steps=(2,),
                          n_pairs=2, n_hidden=5, epochs_ab=1, epochs_ac=1,
                          sparsity=0.4))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sparsity"], 0.4)
        self.assertEqual(rows[0]["k_active"], 2)
        self.assertLessEqual(rows[0]["mean_hidden"], 0.4)
        self.assertLessEqual(rows[0]["fraction_above_half"], 0.4)


if __name__ == "__main__":
    unittest.main()

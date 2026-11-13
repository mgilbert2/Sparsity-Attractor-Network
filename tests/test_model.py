"""Numerical checks for Hopfield dynamics and end-to-end training."""

import unittest

import numpy as np

from src.abac import run
from src.autoencoder import Autoencoder
from src.hopfield import Hopfield, sigmoid
from src.sparsity import k_winners_take_all


def reconstruction_loss(net, x, target):
    """Measure cross-entropy plus the optional hidden population penalty."""
    hidden = net.hidden_state(x)
    output = net.decode(hidden)
    penalty, _ = net.activity_penalty(hidden)
    return -np.sum(target * np.log(output)
                   + (1 - target) * np.log1p(-output)) + penalty


def numerical_gradient(parameter, loss):
    """Estimate derivatives by perturbing each parameter independently."""
    gradient = np.zeros_like(parameter)
    epsilon = 1e-5
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        parameter[index] = original + epsilon
        loss_plus = loss()
        parameter[index] = original - epsilon
        loss_minus = loss()
        parameter[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)
    return gradient


class HopfieldTests(unittest.TestCase):
    """Check energy descent, batched gradients, and numerical stability."""

    def test_each_update_lowers_energy(self):
        """Asynchronous updates must lower the continuous Hopfield energy."""
        layer = Hopfield(4, steps=4, seed=1)
        layer.W_R *= 10
        drive = np.array([[0.2, -0.7, 0.4, 0.1], [-0.3, 0.8, 0.2, -0.5]])
        state, (_, updates) = layer.forward(drive, return_cache=True)

        def energy(hidden):
            """Evaluate recurrent, external-drive, and sigmoid energy terms."""
            entropy = (hidden * np.log(hidden)
                       + (1 - hidden) * np.log1p(-hidden))
            return (-0.5 * np.sum(hidden * (hidden @ layer.W_R), axis=-1)
                    - np.sum(drive * hidden, axis=-1) + entropy.sum(axis=-1))

        states = [update[1] for update in updates] + [state]
        energies = np.array([energy(hidden) for hidden in states])
        self.assertTrue(np.all(np.diff(energies, axis=0) <= 1e-12))
        self.assertTrue(np.any(energies[-1] < energies[0] - 1e-6))

    def test_batched_forward_and_drive_gradient(self):
        """Batch rows must settle independently and have correct derivatives."""
        layer = Hopfield(3, steps=3, seed=2)
        layer.W_R *= 5
        drive = np.array([[0.2, -0.4, 0.6], [-0.8, 0.3, -0.1]])
        upstream = np.array([[0.7, -0.5, 0.4], [-0.6, 0.2, 0.9]])
        state, cache = layer.forward(drive, return_cache=True)
        expected = np.stack([layer.forward(row) for row in drive])
        np.testing.assert_allclose(state, expected, atol=1e-12)
        gradient_drive, _ = layer.backward(upstream, cache)
        numerical = numerical_gradient(
            drive, lambda: np.sum(layer.forward(drive) * upstream))
        np.testing.assert_allclose(gradient_drive, numerical, atol=1e-8)

    def test_invalid_steps(self):
        """Reject sweep counts that cannot describe a fixed number of sweeps."""
        for steps in (-1, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                Hopfield(3, steps=steps)

    def test_extreme_drives_are_finite(self):
        """Saturated sigmoid activations must not cause overflow warnings."""
        with np.errstate(over="raise", invalid="raise"):
            state = Hopfield(3).forward(np.array([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(np.all((state >= 0) & (state <= 1)))


class AutoencoderTests(unittest.TestCase):
    """Check training through recurrence and the original model baseline."""

    def test_all_training_gradients(self):
        """Compare encoder, decoder, and symmetric recurrent updates to differences."""
        self.check_training_gradients(hidden_target=None)

    def test_sparse_training_gradients(self):
        """The hidden activity penalty must backpropagate through every connection."""
        self.check_training_gradients(hidden_target=0.1)

    def test_kwta_training_gradients(self):
        """Gradients through the k-winner readout must match finite differences."""
        self.check_training_gradients(hidden_target=None, sparsity=2 / 3)

    def test_kwta_with_activity_penalty_gradients(self):
        """Hard gating and the soft activity objective must compose correctly."""
        self.check_training_gradients(hidden_target=0.1, sparsity=2 / 3)

    def check_training_gradients(self, hidden_target, sparsity=1.0):
        """Compare a complete training step with numerical objective derivatives."""
        net = Autoencoder(4, 3, lr=0.1, seed=3, hopfield_steps=2,
                          hidden_target=hidden_target, sparsity=sparsity)
        net.hopfield.W_R *= 5
        x = np.array([1.0, 0.0, 1.0, 0.0])
        target = np.array([0.0, 1.0, 1.0, 0.0])
        parameters = [net.W, net.b, net.W_out, net.b_out, net.hopfield.W_R]
        originals = [parameter.copy() for parameter in parameters]

        def loss():
            """Evaluate the objective at the model's current weights."""
            return reconstruction_loss(net, x, target)

        expected = [numerical_gradient(parameter, loss)
                    for parameter in parameters[:-1]]
        # Perturb both directions of a connection to respect Hopfield symmetry.
        recurrent = net.hopfield.W_R
        expected_recurrent = np.zeros_like(recurrent)
        epsilon = 1e-5
        for i in range(len(recurrent)):
            for j in range(i + 1, len(recurrent)):
                original = recurrent[i, j]
                recurrent[i, j] = recurrent[j, i] = original + epsilon
                loss_plus = loss()
                recurrent[i, j] = recurrent[j, i] = original - epsilon
                loss_minus = loss()
                recurrent[i, j] = recurrent[j, i] = original
                # One shared connection occupies two entries in the matrix.
                derivative = (loss_plus - loss_minus) / (4 * epsilon)
                expected_recurrent[i, j] = expected_recurrent[j, i] = derivative
        expected.append(expected_recurrent)

        loss_before = loss()
        net.train_step(x, target)
        for parameter, original, gradient in zip(parameters, originals, expected):
            np.testing.assert_allclose((original - parameter) / net.lr,
                                       gradient, atol=1e-7, rtol=1e-5)
        self.assertLess(loss(), loss_before)
        np.testing.assert_array_equal(recurrent, recurrent.T)
        np.testing.assert_array_equal(np.diag(recurrent), np.zeros(3))
        self.assertGreater(np.linalg.norm(recurrent - originals[-1]), 0)

    def test_population_penalty_gradient(self):
        """Check the activity objective independently, including batched patterns."""
        net = Autoencoder(3, 3, hidden_target=0.1, sparsity_weight=2)
        hidden = np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.2]])
        _, gradient = net.activity_penalty(hidden)
        numerical = numerical_gradient(hidden, lambda: net.activity_penalty(hidden)[0])
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)
        loss, gradient = net.activity_penalty(np.full((2, 3), 0.1))
        self.assertAlmostEqual(loss, 0)
        np.testing.assert_allclose(gradient, 0, atol=1e-12)

    def test_activity_target_lowers_hidden_activity(self):
        """Training with a low target must change measured hidden activity."""
        dense = Autoencoder(3, 4, lr=0.05, seed=7, hopfield_steps=2)
        sparse = Autoencoder(3, 4, lr=0.05, seed=7, hopfield_steps=2,
                             hidden_target=0.1, sparsity_weight=2)
        pattern = np.ones(3)
        for _ in range(100):
            dense.train_step(pattern, pattern)
            sparse.train_step(pattern, pattern)
        dense_activity = dense.hopfield.forward(dense.encode(pattern)).mean()
        sparse_activity = sparse.hopfield.forward(sparse.encode(pattern)).mean()
        self.assertLess(sparse_activity, dense_activity - 0.05)

    def test_zero_steps_matches_original_model(self):
        """Disabling recurrence must preserve the old forward and training rules."""
        self.check_feedforward_equivalence(sparsity=1.0)

    def test_zero_steps_matches_kwta_model(self):
        """Disabling recurrence must preserve the remote k-winner model's rules."""
        self.check_feedforward_equivalence(sparsity=2 / 3)

    def check_feedforward_equivalence(self, sparsity):
        """Compare a forward pass and weight update to the feedforward equations."""
        net = Autoencoder(4, 3, lr=0.2, seed=4, hopfield_steps=0, sparsity=sparsity)
        x = np.array([1.0, 0.0, 0.0, 1.0])
        target = np.array([1.0, 1.0, 0.0, 0.0])
        recurrent_before = net.hopfield.W_R.copy()
        hidden = k_winners_take_all(sigmoid(x @ net.W + net.b), net.k)
        output = sigmoid(hidden @ net.W_out + net.b_out)
        np.testing.assert_allclose(net.forward(x), output, atol=1e-12)
        d_output = output - target
        d_hidden = (d_output @ net.W_out.T) * hidden * (1 - hidden)
        expected = [net.W - net.lr * np.outer(x, d_hidden),
                    net.b - net.lr * d_hidden,
                    net.W_out - net.lr * np.outer(hidden, d_output),
                    net.b_out - net.lr * d_output]
        net.train_step(x, target)
        for actual, wanted in zip([net.W, net.b, net.W_out, net.b_out], expected):
            np.testing.assert_allclose(actual, wanted, atol=1e-12)
        np.testing.assert_array_equal(net.hopfield.W_R, recurrent_before)

    def test_abac_training_and_batched_recall(self):
        """Exercise both training phases and batched probes with recurrent weights."""
        net, task, history = run(n_pairs=2, n_hidden=5, epochs_ab=2, epochs_ac=2)
        self.assertEqual([entry[0] for entry in history], ["AB", "AB", "AC", "AC"])
        for _, ab, ac in history:
            self.assertTrue(0 <= ab <= 1 and 0 <= ac <= 1)
        for list_name in ("AB", "AC"):
            probes = task.probe(list_name)
            expected = np.stack([net.forward(probe) for probe in probes])
            np.testing.assert_allclose(net.forward(probes), expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

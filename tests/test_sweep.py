"""Checks for paired sparsity experiments and saved results."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.sweep import METRICS, summarize, sweep


class SweepTests(unittest.TestCase):
    """Verify controls, summary statistics, and the command-line output files."""

    def test_zero_weight_preserves_matched_control(self):
        """Changing only an inactive target must preserve tasks and training."""
        rows = list(sweep(targets=(None, 0.1), seeds=(3,), hopfield_steps=(0, 2),
                          n_pairs=2, n_hidden=4, epochs_ab=2, epochs_ac=2,
                          sparsity_weight=0))
        self.assertEqual(len(rows), 4)
        for first, second in ((rows[0], rows[1]), (rows[2], rows[3])):
            for metric in METRICS:
                self.assertEqual(first[metric], second[metric])
            self.assertAlmostEqual(first["ab_forgetting"],
                                   first["ab_after_ab"] - first["ab_after_ac"])

    def test_summary_uses_sample_standard_deviation(self):
        """Summary means and deviations must agree with a hand-calculated example."""
        rows = []
        for seed, value in enumerate((0.25, 0.75)):
            row = {metric: value for metric in METRICS}
            row.update(hopfield_steps=3, hidden_target=0.1, seed=seed)
            rows.append(row)
        summary = summarize(rows)[0]
        self.assertEqual(summary["n_seeds"], 2)
        self.assertAlmostEqual(summary["ab_after_ac_mean"], 0.5)
        self.assertAlmostEqual(summary["ab_after_ac_sd"], np.sqrt(0.125))

    def test_cli_saves_raw_and_summary_results(self):
        """A small sweep must save one raw row and one summary per setting."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep.csv"
            completed = subprocess.run(
                [sys.executable, "-m", "src.sweep", "--targets", "none", "0.1",
                 "--seeds", "0", "--hopfield-steps", "0", "--hidden", "4",
                 "--pairs", "2", "--epochs-ab", "1", "--epochs-ac", "1",
                 "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with output.open(newline="", encoding="utf-8") as handle:
                raw = list(csv.DictReader(handle))
            with output.with_name("sweep_summary.csv").open(
                    newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(raw), 2)
            self.assertEqual(raw[0]["hidden_target"], "")
            self.assertEqual(raw[1]["hidden_target"], "0.1")
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary[0]["n_seeds"], "1")
            self.assertEqual(float(summary[0]["ab_after_ac_sd"]), 0)


if __name__ == "__main__":
    unittest.main()

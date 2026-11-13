"""Plot recall and measured hidden activity from a sparsity sweep."""

import numpy as np

from src.sweep import target_label


def plot_summary(summary, path):
    """Compare model settings with bars showing mean and sample SD across seeds."""
    import matplotlib.pyplot as plt

    steps = list(dict.fromkeys(row["hopfield_steps"] for row in summary))
    targets = list(dict.fromkeys(row["hidden_target"] for row in summary))
    lookup = {(row["hopfield_steps"], row["hidden_target"]): row for row in summary}
    panels = (("ab_after_ab", "AB after AB training"),
              ("ab_after_ac", "AB retained after AC training"),
              ("ac_after_ac", "AC after AC training"),
              ("mean_hidden", "Measured mean hidden activity (final probes)"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    positions = np.arange(len(targets))
    width = 0.8 / len(steps)
    for axis, (metric, title) in zip(axes.flat, panels):
        for index, step_count in enumerate(steps):
            group = [lookup[step_count, target] for target in targets]
            means = [row[metric + "_mean"] for row in group]
            deviations = [row[metric + "_sd"] for row in group]
            offset = (index - (len(steps) - 1) / 2) * width
            label = "Feedforward" if step_count == 0 else f"Hopfield ({step_count} sweeps)"
            axis.bar(positions + offset, means, width=width, yerr=deviations,
                     capsize=2, label=label)
        axis.set_title(title)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xticks(positions, [target_label(target) for target in targets])
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Requested mean activity target (none = no penalty)")
    fig.suptitle("Hidden sparsity sweep: mean +/- sample SD across seeds")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

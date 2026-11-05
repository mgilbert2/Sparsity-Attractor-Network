"""AB then AC training with recall tests after every epoch."""

import argparse
from pathlib import Path

import numpy as np

from src.autoencoder import Autoencoder
from src.patterns import Layout, make_task


def recall(net, task, list_name):
    """Fraction of probes whose B/C output is nearest the correct item.

    Candidates are all B and C items, so a wrong-list answer counts
    as an error.
    """
    # The network's B/C output for each probe (A and L given, B/C blank).
    y = net.forward(task.probe(list_name))[:, task.layout.bc]

    # Squared distance from each output to each of the B and C items.
    candidates = np.vstack([task.B, task.C])
    dist = ((y[:, None, :] - candidates[None, :, :]) ** 2).sum(-1)

    # The correct answer for pair i is B_i, or C_i on the AC list.
    correct = np.arange(task.n_pairs)
    if list_name == "AC":
        correct = correct + task.n_pairs

    # Count a pair as recalled if its nearest candidate is the correct one.
    return float((dist.argmin(1) == correct).mean())


def train_epoch(net, task, list_name, rng, with_probes=True):
    """One pass through a list, one backprop step per pattern, shuffled."""
    full = task.full(list_name)
    inputs = [full]
    # Optionally also train on blank-slot probes, with the full pattern
    # as the target, so the network practices filling in B/C.
    if with_probes:
        inputs.append(task.probe(list_name))
    x = np.vstack(inputs)
    t = np.vstack([full] * len(inputs))
    for i in rng.permutation(len(x)):
        net.train_step(x[i], t[i])


def run(n_pairs=8, n_hidden=40, lr=0.2, sparsity=1.0, epochs_ab=100,
        epochs_ac=100, with_probes=True, seed=0):
    """Train on AB, then AC, testing recall on both lists after each epoch.

    Returns the network, the task, and a history of
    (phase, AB recall, AC recall), one entry per epoch.
    """
    task = make_task(n_pairs, Layout(), seed)
    net = Autoencoder(task.layout.size, n_hidden, lr, sparsity, seed)
    rng = np.random.default_rng(seed)
    history = []
    for phase, n_epochs in (("AB", epochs_ab), ("AC", epochs_ac)):
        for _ in range(n_epochs):
            train_epoch(net, task, phase, rng, with_probes)
            history.append((phase, recall(net, task, "AB"),
                            recall(net, task, "AC")))
    return net, task, history


def print_history(history, every):
    """Print recall for the first epoch and every Nth epoch after."""
    print("epoch  phase  AB     AC")
    for i, (phase, ab, ac) in enumerate(history, 1):
        if i % every == 0 or i == 1:
            print(f"{i:5d}  {phase:5s}  {ab:.2f}   {ac:.2f}")


def plot_history(history, path):
    """Save AB and AC recall by epoch, with a line at the list switch."""
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history) + 1)
    ab = [h[1] for h in history]
    ac = [h[2] for h in history]
    switch = sum(1 for h in history if h[0] == "AB")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(epochs, ab, color="black", label="AB")
    ax.plot(epochs, ac, color="black", linestyle="--", label="AC")
    ax.axvline(switch + 0.5, color="gray", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    """Command-line entry point: parse flags, run, print, and plot."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--hidden", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--sparsity", type=float, default=1.0,
                   help="fraction of hidden units active (1.0 = dense)")
    p.add_argument("--epochs-ab", type=int, default=100)
    p.add_argument("--epochs-ac", type=int, default=100)
    p.add_argument("--no-probes", action="store_true",
                   help="train on full patterns only (pure autoencoder)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=10,
                   help="print every N epochs")
    p.add_argument("--plot", type=Path, default=None,
                   help="save a recall curve to this path")
    args = p.parse_args()
    net, _, history = run(args.pairs, args.hidden, args.lr, args.sparsity,
                          args.epochs_ab, args.epochs_ac, not args.no_probes,
                          args.seed)
    print(f"{net.k} of {args.hidden} hidden units active\n")
    print_history(history, args.every)
    if args.plot:
        plot_history(history, args.plot)


if __name__ == "__main__":
    main()

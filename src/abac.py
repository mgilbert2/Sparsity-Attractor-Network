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
    y = net.forward(task.probe(list_name))[:, task.layout.bc]
    candidates = np.vstack([task.B, task.C])
    dist = ((y[:, None, :] - candidates[None, :, :]) ** 2).sum(-1)
    correct = np.arange(task.n_pairs)
    if list_name == "AC":
        correct = correct + task.n_pairs
    return float((dist.argmin(1) == correct).mean())


def train_epoch(net, task, list_name, rng, with_probes=True):
    full = task.full(list_name)
    inputs = [full]
    if with_probes:
        inputs.append(task.probe(list_name))
    x = np.vstack(inputs)
    t = np.vstack([full] * len(inputs))
    for i in rng.permutation(len(x)):
        net.train_step(x[i], t[i])


def run(n_pairs=8, n_hidden=40, lr=0.2, epochs_ab=100, epochs_ac=100,
        with_probes=True, seed=0):
    task = make_task(n_pairs, Layout(), seed)
    net = Autoencoder(task.layout.size, n_hidden, lr, seed)
    rng = np.random.default_rng(seed)
    history = []
    for phase, n_epochs in (("AB", epochs_ab), ("AC", epochs_ac)):
        for _ in range(n_epochs):
            train_epoch(net, task, phase, rng, with_probes)
            history.append((phase, recall(net, task, "AB"),
                            recall(net, task, "AC")))
    return net, task, history


def print_history(history, every):
    print("epoch  phase  AB     AC")
    for i, (phase, ab, ac) in enumerate(history, 1):
        if i % every == 0 or i == 1:
            print(f"{i:5d}  {phase:5s}  {ab:.2f}   {ac:.2f}")


def plot_history(history, path):
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--hidden", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.2)
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
    _, _, history = run(args.pairs, args.hidden, args.lr, args.epochs_ab,
                        args.epochs_ac, not args.no_probes, args.seed)
    print_history(history, args.every)
    if args.plot:
        plot_history(history, args.plot)


if __name__ == "__main__":
    main()

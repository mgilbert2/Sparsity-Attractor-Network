"""Compare hidden activity targets with matched AB/AC tasks and random seeds."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.abac import run


METRICS = ("ab_after_ab", "ab_after_ac", "ac_after_ac", "ab_forgetting",
           "mean_hidden", "fraction_above_half")


def sweep(targets=(None, 0.05, 0.1, 0.2, 0.4), seeds=range(5),
          hopfield_steps=(0, 3), n_pairs=8, n_hidden=40, lr=0.2,
          epochs_ab=100, epochs_ac=100, sparsity_weight=1.0, with_probes=True):
    """Yield one result per setting and seed, keeping all other factors fixed."""
    if epochs_ab < 1 or epochs_ac < 1:
        raise ValueError("Both training phases need at least one epoch")
    for steps in hopfield_steps:
        for target in targets:
            for seed in seeds:
                net, task, history = run(
                    n_pairs=n_pairs, n_hidden=n_hidden, lr=lr,
                    epochs_ab=epochs_ab, epochs_ac=epochs_ac,
                    with_probes=with_probes, seed=seed, hopfield_steps=steps,
                    hidden_target=target, sparsity_weight=sparsity_weight)
                # Test activity on the same partial cues used to measure recall.
                probes = np.vstack([task.probe("AB"), task.probe("AC")])
                hidden = net.hopfield.forward(net.encode(probes))
                ab_after_ab = history[epochs_ab - 1][1]
                ab_after_ac, ac_after_ac = history[-1][1:]
                yield {
                    "hopfield_steps": steps, "hidden_target": target, "seed": seed,
                    "sparsity_weight": sparsity_weight, "n_hidden": n_hidden,
                    "n_pairs": n_pairs, "lr": lr, "epochs_ab": epochs_ab,
                    "epochs_ac": epochs_ac, "with_probes": with_probes,
                    "ab_after_ab": ab_after_ab, "ab_after_ac": ab_after_ac,
                    "ac_after_ac": ac_after_ac,
                    "ab_forgetting": ab_after_ab - ab_after_ac,
                    "mean_hidden": float(hidden.mean()),
                    "fraction_above_half": float((hidden > 0.5).mean()),
                }


def summarize(rows):
    """Report mean and sample standard deviation across seeds for each setting."""
    groups = {}
    for row in rows:
        key = (row["hopfield_steps"], row["hidden_target"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (steps, target), group in groups.items():
        result = {"hopfield_steps": steps, "hidden_target": target,
                  "n_seeds": len(group)}
        for metric in METRICS:
            values = np.array([row[metric] for row in group])
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_sd"] = float(values.std(ddof=1)) if len(group) > 1 else 0.0
        summary.append(result)
    return summary


def target_label(target):
    """Distinguish the unconstrained control from a numerical activity target."""
    return "none" if target is None else f"{target:g}"


def parse_target(value):
    """Parse an activity target or the unconstrained 'none' setting."""
    if value.lower() == "none":
        return None
    try:
        target = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("target must be 'none' or a number") from error
    if not 0 <= target <= 1:
        raise argparse.ArgumentTypeError("target must be between 0 and 1")
    return target


def main():
    """Run the sweep, save raw and summary CSV files, and optionally plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=parse_target, nargs="+",
                        default=[None, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--hopfield-steps", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--epochs-ab", type=int, default=100)
    parser.add_argument("--epochs-ac", type=int, default=100)
    parser.add_argument("--sparsity-weight", type=float, default=1.0)
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("figures/hidden_sparsity.csv"))
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    if min(args.epochs_ab, args.epochs_ac, args.pairs, args.hidden) < 1:
        parser.error("epochs, pairs, and hidden units must be positive")
    if min(args.seeds) < 0 or min(args.hopfield_steps) < 0:
        parser.error("seeds and Hopfield steps must be nonnegative")
    if not np.isfinite(args.sparsity_weight) or args.sparsity_weight < 0:
        parser.error("sparsity weight must be finite and nonnegative")
    for values in (args.targets, args.seeds, args.hopfield_steps):
        if len(values) != len(set(values)):
            parser.error("targets, seeds, and Hopfield step settings must be unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    experiments = sweep(
        targets=args.targets, seeds=args.seeds, hopfield_steps=args.hopfield_steps,
        n_pairs=args.pairs, n_hidden=args.hidden, lr=args.lr,
        epochs_ab=args.epochs_ab, epochs_ac=args.epochs_ac,
        sparsity_weight=args.sparsity_weight, with_probes=not args.no_probes)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for row in experiments:
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()  # Preserve completed experiments if a long sweep stops.
            rows.append(row)
            print(f"steps={row['hopfield_steps']} target={target_label(row['hidden_target'])} "
                  f"seed={row['seed']} AB before={row['ab_after_ab']:.3f} "
                  f"AB final={row['ab_after_ac']:.3f} AC final={row['ac_after_ac']:.3f}",
                  flush=True)

    summary = summarize(rows)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("\nAcross seeds (recall is mean +/- sample SD):")
    print("steps target   AB after AB     AB after AC     AC after AC     mean hidden")
    for row in summary:
        print(f"{row['hopfield_steps']:5d} {target_label(row['hidden_target']):>6s}   "
              f"{row['ab_after_ab_mean']:.3f} +/- {row['ab_after_ab_sd']:.3f}   "
              f"{row['ab_after_ac_mean']:.3f} +/- {row['ab_after_ac_sd']:.3f}   "
              f"{row['ac_after_ac_mean']:.3f} +/- {row['ac_after_ac_sd']:.3f}   "
              f"{row['mean_hidden_mean']:.3f}")
    print(f"Saved {args.output} and {summary_path}")
    if args.plot:
        from src.sweep_plot import plot_summary
        plot_summary(summary, args.plot)


if __name__ == "__main__":
    main()

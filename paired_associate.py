"""A small, visual AB/AC paired-associate network.

This is a rate-based teaching model, not a detailed hippocampal circuit.
Sparse A cues and AB/AC context bits feed Hebbian target weights. Target
units then settle through self-excitation and competition.
"""

import argparse
import math
import random
import sys
import time


N_PAIRS = 4
CUE_BITS = 32
CUE_ACTIVE = 5
CONTEXT_BITS = 16
STEPS = 12
SHADES = " .:-=+*#%@"


def sparse_cue(rng):
    cue = [0] * CUE_BITS
    for index in rng.sample(range(CUE_BITS), CUE_ACTIVE):
        cue[index] = 1
    return cue


class Network:
    def __init__(self):
        rng = random.Random(7)
        self.cues = [sparse_cue(rng) for _ in range(N_PAIRS)]
        self.names = [f"B{i + 1}" for i in range(N_PAIRS)] + [
            f"C{i + 1}" for i in range(N_PAIRS)
        ]
        self.contexts = {
            "AB": [1] * 8 + [0] * 8,
            "AC": [0] * 8 + [1] * 8,
            "none": [0] * CONTEXT_BITS,
        }
        self.weights = [[0] * (CUE_BITS + CONTEXT_BITS)
                        for _ in self.names]

        # Present each A-B and A-C pair once. Each active input bit
        # strengthens its connection to the associated target unit.
        for index, cue in enumerate(self.cues):
            self.learn(cue, self.contexts["AB"], index)
            self.learn(cue, self.contexts["AC"], N_PAIRS + index)

    def learn(self, cue, context, target_index):
        for index, active in enumerate(cue + context):
            self.weights[target_index][index] += active

    def drive(self, cue_index, context_name):
        inputs = self.cues[cue_index] + self.contexts[context_name]
        return [sum(weight * active for weight, active in zip(row, inputs))
                for row in self.weights]

    @staticmethod
    def step(state, drive):
        """Leaky recurrent update with self-excitation and global competition."""
        logits = [input_drive + 1.5 * activity
                  for input_drive, activity in zip(drive, state)]
        peak = max(logits)
        exp_values = [math.exp((value - peak) / 1.2) for value in logits]
        total = sum(exp_values)
        competing = [value / total for value in exp_values]
        return [0.65 * old + 0.35 * new
                for old, new in zip(state, competing)]

    def settle(self, cue_index, context_name):
        state = [0.0] * len(self.names)
        drive = self.drive(cue_index, context_name)
        for _ in range(STEPS):
            state = self.step(state, drive)
        peak = max(state)
        return [name for name, value in zip(self.names, state)
                if peak - value < 0.02]


def summary(network):
    with_context = 0
    without_context = 0
    print("AB/AC paired-associate recall after one presentation per pair")
    print("Cue  Context  Expected  Recalled       Without context")
    for index in range(N_PAIRS):
        for context, expected in (("AB", f"B{index + 1}"),
                                  ("AC", f"C{index + 1}")):
            recalled = network.settle(index, context)
            without = network.settle(index, "none")
            with_context += recalled == [expected]
            without_context += without == [expected]
            print(f"A{index + 1:<3} {context:<8} {expected:<9} "
                  f"{','.join(recalled):<14} {','.join(without)}")
    total = 2 * N_PAIRS
    print(f"\nUnambiguous correct recalls: with context {with_context}/{total}; "
          f"without context {without_context}/{total}")


def bitmap(bits, width):
    return ["".join("#" if bit else "." for bit in bits[start:start + width])
            for start in range(0, len(bits), width)]


def frame_text(network, cue_index, context_name, step_number, state):
    cue = network.cues[cue_index]
    context = network.contexts[context_name]
    drive = network.drive(cue_index, context_name)
    cue_rows = bitmap(cue, 8)
    context_rows = bitmap(context, 8)
    grid = [SHADES[min(9, int(value * 10))] for value in state]
    peak = max(state)
    recalled = (",".join(name for name, value in zip(network.names, state)
                         if peak - value < 0.02) if peak > 0.05 else "settling")
    lines = [
        f"A{cue_index + 1} + {context_name}     step {step_number:02d}/{STEPS}",
        "",
        "Clamped input bits (# = on)",
        "A cue        Context",
    ]
    for row in range(4):
        lines.append(f"{cue_rows[row]}     {context_rows[row] if row < 2 else '        '}")
    lines += [
        "",
        "Target unit activity (denser symbol = more active)",
        "B: " + "  ".join(f"{name}:{shade}" for name, shade
                         in zip(network.names[:N_PAIRS], grid[:N_PAIRS])),
        "C: " + "  ".join(f"{name}:{shade}" for name, shade
                         in zip(network.names[N_PAIRS:], grid[N_PAIRS:])),
        "",
        "Unit  Hebbian drive  Activity",
    ]
    for name, input_drive, value in zip(network.names, drive, state):
        bar = "#" * round(value * 20)
        lines.append(f"{name:<5} {input_drive:>6}        {value:5.2f}  {bar:<20}")
    lines += ["", f"Current recall: {recalled}",
              "Rates update by input + feedback + competition."]
    return "\n".join(lines)


def trial_frames(network, cue_index, context_name):
    state = [0.0] * len(network.names)
    drive = network.drive(cue_index, context_name)
    yield frame_text(network, cue_index, context_name, 0, state)
    for step_number in range(1, STEPS + 1):
        state = network.step(state, drive)
        yield frame_text(network, cue_index, context_name, step_number, state)


def show_frames(frames, delay):
    for frame in frames:
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(frame, flush=True)
        time.sleep(delay)


def play(network, delay):
    print("Enter a cue and context, e.g. A1 AB, A1 AC, or A1 none. q quits.")
    while True:
        try:
            command = input("cue context> ").strip().split()
        except EOFError:
            print()
            return
        if not command:
            continue
        if command[0].lower() in ("q", "quit", "exit"):
            return
        if len(command) != 2 or command[0].upper() not in [
                f"A{i + 1}" for i in range(N_PAIRS)] or \
                command[1].upper() not in ("AB", "AC", "NONE"):
            print("Use A1 through A4 followed by AB, AC, or none.")
            continue
        cue_index = int(command[0][1:]) - 1
        context_name = command[1].upper()
        if context_name == "NONE":
            context_name = "none"
        show_frames(trial_frames(network, cue_index, context_name), delay)


def demo_frames(network):
    for context_name in ("AB", "none", "AC"):
        yield from trial_frames(network, 0, context_name)


def save_gif(frames, path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise SystemExit("GIF export needs Pillow: python3 -m pip install pillow") from error

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    images = []
    for frame in frames:
        image = Image.new("RGB", (900, 700), "#101a26")
        ImageDraw.Draw(image).multiline_text(
            (24, 20), frame, font=font, fill="#e6f0f4", spacing=4)
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=180, loop=0, optimize=True)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="summary",
                        choices=("summary", "play", "movie", "gif"))
    parser.add_argument("--delay", type=float, default=0.18,
                        help="seconds between terminal frames")
    parser.add_argument("--output", default="paired_associate.gif",
                        help="GIF path (gif mode)")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    network = Network()
    if args.mode == "summary":
        summary(network)
    elif args.mode == "play":
        play(network, args.delay)
    elif args.mode == "movie":
        show_frames(demo_frames(network), args.delay)
    else:
        save_gif(demo_frames(network), args.output)


if __name__ == "__main__":
    main()

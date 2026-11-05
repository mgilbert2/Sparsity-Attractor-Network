default:
    @just --list

# Show one-line recall results for every AB/AC pair.
sim:
    python3 paired_associate.py summary

# Choose a cue and context, then watch the target layer settle.
play:
    python3 paired_associate.py play

# Animate AB, no-context, and AC recall in the terminal.
movie:
    python3 paired_associate.py movie

# Save the same animation as an animated GIF (requires Pillow).
gif:
    python3 paired_associate.py gif

set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

# Show available commands
default:
    @echo "just run    Train AB then AC and print recall"
    @echo "just plot   Same, and save a recall plot to figures/"
    @echo "just pure   Train on full patterns only"
    @echo "just task   Print the AB and AC patterns"
    @echo "just sweep  Compare hidden sparsity targets across seeds"
    @echo "just test   Check model gradients and experiment outputs"

run *args:
    uv run python -m src.abac {{args}}

plot *args:
    uv run python -m src.abac --plot figures/abac.png {{args}}

pure *args:
    uv run python -m src.abac --no-probes {{args}}

task:
    uv run python -m src.patterns

sweep *args:
    uv run python -m src.sweep {{args}}

test:
    uv run python -m unittest discover -s tests -v

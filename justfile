# Show available commands
default:
    @echo "just run    Train AB then AC and print recall"
    @echo "just plot   Same, and save a recall plot to figures/"
    @echo "just pure   Train on full patterns only"
    @echo "just task   Print the AB and AC patterns"

run *args:
    uv run python -m src.abac {{args}}

plot *args:
    uv run python -m src.abac --plot figures/abac.png {{args}}

pure *args:
    uv run python -m src.abac --no-probes {{args}}

task:
    uv run python -m src.patterns

# Agent guidelines

## Code is read by humans

This is a research project, and the code is meant to be read and understood by people exploring the model. Prefer readability over cleverness or brevity.

- Use descriptive names, even if longer
- Prefer explicit loops and steps over dense one-liners when it helps the reader
- Give all new code brief comments: a one-line docstring for each function and class, and short comments on steps that aren't obvious
- Comments should explain what a step means for the model, not just what the code does
- Keep files small and focused, so each can be read top to bottom
- Avoid abstractions until they are clearly needed

## Project setup

- Python environment and dependencies are managed with uv
- Common commands are in the justfile
- Source is in `src/`

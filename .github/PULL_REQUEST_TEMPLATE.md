## Summary

<!-- Briefly describe what changed and why. -->

## Checks

- [ ] Unit tests pass (`uv run --locked --no-dev --python 3.12 python -m unittest discover -s tests -v`)
- [ ] Lint, format, and type checks pass (`uv run --locked ruff check . && uv run --locked ruff format --check . && uv run --locked mypy src tests`)
- [ ] Benchmark protocol, grid math, or documentation updates are reflected in `README.md` / `MODEL_DESIGN_BRIEF.md` (if applicable)

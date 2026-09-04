# Reproducibility workflow

`.github/workflows/reproducibility-checks.yml` validates the repository's
locked environment and entrypoint surface.

## What it checks

The workflow provisions the locked `pixi` environment, syncs the `uv.lock`
Python dependencies into that environment via `pixi run sync`, and then runs:

- `pixi run check-r`
- `pixi run validate`

`pixi run validate` expands to:

- `uv run python analysis/tools/publication_manifest.py check`
- `uv run python analysis/tools/publication_manifest.py audit-figures`
- `uv run pytest analysis/tests/ -q`

## Local equivalent

Run the same sequence locally when auditing reproducibility:

```bash
pixi install --locked
pixi run sync
pixi run check-r
pixi run validate
```

`pixi run sync` resolves to
`env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=.pixi/envs/default uv sync --locked --inexact`,
so the locked Python dependencies are applied without pruning pixi-managed
packages such as `rpy2`.

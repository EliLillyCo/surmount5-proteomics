"""Allow ``python -m analysis.pathway`` as a CLI entry point."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

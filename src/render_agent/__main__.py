"""Entry point. Kept free of third-party imports so `selfcheck` and `--version` still work, and
report which import is broken, when a dependency is missing or damaged."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["selfcheck"] and len(args) == 1:
        from render_agent.selfcheck import run_selfcheck

        return run_selfcheck(sys.stdout)
    if args == ["--version"]:
        from render_agent import __version__

        sys.stdout.write(f"render-agent {__version__}\n")
        return 0
    from render_agent.cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())

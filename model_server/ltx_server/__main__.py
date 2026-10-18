"""python -m ltx_server: start the local model server."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, Sequence

from ltx_server.config import ConfigError, Settings, load_settings
from ltx_server.logs import configure_logging
from ltx_server.renderer import FakeRenderer, Renderer
from ltx_server.server import ServerApp


def build_renderer(settings: Settings, environ: Mapping[str, str]) -> Renderer:
    if settings.renderer == "fake":
        return FakeRenderer.from_env(environ)
    if settings.render_mode == "resident":
        from ltx_server.resident_renderer import ResidentLtxRenderer

        return ResidentLtxRenderer(settings)
    from ltx_server.ltx_renderer import LtxRenderer  # imports no torch or ltx_* until load()

    return LtxRenderer(settings)


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        sys.stderr.write("usage: python -m ltx_server   (configure with environment variables)\n")
        return 2
    env = os.environ if environ is None else environ
    configure_logging(logging.INFO)
    try:
        settings = load_settings(env)
    except ConfigError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    app = ServerApp(settings, build_renderer(settings, env))

    def announce(port: int) -> None:
        sys.stdout.write(f"listening on {settings.host}:{port}\n")
        sys.stdout.flush()
        if settings.renderer == "fake":
            sys.stderr.write("NOTE: RENDERER=fake. Output files are not playable videos.\n")

    return app.run(on_started=announce)


if __name__ == "__main__":
    sys.exit(main())

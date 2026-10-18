"""ResidentLtxRenderer: RENDER_MODE=resident. Same lifecycle as LtxRenderer, but the binding keeps
every LTX model built in memory between clips (see resident_binding.py)."""

from __future__ import annotations

from ltx_server.ltx_renderer import Binding, LtxRenderer


class ResidentLtxRenderer(LtxRenderer):
    def _make_binding(self) -> Binding:
        from ltx_server.resident_binding import ResidentBinding  # imports no torch or ltx_*

        return ResidentBinding(self._settings)

"""Local model server for the render agent: one clip per request, model loaded once.

The shell in this package uses the Python standard library only. torch and ltx_* are imported
lazily, and only inside ltx_binding.py.
"""

from __future__ import annotations

__version__ = "0.1.0"

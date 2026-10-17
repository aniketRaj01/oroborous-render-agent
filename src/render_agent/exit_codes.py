from __future__ import annotations

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_AUTH = 3
EXIT_UNEXPECTED = 4
EXIT_SYSTEMIC = 5

# `check` reports any Job API failure as 3; `run` uses 3 only for auth errors.
EXIT_API = EXIT_AUTH
